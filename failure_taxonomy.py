"""
LLM-as-judge failure taxonomy for OSWorld trajectories.

Reads the JSONL trajectory logs produced by agent/instrumented_agent.py,
finds failed episodes, and asks a judge model to classify each failure
into a fixed taxonomy -- with a rationale, so you can spot-check /
human-verify a sample (recommended: 15-20% of judged trajectories,
per standard LLM-as-judge methodology).

Usage:
    export NRP_API_BASE="https://<endpoint>/v1"
    export NRP_API_KEY="<key>"
    python failure_taxonomy.py --trajectory_dir trajectories/ --out failures.jsonl

Taxonomy categories (edit as your Paper 1 pilot data suggests -- these are
a reasonable starting point per the OSWorld / OSWorld 2.0 / WindowsWorld
literature, not a fixed standard):

  GROUNDING_FAILURE       - clicked/typed on the wrong UI element
  STATE_TRACKING_FAILURE  - lost/forgot info carried from an earlier step or app
  PREMATURE_TERMINATION   - declared done/gave up before task was complete
  LOOPING                 - repeated the same or similar action without progress
  PLANNING_FAILURE        - wrong high-level approach from early in the episode
  CROSS_APP_HANDOFF       - failure specifically at an app-switch boundary
  TIMEOUT                 - ran out of steps while still making incremental progress
  OTHER                   - doesn't fit above; judge must explain why
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import openai

TAXONOMY = [
    "GROUNDING_FAILURE",
    "STATE_TRACKING_FAILURE",
    "PREMATURE_TERMINATION",
    "LOOPING",
    "PLANNING_FAILURE",
    "CROSS_APP_HANDOFF",
    "TIMEOUT",
    "OTHER",
]

JUDGE_PROMPT_TEMPLATE = """You are an expert annotator for a research study on why AI agents fail at desktop computer-use tasks.

Below is the full step-by-step trajectory of an agent attempting an OSWorld task, which FAILED.

TASK INSTRUCTION:
{instruction}

TRAJECTORY (each step: model reasoning + action taken):
{trajectory_text}

Classify the PRIMARY reason this trajectory failed into exactly one of these categories:
{taxonomy_list}

Respond with strict JSON only, no other text:
{{
  "primary_category": "<one of the categories above>",
  "confidence": "<low|medium|high>",
  "failing_step_idx": <int, the step index where things went wrong, or -1 if unclear>,
  "rationale": "<1-3 sentences explaining your classification, citing specific steps>"
}}
"""


def load_episode(path: Path) -> Dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    start = next(r for r in records if r["record_type"] == "episode_start")
    steps = [r for r in records if r["record_type"] == "step"]
    end = next((r for r in records if r["record_type"] == "episode_end"), None)
    return {"start": start, "steps": steps, "end": end}


def format_trajectory(steps: List[Dict]) -> str:
    lines = []
    for s in steps:
        lines.append(
            f"[step {s['step_idx']}] response: {str(s.get('model_response'))[:600]}\n"
            f"          action: {s.get('parsed_actions')}"
        )
    return "\n".join(lines)


def judge_episode(client: openai.OpenAI, model: str, episode: Dict) -> Dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=episode["start"]["instruction"],
        trajectory_text=format_trajectory(episode["steps"]),
        taxonomy_list="\n".join(f"- {t}" for t in TAXONOMY),
    )
    # max_tokens set explicitly (not left to the API's default): NRP's
    # qwen3 is a reasoning model that puts chain-of-thought in a separate
    # field from the final answer -- confirmed elsewhere in this project
    # that too small a budget produces null content, not just a short
    # response. temperature=0.0 for judge consistency across a re-run.
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
                # Explicit per-request timeout -- without one, the SDK's
                # default can leave a single hung request blocking for a
                # very long time with no error raised at all. Confirmed
                # 2026-09-17: one episode stalled 30+ minutes on an
                # established-but-silent connection before this was added.
                timeout=90,
            )
            raw = resp.choices[0].message.content
            if not raw:
                raise ValueError("empty content (judge model likely exhausted max_tokens on reasoning)")
            break
        except Exception as e:  # noqa: BLE001 -- one bad episode shouldn't kill a 30+ episode batch
            if attempt == 2:
                return {"primary_category": "OTHER", "confidence": "low",
                        "failing_step_idx": -1, "rationale": f"JUDGE_CALL_FAILED: {e}"}
            time.sleep(5 * (attempt + 1))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"primary_category": "OTHER", "confidence": "low",
                   "failing_step_idx": -1, "rationale": f"JUDGE_PARSE_ERROR: {raw[:300]}"}
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_dir", required=True)
    ap.add_argument("--out", default="failures.jsonl")
    ap.add_argument("--judge_model", default=os.environ.get("NRP_JUDGE_MODEL", "qwen3"))
    ap.add_argument("--resume", action="store_true",
                     help="Skip episodes already present in --out instead of truncating it. "
                          "Confirmed 2026-09-17 worth having: a single stuck request can stall "
                          "the whole batch, and re-judging already-done episodes wastes real "
                          "API calls on a 30+ episode run.")
    args = ap.parse_args()

    client = openai.OpenAI(
        base_url=os.environ["NRP_API_BASE"],
        api_key=os.environ["NRP_API_KEY"],
        # The SDK's own internal retry logic (default max_retries=2)
        # operates independently of judge_episode's explicit retry loop --
        # confirmed 2026-09-17 this compounds badly: 3 outer attempts, each
        # potentially retried 2-3 more times internally by the SDK before
        # the 90s per-call timeout even has a chance to surface, stalled a
        # single episode for 20+ minutes. Disabled here so there's exactly
        # one retry policy in effect, the one judge_episode implements.
        max_retries=0,
    )

    traj_dir = Path(args.trajectory_dir)
    out_records = []
    excluded_infra = 0

    already_judged = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_judged.add(json.loads(line)["episode_id"])
        print(f"Resuming: {len(already_judged)} episode(s) already judged, will skip.")
    else:
        # Truncate/create the output file once, then append per-episode
        # below -- a crash partway through a 30+ episode batch (a real risk
        # given NRP's observed intermittent flakiness) shouldn't lose
        # already-judged work within *this* run; --resume handles picking
        # back up across runs.
        open(args.out, "w", encoding="utf-8").close()

    for path in sorted(traj_dir.glob("*.jsonl")):
        episode = load_episode(path)
        if episode["end"] is None:
            continue  # incomplete run, skip
        if episode["start"]["episode_id"] in already_judged:
            continue
        if episode["end"].get("success"):
            continue  # only judge failures
        if episode["end"].get("abort_reason") == "sustained_api_failure":
            # Circuit-breaker abort (instrumented_agent.py) -- the episode
            # never got a real chance to succeed or fail on its own merits,
            # so judging it would mislabel infra noise as a model capability
            # failure. Excluded from judging entirely, not just scored 0.
            excluded_infra += 1
            continue

        judgment = judge_episode(client, args.judge_model, episode)
        record = {
            "episode_id": episode["start"]["episode_id"],
            "task_id": episode["start"]["task_id"],
            "num_steps": episode["end"].get("num_steps"),
            **judgment,
        }
        out_records.append(record)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{episode['start']['episode_id']}: {judgment['primary_category']} "
              f"({judgment['confidence']})")

    if excluded_infra:
        print(f"\nExcluded {excluded_infra} episode(s) aborted by the circuit "
              f"breaker (sustained_api_failure) -- not judged, not counted below.")

    # quick summary
    from collections import Counter
    counts = Counter(r["primary_category"] for r in out_records)
    print("\n--- Failure taxonomy summary ---")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n} ({100*n/len(out_records):.1f}%)")
    print(f"\nWrote {len(out_records)} judged failures to {args.out}")
    print("Reminder: hand-verify ~15-20% of these against the raw trajectories "
          "before reporting inter-rater agreement in the paper.")


if __name__ == "__main__":
    main()
