"""
Paper 2 training-data curation: turns GROUNDING_FAILURE episodes (identified
by failure_taxonomy.py) into UI-R1-style (screenshot, wrong_click, correct_target)
examples for a click-lands-in-correct-bbox rule-based reward.

Ground truth for "where should the click have gone" comes from two sources,
in priority order (see ROADMAP.md / conversation 2026-10-31 for why):

  1. self-correction mining (--mode selfcorrect): text-only judge call asking
     whether the agent eventually interacts with the SAME conceptual target
     later in its own trajectory. Free (no vision call), higher-trust when
     found (it's the agent's own real success in the same live episode), but
     only covers episodes where the agent happened to recover.

  2. visual bbox annotation (--mode visual): vision-capable judge call that
     looks at the actual screenshot at the failing step and annotates the
     correct target's bounding box directly. Covers everything the
     self-correction pass misses, at the cost of one vision API call per
     episode and being a model's visual judgment rather than a confirmed
     real interaction.

No full accessibility tree is available for already-collected episodes
(instrumented_agent.py only logged a 2000-char excerpt, and OSWorld's own
native traj.jsonl on the PVC has no a11y field at all) -- confirmed
2026-10-31, see ROADMAP.md. A third option (replay specific steps in a live
OSWorld VM to capture a fresh, untruncated a11y tree) is deliberately not
implemented here; flagged as a future ablation, not needed for the first
training set.

Usage:
    python mine_grounding_examples.py --mode selfcorrect \
        --candidates taxonomy_full/grounding_failure_candidates.jsonl \
        --trajectory_dir taxonomy_full/grounding_trajectories \
        --out taxonomy_full/selfcorrect_results.jsonl --resume

    python mine_grounding_examples.py --mode visual \
        --candidates taxonomy_full/grounding_failure_candidates.jsonl \
        --trajectory_dir taxonomy_full/grounding_trajectories \
        --screenshot_dir taxonomy_full/grounding_screenshots \
        --out taxonomy_full/visual_results.jsonl --resume
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import openai

SELFCORRECT_PROMPT = """You are reviewing an AI agent's trajectory on a desktop computer-use task that FAILED due to a grounding error (a misclick / wrong UI element).

TASK INSTRUCTION:
{instruction}

The judge identified step {failing_step_idx} as where the agent first failed to hit its intended UI target. Judge's rationale:
{rationale}

FULL TRAJECTORY (each step: model reasoning + action taken):
{trajectory_text}

Question: at any step AFTER step {failing_step_idx}, does the agent successfully interact with the SAME conceptual UI element it was originally trying to reach at step {failing_step_idx} (e.g. it finally clicks the right field/button/menu item, confirmed by the response text or a visible state change)? This is NOT asking about different elements for different sub-goals later in the task -- only the same target the original failure was about.

Respond with strict JSON only:
{{
  "self_corrected": true/false,
  "correction_step_idx": <int or -1 if false>,
  "correction_coords": [x, y] or null,
  "reasoning": "<1-2 sentences citing the specific step>"
}}
"""

VISUAL_PROMPT = """This screenshot ({width}x{height} pixels) shows the desktop state right before an AI agent took an action while attempting this task:

TASK: {instruction}

The agent intended to interact with a specific UI element but instead clicked at pixel coordinates {wrong_click}, which did not land on the intended target. Judge's rationale for why this was a grounding failure:
{rationale}

Identify the bounding box of the UI element the agent SHOULD have clicked, based on the task and rationale above. Respond with strict JSON only:
{{
  "found": true/false,
  "bbox": [x_min, y_min, x_max, y_max],
  "reasoning": "<one sentence citing what's visible in the screenshot>"
}}
If you cannot confidently locate the correct element in this screenshot, set found=false.
"""


def load_episode(path: Path) -> Dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    start = next(r for r in records if r["record_type"] == "episode_start")
    steps = [r for r in records if r["record_type"] == "step"]
    return {"start": start, "steps": steps}


def format_trajectory(steps: List[Dict]) -> str:
    lines = []
    for s in steps:
        lines.append(
            f"[step {s['step_idx']}] response: {str(s.get('model_response'))[:600]}\n"
            f"          action: {s.get('parsed_actions')}"
        )
    return "\n".join(lines)


def extract_first_click(parsed_actions) -> Optional[List[int]]:
    """Best-effort pull of the first (x, y) pair out of a pyautogui action string."""
    import re
    if not parsed_actions:
        return None
    text = str(parsed_actions)
    m = re.search(r"click\w*\(\s*(\d+)\s*,\s*(\d+)", text)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return None


def call_judge(client: openai.OpenAI, model: str, messages: List[Dict], max_tokens: int) -> Dict:
    # Timeout=400 / single retry policy calibrated 2026-10-27/28 against this
    # same NRP qwen3 endpoint -- see failure_taxonomy.py's judge_episode for
    # the full story (113s baseline latency, concurrency multiplies it, 180s
    # was not enough headroom).
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=400,
            )
            raw = resp.choices[0].message.content
            if not raw:
                raise ValueError("empty content")
            # qwen3 (reasoning model) sometimes prefixes the JSON with
            # leading whitespace/blank lines even when told to respond with
            # strict JSON only -- strip first, and if that's still not
            # parseable, fall back to slicing between the outermost braces
            # rather than failing outright (confirmed 2026-10-31: 35/133
            # first-pass failures were otherwise-valid JSON lost to this).
            text = raw.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                start, end = text.find("{"), text.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(text[start:end + 1])
                raise
        except json.JSONDecodeError:
            return {"_error": f"JUDGE_PARSE_ERROR: {raw[:300]}"}
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"_error": f"JUDGE_CALL_FAILED: {e}"}
            time.sleep(5 * (attempt + 1))


def run_selfcorrect(args, client):
    candidates = [json.loads(l) for l in open(args.candidates)]
    already = set()
    if args.resume and os.path.exists(args.out):
        already = {json.loads(l)["episode_id"] for l in open(args.out) if l.strip()}
        print(f"Resuming: {len(already)} already done.")
    else:
        open(args.out, "w").close()

    for cand in candidates:
        eid, model = cand["episode_id"], cand["model"]
        if eid in already:
            continue
        traj_path = Path(args.trajectory_dir) / f"{model}__{eid}.jsonl"
        if not traj_path.exists():
            print(f"{eid}: MISSING trajectory file, skipping")
            continue
        episode = load_episode(traj_path)
        prompt = SELFCORRECT_PROMPT.format(
            instruction=episode["start"]["instruction"],
            failing_step_idx=cand.get("failing_step_idx"),
            rationale=cand.get("rationale"),
            trajectory_text=format_trajectory(episode["steps"]),
        )
        result = call_judge(client, args.judge_model, [{"role": "user", "content": prompt}], 8192)
        record = {"episode_id": eid, "model": model, "task_id": cand.get("task_id"), **result}
        with open(args.out, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{eid}: self_corrected={result.get('self_corrected')}")


def run_visual(args, client):
    candidates = [json.loads(l) for l in open(args.candidates)]
    already = set()
    if args.resume and os.path.exists(args.out):
        already = {json.loads(l)["episode_id"] for l in open(args.out) if l.strip()}
        print(f"Resuming: {len(already)} already done.")
    else:
        open(args.out, "w").close()

    for cand in candidates:
        eid, model = cand["episode_id"], cand["model"]
        if eid in already:
            continue
        traj_path = Path(args.trajectory_dir) / f"{model}__{eid}.jsonl"
        if not traj_path.exists():
            print(f"{eid}: MISSING trajectory file, skipping")
            continue
        episode = load_episode(traj_path)
        failing_idx = cand.get("failing_step_idx")
        step = next((s for s in episode["steps"] if s["step_idx"] == failing_idx), None)
        if step is None:
            print(f"{eid}: failing_step_idx {failing_idx} not in trajectory, skipping")
            continue
        wrong_click = extract_first_click(step.get("parsed_actions"))
        screenshot_path = Path(args.screenshot_dir) / f"{model}__{eid}__step{failing_idx}.png"
        if not screenshot_path.exists():
            print(f"{eid}: MISSING screenshot, skipping")
            continue
        with open(screenshot_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        from PIL import Image
        with Image.open(screenshot_path) as im:
            width, height = im.size
        prompt = VISUAL_PROMPT.format(
            width=width, height=height,
            instruction=episode["start"]["instruction"],
            wrong_click=wrong_click or "unknown",
            rationale=cand.get("rationale"),
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]
        result = call_judge(client, args.judge_model, messages, 8192)
        record = {
            "episode_id": eid, "model": model, "task_id": cand.get("task_id"),
            "failing_step_idx": failing_idx, "wrong_click": wrong_click, **result,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{eid}: found={result.get('found')} bbox={result.get('bbox')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selfcorrect", "visual"], required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--trajectory_dir", required=True)
    ap.add_argument("--screenshot_dir", default=None, help="required for --mode visual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_model", default=os.environ.get("NRP_JUDGE_MODEL", "qwen3"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    client = openai.OpenAI(
        base_url=os.environ["NRP_API_BASE"],
        api_key=os.environ["NRP_API_KEY"],
        max_retries=0,
    )

    if args.mode == "selfcorrect":
        run_selfcorrect(args, client)
    else:
        if not args.screenshot_dir:
            raise SystemExit("--screenshot_dir is required for --mode visual")
        run_visual(args, client)


if __name__ == "__main__":
    main()
