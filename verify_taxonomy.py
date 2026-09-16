"""
Human verification pass for failure_taxonomy.py's LLM-judge classifications.

Two modes:

  1. Verify (default) -- blind review. Samples ~20% of judged failures,
     shows you the raw trajectory (instruction + steps), and asks for YOUR
     classification BEFORE revealing what the judge said. Blinding matters:
     seeing the judge's answer first anchors your own judgment and inflates
     the agreement rate you end up reporting.

         python verify_taxonomy.py \
             --failures failures.jsonl --trajectory_dir trajectories/ \
             --out human_verification.jsonl

     Re-running with the same --out resumes -- already-verified episodes
     (by episode_id) are skipped, and the sample itself is derived
     deterministically from --seed so re-runs don't drift to a different
     sample. Safe to do a few episodes per sitting.

  2. Report -- once you've verified some episodes:

         python verify_taxonomy.py --report \
             --failures failures.jsonl --out human_verification.jsonl

     Prints raw percent agreement, Cohen's kappa (corrects for chance
     agreement -- what reviewers actually expect for an LLM-as-judge
     validation, not just percent agreement), a confusion matrix, and the
     specific disagreements to go re-read.

Standard target per the project's methodology: verify 15-20% of judged
failures before reporting agreement in the paper.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from failure_taxonomy import TAXONOMY, format_trajectory, load_episode


def load_failures(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_existing_verifications(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["episode_id"]: r for r in records}


def sample_episode_ids(failures: List[Dict], frac: float, seed: int) -> List[str]:
    # Deterministic given (failures.jsonl contents, frac, seed) -- sort first
    # so re-runs don't depend on filesystem iteration order.
    ids = sorted(f["episode_id"] for f in failures)
    rng = random.Random(seed)
    n = max(1, round(len(ids) * frac))
    return sorted(rng.sample(ids, n))


def prompt_human_label(judge_record: Dict, trajectory_dir: Path) -> Optional[Dict]:
    path = trajectory_dir / f"{judge_record['episode_id']}.jsonl"
    if not path.exists():
        print(f"[skip] no trajectory file for {judge_record['episode_id']} at {path}")
        return None

    episode = load_episode(path)
    print("\n" + "=" * 80)
    print(f"episode_id: {judge_record['episode_id']}  (task: {judge_record['task_id']})")
    print(f"TASK INSTRUCTION: {episode['start']['instruction']}")
    print("-" * 80)
    print(format_trajectory(episode["steps"]))
    print("-" * 80)
    print("Classify the PRIMARY reason this failed -- judge's own answer is")
    print("hidden until after you respond, to avoid anchoring:")
    for i, cat in enumerate(TAXONOMY, 1):
        print(f"  {i}. {cat}")
    print("  0. skip this episode")

    while True:
        raw = input("Your category (number): ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(TAXONOMY):
            human_category = TAXONOMY[int(raw) - 1]
            break
        print("Invalid input, try again.")

    notes = input("Notes (optional, enter to skip): ").strip()

    agree = human_category == judge_record["primary_category"]
    print(f"[judge said: {judge_record['primary_category']} "
          f"(confidence={judge_record['confidence']})]")
    print(f"[judge rationale: {judge_record['rationale']}]")
    print("AGREE" if agree else "DISAGREE")

    return {
        "episode_id": judge_record["episode_id"],
        "task_id": judge_record["task_id"],
        "human_category": human_category,
        "judge_category": judge_record["primary_category"],
        "agree": agree,
        "notes": notes,
    }


def run_verify(args):
    failures = load_failures(Path(args.failures))
    by_id = {f["episode_id"]: f for f in failures}
    out_path = Path(args.out)
    already = load_existing_verifications(out_path)

    sampled_ids = sample_episode_ids(failures, args.sample_frac, args.seed)
    remaining = [eid for eid in sampled_ids if eid not in already]

    print(f"Sample size: {len(sampled_ids)} of {len(failures)} judged failures "
          f"({args.sample_frac:.0%}). Already verified: {len(already)}. "
          f"Remaining this session: {len(remaining)}.")

    if not remaining:
        print("Nothing left to verify in this sample. Run with --report to see results.")
        return

    with open(out_path, "a", encoding="utf-8") as f:
        for eid in remaining:
            result = prompt_human_label(by_id[eid], Path(args.trajectory_dir))
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\nWrote verifications to {out_path}. Run again to continue, "
          f"or pass --report once you have enough coverage.")


def cohens_kappa(pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    n = len(pairs)
    po = sum(1 for j, h in pairs if j == h) / n
    categories = set(j for j, h in pairs) | set(h for j, h in pairs)
    pe = 0.0
    for c in categories:
        pj = sum(1 for j, _ in pairs if j == c) / n
        ph = sum(1 for _, h in pairs if h == c) / n
        pe += pj * ph
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, kappa


def kappa_interpretation(kappa: float) -> str:
    # Landis & Koch (1977) benchmarks -- standard citation for this in papers.
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def run_report(args):
    verified = list(load_existing_verifications(Path(args.out)).values())
    if not verified:
        print(f"No verifications found in {args.out} yet.")
        return

    pairs = [(v["judge_category"], v["human_category"]) for v in verified]
    po, kappa = cohens_kappa(pairs)

    print(f"Verified episodes: {len(verified)}")
    print(f"Raw percent agreement: {po:.1%}")
    print(f"Cohen's kappa: {kappa:.3f} ({kappa_interpretation(kappa)})")

    print("\nConfusion matrix (rows=judge, cols=human):")
    cats = sorted(TAXONOMY)
    cm = Counter(pairs)
    header = "judge\\human".ljust(24) + "".join(c[:10].ljust(12) for c in cats)
    print(header)
    for j in cats:
        row = j.ljust(24) + "".join(str(cm.get((j, h), 0)).ljust(12) for h in cats)
        print(row)

    disagreements = [v for v in verified if not v["agree"]]
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s) to re-read:")
        for d in disagreements:
            print(f"  {d['episode_id']}: judge={d['judge_category']} "
                  f"human={d['human_category']}"
                  + (f"  note: {d['notes']}" if d.get("notes") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True, help="failures.jsonl from failure_taxonomy.py")
    ap.add_argument("--trajectory_dir", default="trajectories",
                    help="only needed in verify mode")
    ap.add_argument("--out", default="human_verification.jsonl")
    ap.add_argument("--sample_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true", help="print agreement stats instead of verifying")
    args = ap.parse_args()

    if args.report:
        run_report(args)
    else:
        run_verify(args)


if __name__ == "__main__":
    main()
