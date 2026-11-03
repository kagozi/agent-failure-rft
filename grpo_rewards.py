"""
Reward functions for the grounding-click GRPO run (Paper 2).

TRL's GRPOTrainer calls each reward function as
    reward_func(prompts=..., completions=..., completion_ids=..., **kwargs)
where **kwargs includes every other column in the training dataset --
confirmed directly against trl/trainer/grpo_trainer.py (2026-11-03), not
assumed. So `ground_truth_bbox` (a dataset column, see
prepare_grpo_dataset.py) arrives as a kwarg automatically; no custom
collation needed.

Two reward functions, following the same format+accuracy split used in
TRL's own GRPO-VLM example (huggingface.co/blog/trl-vlm-alignment): a
model that emits a plausible-but-wrong click should get *some* signal
(format_reward) rather than the same flat zero as a model that emits
garbage, otherwise early training has no gradient to climb before it
ever lands a correct click.
"""
import re
from typing import Dict, List, Optional

CLICK_RE = re.compile(r"click\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)


def _completion_text(completion) -> str:
    # Conversational completions come back as [{"role": "assistant", "content": "..."}]
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    return str(completion)


def _extract_click(text: str) -> Optional[tuple]:
    m = CLICK_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def format_reward(completions: List, **kwargs) -> List[float]:
    """1.0 if the completion contains a parseable click(x, y), else 0.0."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        rewards.append(1.0 if _extract_click(text) is not None else 0.0)
    return rewards


def click_in_bbox_reward(completions: List, ground_truth_bbox: List[List[int]], **kwargs) -> List[float]:
    """1.0 if the predicted click falls inside the ground-truth bbox, else 0.0."""
    rewards = []
    for completion, bbox in zip(completions, ground_truth_bbox):
        text = _completion_text(completion)
        click = _extract_click(text)
        if click is None:
            rewards.append(0.0)
            continue
        x, y = click
        x_min, y_min, x_max, y_max = bbox
        rewards.append(1.0 if (x_min <= x <= x_max and y_min <= y <= y_max) else 0.0)
    return rewards
