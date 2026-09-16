"""
Instrumented wrapper around OSWorld's PromptAgent for failure-taxonomy research.

Wraps mm_agents.agent.PromptAgent (or any OSWorld-compatible agent) and logs,
per step: screenshot path, a11y tree snippet, full model reasoning trace,
parsed action, and env feedback -- as structured JSONL trajectories.

Points at an NRP-hosted OpenAI-compatible endpoint by default (qwen3 or
gpt-oss), configured via environment variables:

    export NRP_API_BASE="https://<your-nrp-llm-endpoint>/v1"
    export NRP_API_KEY="<your NRP LLM API key>"
    export NRP_MODEL="qwen3"        # or "gpt-oss", etc.

Usage (from OSWorld repo root, after `pip install -r requirements.txt`):

    from instrumented_agent import InstrumentedAgent
    agent = InstrumentedAgent(trajectory_dir="trajectories/")
    # then use `agent` exactly like mm_agents.agent.PromptAgent in run.py /
    # lib_run_single.py -- it exposes the same .predict() / .reset() interface.
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# NOTE: run this from inside the cloned OSWorld repo (or add it to PYTHONPATH)
# so this import resolves.
from mm_agents.agent import PromptAgent

# Loads NRP_API_BASE / NRP_API_KEY / NRP_MODEL from a .env file in the cwd
# (or any parent directory) if present, so they don't need to be exported
# in every shell session.
load_dotenv()


class InstrumentedAgent:
    """
    Thin instrumentation layer around PromptAgent. Delegates all actual
    model-calling / action-parsing logic to PromptAgent (so you inherit
    OSWorld's battle-tested prompting), but intercepts every step to write
    a structured trajectory log -- which is the actual research artifact
    for the failure-taxonomy paper.
    """

    def __init__(
        self,
        trajectory_dir: str = "trajectories",
        model: Optional[str] = None,
        observation_type: str = "screenshot_a11y_tree",
        action_space: str = "pyautogui",
        max_trajectory_length: int = 15,
        **prompt_agent_kwargs: Any,
    ):
        self.model = model or os.environ.get("NRP_MODEL", "qwen3")
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        # PromptAgent.call_llm checks NRP_API_BASE directly (see the patch
        # in mm_agents/agent.py) and uses it to bypass its per-brand model
        # dispatch, so both env vars just need to be set -- no mirroring
        # into other names required.
        if not os.environ.get("NRP_API_BASE") or not os.environ.get("NRP_API_KEY"):
            raise RuntimeError(
                "NRP_API_BASE and NRP_API_KEY must both be set before "
                "constructing InstrumentedAgent (e.g. NRP_API_BASE="
                "https://ellm.nrp-nautilus.io/v1)."
            )

        self._inner = PromptAgent(
            model=self.model,
            observation_type=observation_type,
            action_space=action_space,
            max_trajectory_length=max_trajectory_length,
            **prompt_agent_kwargs,
        )
        # run.py reads agent.action_space right after construction (to pass
        # into DesktopEnv), so this needs to be a passthrough attribute, not
        # just held on self._inner.
        self.action_space = self._inner.action_space

        self._episode_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._log_path: Optional[Path] = None
        self._step_idx: int = 0
        self._start_time: Optional[float] = None

    # ---- lifecycle -----------------------------------------------------

    def start_episode(self, task_id: str, instruction: str) -> None:
        """Call once per task, before the first predict()."""
        self._episode_id = f"{task_id}_{uuid.uuid4().hex[:8]}"
        self._task_id = task_id
        self._step_idx = 0
        self._start_time = time.time()
        self._log_path = self.trajectory_dir / f"{self._episode_id}.jsonl"

        self._write_record({
            "record_type": "episode_start",
            "episode_id": self._episode_id,
            "task_id": task_id,
            "instruction": instruction,
            "model": self.model,
            "timestamp": self._start_time,
        })

    def reset(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self._inner, "reset"):
            self._inner.reset(*args, **kwargs)

    # ---- the actual instrumented step -----------------------------------

    def predict(self, instruction: str, obs: Dict[str, Any]) -> Any:
        """
        Mirrors PromptAgent.predict(instruction, obs) -> (response, actions).
        Logs the observation, full model response, and parsed actions.
        """
        step_start = time.time()

        response, actions = self._inner.predict(instruction, obs)

        record = {
            "record_type": "step",
            "episode_id": self._episode_id,
            "step_idx": self._step_idx,
            "elapsed_s": round(time.time() - self._start_time, 2) if self._start_time else None,
            "step_latency_s": round(time.time() - step_start, 2),
            "obs_keys": list(obs.keys()) if isinstance(obs, dict) else None,
            # Truncate a11y tree text in the log to keep files manageable;
            # full screenshots should be saved separately by the OSWorld
            # harness (see run.py) and referenced by step_idx here.
            "a11y_tree_excerpt": (obs.get("accessibility_tree") or "")[:2000]
            if isinstance(obs, dict) else None,
            "model_response": response,
            "parsed_actions": actions,
        }
        self._write_record(record)
        self._step_idx += 1
        return response, actions

    def end_episode(self, result: Dict[str, Any]) -> None:
        """
        Call once after the OSWorld harness scores the episode.
        `result` should include at least {"success": bool} from the
        benchmark's evaluator (env.evaluate() in OSWorld's run.py).
        """
        self._write_record({
            "record_type": "episode_end",
            "episode_id": self._episode_id,
            "num_steps": self._step_idx,
            "total_time_s": round(time.time() - self._start_time, 2) if self._start_time else None,
            **result,
        })

    # ---- utils ------------------------------------------------------------

    def _write_record(self, record: Dict[str, Any]) -> None:
        assert self._log_path is not None, "call start_episode() before logging"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
