# VLN/Embodied-Agent PhD Research Scaffold

Working infrastructure for Paper 1 (OSWorld failure taxonomy) and a path
into Paper 2 (rule-based RFT / GRPO), targeting NRP compute.

## Repo layout
```
osworld/              # cloned from xlang-ai/OSWorld (git clone yourself; gitignored here)
agent/
  instrumented_agent.py   # wraps OSWorld's PromptAgent, logs full trajectories as JSONL
eval/
  failure_taxonomy.py     # LLM-as-judge failure classification over logged trajectories
k8s/
  pvc.yaml                 # persistent storage for trajectory logs
  osworld-job.yaml          # runs OSWorld eval on NRP (KVM caveat -- read the file's header)
  grpo-training-job.yaml    # skeleton for Paper 2's RFT training (fill in once framework chosen)
kubevirt_provider/
  (custom OSWorld Provider targeting NRP via KubeVirt -- see its own
  KUBEVIRT_GUIDELINE.md for setup; this is what Step 0 below resolved to)
docs/
  (add your reading notes / paper summaries here as you go -- see learning roadmap)
```

## Step 0 — Confirm KVM support on your NRP namespace — RESOLVED 2026-09-11
Confirmed directly against namespace `gai-lina-group`: KubeVirt v1.7.0 is
deployed cluster-wide, the namespace has full CRUD on
`virtualmachineinstances.kubevirt.io`, and a smoke-test VM reached
`Running` in ~16s with confirmed hardware KVM acceleration (`-accel kvm`
in the guest QEMU process, real `/dev/kvm` char device). NRP support's own
suggestion ("Maybe kubevirt") in reply to the KVM question panned out.

This means: **don't chase OSWorld's built-in `docker` provider's
privileged-container-with-raw-`/dev/kvm` approach** — that needs a
privileged pod, which is a different (and on shared infra, more
questionable) ask than what was actually tested and confirmed working.
Use `kubevirt_provider/` instead — a custom `Provider`/`VMManager`
implementation that boots the same OSWorld Ubuntu disk image via KubeVirt
VMIs. See `kubevirt_provider/KUBEVIRT_GUIDELINE.md` for the full setup
(build+push the containerDisk image, drop the directory into your OSWorld
clone, two small patches to register the provider name).

## Step 1 — Local dev loop (before touching k8s)
Get the pipeline working on a single task locally first -- much faster to
debug than iterating through k8s job submissions.

```bash
git clone https://github.com/xlang-ai/OSWorld.git
cd OSWorld
pip install -r requirements.txt

# point OSWorld's agent at your NRP-hosted LLM
export NRP_API_BASE="https://<your-nrp-llm-endpoint>/v1"   # get from NRP LLM API Keys page
export NRP_API_KEY="<your-key>"
export NRP_MODEL="qwen3"     # or gpt-oss, per nrp.ai/llms model catalog

cp ../agent/instrumented_agent.py mm_agents/instrumented_agent.py
# then edit run.py (or write a small run_instrumented.py) to import
# InstrumentedAgent instead of PromptAgent -- same predict() interface,
# so this should be close to a 1-line swap. Start with 1-2 tasks from
# evaluation_examples/ to confirm the trajectory JSONL looks right.
```

For `--provider_name`, use `kubevirt` (see `kubevirt_provider/
KUBEVIRT_GUIDELINE.md` for the one-time setup: build+push the
containerDisk image, `pip install kubernetes`, register the provider).
This talks to your NRP namespace from your laptop via `kubectl
port-forward` under the hood, so no k8s job submission needed yet for this
step -- just `kubectl` pointed at the `nautilus` context, same as it's
already configured.

## Step 2 — Scale to a batch of tasks
Once a single task round-trips correctly (trajectory JSONL written,
episode scored), pick your 40-60 task pilot subset (mix across OSWorld's
categories: OS, Office, Daily, Professional, Workflow -- see
`evaluation_examples/`) and run the batch, still locally if KVM works on
your own machine, or move to Step 3 for NRP.

## Step 3 — Run on NRP
```bash
kubectl create secret generic nrp-llm-credentials \
  --from-literal=NRP_API_BASE=https://<endpoint>/v1 \
  --from-literal=NRP_API_KEY=<key> \
  --from-literal=NRP_MODEL=qwen3

kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/osworld-job.yaml
kubectl logs -f job/osworld-eval-run
```
Pull trajectories off the PVC once the job completes (see NRP's docs on
mounting/copying from a PVC, or add a small pod that mounts it read-only
and lets you `kubectl cp` out).

## Step 4 — Failure taxonomy
```bash
python eval/failure_taxonomy.py \
  --trajectory_dir trajectories/ \
  --out failures.jsonl \
  --judge_model qwen3
```
Then hand-verify ~15-20% against raw trajectories (open the JSONL, read
the steps, agree/disagree with the judge's category) -- report this
agreement rate in the paper; it's what makes the taxonomy credible to
reviewers rather than an unvalidated LLM opinion.

## Step 5 — Paper 2 on-ramp (later)
Once Paper 1's taxonomy identifies your target failure mode (e.g.
grounding failures), curate 100-300 examples from those failures into
training data, pick a GRPO framework (TRL, verl, or OpenRLHF all have
GRPO support), fill in `k8s/grpo-training-job.yaml`, and request a
multi-GPU NRP allocation for the actual training run.

## Learning-roadmap companion
Keep a running note in `docs/` as you read -- one file per paper, 3-sentence
summary (claim / evidence / gap), so by the time you write Paper 1's related
work section you have a ready-made annotated bibliography instead of
re-reading everything from scratch.
