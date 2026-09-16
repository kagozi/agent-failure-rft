# agent-failure-rft

Infrastructure for running [OSWorld](https://github.com/xlang-ai/OSWorld) computer-use-agent evaluations on the [National Research Platform](https://nrp.ai) (NRP), logging full per-step trajectories, and classifying failures into a taxonomy.

The agent is powered by an NRP-hosted open-weight LLM (OpenAI-compatible API) rather than a commercial API. The desktop VM OSWorld drives is provisioned via [KubeVirt](https://kubevirt.io) on NRP's shared Kubernetes cluster, instead of OSWorld's default Docker-based provider (which needs a privileged container to run QEMU with `/dev/kvm` passthrough manually — not appropriate on shared multi-tenant infra).

## Repo layout

```
kubevirt_provider/           # custom OSWorld Provider/VMManager, targets NRP via KubeVirt
  KUBEVIRT_GUIDELINE.md         # full setup: build the VM image, apply the integration patch, run
  osworld-integration.patch     # git diff against a fresh OSWorld clone -- wires everything in
  provider.py / manager.py      # implements OSWorld's Provider/VMManager interface
  config.py                     # all provider config via env vars
  Dockerfile.containerdisk      # wraps OSWorld's Ubuntu disk as a KubeVirt containerDisk image

instrumented_agent.py        # wraps OSWorld's PromptAgent, logs full step trajectories as JSONL
failure_taxonomy.py          # LLM-as-judge: classifies failed episodes from trajectory JSONL
verify_taxonomy.py           # blind human-verification pass against the judge's classifications
                              # (percent agreement + Cohen's kappa, per-episode confusion matrix)

pvc.yaml                     # k8s PersistentVolumeClaim for trajectory storage
osworld-job.yaml             # k8s Job manifest for running OSWorld eval on NRP
grpo-training-job.yaml       # skeleton k8s Job for a later RL fine-tuning pass

osworld/                     # NOT tracked here -- clone xlang-ai/OSWorld yourself (see Setup)
```

## Setup

```bash
git clone https://github.com/xlang-ai/OSWorld.git osworld
cd osworld
uv venv --python 3.12 .venv   # must be 3.12, not 3.13 -- see note below
uv pip install -r <(grep -vE '^(torch~=|transformers~=|accelerate$)' requirements.txt)
uv pip install kubernetes

git apply ../kubevirt_provider/osworld-integration.patch
cp -r ../kubevirt_provider desktop_env/providers/kubevirt
cp ../instrumented_agent.py .
```

Then follow `kubevirt_provider/KUBEVIRT_GUIDELINE.md` to build and push the containerDisk VM image (one-time, ~20GB) and configure your `.env` (NRP endpoint/key/model, KubeVirt namespace/image).

**Python must be 3.12.** Python 3.13 enabled `ssl.VERIFY_X509_STRICT` by default, which rejects some real-world cluster certificates (including NRP's) that Go's TLS stack — what `kubectl` uses — doesn't require. This surfaces as an opaque `SSLCertVerificationError` from the `kubernetes` Python client even though `kubectl` itself works fine against the same cluster.

## Running an evaluation

```bash
cd osworld
python run.py \
  --provider_name kubevirt \
  --observation_type screenshot_a11y_tree \
  --model qwen3 --max_tokens 4096 \
  --test_all_meta_path evaluation_examples/test_smoke.json \
  --result_dir ./results/smoke_test
```

Trajectories land as JSONL under `<result_dir>/trajectories/`. `--max_tokens` matters more than it looks — see `KUBEVIRT_GUIDELINE.md` for why reasoning models need a much larger budget than OSWorld's default.

## Failure taxonomy pass

```bash
python failure_taxonomy.py --trajectory_dir trajectories/ --out failures.jsonl
python verify_taxonomy.py --failures failures.jsonl --trajectory_dir trajectories/ --out human_verification.jsonl
python verify_taxonomy.py --report --failures failures.jsonl --out human_verification.jsonl
```

## Current status

See `kubevirt_provider/KUBEVIRT_GUIDELINE.md` for what's confirmed working, what's a known blocker, and what's still unverified against a real run.
