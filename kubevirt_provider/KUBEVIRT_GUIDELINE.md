# Configuration of KubeVirt (NRP / Nautilus)

Custom OSWorld provider targeting NRP's shared Kubernetes cluster via
KubeVirt, instead of the Docker provider's privileged-container QEMU
approach. Confirmed working (KVM hardware acceleration, full VMI CRUD
permissions) against namespace `gai-lina-group` on 2026-09-11 — see
`../README.md` for how this fits the overall project plan.

## Why KubeVirt instead of the Docker provider

OSWorld's built-in `docker` provider needs a **privileged** container to
run QEMU with `/dev/kvm` passthrough manually. On a shared multi-tenant
research cluster like NRP, privileged containers are a real isolation risk
and reasonably something a cluster operator/support team would want to
gate carefully — untested here on purpose. KubeVirt sidesteps this
entirely: it's already deployed cluster-wide, VMs are a first-class
Kubernetes object (no privileged pod needed from us), and a smoke test
confirmed real KVM acceleration (`-accel kvm` in the guest QEMU process,
not software fallback).

## Step 1 — Build and push the containerDisk image (one-time)

KubeVirt boots VMs from container images with a qcow2 baked in at
`/disk/` ("containerDisk"). See `Dockerfile.containerdisk` in this
directory for the exact format and build commands. This reuses the exact
same Ubuntu.qcow2 OSWorld's own Docker provider downloads — same guest
image, same pre-baked Flask server on :5000 / VNC on :8006 / etc. — so
there's no new guest-side setup to get right, only a different way of
booting it.

```bash
curl -L -o Ubuntu.qcow2.zip \
  https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip Ubuntu.qcow2.zip
docker build -f Dockerfile.containerdisk -t <your-registry>/osworld-ubuntu:latest .
docker push <your-registry>/osworld-ubuntu:latest
export KUBEVIRT_BASE_IMAGE=<your-registry>/osworld-ubuntu:latest
```

The image is large (20+ GiB uncompressed) — this step takes a while and
needs real disk space. Do it once, not per run.

## Step 2 — Install the Python `kubernetes` client

```bash
pip install kubernetes
```

## Step 3 — Drop this directory into your OSWorld clone

```bash
cp -r kubevirt_provider <path-to-OSWorld-clone>/desktop_env/providers/kubevirt
cp instrumented_agent.py <path-to-OSWorld-clone>/instrumented_agent.py   # or symlink
```

## Step 4 — Apply the integration patch

`osworld-integration.patch` (in this directory) is a real `git diff` taken
against a fresh `xlang-ai/OSWorld` clone, covering everything needed to
wire the provider and the instrumented agent into OSWorld's actual
execution path -- confirmed working end-to-end on 2026-09-15:

- `desktop_env/providers/__init__.py` — registers `"kubevirt"` in
  `create_vm_manager_and_provider`, same pattern as the existing
  `daytona`/`modal` branches.
- `desktop_env/desktop_env.py` — adds `"kubevirt"` to the set of providers
  treated as starting from a clean state each run (fresh containerDisk
  overlay per VMI, same as the Docker provider). Required, not optional --
  `desktop_env.py` raises `ValueError` for any provider name not in
  *either* this set or the vmware/virtualbox "dirty" set.
- `run.py` — swaps `PromptAgent` for `InstrumentedAgent` so every step gets
  logged as JSONL, and passes through `agent.action_space` (`run.py` reads
  this right after construction).
- `lib_run_single.py` — calls `agent.start_episode()` / `agent.end_episode()`
  at the right points in the existing run loop, gated on `hasattr()` so it's
  a no-op for every other agent OSWorld's other run scripts use.
- `mm_agents/agent.py` — adds a generic OpenAI-compatible branch to
  `call_llm()`, gated on `NRP_API_BASE` being set, that takes priority over
  the per-brand model-name dispatch below it. Without this, a model named
  `"qwen*"` gets routed to Alibaba's proprietary `dashscope` SDK
  (hardcoded to Aliyun's cloud, no configurable base URL) instead of NRP's
  OpenAI-compatible endpoint. Also defensively handles NRP's `qwen3` being
  a reasoning model: if `max_tokens` runs out mid-reasoning, `content`
  comes back `null` -- this logs the actual reasoning trace and returns
  `""` instead of crashing on a `None` response.

```bash
cd <path-to-OSWorld-clone>
git apply <path-to-this-scaffold>/kubevirt_provider/osworld-integration.patch
```

## Step 5 — Run

```bash
export KUBEVIRT_NAMESPACE=gai-lina-group   # default, override if needed
export KUBEVIRT_BASE_IMAGE=<your pushed image ref>   # from Step 1

# Local dev loop (harness on your laptop, default mode):
python run.py --provider_name kubevirt --observation_type screenshot_a11y_tree \
  --model qwen3 --max_tokens 4096 \
  --test_all_meta_path evaluation_examples/test_smoke.json \
  --result_dir ./results/smoke_test

# If instead running the harness itself inside the cluster (matches
# ../osworld-job.yaml's pattern):
export KUBEVIRT_IN_CLUSTER=1
```

`--max_tokens` matters more than it looks: NRP's `qwen3` is a reasoning
model, and OSWorld's own default of 1500 is small enough that the whole
budget can be spent on hidden chain-of-thought before any actual answer
comes back (`content: null`). 4096 was enough in testing; watch the logs
for "returned no content" warnings if steps start silently doing nothing.

## Known blocker (unresolved as of 2026-09-15)

The `gai-lina-group` namespace has a `LimitRange` (`gai-lina-group-mem`)
capping every container to 1Gi memory / 100m CPU by default -- far below
what a real desktop VM needs (this provider requests 4Gi/4 cores, matching
the Docker provider's own defaults). KubeVirt's `virtualmachine-controller`
rejects the launcher pod outright (`FailedCreate`, visible via `kubectl
describe vmi`) before it ever reaches `Pending`/`Running`. This blocked the
first real end-to-end run. Needs either: NRP support raising the
namespace's LimitRange for this workload, or finding out whether a
different namespace/quota class is available for KubeVirt-based workloads.
Confirm this before assuming the rest of the pipeline is ready to scale.

## What's unverified

- **Whether all four guest ports (5000/8006/9222/8080) are actually
  reachable through KubeVirt's default masquerade pod-network binding on
  this specific cluster build.** Documented upstream default behavior, and
  the SSL/resource-limit issues above were hit before a real OSWorld VMI
  ever reached `Running`, so this specific check still hasn't happened. If
  a task hangs at `_wait_for_server_ready` once the VM does boot, this is
  the first thing to check — `kubectl exec` into the launcher pod and
  check whether the Flask server process itself started, before suspecting
  network plumbing.
- **Registry push size/reachability.** Pushing a 20+ GiB image to a public
  registry and having every NRP node pull it on VM boot is the "just get
  it working" path, not necessarily the efficient one — if boot latency
  per task turns out to matter (running 40-60 pilot tasks), revisit the
  CDI DataVolume/virtctl import path (persistent PVC-backed disk, no
  per-VM image pull) flagged in `config.py`.
- **Windows tasks.** Only Ubuntu is wired up. OSWorld's Windows qcow2 is a
  separate, larger download — extend `Dockerfile.containerdisk` /
  `config.py` / `manager.py`'s `os_type` handling if/when a task subset
  needs it.
- **Python version.** Must be 3.12, not 3.13 -- Python 3.13 enabled
  `ssl.VERIFY_X509_STRICT` by default, which rejects NRP's cluster
  certificate (missing an Authority/Subject Key Identifier extension that
  Go's TLS stack, what `kubectl` uses, doesn't require). This surfaces as
  `SSLCertVerificationError` from the `kubernetes` Python client, not from
  `kubectl` itself. `.mise.toml` in the real OSWorld repo already pins
  3.12 for unrelated reasons; this is another reason to actually respect
  that pin rather than using whatever Python happens to be on `PATH`.
