# Configuration of KubeVirt (NRP / Nautilus)

Custom OSWorld provider targeting NRP's shared Kubernetes cluster via
KubeVirt, instead of the Docker provider's privileged-container QEMU
approach. Confirmed working (KVM hardware acceleration, full VMI CRUD
permissions) against namespace `gai-lina-group` on 2026-09-11 — see
`../README.md` for how this fits the overall project plan.

**Full end-to-end confirmed 2026-09-15**: one real OSWorld task
(`os/5ea617a3-...`) ran completely through this pipeline -- VMI booted via
KubeVirt, `qwen3` (via NRP) drove real `pyautogui` actions in the guest,
a 17-record trajectory JSONL was written (episode_start + 15 steps +
episode_end), and the episode was scored (0.0 -- a genuine task failure,
not a pipeline failure). Getting there surfaced five real bugs, each
documented below where it's fixed: VMI `resources.limits` (namespace
`LimitRange` rejection), explicit masquerade networking (bridge was the
silent default), draining the port-forward subprocess's stdout, detecting
+ recovering from the port-forward process dying mid-wait, and retry/backoff
on the NRP call path (a reasoning model burning `max_tokens` on hidden
chain-of-thought, or a transient 5xx, previously wasted an entire agent
step with zero recovery attempt -- confirmed this silently ate 13
consecutive steps in one run, which would corrupt a failure taxonomy by
mislabeling API noise as agent capability failure).

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

## Resolved: LimitRange rejection on VMI creation

The `gai-lina-group` namespace has a `LimitRange` (`gai-lina-group-mem`)
that auto-injects a small default *limit* (100m cpu / 1Gi memory, sized
for lightweight batch pods) onto any container that doesn't specify
`resources.limits` explicitly. The original VMI manifest only set
`resources.requests` (4Gi/4 cores, matching the Docker provider's
defaults), so KubeVirt's `virtualmachine-controller` rejected the launcher
pod outright (`FailedCreate`, visible via `kubectl describe vmi`) for
requesting more than the auto-injected limit -- this looked like a
namespace-level quota wall at first, but wasn't. Fixed in `provider.py`'s
`_build_vmi_manifest` by setting `resources.limits` equal to
`resources.requests`, confirmed 2026-09-15 (pod creation succeeds, VMI
reaches `Running`).

## Resolved: guest ports unreachable (wrong network binding)

The original manifest didn't declare `spec.domain.devices.interfaces` /
`spec.networks` at all, on the assumption that leaving it unset would fall
back to masquerade binding (which forwards all ports from the pod IP into
the guest). That assumption was wrong: KubeVirt's real default when
unset is **bridge** binding, where the guest gets its own IP on the
bridge, separate from the pod's own network namespace. `kubectl
port-forward` (and, in `KUBEVIRT_IN_CLUSTER=1` mode, anything else
reaching the VM via the pod IP) can only reach sockets inside the pod's
own netns -- a bridged guest's listening ports are invisible that way, so
`_wait_for_server_ready` timed out every time even though the guest had
booted fine. Fixed by explicitly setting
`interfaces: [{name: default, masquerade: {}}]` and
`networks: [{name: default, pod: {}}]` in `_build_vmi_manifest`. Confirmed
2026-09-15: `/screenshot` answers immediately once this is in place.

## Resolved: port-forward stalling after ~5 minutes

Even with masquerade networking working (confirmed instantly reachable in
manual tests), the real `run.py` path still timed out at
`_wait_for_server_ready` every time -- always after the full timeout, never
sooner, and never reproducible with a manual `kubectl port-forward` run by
hand. Root cause: `_start_port_forward` captured the subprocess's stdout via
`subprocess.PIPE`, read a few lines to confirm the tunnels came up, then
never read from that pipe again. `kubectl port-forward` keeps writing to
stdout for the life of the process (e.g. retry/error output for whichever
of the 4 forwarded ports the guest isn't listening on), and once the OS
pipe buffer fills (~64KB), `kubectl` blocks on its next `write()` -- which
stalls the *entire* port-forward, including the one tunnel (`:5000`)
`_wait_for_server_ready` actually needs. Manual reproductions never hit
this because they either redirected output to a file (unbounded) or exited
before the buffer filled. Fixed by draining the pipe continuously in a
background thread for the life of the process.

## Resolved: kubectl port-forward dying silently mid-wait

The pipe-draining fix above didn't fix the timeout on its own -- caught it
live by curling the exact tunnel `run.py` had just created while it was
still inside its 5-minute wait, and found `kubectl port-forward` simply
wasn't running anymore (absent from `ps aux`), while `run.py` itself was
still alive, still retrying HTTP requests against the now-dead local port.
`kubectl port-forward` established the tunnel successfully (all 4
"Forwarding from" lines seen) and then exited on its own some time later
with nothing logged -- consistent with an idle/connection timeout
somewhere between the local kubectl process and the API server, though the
exact upstream cause wasn't tracked further. Manual reproductions never
caught this because they were always short-lived, freshly-started tunnels.
First fix attempt only checked `self._port_forward_proc.poll()` inside
`_wait_for_server_ready`'s loop -- which covers the initial boot wait, but
not the rest of the episode. Confirmed 2026-09-15 in a follow-up run: the
tunnel died again, this time mid-episode (after 2 real agent steps had
already succeeded), and every OSWorld controller call (screenshot, a11y
tree, action execution, recording) goes straight to `http://localhost:<port>`
with no awareness of this provider's tunnel at all, so nothing else would
ever notice or recover. Fixed properly with a persistent watchdog thread
(`_port_forward_watchdog`) that runs for the entire VM lifetime (started
in `start_emulator`, stopped in `stop_emulator`), plus a shared
`_restart_port_forward_if_dead` helper (behind a lock, since both the
watchdog and the initial readiness wait can independently notice a dead
tunnel) so a restart doesn't hand out new local port numbers -- callers
outside this provider have already cached the port from `get_ip_address()`
and would be silently stranded if it changed.

## What's unverified
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
