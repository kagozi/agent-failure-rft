"""KubeVirt provider configuration.

All values are read from environment variables so the provider stays a
drop-in, following the same pattern as OSWorld's other cloud providers
(compare desktop_env/providers/fastvm/config.py). No code changes needed
to retarget namespace/storage/image between dev and a future NRP job.

Required:
  KUBEVIRT_BASE_IMAGE     — container image reference holding the OSWorld
                             Ubuntu disk as a KubeVirt containerDisk (built
                             with Dockerfile.containerdisk in this same
                             directory -- see KUBEVIRT_GUIDELINE.md Step 1).
                             There is no default: the provider refuses to
                             start a VM without this, since booting the
                             wrong/empty image just produces a VM with no
                             OSWorld guest server on :5000.

Optional:
  KUBEVIRT_NAMESPACE         — default "gai-lina-group" (confirmed working
                                NRP/Nautilus namespace as of 2026-09-11).
  KUBEVIRT_VM_MEMORY          — default "4Gi", matches Docker provider's
                                 RAM_SIZE default.
  KUBEVIRT_VM_CPU_CORES        — default "4", matches Docker provider's
                                  CPU_CORES default.
  KUBEVIRT_IN_CLUSTER          — "1" if the OSWorld harness (run.py) itself
                                  runs inside a pod in KUBEVIRT_NAMESPACE
                                  (e.g. via the existing osworld-job.yaml
                                  pattern). In that case the VM's pod-network
                                  IP is reachable directly, no forwarding
                                  needed. Default "0" -- assumes the harness
                                  runs on your laptop (the current, Step-1
                                  local-dev-loop use case per README.md),
                                  and the provider shells out to
                                  `kubectl port-forward` against the VM's
                                  virt-launcher pod to reach it.
  KUBEVIRT_LOCAL_PORT_BASE      — default 15000. Starting point for local
                                   port-forward allocation (avoids clobbering
                                   anything already using 5000/8006/9222/8080
                                   locally).
  KUBEVIRT_BOOT_TIMEOUT_SEC      — default 300. Wait budget for the VMI to
                                    reach phase=Running.
  KUBEVIRT_READY_TIMEOUT_SEC     — default 300. Wait budget for the in-guest
                                    OSWorld Flask server on :5000 to answer
                                    /screenshot, after networking is up.
  KUBEVIRT_KUBECONFIG            — optional explicit kubeconfig path; if
                                    unset, uses the default kubeconfig
                                    resolution (~/.kube/config / in-cluster).

Known unverified item (flag honestly, don't assume): whether NRP's CDI
upload proxy is reachable from outside the cluster. That question only
matters if you use the alternative DataVolume/virtctl image-import path
instead of the containerDisk path above -- this provider defaults to
containerDisk specifically to avoid depending on that unverified proxy.
See KUBEVIRT_GUIDELINE.md for the tradeoff.
"""
from __future__ import annotations

import os

NAMESPACE = os.environ.get("KUBEVIRT_NAMESPACE", "gai-lina-group")

BASE_IMAGE = os.environ.get("KUBEVIRT_BASE_IMAGE") or None

VM_MEMORY = os.environ.get("KUBEVIRT_VM_MEMORY", "4Gi")
VM_CPU_CORES = os.environ.get("KUBEVIRT_VM_CPU_CORES", "4")

IN_CLUSTER = os.environ.get("KUBEVIRT_IN_CLUSTER", "0") == "1"
LOCAL_PORT_BASE = int(os.environ.get("KUBEVIRT_LOCAL_PORT_BASE", "15000"))

BOOT_TIMEOUT_SEC = float(os.environ.get("KUBEVIRT_BOOT_TIMEOUT_SEC", "300"))
READY_TIMEOUT_SEC = float(os.environ.get("KUBEVIRT_READY_TIMEOUT_SEC", "300"))

KUBECONFIG = os.environ.get("KUBEVIRT_KUBECONFIG") or None

# Guest ports -- fixed by the OSWorld base image itself (baked into the
# qcow2's systemd units), not configurable per-VM. Mirrors
# desktop_env/desktop_env.py's own defaults.
SERVER_PORT = 5000
CHROMIUM_PORT = 9222
VNC_PORT = 8006
VLC_PORT = 8080

API_GROUP = "kubevirt.io"
API_VERSION = "v1"
VMI_PLURAL = "virtualmachineinstances"
