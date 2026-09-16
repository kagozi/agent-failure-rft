"""KubeVirt implementation of OSWorld's `Provider` interface.

Maps the five abstract methods (desktop_env/providers/base.py) onto a
KubeVirt VirtualMachineInstance (VMI) running on NRP's Nautilus cluster.

Mechanism, confirmed working end-to-end against namespace "gai-lina-group"
on 2026-09-11 (see chat history / project memory, not re-derived here):
  - KubeVirt v1.7.0 is deployed cluster-wide; this namespace has full CRUD
    on virtualmachineinstances.kubevirt.io.
  - `devices.kubevirt.io/kvm` is advertised on effectively every worker
    node -- a smoke-test VMI reached Running in ~16s and its QEMU process
    was confirmed running with `-accel kvm` (hardware acceleration, not
    software emulation).
  - This is architecturally cleaner than OSWorld's own Docker provider,
    which needs a *privileged* container to run QEMU manually
    (desktop_env/providers/docker/provider.py). KubeVirt VMs are a
    first-class Kubernetes object; no privileged pod required from us.

Not yet verified against a real OSWorld task (only a bare demo VM image
was smoke-tested) -- treat the boot/readiness timing and the masquerade
port-forwarding assumption below as things to confirm on first real run,
same spirit as the project's existing KVM-caveat comments in
../osworld-job.yaml.

Networking: KubeVirt's default pod-network interface uses masquerade
binding, which by default forwards *all* ports from the pod IP into the
guest (see KubeVirt docs on Masquerade binding) -- unverified specifically
for ports 5000/8006/9222/8080 against this cluster's KubeVirt build, but
this is documented upstream default behavior, not a guess unique to this
setup.

Two deployment modes (desktop_env.py always calls
`provider.start_emulator(path_to_vm, headless, os_type)` positionally, so
mode is picked via env var, not a constructor arg -- see config.py):
  - KUBEVIRT_IN_CLUSTER=1: assumes the OSWorld harness (run.py) itself runs
    inside a pod in the same namespace (matches ../osworld-job.yaml). Talks
    to the VMI's pod-network IP directly.
  - KUBEVIRT_IN_CLUSTER unset (default): assumes the harness runs on your
    laptop (README.md's current Step 1 -- local dev loop before touching
    k8s). Shells out to `kubectl port-forward` against the VM's
    virt-launcher pod, same idea as the daytona provider's SSH tunnel
    (desktop_env/providers/daytona/provider.py) but using kubectl instead
    of SSH, since KubeVirt VMI pods are ordinary pods `kubectl
    port-forward` already understands -- no virtctl binary required.
"""
from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
import uuid
from typing import Optional

import requests
from kubernetes import client, config as kube_config
from kubernetes.client.rest import ApiException

from desktop_env.providers.base import Provider

from . import config

logger = logging.getLogger("desktopenv.providers.kubevirt.KubeVirtProvider")
logger.setLevel(logging.INFO)


def _load_kube_clients():
    try:
        if config.KUBECONFIG:
            kube_config.load_kube_config(config_file=config.KUBECONFIG)
        else:
            kube_config.load_kube_config()
    except Exception:
        kube_config.load_incluster_config()
    return client.CustomObjectsApi(), client.CoreV1Api()


def _free_local_port(start: int) -> int:
    port = start
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError(f"No free local port found starting from {start}")


def _build_vmi_manifest(name: str, image: str) -> dict:
    return {
        "apiVersion": f"{config.API_GROUP}/{config.API_VERSION}",
        "kind": "VirtualMachineInstance",
        "metadata": {
            "name": name,
            "namespace": config.NAMESPACE,
            "labels": {"app": "osworld-agent-vm"},
        },
        "spec": {
            "domain": {
                "resources": {
                    "requests": {
                        "memory": config.VM_MEMORY,
                        "cpu": config.VM_CPU_CORES,
                    },
                    # Explicit limits are required here: the namespace has a
                    # LimitRange that auto-injects a small default *limit*
                    # (100m cpu / 1Gi memory, sized for lightweight batch
                    # pods) onto any container that doesn't specify one --
                    # then rejects our (larger) request for exceeding that
                    # injected limit. Setting limits == requests avoids
                    # triggering the default injection.
                    "limits": {
                        "memory": config.VM_MEMORY,
                        "cpu": config.VM_CPU_CORES,
                    },
                },
                "devices": {
                    "disks": [
                        {"name": "containerdisk", "disk": {"bus": "virtio"}},
                    ],
                    # Explicit masquerade binding is required. Leaving this
                    # unset doesn't fall back to masquerade (as this
                    # provider originally assumed) -- KubeVirt's real
                    # default is *bridge* binding, where the guest gets its
                    # own IP on the bridge, separate from the pod's own
                    # network namespace. `kubectl port-forward` (and, in
                    # IN_CLUSTER mode, anything else reaching the VM via
                    # the pod IP) only sees sockets inside the pod's own
                    # netns, so a bridged guest's ports are unreachable
                    # that way. Masquerade NATs guest traffic through the
                    # pod's own IP via iptables, which is what actually
                    # makes the guest's ports visible on the pod IP.
                    # Confirmed 2026-09-15: with bridge (the unset
                    # default), _wait_for_server_ready always timed out;
                    # explicit masquerade fixed it.
                    "interfaces": [
                        {"name": "default", "masquerade": {}},
                    ],
                },
            },
            "networks": [
                {"name": "default", "pod": {}},
            ],
            "terminationGracePeriodSeconds": 30,
            "volumes": [
                {
                    "name": "containerdisk",
                    "containerDisk": {"image": image},
                },
            ],
        },
    }


class KubeVirtProvider(Provider):
    def __init__(self, region: Optional[str] = None):
        super().__init__(region)
        self.custom_api, self.core_api = _load_kube_clients()

        self.vmi_name: Optional[str] = None
        self.vm_ip: Optional[str] = None
        self.server_port: Optional[int] = None
        self.chromium_port: Optional[int] = None
        self.vnc_port: Optional[int] = None
        self.vlc_port: Optional[int] = None
        self._port_forward_proc: Optional[subprocess.Popen] = None

    # ---- Provider interface ------------------------------------------------

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        if os_type != "Ubuntu":
            raise NotImplementedError(
                "KubeVirtProvider only has an Ubuntu containerDisk wired up so "
                "far -- build+push a Windows containerDisk and extend "
                "config.py/manager.py before using os_type='Windows'."
            )
        if not path_to_vm:
            raise RuntimeError(
                "No base image reference (path_to_vm). Set KUBEVIRT_BASE_IMAGE "
                "-- see KUBEVIRT_GUIDELINE.md Step 1 to build/push it."
            )

        self.vmi_name = f"osworld-{uuid.uuid4().hex[:10]}"
        manifest = _build_vmi_manifest(self.vmi_name, path_to_vm)
        logger.info("Creating VMI %s from image %s", self.vmi_name, path_to_vm)
        self.custom_api.create_namespaced_custom_object(
            group=config.API_GROUP,
            version=config.API_VERSION,
            namespace=config.NAMESPACE,
            plural=config.VMI_PLURAL,
            body=manifest,
        )

        try:
            pod_ip = self._wait_for_running_and_get_pod_ip()

            if config.IN_CLUSTER:
                self.vm_ip = pod_ip
                self.server_port = config.SERVER_PORT
                self.chromium_port = config.CHROMIUM_PORT
                self.vnc_port = config.VNC_PORT
                self.vlc_port = config.VLC_PORT
            else:
                self._start_port_forward()
                self.vm_ip = "localhost"

            self._wait_for_server_ready()
        except Exception:
            logger.error("start_emulator failed, cleaning up VMI %s", self.vmi_name)
            self.stop_emulator(path_to_vm)
            raise

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.vm_ip, self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started -- call start_emulator first")
        return f"{self.vm_ip}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        # Matches the Docker provider precedent (desktop_env/providers/docker/
        # provider.py): every episode gets a fresh VM from the pristine base
        # containerDisk image (KubeVirt auto-creates an ephemeral
        # copy-on-write layer per VMI, same idea as Docker's read-only qcow2
        # mount), so there is no persistent snapshot to save.
        raise NotImplementedError("Snapshots not available for the KubeVirt provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        # Also matches the Docker provider: "revert" just means tear down
        # and let the next start_emulator() call boot a clean instance.
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        if self._port_forward_proc is not None:
            logger.info("Stopping port-forward for VMI %s", self.vmi_name)
            self._port_forward_proc.terminate()
            try:
                self._port_forward_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._port_forward_proc.kill()
            self._port_forward_proc = None

        if self.vmi_name is not None:
            logger.info("Deleting VMI %s", self.vmi_name)
            try:
                self.custom_api.delete_namespaced_custom_object(
                    group=config.API_GROUP,
                    version=config.API_VERSION,
                    namespace=config.NAMESPACE,
                    plural=config.VMI_PLURAL,
                    name=self.vmi_name,
                )
            except ApiException as e:
                if e.status != 404:
                    logger.error("Error deleting VMI %s: %s", self.vmi_name, e)
            self.vmi_name = None

        self.vm_ip = None
        self.server_port = self.chromium_port = self.vnc_port = self.vlc_port = None

    # ---- internals ----------------------------------------------------------

    def _wait_for_running_and_get_pod_ip(self) -> str:
        deadline = time.monotonic() + config.BOOT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            obj = self.custom_api.get_namespaced_custom_object(
                group=config.API_GROUP,
                version=config.API_VERSION,
                namespace=config.NAMESPACE,
                plural=config.VMI_PLURAL,
                name=self.vmi_name,
            )
            status = obj.get("status", {})
            phase = status.get("phase")
            if phase == "Running":
                interfaces = status.get("interfaces") or []
                if interfaces and interfaces[0].get("ipAddress"):
                    ip = interfaces[0]["ipAddress"]
                    logger.info("VMI %s Running at pod IP %s", self.vmi_name, ip)
                    return ip
            elif phase == "Failed":
                raise RuntimeError(f"VMI {self.vmi_name} entered Failed phase: {status}")
            time.sleep(3)
        raise TimeoutError(
            f"VMI {self.vmi_name} did not reach Running with an IP within "
            f"{config.BOOT_TIMEOUT_SEC:.0f}s"
        )

    def _launcher_pod_name(self) -> str:
        pods = self.core_api.list_namespaced_pod(
            namespace=config.NAMESPACE,
            label_selector=f"kubevirt.io=virt-launcher,vm.kubevirt.io/name={self.vmi_name}",
        )
        if not pods.items:
            raise RuntimeError(f"No virt-launcher pod found for VMI {self.vmi_name}")
        return pods.items[0].metadata.name

    def _start_port_forward(self):
        pod_name = self._launcher_pod_name()
        base = config.LOCAL_PORT_BASE
        self.server_port = _free_local_port(base)
        self.chromium_port = _free_local_port(self.server_port + 1)
        self.vnc_port = _free_local_port(self.chromium_port + 1)
        self.vlc_port = _free_local_port(self.vnc_port + 1)

        port_pairs = [
            f"{self.server_port}:{config.SERVER_PORT}",
            f"{self.chromium_port}:{config.CHROMIUM_PORT}",
            f"{self.vnc_port}:{config.VNC_PORT}",
            f"{self.vlc_port}:{config.VLC_PORT}",
        ]
        cmd = ["kubectl", "port-forward", "-n", config.NAMESPACE, f"pod/{pod_name}", *port_pairs]
        logger.info("Starting port-forward: %s", " ".join(cmd))
        self._port_forward_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # kubectl prints one "Forwarding from ..." line per port pair once the
        # tunnel is actually up; give it a few seconds rather than assuming.
        deadline = time.monotonic() + 15
        forwarding_lines = 0
        while time.monotonic() < deadline and forwarding_lines < len(port_pairs):
            line = self._port_forward_proc.stdout.readline()
            if not line:
                if self._port_forward_proc.poll() is not None:
                    raise RuntimeError(
                        f"kubectl port-forward exited early (code "
                        f"{self._port_forward_proc.returncode}) -- is the VMI's "
                        f"launcher pod still starting, or was the pod name stale?"
                    )
                continue
            if "Forwarding from" in line:
                forwarding_lines += 1
        if forwarding_lines < len(port_pairs):
            raise TimeoutError("kubectl port-forward did not confirm all 4 tunnels in time")

        # Keep draining stdout for the life of the process. If nothing reads
        # it, the OS pipe buffer (~64KB) eventually fills from kubectl's own
        # ongoing log/retry output (e.g. connection resets on whichever of
        # the 4 ports the guest isn't listening on yet) and kubectl blocks on
        # its next write() -- which stalls the *entire* port-forward,
        # including the tunnels that were otherwise fine. Confirmed
        # 2026-09-15: without this, _wait_for_server_ready always timed out
        # against a real multi-port forward, even though the same command
        # worked fine manually with output redirected to a file instead of
        # captured via PIPE.
        threading.Thread(
            target=self._drain_port_forward_output, daemon=True
        ).start()

    def _drain_port_forward_output(self):
        proc = self._port_forward_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                logger.debug("[port-forward] %s", line.rstrip())
        except (ValueError, OSError):
            pass  # pipe closed during teardown

    def _wait_for_server_ready(self):
        deadline = time.monotonic() + config.READY_TIMEOUT_SEC
        last_err = None
        restarts = 0
        while time.monotonic() < deadline:
            # Recomputed every iteration, not hoisted above the loop --
            # a restart below can reassign self.server_port to a different
            # local port than what we started with.
            url = f"http://{self.vm_ip}:{self.server_port}/screenshot"
            # kubectl port-forward can die mid-wait (e.g. an idle/connection
            # timeout somewhere between here and the API server) without
            # printing anything actionable -- confirmed 2026-09-15, it
            # establishes the tunnel successfully, then exits silently a
            # short time later, and every request against the now-dead
            # local port fails with ConnectionError for the rest of the
            # timeout. If we're doing our own port-forwarding (not
            # IN_CLUSTER) and the process has exited, restart the tunnel
            # instead of continuing to hammer a dead port.
            if self._port_forward_proc is not None and self._port_forward_proc.poll() is not None:
                restarts += 1
                logger.warning(
                    "kubectl port-forward exited unexpectedly (code %s) -- "
                    "restarting tunnel (attempt %d)",
                    self._port_forward_proc.returncode, restarts,
                )
                self._start_port_forward()
            try:
                r = requests.get(url, timeout=(10, 10))
                if r.status_code == 200:
                    logger.info("OSWorld guest server ready at %s", url)
                    return
                last_err = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001
                last_err = type(e).__name__
            time.sleep(3)
        raise TimeoutError(
            f"OSWorld guest server at {url} not ready after "
            f"{config.READY_TIMEOUT_SEC:.0f}s ({restarts} port-forward "
            f"restart(s), last error: {last_err}). If this "
            f"is the first run against a new containerDisk image, confirm the "
            f"guest's systemd units for the Flask server/VNC actually started "
            f"-- e.g. `kubectl exec` into the launcher pod's compute container "
            f"and check console logs, same idea as the KVM smoke test."
        )
