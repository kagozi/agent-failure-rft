"""KubeVirt VM "manager" -- resolves which base image to boot.

Deliberately minimal, matching desktop_env/providers/docker/manager.py's
own precedent: that manager doesn't maintain a real VM pool/registry either
(add_vm/delete_vm/occupy_vm are no-ops) because the Docker/KubeVirt-style
providers boot a fresh, disposable instance per run rather than checking
one out of a shared pool the way the VMware/VirtualBox providers do.

Unlike the Docker manager, this one does NOT download and unzip the
Ubuntu.qcow2 itself -- that download-and-unzip step still has to happen
once, offline, as part of building the containerDisk image (see
Dockerfile.containerdisk + KUBEVIRT_GUIDELINE.md Step 1). Baking the image
is a one-time human action (build + docker/podman push), not something to
redo inside every DesktopEnv() construction.
"""
import logging

from desktop_env.providers.base import VMManager

from . import config

logger = logging.getLogger("desktopenv.providers.kubevirt.KubeVirtVMManager")
logger.setLevel(logging.INFO)


class KubeVirtVMManager(VMManager):
    def __init__(self, registry_path: str = ""):
        pass

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, region=None, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, region=None, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return config.BASE_IMAGE

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, os_type: str = "Ubuntu", region=None, screen_size=(1920, 1080), **kwargs) -> str:
        if os_type != "Ubuntu":
            raise NotImplementedError(
                "Only an Ubuntu containerDisk is wired up so far -- see "
                "provider.py's start_emulator for the same restriction."
            )
        if not config.BASE_IMAGE:
            raise RuntimeError(
                "KUBEVIRT_BASE_IMAGE is not set. Build and push the OSWorld "
                "containerDisk image first (Dockerfile.containerdisk + "
                "KUBEVIRT_GUIDELINE.md Step 1), then export "
                "KUBEVIRT_BASE_IMAGE=<your pushed image ref>."
            )
        logger.info("Using KubeVirt containerDisk image: %s", config.BASE_IMAGE)
        return config.BASE_IMAGE
