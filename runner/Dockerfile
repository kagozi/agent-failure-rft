# Runner image for executing the OSWorld harness *inside* the NRP cluster
# (KUBEVIRT_IN_CLUSTER=1), as a k8s Job -- distinct from the VM's own
# containerDisk image (kubevirt_provider/Dockerfile.containerdisk), which
# is what OSWorld's agent actually drives. This image just needs to run
# run.py and talk to the Kubernetes API + the NRP LLM endpoint.
#
# Build from the REPO ROOT (not this directory), so it can COPY in
# kubevirt_provider/ and instrumented_agent.py:
#   docker build --platform linux/amd64 -f runner/Dockerfile \
#     -t ghcr.io/kagozi/osworld-runner:latest .
#   docker push ghcr.io/kagozi/osworld-runner:latest
#
# Must be Python 3.12, not 3.13 -- see KUBEVIRT_GUIDELINE.md's note on
# ssl.VERIFY_X509_STRICT rejecting NRP's cluster certificate under 3.13.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pinned to the exact commit this scaffold's patches were developed and
# confirmed working against (2026-09-15) -- OSWorld's own history isn't
# under our control, and an unpinned `git clone` could silently pull in
# upstream changes that no longer apply cleanly against our patch.
ARG OSWORLD_COMMIT=b138d348256078fa634fc3b73567a7337c793e6b
RUN git clone https://github.com/xlang-ai/OSWorld.git osworld \
 && cd osworld \
 && git checkout "${OSWORLD_COMMIT}"

WORKDIR /workspace/osworld

COPY kubevirt_provider/osworld-integration.patch /tmp/osworld-integration.patch
RUN git apply /tmp/osworld-integration.patch

COPY kubevirt_provider/__init__.py kubevirt_provider/config.py kubevirt_provider/manager.py kubevirt_provider/provider.py \
     desktop_env/providers/kubevirt/
COPY instrumented_agent.py .

# Mirrors the local dev setup: torch/transformers/accelerate are only
# needed by local open-weight-model agents we don't use (we call NRP's
# hosted API instead) -- confirmed via grep, not assumption. pynput is
# additionally excluded here (Linux-only, not needed on macOS dev
# machines): it pulls in evdev, which needs Linux kernel headers to build
# a native wheel and isn't imported anywhere in our code path either --
# it's for human-baseline input capture (run_single_example_human), not
# the automated PromptAgent path this runner drives.
RUN grep -vE '^(torch~=|transformers~=|accelerate$|pynput~=)' requirements.txt > /tmp/requirements-runner.txt \
 && pip install --no-cache-dir -r /tmp/requirements-runner.txt \
 && pip install --no-cache-dir kubernetes

ENTRYPOINT ["python", "run.py"]
