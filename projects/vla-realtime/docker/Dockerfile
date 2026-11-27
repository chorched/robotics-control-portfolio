# C1t smoke image: LeRobot (smolvla) + LIBERO on Blackwell (sm_120) via CUDA 12.8 wheels
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

ARG LEROBOT_TAG=v0.4.4
ARG LIBERO_SHA=8f1084e3132a39270c3a13ebe37270a43ece2a01
ARG SMOLVLA_REV=6721902bc4d61e50a3bfdb11dfb4cb626f05d102

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        git curl ca-certificates build-essential cmake pkg-config \
        ffmpeg \
        libegl1 libgl1 libgles2 libosmesa6 libglew-dev libglfw3 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install -U pip setuptools wheel

# Blackwell (compute 12.0) needs the cu128 wheel index
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0

# LeRobot pinned at tag commit (recorded in c1_smoke.json)
RUN git clone --depth 1 --branch ${LEROBOT_TAG} https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip install -e "/opt/lerobot[smolvla]"

# LIBERO pinned at master commit 2025-03-15; robosuite/bddl installed by hand so the
# legacy requirements.txt (numpy==1.22.4, torch from 2023) does not fight the cu128 torch.
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
    && cd /opt/LIBERO && git checkout ${LIBERO_SHA}
RUN pip install robosuite==1.4.1 bddl==1.0.1 easydict future cloudpickle \
        "gym==0.25.2" hydra-core matplotlib thop \
    && cd /opt/LIBERO && pip install -e . --no-deps

# Policy weights pinned to the Hub revision (cache lives in the image => Docker default location)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('HuggingFaceVLA/smolvla_libero', revision='${SMOLVLA_REV}', local_dir='/opt/models/smolvla_libero'); \
snapshot_download('HuggingFaceTB/SmolVLM2-500M-Instruct', local_dir='/opt/models/SmolVLM2-500M-Instruct')"

WORKDIR /work
CMD ["bash"]
