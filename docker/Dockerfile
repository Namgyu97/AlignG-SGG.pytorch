# AlignG-SGG Docker image
# Verified with: CUDA 12.1, PyTorch 2.5.1, Python 3.12 on RTX 4090.
#
# Build:
#   docker build -t alignG-sgg:cu121 -f docker/Dockerfile .
#
# Run (mount your repo + datasets):
#   docker run --gpus all -it --rm \
#       -v $PWD:/workspace/AlignG-SGG.pytorch \
#       -v /path/to/datasets:/workspace/AlignG-SGG.pytorch/datasets \
#       alignG-sgg:cu121

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential cmake \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        python3.12 python3.12-dev python3.12-venv python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# PyTorch matching the verified env. Use cu121 wheels.
RUN pip install --upgrade pip \
 && pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace/AlignG-SGG.pytorch
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Project source is mounted at runtime; build the csrc extension when entering.
CMD ["/bin/bash", "-lc", \
     "python setup.py build develop && /bin/bash"]
