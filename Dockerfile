# CUDA image for J3PA training, evaluation, and visualization.
# The CUDA toolkit matches the CUDA 13.0 build of PyTorch 2.13, which compiles PyTorch3D.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git ninja-build ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /opt/j3pa
COPY pyproject.toml uv.lock README.md ./

# No GPU is visible during the build, so FORCE_CUDA builds the PyTorch3D CUDA kernels anyway.
# 8.0 covers A100 (and runs on 8.6), 9.0 is H100, PTX for anything newer.
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" FORCE_CUDA=1 MAX_JOBS=16 \
    uv sync --frozen --no-dev --no-install-project

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN python -c "import torch, pytorch3d, h5py, hdf5plugin, numba, zod; \
from pytorch3d.ops import ball_query, knn_points; \
assert torch.version.cuda is not None; print(torch.__version__, pytorch3d.__version__)"

WORKDIR /app
ENTRYPOINT []
CMD ["/bin/bash"]
