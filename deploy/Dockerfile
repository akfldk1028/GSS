# GSS API Server — CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Conda env
RUN conda create -n gss python=3.10 -y
SHELL ["conda", "run", "-n", "gss", "/bin/bash", "-c"]

WORKDIR /app
COPY . /app

# PyTorch + CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# GSS + server deps
RUN pip install -e ".[server]"

# gsplat (must be after torch)
RUN pip install gsplat

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "gss", \
     "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
