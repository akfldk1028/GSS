#!/usr/bin/env bash
# ============================================================
# GSS Fresh Linux Setup — 전체 환경 + 데이터 + SLAM 백엔드
# Usage: bash deploy/setup_fresh_linux.sh
# ============================================================
set -euo pipefail

GSS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GSS_ROOT"
echo "=== GSS Setup: $GSS_ROOT ==="

# ── 0. System prereqs ───────────────────────────────────────
echo ""
echo ">>> [0/8] System check"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed. NVIDIA driver 필요."; exit 1;
}
echo "  gcc: $(gcc --version | head -1)"
echo "  python: $(python3 --version)"

# ── 1. Conda env (gss) ─────────────────────────────────────
echo ""
echo ">>> [1/8] Main conda env (gss)"
if conda env list | grep -q "^gss "; then
    echo "  gss env already exists, skipping"
else
    conda create -n gss python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate gss

# PyTorch + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# gsplat
pip install gsplat

# GSS editable install (core + server + mesh + dev)
pip install -e ".[dev,server,mesh,grpc]"
pip install pytest-asyncio

echo "  Verify:"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "import gss; print('  gss OK')"

# ── 2. gRPC proto compile ──────────────────────────────────
echo ""
echo ">>> [2/8] Compile gRPC protos"
python -m server.grpc.compile_protos || echo "  (skipped — grpcio-tools not available)"

# ── 3. Sample data download ────────────────────────────────
echo ""
echo ">>> [3/8] Sample data"
mkdir -p data/raw data/interim data/processed

# point_cloud_jh.ply — 원본이 있으면 복사, 없으면 안내
if [ ! -f data/raw/point_cloud_jh.ply ]; then
    echo "  WARNING: data/raw/point_cloud_jh.ply 없음 — WSL에서 복사하거나 직접 준비"
    echo "    cp /path/to/point_cloud_jh.ply data/raw/"
fi

# INRIA pre-trained models (optional, 14GB)
read -p "  INRIA pretrained 13 scenes 다운로드? (14GB) [y/N]: " dl_inria
if [[ "$dl_inria" =~ ^[Yy]$ ]]; then
    python scripts/download_sample_data.py download -d inria_pretrained
fi

# ── 4. Clone SLAM backends ─────────────────────────────────
echo ""
echo ">>> [4/8] Clone SLAM backends"
mkdir -p clone

# MASt3R-SLAM (1순위)
if [ ! -d clone/MASt3R-SLAM ]; then
    git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git clone/MASt3R-SLAM
else
    echo "  MASt3R-SLAM already cloned"
fi

# SLAM3R
if [ ! -d clone/SLAM3R ]; then
    git clone https://github.com/PKU-VCL-3DV/SLAM3R.git clone/SLAM3R
else
    echo "  SLAM3R already cloned"
fi

# MonoGS
if [ ! -d clone/MonoGS ]; then
    git clone --recursive https://github.com/muskie82/MonoGS.git clone/MonoGS
else
    echo "  MonoGS already cloned"
fi

# PlanarGS
if [ ! -d clone/PlanarGS ]; then
    git clone --recursive https://github.com/SJTU-ViSYS-team/PlanarGS.git clone/PlanarGS
else
    echo "  PlanarGS already cloned"
fi

# 3DGRUT
if [ ! -d clone/3dgrut ]; then
    git clone --recursive https://github.com/nv-tlabs/3dgrut.git clone/3dgrut
else
    echo "  3DGRUT already cloned"
fi

# ── 5. MASt3R-SLAM env + checkpoints ──────────────────────
echo ""
echo ">>> [5/8] MASt3R-SLAM conda env + checkpoints"

if conda env list | grep -q "^mast3r_slam "; then
    echo "  mast3r_slam env already exists"
else
    conda create -n mast3r_slam python=3.11 -y
fi

# Build in subshell to not mess up current env
(
    eval "$(conda shell.bash hook)"
    conda activate mast3r_slam
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

    cd clone/MASt3R-SLAM
    git submodule update --init --recursive
    pip install -e thirdparty/mast3r
    pip install -e thirdparty/in3d
    pip install --no-build-isolation -e .

    # Checkpoints (~2GB)
    mkdir -p checkpoints
    cd checkpoints
    [ -f MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth ] || \
        wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    [ -f MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth ] || \
        wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
    [ -f MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl ] || \
        wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
    echo "  MASt3R-SLAM setup complete"
)

# ── 6. SLAM3R env ──────────────────────────────────────────
echo ""
echo ">>> [6/8] SLAM3R conda env"

if conda env list | grep -q "^slam3r "; then
    echo "  slam3r env already exists"
else
    conda create -n slam3r python=3.11 cmake=3.14.0 -y
fi

(
    eval "$(conda shell.bash hook)"
    conda activate slam3r
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
    cd clone/SLAM3R
    pip install -r requirements.txt
    # CUDA extension
    cd slam3r/pos_embed/curope/ && python setup.py build_ext --inplace && cd "$GSS_ROOT"
    echo "  SLAM3R setup complete"
)

# ── 7. MonoGS env ──────────────────────────────────────────
echo ""
echo ">>> [7/8] MonoGS conda env"

if conda env list | grep -q "^MonoGS "; then
    echo "  MonoGS env already exists"
else
    (
        cd clone/MonoGS
        conda env create -f environment.yml
    )
fi

# ── 8. Verify ──────────────────────────────────────────────
echo ""
echo ">>> [8/8] Final verification"
conda activate gss

echo "--- pytest ---"
python -m pytest tests/ --tb=short -q 2>&1 | tail -3

echo "--- pipeline test ---"
if [ -f data/raw/point_cloud_jh.ply ]; then
    gss run --config configs/pipeline_import.yaml
    echo "  Output:"
    ls -lh data/processed/
else
    echo "  (skipped — no PLY file)"
fi

echo ""
echo "============================================================"
echo "  GSS Setup Complete!"
echo ""
echo "  Conda envs:"
echo "    gss          — main pipeline + server"
echo "    mast3r_slam  — MASt3R-SLAM (CVPR 2025)"
echo "    slam3r       — SLAM3R (CVPR 2025)"
echo "    MonoGS       — MonoGS (CVPR 2024)"
echo ""
echo "  Run:"
echo "    conda activate gss"
echo "    gss run --config configs/pipeline_import.yaml"
echo "    gss run --config configs/pipeline_slam.yaml"
echo "    uvicorn server.app:app --host 0.0.0.0 --port 8000"
echo "============================================================"
