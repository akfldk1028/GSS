#!/usr/bin/env bash
# GSS API — bare-metal Linux setup (Ubuntu 22.04+)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="gss"

echo "=== GSS Server Setup ==="
echo "Repo: $REPO_DIR"

# 1. Conda env
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env '${ENV_NAME}'..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 2. PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. GSS + server
cd "$REPO_DIR"
pip install -e ".[server]"

# 4. gsplat
pip install gsplat

# 5. Systemd unit (optional)
UNIT_FILE=/etc/systemd/system/gss-api.service
if [ ! -f "$UNIT_FILE" ]; then
    echo "Installing systemd service..."
    sudo tee "$UNIT_FILE" > /dev/null <<UNIT
[Unit]
Description=GSS API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$(conda info --base)/envs/${ENV_NAME}/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0
Environment=GSS_DATA_ROOT=$REPO_DIR/data
Environment=GSS_JOBS_DIR=$REPO_DIR/data/jobs

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload
    echo "Service installed. Start with: sudo systemctl start gss-api"
fi

echo "=== Done. Run: uvicorn server.app:app --host 0.0.0.0 --port 8000 ==="
