#!/bin/bash
# GSS Environment Setup Script
set -e

echo "=== GSS Pipeline Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python: $python_version"
if [ "$(echo "$python_version < 3.10" | bc)" -eq 1 ]; then
    echo "ERROR: Python 3.10+ required"
    exit 1
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install core dependencies
echo "Installing core dependencies..."
pip install -e ".[dev]"

# Check CUDA
echo ""
echo "=== GPU Check ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "torch not installed (install manually for GPU steps)"

# Check COLMAP
echo ""
echo "=== COLMAP Check ==="
colmap --version 2>/dev/null || echo "COLMAP not found (install for S02)"

echo ""
echo "=== Setup Complete ==="
echo "Run: gss info        # show pipeline structure"
echo "Run: gss run          # run full pipeline"
echo "Run: gss run-step extract_frames  # run single step"
