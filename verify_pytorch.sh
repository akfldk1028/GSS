#!/bin/bash

# Verification script for PyTorch CUDA installation
# This script checks that PyTorch is installed with CUDA support (cu121 or cu124)

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "PyTorch CUDA Verification"
echo "========================================"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "^gss "; then
    echo -e "${RED}Error: Conda environment 'gss' not found.${NC}"
    echo "Please run: bash .auto-claude/specs/013-fix-python-env/init.sh"
    exit 1
fi

echo -e "${GREEN}✓ Conda environment 'gss' exists${NC}"
echo ""

# Check PyTorch version
echo "Checking PyTorch version..."
TORCH_VERSION=$(conda run -n gss python -c "import torch; print(torch.__version__)" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch version: $TORCH_VERSION${NC}"

    # Check if it's CUDA version (cu121 or cu124)
    if [[ "$TORCH_VERSION" == *"cu121"* ]]; then
        echo -e "${GREEN}✓ CUDA 12.1 support detected (cu121)${NC}"
    elif [[ "$TORCH_VERSION" == *"cu124"* ]]; then
        echo -e "${GREEN}✓ CUDA 12.4 support detected (cu124)${NC}"
    elif [[ "$TORCH_VERSION" == *"cu"* ]]; then
        echo -e "${YELLOW}⚠ CUDA support detected but not cu121/cu124: $TORCH_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ CPU-only version detected. Expected cu121 or cu124.${NC}"
    fi
else
    echo -e "${RED}✗ Failed to import torch${NC}"
    echo "$TORCH_VERSION"
    exit 1
fi

echo ""
echo "========================================"
echo "Verification Complete"
echo "========================================"
echo ""
echo "Expected: PyTorch version with cu121 or cu124"
echo "Actual: $TORCH_VERSION"
echo ""
