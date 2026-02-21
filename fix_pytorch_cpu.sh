#!/bin/bash

# PyTorch DLL Fix - Approach 2: Use CPU-Only PyTorch
# This approach uses CPU-only PyTorch which has fewer DLL dependencies

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "PyTorch DLL Fix - CPU-Only Approach"
echo "========================================"
echo ""
echo -e "${YELLOW}WARNING: This will install CPU-only PyTorch${NC}"
echo -e "${YELLOW}CUDA will NOT be available${NC}"
echo ""
echo "This script will:"
echo "  1. Remove current PyTorch installation"
echo "  2. Install CPU-only PyTorch (pip)"
echo "  3. Verify imports work"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found.${NC}"
    exit 1
fi

# Check if gss environment exists
if ! conda env list | grep -q "^gss "; then
    echo -e "${RED}Error: 'gss' conda environment not found.${NC}"
    echo "Please run init.sh first to create the environment."
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 1: Removing current PyTorch...${NC}"
conda run -n gss pip uninstall -y torch torchvision 2>/dev/null || true
echo -e "${GREEN}✓ Removed${NC}"

echo ""
echo -e "${YELLOW}Step 2: Installing CPU-only PyTorch...${NC}"
conda run -n gss pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo -e "${GREEN}✓ Installed${NC}"

echo ""
echo -e "${YELLOW}Step 3: Reinstalling gsplat...${NC}"
conda run -n gss pip install --force-reinstall gsplat
echo -e "${GREEN}✓ Installed${NC}"

echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo ""

# Test PyTorch import
echo -n "Testing PyTorch import... "
if conda run -n gss python -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ SUCCESS${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo ""
    echo "PyTorch still fails to import. This may be a system-level issue."
    echo "Check if Visual C++ redistributables are installed:"
    echo "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
    exit 1
fi

# Test CUDA availability (should be False)
echo -n "Testing CUDA availability... "
CUDA_AVAILABLE=$(conda run -n gss python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo -e "${YELLOW}CPU-only mode (expected)${NC}"
else
    echo -e "${GREEN}CUDA Available (unexpected)${NC}"
fi

# Test gsplat import
echo -n "Testing gsplat import... "
if conda run -n gss python -c "import gsplat; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ SUCCESS${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

# Test matplotlib import
echo -n "Testing matplotlib import... "
if conda run -n gss python -c "import matplotlib; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ SUCCESS${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

# Test gss package import
echo -n "Testing gss package import... "
if conda run -n gss python -c "import gss; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ SUCCESS${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

echo ""
echo "========================================"
echo "Fix Complete!"
echo "========================================"
echo ""
echo -e "${YELLOW}NOTE: CUDA is NOT available in CPU-only mode${NC}"
echo "If you need CUDA, try installing Visual C++ redistributables and then:"
echo "  ${YELLOW}bash fix_pytorch_conda.sh${NC}"
echo ""
