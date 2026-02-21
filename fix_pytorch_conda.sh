#!/bin/bash

# PyTorch DLL Fix - Approach 1: Use Conda's PyTorch Package
# This approach uses conda's PyTorch instead of pip, which has better DLL dependency management on Windows

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "PyTorch DLL Fix - Conda Approach"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Remove pip-installed PyTorch"
echo "  2. Reinstall PyTorch via conda (better DLL management)"
echo "  3. Verify imports work"
echo ""

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

echo -e "${YELLOW}Step 1: Removing pip-installed PyTorch...${NC}"
conda run -n gss pip uninstall -y torch torchvision 2>/dev/null || true
echo -e "${GREEN}✓ Removed${NC}"

echo ""
echo -e "${YELLOW}Step 2: Installing PyTorch via conda...${NC}"
conda install -n gss pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
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
    echo "PyTorch still fails to import. Try the CPU-only approach:"
    echo "  bash fix_pytorch_cpu.sh"
    exit 1
fi

# Test CUDA availability
echo -n "Testing CUDA availability... "
CUDA_AVAILABLE=$(conda run -n gss python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}✓ CUDA Available${NC}"
else
    echo -e "${YELLOW}⚠ CUDA Not Available (CPU mode)${NC}"
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
echo "If all tests passed, the PyTorch DLL issue is resolved."
echo "If PyTorch still fails, try the CPU-only approach:"
echo "  ${YELLOW}bash fix_pytorch_cpu.sh${NC}"
echo ""
