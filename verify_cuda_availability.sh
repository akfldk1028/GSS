#!/bin/bash

# Verification script for torch CUDA availability
# This script checks that torch.cuda.is_available() returns True

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Torch CUDA Availability Verification"
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

# Check CUDA availability
echo "Checking torch CUDA availability..."
CUDA_AVAILABLE=$(conda run -n gss python -c "import torch; print(torch.cuda.is_available())" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully imported torch${NC}"

    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        echo -e "${GREEN}✓ CUDA is AVAILABLE${NC}"

        # Get additional CUDA info
        CUDA_VERSION=$(conda run -n gss python -c "import torch; print(torch.version.cuda)" 2>&1)
        DEVICE_COUNT=$(conda run -n gss python -c "import torch; print(torch.cuda.device_count())" 2>&1)
        DEVICE_NAME=$(conda run -n gss python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1)

        echo ""
        echo "CUDA Information:"
        echo "  CUDA Version: $CUDA_VERSION"
        echo "  Device Count: $DEVICE_COUNT"
        echo "  Device Name: $DEVICE_NAME"
    else
        echo -e "${RED}✗ CUDA is NOT AVAILABLE${NC}"
        echo -e "${YELLOW}Expected: True${NC}"
        echo -e "${YELLOW}Actual: $CUDA_AVAILABLE${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Failed to import torch or check CUDA availability${NC}"
    echo "$CUDA_AVAILABLE"
    exit 1
fi

echo ""
echo "========================================"
echo "Verification Complete"
echo "========================================"
echo ""
echo "Expected: True"
echo "Actual: $CUDA_AVAILABLE"
echo ""
echo -e "${GREEN}✓ SUBTASK-6 VERIFICATION PASSED${NC}"
echo ""
