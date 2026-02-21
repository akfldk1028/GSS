#!/bin/bash

# Verification Script for GSS Test Suite
# Subtask 8: Run existing tests to verify environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "GSS Test Suite Verification"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda or Miniconda.${NC}"
    exit 1
fi

# Check if gss environment exists
if ! conda env list | grep -q "^gss "; then
    echo -e "${RED}Error: conda environment 'gss' not found.${NC}"
    echo "Please run the init.sh script first to create the environment."
    exit 1
fi

echo -e "${GREEN}✓ conda environment 'gss' found${NC}"
echo ""

# Run pytest with verbose output
echo -e "${YELLOW}Running pytest tests/ -v...${NC}"
echo ""
conda run -n gss pytest tests/ -v

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo "========================================"
    echo ""
    echo "Environment verification complete."
    echo "The GSS environment is ready for use."
else
    echo -e "${RED}✗ Some tests failed!${NC}"
    echo "========================================"
    echo ""
    echo "Please review the test output above to identify issues."
    exit 1
fi

echo ""
