#!/bin/bash
# Helper script to create the GSS conda environment
# Run this script manually from the project root

echo "========================================"
echo "Creating GSS Conda Environment"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH"
    echo "Please ensure Anaconda or Miniconda is installed and in your PATH"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "ERROR: environment.yml not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "Creating conda environment from environment.yml..."
echo "This may take several minutes..."
echo

conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "SUCCESS: Conda environment 'gss' created"
    echo "========================================"
    echo
    echo "Next steps:"
    echo "1. Activate the environment: conda activate gss"
    echo "2. Verify installation: python -c \"import torch; print(torch.cuda.is_available())\""
    echo
else
    echo
    echo "========================================"
    echo "ERROR: Failed to create conda environment"
    echo "========================================"
    echo
    echo "Please check the error messages above"
    echo
fi
