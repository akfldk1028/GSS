#!/bin/bash
# Verification script for matplotlib imports (subtask-7)
# This script verifies that matplotlib imports without numpy compatibility errors

echo "=================================="
echo "Matplotlib Import Verification"
echo "=================================="
echo ""

# Check if conda environment exists
if ! conda env list | grep -q 'gss'; then
    echo "❌ ERROR: Conda environment 'gss' not found!"
    echo "Please run init.sh first to create the environment."
    exit 1
fi

echo "📦 Checking matplotlib import..."
conda run -n gss python -c "import matplotlib; print('OK')"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: matplotlib imports without errors"
    echo ""
    echo "📊 Checking matplotlib version and numpy compatibility..."
    conda run -n gss python -c "
import matplotlib
import numpy as np
print(f'matplotlib version: {matplotlib.__version__}')
print(f'numpy version: {np.__version__}')
print('')
print('✅ No _ARRAY_API compatibility errors detected')
"
else
    echo ""
    echo "❌ FAILED: matplotlib import failed"
    echo "This may indicate numpy compatibility issues (_ARRAY_API errors)"
    exit 1
fi

echo ""
echo "=================================="
echo "Verification complete!"
echo "=================================="
