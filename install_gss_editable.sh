#!/bin/bash
# Install gss package in editable mode in the conda environment

set -e

echo "Installing gss package in editable mode..."
conda run -n gss pip install -e .

echo "Verifying installation..."
conda run -n gss python -c "import gss; print('OK')"

echo "gss package installation complete!"
