# Conda Environment Setup Instructions

## ⚠️ MANUAL ACTION REQUIRED

The automated system cannot execute `conda` commands due to security restrictions.
You must create the conda environment manually.

## Quick Setup (Recommended)

### On Windows:
```bash
./create_conda_env.bat
```

### On Linux/Mac:
```bash
chmod +x create_conda_env.sh
./create_conda_env.sh
```

## Manual Setup

If the scripts don't work, follow these steps:

### Step 1: Create the Environment
```bash
conda env create -f environment.yml
```

This will:
- Create a new conda environment named `gss`
- Install Python 3.10
- Install PyTorch with CUDA 12.1 support
- Install all required dependencies (numpy<2.0, matplotlib, gsplat, etc.)
- Install the gss package in editable mode

### Step 2: Verify Installation
```bash
# Activate the environment
conda activate gss

# Verify the environment exists
conda env list | grep gss

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check other imports
python -c "import gsplat; import matplotlib; import gss; print('All imports OK')"
```

### Expected Output:
- `conda env list | grep gss` should show the gss environment
- `torch.cuda.is_available()` should return `True`
- All imports should work without errors

## Troubleshooting

### If the environment already exists:
```bash
# Remove the old environment
conda env remove -n gss

# Create it again
conda env create -f environment.yml
```

### If you get CUDA errors:
- Ensure you have NVIDIA drivers installed (version 581.83 or compatible)
- Check CUDA version: `nvidia-smi`
- The environment.yml uses CUDA 12.1, which should work with your driver

### If numpy compatibility issues persist:
- Verify numpy version: `conda run -n gss python -c "import numpy; print(numpy.__version__)"`
- Should be < 2.0 (e.g., 1.26.x)

## After Setup

Once the environment is created, the automated system can continue with the remaining subtasks:
- subtask-3: Install PyTorch with CUDA support (already done via environment.yml)
- subtask-4: Install gsplat (already done via environment.yml)
- subtask-5: Install gss package in editable mode (already done via environment.yml)
- subtask-6: Verify torch CUDA availability
- subtask-7: Verify matplotlib imports
- subtask-8: Run existing tests

## Status Update

After creating the environment, update the task status:
1. Verify the environment exists: `conda env list | grep gss`
2. If successful, the automated system will detect the environment and continue
