# PyTorch DLL Loading Fix Guide

## Problem

PyTorch fails to import in the `gss` conda environment with the error:
```
OSError: [WinError 182] Error loading "fbgemm.dll" or one of its dependencies
```

This prevents both PyTorch and gsplat from working, blocking the core functionality of the GSS pipeline.

## Root Cause

The current setup uses pip to install PyTorch with CUDA support, which can have DLL dependency issues on Windows. The pip wheels may be missing or have incompatible Visual C++ runtime or CUDA library dependencies.

## Solutions (Try in Order)

### ✅ Approach 1: Use Conda's PyTorch (RECOMMENDED)

Conda packages have better DLL dependency management on Windows.

**Windows:**
```batch
fix_pytorch_conda.bat
```

**Linux/Mac:**
```bash
bash fix_pytorch_conda.sh
```

**What it does:**
1. Removes pip-installed PyTorch
2. Installs PyTorch via conda (better DLL handling)
3. Reinstalls gsplat
4. Verifies all imports work

**Expected outcome:** PyTorch imports successfully with CUDA support

---

### ⚠️ Approach 2: CPU-Only PyTorch (FALLBACK)

If Approach 1 fails, use CPU-only PyTorch which has fewer DLL dependencies.

**Windows:**
```batch
fix_pytorch_cpu.bat
```

**Linux/Mac:**
```bash
bash fix_pytorch_cpu.sh
```

**What it does:**
1. Removes current PyTorch
2. Installs CPU-only PyTorch (pip)
3. Reinstalls gsplat
4. Verifies imports work

**Expected outcome:** PyTorch imports successfully, but `torch.cuda.is_available()` returns `False`

**Note:** This changes the acceptance criteria - CUDA won't be available, but torch and gsplat will work.

---

### 🔍 Diagnostic Script

If both approaches fail, run the diagnostic script to identify system-level issues:

**Windows:**
```batch
C:\Users\User\anaconda3\envs\gss\python.exe diagnose_torch_dlls.py
```

**What it checks:**
- Python version and platform
- DLL file existence (fbgemm.dll, c10.dll)
- Visual C++ redistributables
- CUDA paths in system PATH
- CUDA runtime DLLs

**Common fix:** If Visual C++ redistributables are missing, install them:
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and reboot
- Then try Approach 1 again

---

## Verification

After applying a fix, verify all imports work:

```bash
# Windows
C:\Users\User\anaconda3\envs\gss\python.exe test_imports_simple.py

# Or via conda (if conda commands work)
conda run -n gss python test_imports_simple.py
```

**Expected output:**
```
Python: 3.10.19 | packaged by conda-forge | ...
--------------------------------------------------
PASS: torch 2.x.x
  CUDA available: True (or False for CPU-only)
PASS: gsplat
PASS: matplotlib
PASS: gss
```

---

## Files Created

### Configuration Files
- `environment.yml` - Updated with conda PyTorch and build package
- `environment_conda_pytorch.yml` - Conda PyTorch configuration
- `environment_cpu_pytorch.yml` - CPU-only PyTorch configuration

### Fix Scripts
- `fix_pytorch_conda.sh` / `.bat` - Approach 1 (conda PyTorch)
- `fix_pytorch_cpu.sh` / `.bat` - Approach 2 (CPU-only PyTorch)
- `diagnose_torch_dlls.py` - Diagnostic script

### Test Scripts
- `test_imports_simple.py` - Verify all imports work

---

## Additional Fixes Applied

### ✅ Lint Errors Fixed
Removed unused imports in `src/gss/steps/s01_extract_frames/step.py`:
- Line 6: `from pathlib import Path` (removed)
- Line 30: `import numpy as np` (removed)

Verification:
```bash
ruff check src/gss/steps/s01_extract_frames/step.py
# Result: All checks passed!
```

### ✅ Build Module Added
Added `python-build>=0.10` to `environment.yml` to fix:
```bash
python -m build
# No longer fails with "No module named build"
```

---

## Quick Start

**For most users (Windows):**
1. Run `fix_pytorch_conda.bat`
2. If that fails, run `fix_pytorch_cpu.bat`
3. If both fail, run diagnostics: `C:\Users\User\anaconda3\envs\gss\python.exe diagnose_torch_dlls.py`

**For Linux/Mac users:**
1. Run `bash fix_pytorch_conda.sh`
2. If that fails, run `bash fix_pytorch_cpu.sh`

---

## Success Criteria

After applying fixes, ALL of the following must pass:

1. ✅ PyTorch imports: `python -c "import torch; print('OK')"`
2. ✅ CUDA available: `python -c "import torch; print(torch.cuda.is_available())"` → True (or False for CPU-only)
3. ✅ gsplat imports: `python -c "import gsplat; print('OK')"`
4. ✅ matplotlib imports: `python -c "import matplotlib; print('OK')"`
5. ✅ gss package imports: `python -c "import gss; print('OK')"`
6. ✅ Tests pass: `pytest tests/ -v` → 1 passed
7. ✅ Lint passes: `ruff check src/gss/steps/s01_extract_frames/step.py` → All checks passed!

---

## If Nothing Works

If all approaches fail:

1. **Check system requirements:**
   - Visual C++ 2015-2022 Redistributable (x64) installed
   - CUDA 12.1 or compatible driver installed (for CUDA support)
   - Windows 10/11 64-bit

2. **Try manual installation:**
   ```batch
   conda activate gss

   # Remove PyTorch
   pip uninstall torch torchvision

   # Try different PyTorch version
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

   # Or try PyTorch 2.3.x
   pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Contact support:**
   - Document your findings from `diagnose_torch_dlls.py`
   - Note which approaches you tried
   - Include error messages from failed import attempts

---

## Summary

- **Easy fix:** Run `fix_pytorch_conda.bat` (Windows) or `bash fix_pytorch_conda.sh` (Linux/Mac)
- **Fallback:** Run CPU-only version if CUDA version fails
- **Lint errors:** Already fixed in `src/gss/steps/s01_extract_frames/step.py`
- **Build module:** Already added to `environment.yml`
