@echo off
REM PyTorch DLL Fix - Approach 1: Use Conda's PyTorch Package
REM This approach uses conda's PyTorch instead of pip, which has better DLL dependency management on Windows

echo ========================================
echo PyTorch DLL Fix - Conda Approach
echo ========================================
echo.
echo This script will:
echo   1. Remove pip-installed PyTorch
echo   2. Reinstall PyTorch via conda (better DLL management)
echo   3. Verify imports work
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: conda not found.
    exit /b 1
)

echo Step 1: Removing pip-installed PyTorch...
conda run -n gss pip uninstall -y torch torchvision 2>nul
echo Done.

echo.
echo Step 2: Installing PyTorch via conda...
conda install -n gss pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
echo Done.

echo.
echo Step 3: Reinstalling gsplat...
conda run -n gss pip install --force-reinstall gsplat
echo Done.

echo.
echo ========================================
echo Verification
echo ========================================
echo.

echo Testing PyTorch import...
conda run -n gss python -c "import torch; print(f'torch {torch.__version__}')"
if %errorlevel% neq 0 (
    echo FAILED
    echo.
    echo PyTorch still fails to import. Try the CPU-only approach:
    echo   fix_pytorch_cpu.bat
    exit /b 1
)
echo SUCCESS

echo.
echo Testing CUDA availability...
conda run -n gss python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo Testing gsplat import...
conda run -n gss python -c "import gsplat; print('OK')"

echo.
echo Testing matplotlib import...
conda run -n gss python -c "import matplotlib; print('OK')"

echo.
echo Testing gss package import...
conda run -n gss python -c "import gss; print('OK')"

echo.
echo ========================================
echo Fix Complete!
echo ========================================
echo.
echo If all tests passed, the PyTorch DLL issue is resolved.
echo If PyTorch still fails, try the CPU-only approach:
echo   fix_pytorch_cpu.bat
echo.
pause
