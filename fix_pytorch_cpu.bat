@echo off
REM PyTorch DLL Fix - Approach 2: Use CPU-Only PyTorch
REM This approach uses CPU-only PyTorch which has fewer DLL dependencies

echo ========================================
echo PyTorch DLL Fix - CPU-Only Approach
echo ========================================
echo.
echo WARNING: This will install CPU-only PyTorch
echo CUDA will NOT be available
echo.
echo This script will:
echo   1. Remove current PyTorch installation
echo   2. Install CPU-only PyTorch (pip)
echo   3. Verify imports work
echo.
set /p CONFIRM="Continue? (y/n): "
if /i not "%CONFIRM%"=="y" exit /b 0

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: conda not found.
    exit /b 1
)

echo.
echo Step 1: Removing current PyTorch...
conda run -n gss pip uninstall -y torch torchvision 2>nul
echo Done.

echo.
echo Step 2: Installing CPU-only PyTorch...
conda run -n gss pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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
    echo PyTorch still fails to import. This may be a system-level issue.
    echo Check if Visual C++ redistributables are installed:
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    exit /b 1
)
echo SUCCESS

echo.
echo Testing CUDA availability (should be False)...
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
echo NOTE: CUDA is NOT available in CPU-only mode
echo If you need CUDA, try installing Visual C++ redistributables and then:
echo   fix_pytorch_conda.bat
echo.
pause
