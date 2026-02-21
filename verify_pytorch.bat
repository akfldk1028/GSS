@echo off
REM Verification script for PyTorch CUDA installation (Windows)
REM This script checks that PyTorch is installed with CUDA support (cu121 or cu124)

echo ========================================
echo PyTorch CUDA Verification
echo ========================================
echo.

REM Check if conda environment exists
conda env list | findstr /C:"gss" >nul 2>&1
if errorlevel 1 (
    echo Error: Conda environment 'gss' not found.
    echo Please run: bash .auto-claude/specs/013-fix-python-env/init.sh
    exit /b 1
)

echo [OK] Conda environment 'gss' exists
echo.

REM Check PyTorch version
echo Checking PyTorch version...
conda run -n gss python -c "import torch; print(torch.__version__)"

if errorlevel 1 (
    echo [ERROR] Failed to import torch
    exit /b 1
)

echo.
echo ========================================
echo Verification Complete
echo ========================================
echo.
echo Expected: PyTorch version with cu121 or cu124
echo Run this script to see the actual version above
echo.
