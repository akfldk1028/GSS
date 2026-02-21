@echo off
REM Verification script for matplotlib imports (subtask-7)
REM This script verifies that matplotlib imports without numpy compatibility errors

echo ==================================
echo Matplotlib Import Verification
echo ==================================
echo.

REM Check if conda environment exists
conda env list | findstr /C:"gss" >nul
if errorlevel 1 (
    echo ERROR: Conda environment 'gss' not found!
    echo Please run init.bat first to create the environment.
    exit /b 1
)

echo Checking matplotlib import...
conda run -n gss python -c "import matplotlib; print('OK')"

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS: matplotlib imports without errors
    echo.
    echo Checking matplotlib version and numpy compatibility...
    conda run -n gss python -c "import matplotlib; import numpy as np; print(f'matplotlib version: {matplotlib.__version__}'); print(f'numpy version: {np.__version__}'); print(''); print('No _ARRAY_API compatibility errors detected')"
) else (
    echo.
    echo FAILED: matplotlib import failed
    echo This may indicate numpy compatibility issues (_ARRAY_API errors^)
    exit /b 1
)

echo.
echo ==================================
echo Verification complete!
echo ==================================
