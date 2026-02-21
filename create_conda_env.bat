@echo off
REM Helper script to create the GSS conda environment
REM Run this script manually from the project root

echo ========================================
echo Creating GSS Conda Environment
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: conda not found in PATH
    echo Please ensure Anaconda or Miniconda is installed and in your PATH
    pause
    exit /b 1
)

REM Check if environment.yml exists
if not exist "environment.yml" (
    echo ERROR: environment.yml not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo Creating conda environment from environment.yml...
echo This may take several minutes...
echo.

conda env create -f environment.yml

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS: Conda environment 'gss' created
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Activate the environment: conda activate gss
    echo 2. Verify installation: python -c "import torch; print(torch.cuda.is_available())"
    echo.
) else (
    echo.
    echo ========================================
    echo ERROR: Failed to create conda environment
    echo ========================================
    echo.
    echo Please check the error messages above
    echo.
)

pause
