@echo off
REM Verification script for torch CUDA availability (Windows)
REM This script checks that torch.cuda.is_available() returns True

echo ========================================
echo Torch CUDA Availability Verification
echo ========================================
echo.

REM Check if conda environment exists
conda env list | findstr /B "gss " >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda environment 'gss' not found.
    echo Please run: .auto-claude\specs\013-fix-python-env\init.sh
    exit /b 1
)

echo [OK] Conda environment 'gss' exists
echo.

REM Check CUDA availability
echo Checking torch CUDA availability...
conda run -n gss python -c "import torch; print(torch.cuda.is_available())" > cuda_check.tmp 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to import torch or check CUDA availability
    type cuda_check.tmp
    del cuda_check.tmp
    exit /b 1
)

set /p CUDA_AVAILABLE=<cuda_check.tmp
del cuda_check.tmp

echo [OK] Successfully imported torch
echo.

if "%CUDA_AVAILABLE%"=="True" (
    echo [OK] CUDA is AVAILABLE
    echo.

    REM Get additional CUDA info
    echo CUDA Information:
    echo   CUDA Version:
    conda run -n gss python -c "import torch; print('    ' + str(torch.version.cuda))"
    echo   Device Count:
    conda run -n gss python -c "import torch; print('    ' + str(torch.cuda.device_count()))"
    echo   Device Name:
    conda run -n gss python -c "import torch; print('    ' + str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'))"

    echo.
    echo ========================================
    echo Verification Complete
    echo ========================================
    echo.
    echo Expected: True
    echo Actual: %CUDA_AVAILABLE%
    echo.
    echo [OK] SUBTASK-6 VERIFICATION PASSED
    echo.
) else (
    echo [ERROR] CUDA is NOT AVAILABLE
    echo Expected: True
    echo Actual: %CUDA_AVAILABLE%
    exit /b 1
)
