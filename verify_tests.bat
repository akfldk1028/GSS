@echo off
REM Verification Script for GSS Test Suite
REM Subtask 8: Run existing tests to verify environment

echo ========================================
echo GSS Test Suite Verification
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda not found. Please install Anaconda or Miniconda.
    exit /b 1
)

REM Check if gss environment exists
conda env list | findstr /B "gss " >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda environment 'gss' not found.
    echo Please run the init.sh script first to create the environment.
    exit /b 1
)

echo [OK] conda environment 'gss' found
echo.

REM Run pytest with verbose output
echo Running pytest tests/ -v...
echo.
call conda run -n gss pytest tests/ -v

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo [OK] All tests passed!
    echo ========================================
    echo.
    echo Environment verification complete.
    echo The GSS environment is ready for use.
    echo.
) else (
    echo.
    echo ========================================
    echo [FAILED] Some tests failed!
    echo ========================================
    echo.
    echo Please review the test output above to identify issues.
    exit /b 1
)
