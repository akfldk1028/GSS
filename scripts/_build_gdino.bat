@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set DISTUTILS_USE_SDK=1
cd /d C:\DK\GSS\clone\PlanarGS\submodules\groundedsam
echo === Building GroundingDINO ===
conda run -n planargs pip install --no-build-isolation -e GroundingDINO 2>&1
echo === ERRORLEVEL=%ERRORLEVEL% ===
conda run -n planargs python -c "import groundingdino; print('groundingdino OK')" 2>&1
