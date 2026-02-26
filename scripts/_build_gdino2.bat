@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set DISTUTILS_USE_SDK=1
cd /d C:\DK\GSS\clone\PlanarGS\submodules\groundedsam
conda run -n planargs pip install --no-build-isolation -e GroundingDINO > C:\DK\GSS\gdino_build.log 2>&1
echo ERRORLEVEL=%ERRORLEVEL% >> C:\DK\GSS\gdino_build.log
