@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set DISTUTILS_USE_SDK=1
set FORCE_CUDA=1
cd /d C:\DK\GSS
echo === Building pytorch3d from submodule ===
conda run -n planargs pip install -e clone\PlanarGS\submodules\pytorch3d --no-build-isolation 2>&1
echo ERRORLEVEL=%ERRORLEVEL%
