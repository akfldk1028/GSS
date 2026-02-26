@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set DISTUTILS_USE_SDK=1
cd /d C:\DK\GSS
echo === Building diff-plane-rasterization ===
conda run -n planargs pip install clone\PlanarGS\submodules\diff-plane-rasterization --no-build-isolation 2>&1
echo ERRORLEVEL=%ERRORLEVEL%
echo === Verifying ===
conda run -n planargs python -c "import diff_plane_rasterization; print('diff_plane_rasterization OK')"
