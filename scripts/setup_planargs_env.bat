@echo off
REM Setup PlanarGS conda environment for GSS pipeline
REM Run from GSS project root: scripts\setup_planargs_env.bat

echo === PlanarGS Environment Setup ===

REM Step 1: Create conda env
echo [1/5] Creating conda environment 'planargs'...
call conda create -n planargs python=3.10 -y
if %ERRORLEVEL% neq 0 (echo FAILED: conda create & exit /b 1)

REM Step 2: Install PyTorch (cu124 for RTX 5080 compat)
echo [2/5] Installing PyTorch 2.4.1+cu124...
call conda run -n planargs pip install cmake==3.20.*
call conda run -n planargs pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% neq 0 (echo FAILED: PyTorch install & exit /b 1)

REM Step 3: Install PlanarGS requirements
echo [3/5] Installing PlanarGS requirements...
call conda run -n planargs pip install -r clone\PlanarGS\requirements.txt
if %ERRORLEVEL% neq 0 (echo FAILED: requirements.txt & exit /b 1)

REM Step 4: Build CUDA submodules (requires MSVC)
echo [4/5] Building CUDA submodules (requires MSVC environment)...
echo NOTE: If this fails, run from a "Developer Command Prompt for VS" or
echo       run "vcvarsall.bat amd64" first.

call conda run -n planargs pip install -e clone\PlanarGS\submodules\simple-knn --no-build-isolation
if %ERRORLEVEL% neq 0 (echo WARNING: simple-knn build failed)

call conda run -n planargs pip install -e clone\PlanarGS\submodules\pytorch3d --no-build-isolation
if %ERRORLEVEL% neq 0 (echo WARNING: pytorch3d build failed)

call conda run -n planargs pip install clone\PlanarGS\submodules\diff-plane-rasterization --no-build-isolation
if %ERRORLEVEL% neq 0 (echo WARNING: diff-plane-rasterization build failed)

REM Step 5: GroundedSAM
echo [5/5] Installing GroundedSAM...
if not exist clone\PlanarGS\submodules\groundedsam (
    cd clone\PlanarGS\submodules
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git groundedsam
    cd ..\..\..
)
call conda run -n planargs pip install -e clone\PlanarGS\submodules\groundedsam\segment_anything
call conda run -n planargs pip install --no-build-isolation -e clone\PlanarGS\submodules\groundedsam\GroundingDINO
if %ERRORLEVEL% neq 0 (echo WARNING: GroundedSAM build failed)

echo.
echo === Setup Complete ===
echo.
echo Checkpoints still needed (~4.2GB total) in clone\PlanarGS\ckpt\:
echo   1. DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth (~1GB)
echo      URL: https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
echo   2. groundingdino_swint_ogc.pth (~700MB)
echo      URL: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
echo   3. sam_vit_h_4b8939.pth (~2.5GB)
echo      URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
echo.
echo Verify: conda run -n planargs python -c "import torch; print(torch.cuda.is_available())"
echo Verify: conda run -n planargs python -c "import diff_plane_rasterization; print('OK')"
echo Verify: conda run -n planargs python -c "import simple_knn; print('OK')"
