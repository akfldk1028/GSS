# Step 03 (3DGRUT): NVIDIA 3D Gaussian Ray Tracing

## Purpose
Train a 3D Gaussian scene representation using NVIDIA's 3DGRUT framework.
Alternative to the default gsplat-based s03_gaussian_splatting step.

3DGRUT supports both rasterization (3DGUT) and ray-tracing (3DGRT) backends,
with built-in USDZ export for Omniverse/ARKit integration.

## Inputs
| Field | Type | Source |
|-------|------|--------|
| frames_dir | Path | s01_extract_frames — training images |
| sparse_dir | Path | s02_colmap — COLMAP sparse reconstruction |

## Outputs
| Field | Type | Destination |
|-------|------|-------------|
| model_path | Path (PLY) | s04~s08 downstream pipeline |
| checkpoint_path | Path (.pt) | gRPC render server |
| usdz_path | Path (USDZ) | Omniverse / ARKit direct load |
| num_gaussians | int | Metadata |
| training_iterations | int | Metadata |

## Tools Used
- **3DGRUT** (clone/3dgrut/) — Hydra-based trainer with COLMAP dataset support
- **subprocess** — Invokes 3DGRUT training as isolated process (separate CUDA context)

## Configuration
- `method`: `3dgut` (rasterization, faster) or `3dgrt` (ray tracing, higher quality)
- `config_name`: Hydra config path (default: `apps/colmap_3dgut`)
- `num_iterations`: Training iterations (default: 30000)
- `export_ply` / `export_usdz`: Toggle output formats

## Pipeline
Used in `configs/pipeline_3dgrut.yaml` as drop-in replacement for s03_gaussian_splatting.
