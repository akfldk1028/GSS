# Step 03-alt: PlanarGS (NeurIPS 2025)

## Purpose
Replaces s03 (gsplat) + s04 (depth render) + s05 (TSDF fusion) with PlanarGS,
which uses co-planarity loss + GroundedSAM + DUSt3R priors for 2.5x better
indoor surface reconstruction (Chamfer distance).

## Pipeline Position
```
s01 → s02 → s03_planargs → s06 → s07
```
Config: `configs/pipeline_planargs.yaml`

## Input
| Field | Type | Description |
|-------|------|-------------|
| `frames_dir` | Path | Training images (from s01) |
| `sparse_dir` | Path | COLMAP sparse/0/ reconstruction (from s02) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `surface_points_path` | Path | Mesh vertices as PLY point cloud (for s06) |
| `mesh_path` | Path | Full triangle mesh (`tsdf_fusion_post.ply`) |
| `metadata_path` | Path | Run metadata JSON |
| `num_surface_points` | int | Number of surface points extracted |

## Config (`configs/steps/s03_planargs.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `planargs_repo` | `clone/PlanarGS` | Path to PlanarGS repository |
| `conda_env` | `planargs` | Conda env name |
| `group_size` | 25 | DUSt3R group size (25 for 16GB, 40 for 24GB VRAM) |
| `text_prompts` | `wall. floor. door...` | GroundedSAM detection text |
| `iterations` | 30000 | GS training iterations |
| `voxel_size` | 0.02 | TSDF voxel size for mesh extraction |
| `max_depth` | 100.0 | Max depth for TSDF fusion |
| `skip_geomprior` | false | Skip DUSt3R step |
| `skip_lp3` | false | Skip GroundedSAM step |
| `skip_train` | false | Skip training step |

## Subprocess Pipeline
1. `run_geomprior.py` -- DUSt3R depth/normal prior generation
2. `run_lp3.py` -- GroundedSAM planar mask extraction
3. `train.py` -- 30K iteration Gaussian splatting with co-planarity loss
4. `render.py` -- Depth rendering + TSDF mesh extraction

## Tools Used
- PlanarGS (`clone/PlanarGS`) via subprocess in `planargs` conda env
- DUSt3R: Geometric depth/normal priors
- GroundedSAM: Planar region segmentation
- diff-plane-rasterization: Plane-aware Gaussian rasterizer
- Open3D: Mesh to point cloud conversion

## Test Results (Replica room0, 2026-02-23)
| Substep | Time | Result |
|---------|------|--------|
| geomprior (DUSt3R) | ~1 min | 14 depth/normal maps |
| lp3 (GroundedSAM) | ~30s | 14 planar masks |
| train (30K iters) | ~35 min | PSNR 40.13, 481K gaussians |
| render (TSDF) | ~7 min | 263K clusters |
| **Total** | **~45 min** | **21.4M surface points** |

Output quality: Clear rectangular room with flat walls, floor, ceiling visible.

### Surface Reconstruction
![PlanarGS Surface](../../../../../../docs/images/planargs_surface.png)

### Plane Detection (s06 v4: Manhattan + merge + position filter)
- **3 walls, 1 floor, 2 ceilings** (from 8w/7f/0c before improvements)
- Manhattan World alignment auto-detects axis, coplanar merge reduces fragmentation
- Position-based classification separates furniture from real floor/ceiling

![Planes 3D](../../../../../../docs/images/planargs_planes_3d.png)
![Planes Top-Down](../../../../../../docs/images/planargs_planes_topdown.png)
![Planes Front](../../../../../../docs/images/planargs_planes_front.png)

## Environment Setup
Requires separate conda environment (`planargs`).
- Setup: `scripts/setup_planargs_env.bat`
- CUDA builds: `scripts/_build_planargs_cuda.bat`
- Verify: `scripts/_verify_planargs.py`

## Dependencies (planargs conda env)
- `torch 2.8.0+cu128`
- `diff-plane-rasterization` (CUDA, from submodules)
- `simple-knn` (CUDA, from submodules)
- `pytorch3d 0.7.9` (CUDA, from submodules)
- `segment_anything`, `groundingdino` (GroundedSAM)
- `transformers<5`, `open3d`

## Patches Applied (torch 2.8 compatibility)
- `dust3r/model.py`: `weights_only=False`
- `train.py`: `weights_only=False`
- `lp3/run_groundedsam.py`: `weights_only=False`
- `run_geomprior.py`: IMAGE_EXTS filter
- `GroundingDINO ms_deform_attn_cuda.cu`: deprecated API fix
