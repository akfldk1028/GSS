# S03: 3D Gaussian Splatting (gsplat)

Train 2DGS/3DGS model from COLMAP reconstruction using gsplat.

> **Note**: For indoor scenes, prefer `s03_planargs` (PlanarGS) which replaces s03+s04+s05 with
> co-planarity-aware reconstruction. See `configs/pipeline_planargs.yaml`.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `frames_dir` | Path | Training images |
| `sparse_dir` | Path | COLMAP sparse reconstruction |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `model_path` | Path | Trained model (point_cloud.ply) |
| `num_gaussians` | int | Number of Gaussians |
| `training_iterations` | int | Iterations completed |

Also saves `model.pt` checkpoint for S04 depth rendering.

## Config (`configs/steps/s03_gaussian_splatting.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `method` | 2dgs | 2dgs / 3dgs |
| `iterations` | 30000 | Training iterations |
| `learning_rate` | 1.6e-4 | Position learning rate |
| `densify_until` | 15000 | Densification cutoff |
| `sh_degree` | 3 | Spherical harmonics degree |

## Dependencies
- `torch` (with CUDA)
- `gsplat`
