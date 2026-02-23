# S04: Depth/Normal Rendering

Render depth and normal maps from trained Gaussian model.

> **Note**: In PlanarGS pipeline (`pipeline_planargs.yaml`), this step is skipped.
> PlanarGS handles depth rendering internally within `s03_planargs`.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `model_path` | Path | Trained model (.ply) |
| `sparse_dir` | Path | COLMAP sparse dir (camera poses) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `depth_dir` | Path | Depth maps (.npy) |
| `normal_dir` | Path or None | Normal maps (.npy) |
| `num_views` | int | Views rendered |
| `poses_file` | Path | poses.json (intrinsics + per-view c2w) |

## Config (`configs/steps/s04_depth_render.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `num_views` | 400 | Views to render |
| `render_normals` | true | Render normal maps |
| `render_resolution_scale` | 1.0 | Resolution scale |
| `view_selection` | uniform | uniform / coverage / all |

## Dependencies
- `torch`, `gsplat`
