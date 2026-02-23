# S05: TSDF Fusion

Fuse depth maps into volumetric TSDF and extract surface point cloud.

> **Note**: In PlanarGS pipeline (`pipeline_planargs.yaml`), this step is skipped.
> PlanarGS handles TSDF fusion internally within `s03_planargs`.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `depth_dir` | Path | Depth maps from S04 |
| `poses_file` | Path | poses.json from S04 |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `tsdf_dir` | Path | `data/interim/s05_tsdf/` |
| `surface_points_path` | Path | surface_points.ply |
| `num_surface_points` | int | Points extracted |
| `metadata_path` | Path | metadata.json |

## Config (`configs/steps/s05_tsdf_fusion.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `voxel_size` | 0.006 | Voxel size (6mm) |
| `sdf_trunc` | 0.04 | SDF truncation (m) |
| `depth_trunc` | 5.0 | Max depth (m) |
| `depth_scale` | 1.0 | Depth unit scale |
| `use_gpu` | false | GPU pipeline |

## Dependencies
- `open3d >= 0.17`
