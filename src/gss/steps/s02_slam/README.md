# Step 02-alt: GS-SLAM

Replaces COLMAP(s02) + 3DGS training(s03) with a single real-time SLAM step.

## Purpose
Eliminate the 90%+ pipeline bottleneck by using monocular SLAM systems that
simultaneously estimate camera poses and reconstruct dense 3D geometry.

## Backends

| Backend | Paper | Speed | Output |
|---------|-------|-------|--------|
| MASt3R-SLAM | CVPR 2025 | ~15 FPS | Dense 3D pointmap + poses |
| MonoGS | CVPR 2024 Highlight | 3-10 FPS | 3DGS PLY + poses |
| SLAM3R | CVPR 2025 Highlight | 20+ FPS | Dense point cloud |

## Inputs
- `frames_dir` — Directory of images (s01 output)

## Outputs
- `surface_points_path` — Dense point cloud PLY with normals (s06 compatible)
- `metadata_path` — JSON metadata
- `camera_poses_path` — Camera poses (optional, backend-dependent)
- `num_surface_points` — Point count

## Post-processing
1. Open3D voxel downsample (configurable voxel size)
2. Open3D normal estimation + consistent orientation (if SLAM output lacks normals)

## Tools Used
- SLAM backend (conda subprocess, Pattern C)
- Open3D (point cloud processing)
