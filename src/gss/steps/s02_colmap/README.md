# S02: COLMAP SfM

Structure-from-Motion for camera pose estimation. Uses pycolmap Python API with COLMAP CLI fallback.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `frames_dir` | Path | Extracted frames from S01 |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `sparse_dir` | Path | COLMAP sparse reconstruction |
| `cameras_file` | Path | cameras.bin |
| `images_file` | Path | images.bin |
| `num_registered` | int | Registered images |
| `num_points3d` | int | 3D points in sparse cloud |

## Config (`configs/steps/s02_colmap.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `use_gpu` | true | GPU feature extraction |
| `matcher` | sequential | sequential / exhaustive / vocab_tree |
| `match_window` | 10 | Sequential overlap window |
| `single_camera` | true | Share intrinsics |
| `camera_model` | OPENCV | PINHOLE / OPENCV / RADIAL |
| `max_num_features` | 8192 | SIFT features per image |

## Dependencies
- `pycolmap` (preferred) or COLMAP CLI in PATH
