# Step 00: Import External PLY

## Purpose
Import an external 3DGS Gaussian PLY or plain point cloud PLY into the GSS pipeline,
producing `surface_points.ply` + `metadata.json` compatible with s06 Plane Extraction.

This enables skipping steps s01-s05 when starting from a pre-trained 3DGS model or
an existing point cloud.

## PLY Format Auto-Detection
- **Gaussian PLY**: Contains `f_dc_0`, `scale_0`, `rot_0` properties. XYZ centers are
  extracted, with optional opacity-based filtering (`sigmoid(opacity) >= min_opacity`).
- **Plain point cloud PLY**: Standard XYZ (+ optional normals/colors). Loaded directly
  via Open3D.

## Inputs
| Field | Type | Description |
|-------|------|-------------|
| `ply_path` | Path | Path to input PLY file |

## Outputs
| Field | Type | Description |
|-------|------|-------------|
| `surface_points_path` | Path | `surface_points.ply` (Open3D point cloud) |
| `metadata_path` | Path | `metadata.json` with source info |
| `num_surface_points` | int | Number of output points |

## Processing Pipeline
1. Detect PLY format (Gaussian vs plain)
2. Load points (with opacity filtering for Gaussian PLY)
3. Voxel downsampling (optional)
4. Statistical outlier removal (optional)
5. Normal estimation (if missing)

## Tools Used
- `plyfile` — Read Gaussian PLY custom attributes (SH, scale, rotation, opacity)
- `open3d` — Point cloud processing (outlier removal, normals, voxel downsample, I/O)
