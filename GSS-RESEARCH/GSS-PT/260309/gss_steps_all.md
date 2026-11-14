# GSS Pipeline Steps Overview

---

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

---

# S01: Extract Frames

Extract frames from video with blur filtering and optional resize.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `video_path` | Path | Input video file (.mp4, .avi, etc.) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `frames_dir` | Path | `data/interim/s01_frames/` |
| `frame_count` | int | Number of extracted frames |
| `fps_used` | float | Effective extraction FPS |
| `frame_list` | list[str] | Frame filenames |

## Config (`configs/steps/s01_extract_frames.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `target_fps` | 2.0 | Target extraction rate |
| `max_frames` | 2000 | Maximum frames to extract |
| `output_format` | png | Image format (png/jpg) |
| `blur_threshold` | 100.0 | Laplacian variance threshold |
| `resize_width` | null | Resize width (null = original) |

## Dependencies
- `opencv-python >= 4.8`
- `numpy`

---

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

---

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

---

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

---

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

---

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

---

# S06: Plane Extraction

Iterative RANSAC plane segmentation with Manhattan World alignment, coplanar merging,
position-based classification, and clean rectangular boundaries.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `surface_points_path` | Path | surface_points.ply from S05 or S03_planargs |
| `metadata_path` | Path | TSDF metadata.json |

Input comes from either:
- **기존 pipeline**: S05 TSDF fusion output (`configs/pipeline.yaml`)
- **PlanarGS pipeline**: S03_planargs mesh vertices (`configs/pipeline_planargs.yaml`)

## Output
| Field | Type | Description |
|-------|------|-------------|
| `planes_file` | Path | planes.json |
| `boundaries_file` | Path | boundaries.json |
| `num_planes` | int | Total planes |
| `num_walls` | int | Wall count |
| `num_floors` | int | Floor count |
| `num_ceilings` | int | Ceiling count |

Also saves `manhattan_alignment.json` (rotation matrix) when `normalize_coords=True`.

## Processing Pipeline

```
Point Cloud
  → Manhattan World Alignment (C)     # Auto-detect axes, rotate to axis-aligned
  → RANSAC Plane Segmentation         # Sequential plane extraction
  → Position-based Classification (D) # floor/ceiling/furniture by height
  → Coplanar Merging (A)              # Union-Find + centroid separation
  → Architectural Filtering (E)       # Small walls/ceilings → other
  → Boundary Extraction (B)           # min_rotated_rect for arch, skip other
  → Inverse Transform                 # Boundaries back to original coords
```

### A. Coplanar Merging
- Union-Find for transitive merging (A+B merge, B+C merge → A+B+C)
- Criteria: same label + normal angle < threshold + centroid-to-plane distance < threshold
- SVD refit after merge for accurate plane equation

### B. Boundary Extraction
- Wall/Floor/Ceiling: Shapely `minimum_rotated_rectangle` → clean 5-vertex rectangles
- Other: boundary skipped (not used in BIM export)

### C. Manhattan World Alignment
- Estimate normals → SVD to find dominant vertical axis
- SVD on horizontal normals → dominant forward axis
- Build orthogonal rotation matrix R → axis-aligned point cloud
- After extraction, inverse-transform boundaries to original coordinates

### D. Position-based Classification
- Horizontal planes classified by centroid height (up-axis position):
  - Bottom 25% → floor
  - Top 25% → ceiling
  - Middle → other (furniture)

### E. Architectural Filtering
- Walls with < `wall_min_ratio` × max wall inliers → reclassified as other
- Same for ceilings

## Config (`configs/steps/s06_plane_extraction.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `max_planes` | 30 | Max planes to extract |
| `distance_threshold` | 0.02 | RANSAC inlier dist |
| `min_inliers` | 500 | Min points per plane |
| `ransac_iterations` | 1000 | RANSAC iterations |
| `angle_threshold` | 15.0 | Classification angle (deg) |
| `simplify_tolerance` | 0.05 | Douglas-Peucker tolerance |
| `up_axis` | `z` | Up axis (auto-detected when normalize_coords=True) |
| `merge_coplanar` | true | Enable coplanar merging |
| `merge_angle_threshold` | 10.0 | Max angle between normals for merge (deg) |
| `merge_distance_threshold` | 1.5 | Max centroid separation for merge |
| `wall_min_ratio` | 0.1 | Min inlier ratio vs largest wall/ceiling |
| `normalize_coords` | true | Manhattan World alignment |

### Recommended config for PlanarGS (Replica room0)
```yaml
distance_threshold: 0.3
min_inliers: 1000
ransac_iterations: 2000
simplify_tolerance: 0.5
merge_coplanar: true
merge_angle_threshold: 10.0
merge_distance_threshold: 1.5
wall_min_ratio: 0.1
normalize_coords: true
```

## E2E Results (Replica room0, 21.4M points)

| Metric | Before | After |
|--------|--------|-------|
| Walls | 8 (duplicates) | 3 (correct) |
| Floors | 7 (fragmented) | 1 |
| Ceilings | 0 | 2 |
| Boundary | irregular polygons | clean rectangles |
| Time | ~672s | ~101s |

## References
- **VERTICAL** (ISPRS 2025): Coplanar face merging with shared edge detection
- **Cloud2BIM** (2503.11498): 2D histogram wall detection, RANSAC→IFC pipeline
- **Furukawa et al.** (ECCV 2016): Normal histogram → Manhattan World axes

## Dependencies
- `open3d`, `shapely`, `numpy`
- `alphashape` (optional, only for "other" planes if boundary needed)

---

# Step 06b: Plane Regularization

Geometric cleanup between RANSAC plane extraction (s06) and IFC export (s07).

## Purpose
- Snap noisy normals to dominant axes (Manhattan ±X/±Z or data-driven cluster mode)
- Unify floor/ceiling heights across coplanar planes
- Detect wall thickness from parallel plane pairs (Manhattan + arbitrary angles)
- Synthesize missing walls from floor boundary outline (AABB or ConvexHull)
- Trim/extend wall center-lines to meet at corners with direction-constrained snapping
- Detect enclosed room spaces for IfcSpace generation
- Classify walls as interior/exterior for building envelope analysis

## Inputs
| Field | Source | Description |
|-------|--------|-------------|
| `planes_file` | s06 | planes.json with detected planes |
| `boundaries_file` | s06 | boundaries.json with boundary polylines |

Also reads `manhattan_alignment.json` from `data/interim/s06_planes/`.

## Outputs
| Field | Description |
|-------|-------------|
| `planes_file` | Regularized planes.json (same schema, backward-compatible) |
| `boundaries_file` | Regularized boundaries.json |
| `walls_file` | walls.json with center-lines, thickness, height |
| `spaces_file` | spaces.json with room polygons (optional) |

## Sub-modules (execution order)
1. **A. Normal Snapping** (`_snap_normals.py`) -- Manhattan mode: ±X/±Z. Cluster mode: discover dominant directions from data via greedy angle clustering.
2. **B. Height Snapping** (`_snap_heights.py`) -- cluster floor/ceiling heights
3. **C. Wall Thickness** (`_wall_thickness.py`) -- parallel pair detection + center-lines. Supports both Manhattan (string axis "x"/"z") and arbitrary-angle walls (vector-based grouping, "oblique:<angle>" axis label).
4. **C2. Wall Closure** (`_wall_closure.py`) -- synthesize missing walls. Manhattan: AABB edges. Cluster: ConvexHull edges for arbitrary building shapes.
5. **D. Intersection Trimming** (`_intersection_trimming.py`) -- endpoint snapping to corners (see below)
6. **H2. Polyline Merging** (`_polyline_walls.py`) -- merge collinear 2-point walls sharing an endpoint into N-point polyline center-lines (disabled by default, `enable_polyline_merging: true`)
7. **I. Column Detection** (`_column_detection.py`) -- reclassify short walls as columns (disabled by default)
8. **E. Space Detection** (`_space_detection.py`) -- Shapely polygonize to room boundaries
9. **G. Exterior Classification** (`_exterior_classification.py`) -- ConvexHull-based interior/exterior wall labeling (disabled by default)
10. **F. Opening Detection** (`_opening_detection.py`) -- Cloud2BIM histogram void detection for doors/windows (disabled by default). Surface points are rotated from COLMAP → Manhattan space via `manhattan_rotation` before inlier extraction. u coordinate measured from p1 in p1→p2 direction (matches s07 opening builder convention).

### D. Intersection Trimming Details

Intersection trimming runs several passes to produce clean wall corners:

1. **Pairwise intersection**: For each pair of non-parallel walls, compute the
   infinite-line intersection in XZ and snap/extend/trim the nearest endpoint.
2. **Direction-constrained snapping**: When snapping an endpoint to an intersection,
   the wall stays on its line. Manhattan walls: X-normal keeps X constant, Z-normal
   keeps Z constant. Oblique walls: project onto the wall's center-line direction.
3. **Wall straightness enforcement**: A post-process ensures both endpoints of a wall
   project to the same position along the wall's normal direction (preventing drift).
4. **Corner reconnection**: For Manhattan walls, endpoints snap to ideal (wall_x, wall_z)
   corners. For general walls, line-line intersections are used.
5. **Junction clustering**: Nearby endpoints (from 3+ walls meeting at one
   point) are clustered to a single centroid.
6. Steps 3-4 run again after clustering to maintain alignment.

### G. Exterior Classification

Uses the convex hull of wall center-line midpoints to classify walls:
- Midpoints on or near the hull boundary → **exterior** (building perimeter)
- Midpoints inside the hull → **interior** (room partitions)
- `is_exterior` field added to each wall in walls.json
- `extract_building_footprint()` orders exterior wall endpoints into a polygon
- Disabled by default (`enable_exterior_classification: false`)

## Normal Modes

| Mode | Config | Description |
|------|--------|-------------|
| Manhattan | `normal_mode: "manhattan"` (default) | Snaps to ±X/±Z. Best for interior scans of axis-aligned rooms. |
| Cluster | `normal_mode: "cluster"` | Discovers dominant wall directions from data. Required for non-orthogonal buildings, exterior scans, or 45°/60° walls. |

Cluster mode uses greedy angle clustering with `cluster_angle_tolerance` (default 15°).
Both modes are backward compatible — Manhattan data produces identical results in either mode.

## N-Point Polyline Walls

When `enable_polyline_merging: true`, collinear walls sharing an endpoint are merged
into N-point polyline center-lines (e.g., L-shaped wall = 3 points). All downstream
code supports N-point walls:

- `_transform_walls_from_manhattan()`: converts each point to 3D + inverse rotation
- `_rebuild_wall_boundary()`: strip boundary (N bottom + N top reversed + close)
- `visualize_planes_3d.py`: renders one quad per segment
- s07 `_wall_builder.py`: already supports polyline profiles via `IfcArbitraryClosedProfileDef`

## Coordinate System

All processing runs in Manhattan-aligned Y-up space. The Manhattan rotation
preserves the Y axis, so `floor_h` / `ceil_h` values work identically in both
Manhattan and original (COLMAP) coordinate systems. Results are transformed back
to original coordinates for backward-compatible output.

## Visualization

`scripts/visualize_planes_3d.py` provides 3D BIM visualization of s06b output:

- **Default**: Manhattan-aligned coordinates (clean axis-aligned walls).
- `--original`: Use COLMAP original coordinates instead.
- `--save`: Render to PNG files (top-down, isometric, front elevation) via matplotlib.
- Without `--save`: Interactive Open3D window (rotate/zoom/pan).
- Auto-displays stats: wall count (detected vs. synthetic), floor/ceiling heights, room dimensions.
- Deduplicates synthetic walls that overlap with detected walls before rendering.

## Tools Used
- NumPy (linear algebra, coordinate transforms)
- Shapely (polygonize, unary_union for space detection)
- SciPy (ConvexHull for wall closure outline estimation, exterior classification)

## References
- Cloud2BIM (AiC 2025): wall thickness, intersection snapping, space generation
- Yu et al. (Visual Computer 2024): Manhattan normal regularization

---

# Step 06c: Building Extraction

Exterior building reconstruction from RANSAC planes and point clouds.

## Purpose

Processes exterior-scanned buildings to extract:
- Ground plane separation
- Facade (exterior wall face) grouping
- 2D building footprint (concave hull for non-convex shapes)
- Roof structure (ridges, eaves, valleys)
- Storey count from facade patterns

## Inputs

| Input | Source | Description |
|-------|--------|-------------|
| `planes.json` | s06 | RANSAC-detected planes |
| `boundaries.json` | s06 | Plane boundary polygons |
| `surface_points.ply` | s05/s06 | Point cloud (optional) |

## Outputs

| Output | Description |
|--------|-------------|
| `planes.json` | Updated with ground/roof labels |
| `boundaries.json` | Updated boundaries |
| `building_context.json` | Footprint, facades, roof structure, storeys |
| `building_points.ply` | Segmented building points (optional) |

## Sub-modules

| Phase | Module | Description |
|-------|--------|-------------|
| A | `_ground_separation` | Lowest wide horizontal plane → ground |
| B | `_building_segmentation` | DBSCAN density clustering (optional) |
| C | `_facade_detection` | Normal-grouped vertical plane clusters |
| D | `_footprint_extraction` | Alpha shape → concave 2D outline |
| E | `_roof_structuring` | Plane intersection → ridge/eave/valley |
| F | `_storey_detection` | Height histogram → floor levels (optional) |

## Tools Used

- numpy (all modules)
- scipy (DBSCAN fallback, ConvexHull)
- shapely (concave_hull, polygon operations)
- alphashape (optional, for alpha shape computation)
- open3d (optional, point cloud I/O)

---

# S07: IFC/BIM Export

Generate IFC4 BIM file from center-line wall data (Cloud2BIM pattern).

## Input
| Field | Type | Description |
|-------|------|-------------|
| `walls_file` | Path | walls.json from S06b (center-lines + thickness) |
| `spaces_file` | Path | spaces.json from S06b (room polygons, optional) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `ifc_path` | Path | Generated .ifc file |
| `num_walls` | int | IfcWall objects |
| `num_slabs` | int | IfcSlab objects |
| `num_spaces` | int | IfcSpace objects |
| `num_openings` | int | IfcOpeningElement objects (doors + windows) |
| `ifc_version` | str | IFC schema version |

## IFC Hierarchy
```
IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey
  ├─ IfcWall (center_line_2d → extruded rectangle)
  │   └─ IfcRelVoidsElement → IfcOpeningElement
  │        └─ IfcRelFillsElement → IfcDoor / IfcWindow
  ├─ IfcSlab (floor + ceiling from boundary_2d)
  └─ IfcSpace (room polygons, aggregated via IfcRelAggregates)
```

## Sub-modules
| Module | Description |
|--------|-------------|
| `_ifc_builder.py` | Project/Site/Building/Storey hierarchy + contexts |
| `_wall_builder.py` | Center-line → IfcWall (extruded rectangle profile) |
| `_slab_builder.py` | Floor/ceiling IfcSlab from boundary_2d |
| `_space_builder.py` | IfcSpace from room polygons |
| `_opening_builder.py` | IfcOpeningElement + IfcDoor/IfcWindow from openings data. Opening z is relative to wall placement (not absolute). Fill elements get distinct geometry. |

## Config (`configs/steps/s07_ifc_export.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `ifc_version` | IFC4 | Schema version |
| `project_name` | GSS_BIM | IFC project name |
| `building_name` | Building | Building name |
| `default_wall_thickness` | 0.2 | Wall thickness (m) |
| `default_slab_thickness` | 0.3 | Slab thickness (m) |
| `include_synthetic_walls` | true | Include synthetic walls in export |
| `create_spaces` | true | Create IfcSpace objects |

## Dependencies
- `ifcopenshell >= 0.7`

---

# S08: Mesh Export (IFC → GLB / USD)

Export IFC geometry to universal 3D formats for digital twin platforms.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `ifc_path` | Path | .ifc file from S07 |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `glb_path` | Path | Exported .glb file |
| `usd_path` | Path | Exported .usdc file |
| `usdz_path` | Path | Exported .usdz file (optional) |
| `num_meshes` | int | Total mesh count |
| `num_vertices` | int | Total vertex count |
| `num_faces` | int | Total face count |

## Formats
- **GLB** — Universal glTF Binary. Supported by all major 3D viewers, UE5, Unity, Three.js, Blender.
- **USDC** — Binary USD. Required for NVIDIA Omniverse and Isaac Sim.
- **USDZ** — Packaged USD. Required for Apple Vision Pro / ARKit.

## Config (`configs/steps/s08_mesh_export.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `export_glb` | true | Export GLB file |
| `export_usd` | true | Export USDC file |
| `export_usdz` | false | Export USDZ package |
| `color_scheme` | by_class | Color by IFC class |
| `include_spaces` | false | Include IfcSpace geometry |
| `usd_up_axis` | Y | USD up axis ("Y" or "Z"). Y-up is Isaac Sim / Omniverse standard |
| `usd_meters_per_unit` | 1.0 | USD scale factor |
| `usd_double_sided` | true | Double-sided meshes for thin BIM walls |

## Coordinate Transform (Z-up → Y-up)

IFC uses **Z-up** convention. GLB (glTF) and USD with `up_axis=Y` use **Y-up**.

The transform applied: `(x, y, z) → (x, z, -y)`

| Source | Target | Transform |
|--------|--------|-----------|
| IFC Z-up | GLB Y-up | Always applied |
| IFC Z-up | USD Y-up | Applied (default) |
| IFC Z-up | USD Z-up | No transform (`usd_up_axis: "Z"`) |

## Graceful Degradation

Both `trimesh` and `usd-core` are checked at runtime:
- **trimesh** missing → GLB export skipped with warning
- **usd-core** missing → USD/USDZ export skipped with warning
- **usd-core OSError** (e.g., DLL init failure on Windows) → warning with version suggestion

## USD Details
- Vertex normals: area-weighted per-vertex normals via `_compute_normals()`
- `doubleSided=True`: thin BIM walls visible from both sides (configurable)
- `displayName` metadata on mesh prims for Omniverse UI
- Materials: `UsdPreviewSurface` PBR (roughness=0.7, metallic=0.0, per-class diffuseColor)
- `MaterialBindingAPI.Apply()` called before `Bind()` (Omniverse spec compliance)
- Material deduplication by color key (RGBA)
- Face vertex counts: all triangles (`[3] * n_faces`)

## Cross-Platform (Windows → Linux)
- USDZ packaging uses POSIX paths (`Path.as_posix()`) to avoid backslash issues
- GLB node names sanitized: all non-alphanumeric chars replaced with underscore
- USDC binary format is OS-independent (portable between Windows/Linux/macOS)
- Default Y-up matches Isaac Sim / Omniverse convention (no axis fix needed on import)

## Visualization
```bash
# Single file
python scripts/visualize_mesh.py data/processed/GSS_BIM.glb
python scripts/visualize_mesh.py data/processed/GSS_BIM.usdc

# Compare GLB vs USD side by side
python scripts/visualize_mesh.py --compare --save

# Save PNG renders
python scripts/visualize_mesh.py data/processed/GSS_BIM.glb --save
```

Output: `docs/images/mesh_*_isometric.png`, `mesh_*_topdown.png`

## Dependencies
- `ifcopenshell >= 0.7` — IFC geometry extraction
- `trimesh >= 4.0` — GLB export (graceful skip if missing)
- `usd-core >= 24.0` (optional) — USD/USDZ export (graceful skip if missing)
- Note: `usd-core 24.11` recommended on Windows (26.x has DLL init issues)
