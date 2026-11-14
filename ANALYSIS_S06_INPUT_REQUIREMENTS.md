# GSS Plane Extraction (s06) Input Requirements Analysis

## Summary
Comprehensive analysis of what data formats s06 (plane extraction) needs and what a GS-SLAM replacement would need to output for full compatibility with the GSS pipeline.

## Key Findings

### 1. s06 Input Contract (Minimal Requirements)
**Source**: `/c/DK/GSS/src/gss/steps/s06_plane_extraction/contracts.py`

s06 requires exactly two files:
- `surface_points_path`: Path to `surface_points.ply` (XYZ point cloud only)
- `metadata_path`: Path to `metadata.json` (processing metadata)

**Critical Discovery**: s06 does NOT need Gaussian properties (f_dc_*, scale_*, rot_*, opacity). It only extracts XYZ coordinates.

### 2. s00_import_ply Processing Pipeline
**Source**: `/c/DK/GSS/src/gss/steps/s00_import_ply/step.py`

The import step performs the following sequence:

**Step 2.1: Format Detection** (`_detect_ply_format`):
- Checks for Gaussian markers: `f_dc_0`, `scale_0`, `rot_0`
- Returns either `"gaussian_splat"` or `"pointcloud"`
- No error if markers missing; graceful fallback to plain point cloud

**Step 2.2: Point Loading**:
- **For Gaussian PLY**: Extracts XYZ centers, applies sigmoid opacity filtering (threshold >= min_opacity, default 0.5)
  - Formula: `opacity = 1.0 / (1.0 + exp(-raw_opacity))`
  - Filters before further processing
- **For Plain PLY**: Loads via Open3D (no filtering)

**Step 2.3: Processing Pipeline** (all optional, defaults shown):
1. Voxel downsampling (default: 0.0 = disabled)
   - Config: `voxel_downsample: 0.0`
2. Statistical outlier removal (default: enabled)
   - Config: `remove_outliers: true, outlier_nb_neighbors: 20`
3. Normal estimation via KDTree (default: enabled)
   - Config: `estimate_normals: true`
   - Search parameters: radius=0.1, max_nn=30

**Step 2.4: Output Generation**:
- Saves processed point cloud as `surface_points.ply` (Open3D format, XYZ + optional normals)
- Saves metadata.json with source tracking:
  ```json
  {
    "source": "<original_ply_path>",
    "source_format": "gaussian_splat|pointcloud",
    "num_surface_points": <count>,
    "min_opacity": 0.5 or null,
    "voxel_downsample": 0.0,
    "outlier_removal": true
  }
  ```

### 3. s06 Processing Pipeline
**Source**: `/c/DK/GSS/src/gss/steps/s06_plane_extraction/step.py`

Given the point cloud from s00, s06 performs:

**Step 3.1: Optional Manhattan World Alignment**:
- SVD analysis of plane normals to detect axes
- Finds vertical (up) and horizontal axes
- Aligns coordinate system to orthogonal axes
- Config: `normalize_coords: true` (default)

**Step 3.2: RANSAC Plane Segmentation**:
- Iteratively fits planes to point cloud
- Key parameters:
  - `max_planes`: 30
  - `distance_threshold`: 0.02 (inlier distance in meters)
  - `min_inliers`: 500 (minimum points per plane)
  - `ransac_iterations`: 1000

**Step 3.3: Plane Classification**:
- Classifies by normal vector angle and position:
  - **Wall**: 80-100° from up-axis OR 20-100° from up-axis in middle height band
  - **Floor**: 0-10° from up-axis AND in bottom 25% of point cloud
  - **Ceiling**: 170-180° from up-axis AND in top 25% of point cloud
  - **Other**: everything else (furniture, etc.)
- Angle threshold: 15 degrees
- Config: `angle_threshold: 15.0`

**Step 3.4: Coplanar Merging** (Union-Find algorithm):
- Merges geometrically coplanar planes with same label
- Criteria:
  - Normal angle difference < 10 degrees
  - Centroid separation < 1.5 meters
- Transitive merging (A≈B and B≈C → A,B,C merged)
- Config: `merge_coplanar: true, merge_angle_threshold: 10.0, merge_distance_threshold: 1.5`

**Step 3.5: Boundary Extraction**:
- Different strategies by plane type:
  - **Walls/Floors/Ceilings**: `minimum_rotated_rectangle` (tight axis-aligned bounding polygon)
  - **Furniture/Other**: `alphashape` (concave hull for irregular objects)
- Returns 3D polylines with simplification
- Config: `simplify_tolerance: 0.05`

### 4. s06 Output Format
**Source**: `/c/DK/GSS/src/gss/steps/s06_plane_extraction/contracts.py`

Two JSON files are produced:

**planes.json** - Array of DetectedPlane objects:
```json
[
  {
    "id": 0,
    "normal": [0.0, 0.0, 1.0],
    "d": -1.5,
    "label": "floor",
    "num_inliers": 15000,
    "boundary_3d": [[0, 0, -1.5], [10, 0, -1.5], [10, 5, -1.5], [0, 5, -1.5]]
  },
  ...
]
```

**boundaries.json** - Collection of boundary polylines (same structure, typically)

**Output Metrics**:
- `num_planes`: Total planes
- `num_walls`: Wall count
- `num_floors`: Floor count
- `num_ceilings`: Ceiling count

### 5. Downstream Requirements (s06b & s07)

**s06b (Plane Regularization)** - `/c/DK/GSS/src/gss/steps/s06b_plane_regularization/contracts.py`

Requires:
- `planes_file`: planes.json from s06
- `boundaries_file`: boundaries.json from s06

Produces:
- Regularized planes.json, boundaries.json
- `walls.json`: Wall segments with center-lines and thickness
- `spaces.json`: Room polygons with coordinate scale
- `columns.json`: Detected architectural columns

**s07 (IFC Export)** - `/c/DK/GSS/src/gss/steps/s07_ifc_export/contracts.py`

Requires (mandatory and optional):
- `walls_file`: walls.json from s06b (MANDATORY)
- `spaces_file`: spaces.json from s06b (optional)
- `planes_file`: planes.json for roof detection (optional)
- `building_context_file`: building_context.json from s06c (optional, exterior only)
- `columns_file`: columns.json from s06b (optional)

### 6. Data Format Compatibility
All pipelines (regular TSDF, PlanarGS, 3DGRUT, import) converge at s06 by producing identical outputs:
- Same `surface_points.ply` structure (XYZ point cloud via Open3D)
- Same `metadata.json` structure
- No format divergence throughout pipeline

### 7. GS-SLAM Replacement Requirements

For a GS-SLAM replacement to work with GSS:

**Required Output**:
1. PLY file with XYZ coordinates (XYZ + optional colors/normals)
2. Point density: 50k-500k points typical for indoor scenes
3. Scale: Real-world meters (RANSAC distance_threshold = 0.02m = 2cm)
4. Normal vectors: Optional (s00 estimates if missing via KDTree)

**Recommended Characteristics**:
- Minimize outliers (s00 removes statistical outliers, but garbage in = garbage out)
- Axis-aligned or near-axis-aligned (Manhattan alignment helps but not required)
- Reasonably uniform point distribution (extreme density spikes cause RANSAC issues)

**Configuration Tuning for Different Scenarios**:

```yaml
# Fast processing (fewer/larger planes)
s06_plane_extraction:
  max_planes: 15
  distance_threshold: 0.03
  min_inliers: 1000
  ransac_iterations: 500

# High detail (more/smaller planes, slower)
s06_plane_extraction:
  max_planes: 50
  distance_threshold: 0.01
  min_inliers: 200
  ransac_iterations: 2000

# Furniture-heavy scenes (preserve clutter)
s06_plane_extraction:
  max_planes: 50
  merge_coplanar: false  # keep planes separate

# Clean architectural (aggressive merging)
s06_plane_extraction:
  max_planes: 30
  merge_angle_threshold: 15.0  # more tolerance
  merge_distance_threshold: 2.0  # larger distance
```

## Data Flow Diagram

```
GS-SLAM Replacement
  ↓ (output PLY with XYZ)
  │
s00_import_ply
  │ (format detection + preprocessing)
  │ - Auto-detects Gaussian vs plain PLY
  │ - Filters opacity (if Gaussian)
  │ - Removes outliers (if enabled)
  │ - Estimates normals (if enabled)
  ↓
surface_points.ply + metadata.json
  ↓
s06_plane_extraction
  │ (RANSAC + classification)
  │ - Manhattan alignment (if enabled)
  │ - Iterative plane fitting
  │ - Wall/floor/ceiling classification
  │ - Coplanar merging (Union-Find)
  │ - Boundary extraction
  ↓
planes.json + boundaries.json
  ↓
s06b_plane_regularization
  │ (geometry cleanup)
  ↓
walls.json + spaces.json + columns.json
  ↓
s07_ifc_export
  │ (BIM generation)
  ↓
model.ifc
  ↓
s08_mesh_export
  │ (GLB + USD/USDZ conversion)
  ↓
Digital Twin Assets (GLB, USDC, USDZ)
```

## Critical Compatibility Points

| Aspect | Requirement | Note |
|--------|-------------|------|
| **Point Format** | XYZ coordinates only | Gaussian properties auto-extracted, others ignored |
| **Point Density** | 50k-500k points | Fewer → fewer planes; More → slower processing |
| **Point Scale** | Real-world meters | RANSAC threshold = 0.02m internally |
| **Normals** | Optional | s00 estimates via KDTree if missing |
| **Outliers** | Minimized | s00 filters but GS-SLAM should avoid extreme outliers |
| **Coordinate Axes** | +Z up preferred | Manhattan alignment helps but not required |
| **Colors** | Optional | Not used by s06 onward |
| **PLY Format** | Gaussian or Plain | Auto-detected, either works |

## Files Referenced

Core contracts and implementations:
- `/c/DK/GSS/src/gss/steps/s06_plane_extraction/contracts.py` - s06 I/O contract
- `/c/DK/GSS/src/gss/steps/s06_plane_extraction/step.py` - s06 implementation (RANSAC, classification, merging)
- `/c/DK/GSS/src/gss/steps/s06_plane_extraction/config.py` - s06 tunable parameters
- `/c/DK/GSS/src/gss/steps/s00_import_ply/step.py` - s00 implementation (format detection, preprocessing)
- `/c/DK/GSS/src/gss/steps/s00_import_ply/contracts.py` - s00 I/O contract
- `/c/DK/GSS/src/gss/steps/s00_import_ply/config.py` - s00 tunable parameters
- `/c/DK/GSS/src/gss/steps/s06b_plane_regularization/contracts.py` - s06b I/O contract
- `/c/DK/GSS/src/gss/steps/s07_ifc_export/contracts.py` - s07 I/O contract

## Exploration Status
✅ Complete - All key findings documented, no further exploration needed
