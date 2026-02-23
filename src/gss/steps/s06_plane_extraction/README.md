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
