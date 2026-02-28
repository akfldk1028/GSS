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
6. **E. Space Detection** (`_space_detection.py`) -- Shapely polygonize to room boundaries
7. **G. Exterior Classification** (`_exterior_classification.py`) -- ConvexHull-based interior/exterior wall labeling (disabled by default)
8. **F. Opening Detection** (`_opening_detection.py`) -- Cloud2BIM histogram void detection for doors/windows (disabled by default). Surface points are rotated from COLMAP → Manhattan space via `manhattan_rotation` before inlier extraction. u coordinate measured from p1 in p1→p2 direction (matches s07 opening builder convention).

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
