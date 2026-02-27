# Step 06b: Plane Regularization

Geometric cleanup between RANSAC plane extraction (s06) and IFC export (s07).

## Purpose
- Snap noisy normals to exact Manhattan axes
- Unify floor/ceiling heights across coplanar planes
- Detect wall thickness from parallel plane pairs
- Synthesize missing walls from floor boundary outline
- Trim/extend wall center-lines to meet at corners with axis-constrained snapping
- Detect enclosed room spaces for IfcSpace generation

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
1. **A. Normal Snapping** (`_snap_normals.py`) -- wall normals to +/-X/+/-Z, floor/ceiling to +/-Y
2. **B. Height Snapping** (`_snap_heights.py`) -- cluster floor/ceiling heights
3. **C. Wall Thickness** (`_wall_thickness.py`) -- parallel pair detection + center-lines
4. **C2. Wall Closure** (`_wall_closure.py`) -- synthesize missing walls from floor boundary or wall AABB
5. **D. Intersection Trimming** (`_intersection_trimming.py`) -- endpoint snapping to corners (see below)
6. **E. Space Detection** (`_space_detection.py`) -- Shapely polygonize to room boundaries
7. **F. Opening Detection** (`_opening_detection.py`) -- Phase 2 stub, disabled by default

### D. Intersection Trimming Details

Intersection trimming runs several passes to produce clean Manhattan-aligned corners:

1. **Pairwise intersection**: For each pair of perpendicular walls, compute the
   infinite-line intersection in XZ and snap/extend/trim the nearest endpoint.
2. **Axis-constrained snapping**: When snapping an endpoint to an intersection,
   the wall's constant coordinate is preserved. X-normal walls keep X constant
   (only Z changes); Z-normal walls keep Z constant (only X changes). This
   prevents walls from leaning at corners.
3. **Axis alignment enforcement**: A post-process averages each wall's normal
   coordinate across both endpoints, ensuring the wall stays perfectly straight
   (e.g., both endpoints of an X-normal wall share the same X value).
4. **Corner reconnection**: After axis alignment, perpendicular wall endpoints
   are reconnected to the ideal corner point (wall_x, wall_z) where an
   X-normal wall's X position meets a Z-normal wall's Z position.
5. **Junction clustering**: Nearby endpoints (from 3+ walls meeting at one
   point) are clustered to a single centroid.
6. Steps 3-4 run again after clustering to maintain alignment.

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
- SciPy (ConvexHull for wall closure outline estimation)

## References
- Cloud2BIM (AiC 2025): wall thickness, intersection snapping, space generation
- Yu et al. (Visual Computer 2024): Manhattan normal regularization
