# Step 06b: Plane Regularization

Geometric cleanup between RANSAC plane extraction (s06) and IFC export (s07).

## Purpose
- Snap noisy normals to exact Manhattan axes
- Unify floor/ceiling heights across coplanar planes
- Detect wall thickness from parallel plane pairs
- Trim/extend wall center-lines to meet at corners
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
1. **A. Normal Snapping** (`_snap_normals.py`) — wall normals → ±X/±Z, floor/ceiling → ±Y
2. **B. Height Snapping** (`_snap_heights.py`) — cluster floor/ceiling heights
3. **C. Wall Thickness** (`_wall_thickness.py`) — parallel pair detection + center-lines
4. **D. Intersection Trimming** (`_intersection_trimming.py`) — endpoint snapping to corners
5. **E. Space Detection** (`_space_detection.py`) — Shapely polygonize → room boundaries
6. **F. Opening Detection** (`_opening_detection.py`) — Phase 2 stub, disabled by default

## Tools Used
- NumPy (linear algebra, coordinate transforms)
- Shapely (polygonize, unary_union for space detection)

## References
- Cloud2BIM (AiC 2025): wall thickness, intersection snapping, space generation
- Yu et al. (Visual Computer 2024): Manhattan normal regularization
