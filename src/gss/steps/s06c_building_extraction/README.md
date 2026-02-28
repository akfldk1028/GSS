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
