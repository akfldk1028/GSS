# GSS - 3D Gaussian Splatting to BIM Pipeline

## Architecture
네 가지 파이프라인 지원:
- **기존**: Video → Frames → COLMAP → gsplat(2DGS) → Depth → TSDF → Planes → PlaneRegularization → IFC → MeshExport
- **PlanarGS**: Video → Frames → COLMAP → PlanarGS(subprocess) → Planes → PlaneRegularization → IFC → MeshExport
- **Exterior**: Video → ... → Planes → **BuildingExtraction** → PlaneRegularization → IFC → MeshExport (외부 건물)
- **Import**: PLY import → Planes → PlaneRegularization → IFC → MeshExport (기존 3DGS/PlanarGS 결과물 직접 투입)

## Project Structure
```
src/gss/core/          - Pipeline runner, BaseStep ABC, shared contracts
src/gss/steps/s00~s08/ - Each step: __init__.py + step.py + config.py + contracts.py + README.md
src/gss/steps/s03_planargs/ - PlanarGS wrapper (replaces s03+s04+s05)
src/gss/steps/s06b_plane_regularization/ - Geometric cleanup (7 sub-modules: A~F)
src/gss/steps/s06c_building_extraction/ - Exterior building reconstruction (6 sub-modules: A~F)
src/gss/steps/s08_mesh_export/  - IFC → GLB/USD export for digital twin platforms
src/gss/utils/         - Shared utilities (I/O, geometry, subprocess)
configs/               - YAML configs (pipeline.yaml, pipeline_planargs.yaml, pipeline_import.yaml, per-step)
data/                  - raw/ → interim/s01~s06/ → processed/
clone/PlanarGS/        - PlanarGS repo (separate conda env)
```

## Step Pattern (MUST follow)
Every step inherits `BaseStep[InputT, OutputT, ConfigT]` from `src/gss/core/step_base.py`.
Each step module has exactly 5 files:
- `__init__.py` - Re-exports Step, Input, Output, Config classes
- `contracts.py` - Pydantic Input/Output models (I/O contract)
- `config.py` - Pydantic Config model (all tunable params)
- `step.py` - Step class with `run()` and `validate_inputs()`
- `README.md` - Purpose, inputs, outputs, tools used
- (s06b has additional `_*.py` sub-modules for each regularization phase A~F)
- (s06c has additional `_*.py` sub-modules for building extraction phases A~F)
- (s08 has additional `_*.py` sub-modules: `_ifc_to_mesh`, `_glb_writer`, `_usd_writer`)

## Tech Stack
| Phase | Tool | Package | Pipeline |
|-------|------|---------|----------|
| SfM | hloc + COLMAP | pycolmap | both |
| 3DGS | gsplat (2DGS mode) | gsplat | 기존 |
| 3DGS+Depth+TSDF | PlanarGS (subprocess) | diff-plane-rasterization | PlanarGS |
| TSDF | Open3D ScalableTSDFVolume | open3d | 기존 |
| Planes | Open3D RANSAC + alphashape | open3d, alphashape, shapely | both |
| Regularization | Normal/height snap, wall thickness, space detection, opening detection | numpy, shapely | both |
| Building Extraction | Ground separation, facade detection, footprint, roof structure | numpy, shapely, scipy | exterior |
| BIM | IfcOpenShell | ifcopenshell | both |
| Mesh Export | GLB (trimesh), USD/USDZ (usd-core) | trimesh, usd-core | both |

## Four Pipeline Configs
- `configs/pipeline.yaml` — 기존 9-step (s01→s02→s03→s04→s05→s06→s06b→s07→s08)
- `configs/pipeline_planargs.yaml` — PlanarGS 7-step (s01→s02→s03_planargs→s06→s06b→s07→s08)
- `configs/pipeline_exterior.yaml` — Exterior 10-step (s01→...→s06→**s06c**→s06b→s07→s08)
- `configs/pipeline_import.yaml` — Import 5-step (s00→s06→s06b→s07→s08)

## Two Conda Environments
- **기존 (gss)**: torch 2.8.0+cu128, gsplat 1.5.3
- **PlanarGS (planargs)**: torch 2.8.0+cu128, diff-plane-rasterization, simple-knn, pytorch3d, transformers<5

## Conventions
- Python 3.10+, Pydantic v2, type hints everywhere
- Config in YAML, never hardcode parameters in step code
- Data artifacts flow through `data/interim/s{NN}_{name}/` (s06b → `s06b_plane_regularization/`)
- Run: `gss run` (full pipeline) or `gss run-step <name>` (single step)
- Run PlanarGS: `gss run --config configs/pipeline_planargs.yaml`
- Run Import: `gss run --config configs/pipeline_import.yaml`
- Test: `pytest tests/`
- Lint: `ruff check src/`

## Domain Terms
- **3DGS**: 3D Gaussian Splatting - scene representation using Gaussian primitives
- **2DGS**: 2D variant using flat disks for better surface geometry
- **TSDF**: Truncated Signed Distance Field - volumetric surface representation
- **SfM**: Structure from Motion - camera pose estimation from images
- **IFC**: Industry Foundation Classes - BIM data exchange format
- **GLB**: glTF Binary - universal 3D format (viewers, game engines)
- **USD/USDC**: Universal Scene Description - Omniverse/Isaac Sim format
- **USDZ**: Packaged USD - Apple Vision Pro / ARKit format
- **LOD**: Level of Development (200=basic geometry, 300=openings, 400=detailed)
- **PlanarGS**: Co-planarity-aware Gaussian Splatting (NeurIPS 2025)

## s06b Regularization Sub-modules
| Phase | Module | Description |
|-------|--------|-------------|
| A | `_snap_normals` | Wall normals → Manhattan axes |
| B | `_snap_heights` | Floor/ceiling height clustering |
| C | `_wall_thickness` | Parallel pair detection + center-lines |
| C2 | `_wall_closure` | Synthesize missing walls from floor boundary |
| D | `_intersection_trimming` | Snap wall endpoints to corners |
| E | `_space_detection` | Polygonize center-lines → room boundaries |
| F | `_opening_detection` | Door/window detection via Cloud2BIM histogram (disabled by default) |

## s06c Building Extraction Sub-modules
| Phase | Module | Description |
|-------|--------|-------------|
| A | `_ground_separation` | Lowest wide horizontal plane → ground label |
| B | `_building_segmentation` | DBSCAN density clustering (optional, disabled by default) |
| C | `_facade_detection` | Normal-grouped vertical plane clusters → facades |
| D | `_footprint_extraction` | Alpha shape / concave hull → 2D building outline |
| E | `_roof_structuring` | Plane intersection → ridge/eave/valley lines |
| F | `_storey_detection` | Height histogram → floor levels (optional, disabled by default) |

## s07 IFC Hierarchy
```
IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey
  ├─ IfcWall (center_line_2d → extruded rectangle)
  │   └─ IfcRelVoidsElement → IfcOpeningElement
  │        └─ IfcRelFillsElement → IfcDoor / IfcWindow
  ├─ IfcSlab (floor + ceiling from boundary_2d)
  └─ IfcSpace (room polygons from boundary_2d)
```

## s08 Mesh Export
- IFC → GLB (trimesh) + USDC (usd-core, optional) + USDZ (optional)
- `_ifc_to_mesh.py`: ifcopenshell.geom → MeshData(name, verts, faces, color, ifc_class)
- `_glb_writer.py`: trimesh.Scene → GLB (Z-up → Y-up: `(x,y,z)→(x,z,-y)`)
- `_usd_writer.py`: pxr UsdGeom.Mesh + UsdPreviewSurface materials → USDC (Y-up 시 동일 변환)
- Both trimesh and usd-core: graceful degradation (`_has_trimesh()`, `_has_pxr()`)
