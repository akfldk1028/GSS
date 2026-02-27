# GSS - 3D Gaussian Splatting to BIM Pipeline

## Architecture
세 가지 파이프라인 지원:
- **기존**: Video → Frames → COLMAP → gsplat(2DGS) → Depth → TSDF → Planes → PlaneRegularization → IFC
- **PlanarGS**: Video → Frames → COLMAP → PlanarGS(subprocess) → Planes → PlaneRegularization → IFC
- **Import**: PLY import → Planes → PlaneRegularization → IFC (기존 3DGS/PlanarGS 결과물 직접 투입)

## Project Structure
```
src/gss/core/          - Pipeline runner, BaseStep ABC, shared contracts
src/gss/steps/s00~s07/ - Each step: __init__.py + step.py + config.py + contracts.py + README.md
src/gss/steps/s03_planargs/ - PlanarGS wrapper (replaces s03+s04+s05)
src/gss/steps/s06b_plane_regularization/ - Geometric cleanup (6 sub-modules)
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
- (s06b has additional `_*.py` sub-modules for each regularization phase)

## Tech Stack
| Phase | Tool | Package | Pipeline |
|-------|------|---------|----------|
| SfM | hloc + COLMAP | pycolmap | both |
| 3DGS | gsplat (2DGS mode) | gsplat | 기존 |
| 3DGS+Depth+TSDF | PlanarGS (subprocess) | diff-plane-rasterization | PlanarGS |
| TSDF | Open3D ScalableTSDFVolume | open3d | 기존 |
| Planes | Open3D RANSAC + alphashape | open3d, alphashape, shapely | both |
| Regularization | Normal/height snap, wall thickness, space detection | numpy, shapely | both |
| BIM | IfcOpenShell | ifcopenshell | both |

## Three Pipeline Configs
- `configs/pipeline.yaml` — 기존 8-step (s01→s02→s03→s04→s05→s06→s06b→s07)
- `configs/pipeline_planargs.yaml` — PlanarGS 6-step (s01→s02→s03_planargs→s06→s06b→s07)
- `configs/pipeline_import.yaml` — Import 4-step (s00→s06→s06b→s07)

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
- **LOD**: Level of Development (200=basic geometry, 300=openings, 400=detailed)
- **PlanarGS**: Co-planarity-aware Gaussian Splatting (NeurIPS 2025)
