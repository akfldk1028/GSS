# GSS - 3D Gaussian Splatting to BIM Pipeline

## Architecture
모듈화된 파이프라인: Video → Frames → COLMAP → 3DGS(2DGS) → Depth → TSDF → Planes → IFC

## Project Structure
```
src/gss/core/       - Pipeline runner, BaseStep ABC, shared contracts
src/gss/steps/s01~s07/ - Each step: step.py + config.py + contracts.py
src/gss/utils/       - Shared utilities (I/O, geometry, subprocess)
configs/             - YAML configs (pipeline.yaml + per-step)
data/                - raw/ → interim/s01~s06/ → processed/
```

## Step Pattern (MUST follow)
Every step inherits `BaseStep[InputT, OutputT, ConfigT]` from `src/gss/core/step_base.py`.
Each step module has exactly 4 files:
- `contracts.py` - Pydantic Input/Output models (I/O contract)
- `config.py` - Pydantic Config model (all tunable params)
- `step.py` - Step class with `run()` and `validate_inputs()`
- `README.md` - Purpose, inputs, outputs, tools used

## Tech Stack
| Phase | Tool | Package |
|-------|------|---------|
| SfM | hloc + COLMAP | pycolmap |
| 3DGS | gsplat (2DGS mode) | gsplat |
| TSDF | Open3D ScalableTSDFVolume | open3d |
| Planes | Open3D RANSAC + alphashape | open3d, alphashape, shapely |
| BIM | IfcOpenShell | ifcopenshell |

## Conventions
- Python 3.10+, Pydantic v2, type hints everywhere
- Config in YAML, never hardcode parameters in step code
- Data artifacts flow through `data/interim/s{NN}_{name}/`
- Run: `gss run` (full pipeline) or `gss run-step <name>` (single step)
- Test: `pytest tests/`
- Lint: `ruff check src/`

## Domain Terms
- **3DGS**: 3D Gaussian Splatting - scene representation using Gaussian primitives
- **2DGS**: 2D variant using flat disks for better surface geometry
- **TSDF**: Truncated Signed Distance Field - volumetric surface representation
- **SfM**: Structure from Motion - camera pose estimation from images
- **IFC**: Industry Foundation Classes - BIM data exchange format
- **LOD**: Level of Development (200=basic geometry, 300=openings, 400=detailed)
