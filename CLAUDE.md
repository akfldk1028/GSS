# GSS - 3D Gaussian Splatting to BIM Pipeline

## Architecture
여섯 가지 파이프라인 지원:
- **기존**: Video → Frames → COLMAP → gsplat(2DGS) → Depth → TSDF → Planes → PlaneRegularization → IFC → MeshExport
- **PlanarGS**: Video → Frames → COLMAP → PlanarGS(subprocess) → Planes → PlaneRegularization → IFC → MeshExport
- **3DGRUT**: Video → Frames → COLMAP → **3DGRUT**(subprocess) → Planes → PlaneRegularization → IFC → MeshExport (NVIDIA ray tracing)
- **SLAM**: Video → Frames → **GS-SLAM**(subprocess) → Planes → PlaneRegularization → IFC → MeshExport (COLMAP+3DGS 대체, 실시간)
- **Exterior**: Video → ... → Planes → **BuildingExtraction** → PlaneRegularization → IFC → MeshExport (외부 건물)
- **Import**: PLY import → Planes → PlaneRegularization → IFC → MeshExport (기존 3DGS/PlanarGS 결과물 직접 투입)

## Project Structure
```
src/gss/core/          - Pipeline runner, BaseStep ABC, shared contracts
src/gss/steps/s00~s08/ - Each step: __init__.py + step.py + config.py + contracts.py + README.md
src/gss/steps/s02_slam/     - GS-SLAM wrapper (replaces s02+s03, MASt3R-SLAM/MonoGS/SLAM3R)
src/gss/steps/s03_3dgrut/   - NVIDIA 3DGRUT training (alternative to gsplat)
src/gss/steps/s03_planargs/ - PlanarGS wrapper (replaces s03+s04+s05)
src/gss/steps/s06b_plane_regularization/ - Geometric cleanup (7 sub-modules: A~F)
src/gss/steps/s06c_building_extraction/ - Exterior building reconstruction (6 sub-modules: A~F)
src/gss/steps/s08_mesh_export/  - IFC → GLB/USD export for digital twin platforms
src/gss/utils/         - Shared utilities (I/O, geometry, subprocess)
configs/               - YAML configs (pipeline.yaml, pipeline_planargs.yaml, pipeline_import.yaml, per-step)
data/                  - raw/ → interim/s01~s06/ → processed/
clone/PlanarGS/        - PlanarGS repo (separate conda env)
clone/MASt3R-SLAM/     - MASt3R-SLAM repo (separate conda env)
clone/MonoGS/          - MonoGS repo (separate conda env)
clone/SLAM3R/          - SLAM3R repo (separate conda env)
server/                - FastAPI API server (wraps pipeline, never modifies src/gss/)
server/routes/         - REST + WebSocket endpoints
server/grpc/           - gRPC render server (3DGRUT real-time rendering)
deploy/                - Dockerfile, docker-compose, nginx, setup_linux.sh
tests/test_server/     - Server unit tests (37 tests)
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
| SfM | hloc + COLMAP | pycolmap | 기존/PlanarGS/3DGRUT |
| SLAM | MASt3R-SLAM / MonoGS / SLAM3R | subprocess | SLAM |
| 3DGS | gsplat (2DGS mode) | gsplat | 기존 |
| 3DGS | 3DGRUT (rasterization/ray tracing) | hydra-core | 3DGRUT |
| 3DGS+Depth+TSDF | PlanarGS (subprocess) | diff-plane-rasterization | PlanarGS |
| TSDF | Open3D ScalableTSDFVolume | open3d | 기존 |
| Planes | Open3D RANSAC + alphashape | open3d, alphashape, shapely | both |
| Regularization | Normal/height snap, wall thickness, space detection, opening detection | numpy, shapely | both |
| Building Extraction | Ground separation, facade detection, footprint, roof structure | numpy, shapely, scipy | exterior |
| BIM | IfcOpenShell | ifcopenshell | both |
| Mesh Export | GLB (trimesh), USD/USDZ (usd-core) | trimesh, usd-core | both |

## Six Pipeline Configs
- `configs/pipeline.yaml` — 기존 9-step (s01→s02→s03→s04→s05→s06→s06b→s07→s08)
- `configs/pipeline_planargs.yaml` — PlanarGS 7-step (s01→s02→s03_planargs→s06→s06b→s07→s08)
- `configs/pipeline_3dgrut.yaml` — 3DGRUT 7-step (s01→s02→s03_3dgrut→s06→s06b→s07→s08)
- `configs/pipeline_slam.yaml` — SLAM 6-step (s01→s02_slam→s06→s06b→s07→s08)
- `configs/pipeline_exterior.yaml` — Exterior 10-step (s01→...→s06→**s06c**→s06b→s07→s08)
- `configs/pipeline_import.yaml` — Import 5-step (s00→s06→s06b→s07→s08)

## Conda Environments
- **기존 (gss)**: torch 2.8.0+cu128, gsplat 1.5.3
- **PlanarGS (planargs)**: torch 2.8.0+cu128, diff-plane-rasterization, simple-knn, pytorch3d, transformers<5
- **SLAM (mast3r_slam / monogs / slam3r)**: backend별 별도 conda env (Pattern C subprocess)

## Conventions
- Python 3.10+, Pydantic v2, type hints everywhere
- Config in YAML, never hardcode parameters in step code
- Data artifacts flow through `data/interim/s{NN}_{name}/` (s06b → `s06b_plane_regularization/`)
- Run: `gss run` (full pipeline) or `gss run-step <name>` (single step)
- Run PlanarGS: `gss run --config configs/pipeline_planargs.yaml`
- Run SLAM: `gss run --config configs/pipeline_slam.yaml`
- Run Import: `gss run --config configs/pipeline_import.yaml`
- Run Server: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- Test: `pytest tests/`
- Test Server: `pytest tests/test_server/`
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
- **3DGRUT**: NVIDIA 3D Gaussian Ray Tracing Utility Toolkit (rasterization + ray tracing)
- **3DGUT**: 3DGRUT rasterization mode (faster)
- **3DGRT**: 3DGRUT ray tracing mode (higher quality)
- **GS-SLAM**: Gaussian Splatting SLAM - simultaneous localization + 3D reconstruction
- **MASt3R-SLAM**: Dense monocular SLAM using MASt3R features (CVPR 2025)
- **MonoGS**: Monocular Gaussian Splatting SLAM (CVPR 2024 Highlight)
- **SLAM3R**: Feed-forward SLAM, no training loop (CVPR 2025 Highlight)

## s06b Regularization Sub-modules
| Phase | Module | Description |
|-------|--------|-------------|
| A | `_snap_normals` | Wall normals → Manhattan axes |
| B | `_snap_heights` | Floor/ceiling height clustering |
| C | `_wall_thickness` | Parallel pair detection + center-lines |
| C2 | `_wall_closure` | Synthesize missing walls from floor boundary |
| D | `_intersection_trimming` | Snap wall endpoints to corners |
| H2 | `_polyline_walls` | Merge collinear walls → N-point polyline center-lines |
| I | `_column_detection` | Reclassify short walls as columns |
| E | `_space_detection` | Polygonize center-lines → room boundaries |
| G | `_exterior_classification` | Interior/exterior wall labeling (disabled by default) |
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
- `_glb_writer.py`: trimesh.Scene → GLB (Z-up → Y-up: `(x,y,z)→(x,z,-y)`, sanitized node names)
- `_usd_writer.py`: UsdGeom.Mesh + normals + doubleSided + displayName + UsdPreviewSurface → USDC
- Default: `usd_up_axis=Y` (Isaac Sim/Omniverse standard), `usd_double_sided=true`
- Cross-platform: USDZ POSIX paths, GLB node sanitization, OSError graceful warning
- Both trimesh and usd-core: graceful degradation (`_has_trimesh()`, `_has_pxr()`)

## Server (API + Real-time Digital Twin)
`server/` wraps the pipeline as a FastAPI service. **Never modifies `src/gss/`.**

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | GPU status, queue size |
| POST | /scan | Upload PLY/MP4 → job_id |
| GET | /jobs/{id} | Job status + download URLs |
| WS | /ws/jobs/{id} | Real-time step progress (Phase 1) |
| WS | /ws/stream/{id} | Gaussian binary streaming (Phase 2) |
| GET | /nucleus/status | Omniverse Nucleus connection (Phase 3) |
| POST | /nucleus/connect | Connect to Nucleus server |
| POST | /nucleus/push/{id} | Push USD to Nucleus |
| GET | /sensors/ | Sensor state (Phase 4) |
| POST | /sensors/mqtt/connect | Start MQTT bridge |
| WS | /sensors/ws | Sensor real-time stream |
| GET | /render/models | List loaded gRPC render models |
| POST | /render/load | Load 3DGRUT model for rendering |
| WS | /ws/render/{model_id} | Real-time camera→frame rendering |

### Server Files
```
server/
├── app.py              - FastAPI app + lifespan
├── config.py           - ServerSettings (GSS_ env prefix, pydantic-settings)
├── models.py           - Pydantic request/response schemas
├── jobs.py             - JobManager (asyncio.Queue)
├── events.py           - EventBus (pub/sub per job)
├── worker.py           - GPU worker (ThreadPoolExecutor → run_pipeline)
├── splat_encoder.py    - Gaussian → 48B binary (Phase 2)
├── spatial_index.py    - CuboidIndex view-adaptive query (Phase 2)
├── nucleus.py          - Omniverse Nucleus connection + USD push (Phase 3)
├── mqtt.py             - MQTT bridge + TwinState (Phase 4)
├── grpc/
│   ├── protos/{common,render}.proto  - gRPC service definitions
│   ├── generated/                    - protoc output (gitignored)
│   ├── render_server.py              - gRPC render servicer (3DGRUT)
│   ├── render_client.py              - gRPC client (FastAPI→gRPC bridge)
│   └── compile_protos.py             - Proto compilation script
└── routes/{health,scan,ws,stream,nucleus_sync,sensors,render}.py
```

### Install & Run
```bash
pip install -e ".[server]"              # core server
pip install -e ".[server,realtime]"     # + MQTT (aiomqtt)
pip install -e ".[server,grpc]"         # + gRPC rendering
uvicorn server.app:app --host 0.0.0.0 --port 8000
python -m server.grpc.compile_protos    # compile proto files
python -m server.grpc.render_server     # standalone gRPC server
```

### Deploy
- `deploy/Dockerfile` — CUDA 12.1 + conda + uvicorn
- `deploy/docker-compose.yml` — GPU-reserved service
- `deploy/setup_linux.sh` — Bare-metal + systemd
- `deploy/nginx.conf` — Reverse proxy (2G upload, 1h timeout)
