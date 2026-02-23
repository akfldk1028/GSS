# GSS — 3DGS to BIM Pipeline

Video → 3D Gaussian Splatting → Surface Reconstruction → BIM (IFC) 자동 변환 파이프라인.

## Two Pipelines

### 기존 (7-step)
```
s01 → s02 → s03(gsplat) → s04(depth) → s05(TSDF) → s06 → s07
```
Config: `configs/pipeline.yaml`

### PlanarGS (5-step, 권장)
```
s01 → s02 → s03_planargs(PlanarGS) → s06 → s07
```
Config: `configs/pipeline_planargs.yaml`

## Pipeline Steps

| # | Step | Description | Pipeline |
|---|------|-------------|----------|
| s01 | extract_frames | 비디오에서 프레임 추출 (OpenCV) | both |
| s02 | colmap | SfM 카메라 포즈 추정 (pycolmap) | both |
| s03 | gaussian_splatting | 2DGS 학습 (gsplat) | 기존 |
| s03_planargs | planargs | PlanarGS subprocess (NeurIPS 2025) | PlanarGS |
| s04 | depth_render | Depth/Normal 맵 렌더링 (gsplat) | 기존 |
| s05 | tsdf_fusion | TSDF 볼륨 통합 (Open3D) | 기존 |
| s06 | plane_extraction | 평면 추출 + Manhattan 정렬 + 병합 + 경계 | both |
| s07 | ifc_export | IFC4 BIM 파일 생성 (IfcOpenShell) | both |

## Quick Start

```bash
pip install -e .
gss info                                        # Show pipeline steps
gss run                                         # Run 기존 pipeline
gss run --config configs/pipeline_planargs.yaml  # Run PlanarGS pipeline
gss run-step plane_extraction                    # Run single step
```

## E2E Test Results (Replica room0)

| Metric | PlanarGS |
|--------|----------|
| Surface points | 21.4M |
| PSNR | 40.13 |
| Walls detected | 3 |
| Floors detected | 1 |
| Ceilings detected | 2 |
| s06 time | ~100s |
| Total time | ~45min |

## Tech Stack

| Phase | Tool | Package |
|-------|------|---------|
| SfM | COLMAP | pycolmap |
| 3DGS (기존) | gsplat 2DGS | gsplat |
| 3DGS (PlanarGS) | PlanarGS subprocess | diff-plane-rasterization |
| TSDF | Open3D ScalableTSDFVolume | open3d |
| Planes | RANSAC + Manhattan + coplanar merge | open3d, shapely |
| BIM | IfcOpenShell | ifcopenshell |

## s06 Plane Extraction Features

- **Manhattan World Alignment**: normal histogram → auto axis detection → axis-aligned RANSAC
- **Coplanar Merging**: Union-Find + centroid separation → SVD refit
- **Position-based Classification**: horizontal planes → floor/ceiling/furniture by height
- **Architectural Filtering**: small walls/ceilings (< 10% of max) → reclassified as furniture
- **Clean Boundaries**: minimum_rotated_rectangle for wall/floor/ceiling (5-vertex rectangles)

## Project Structure

```
src/gss/core/          - Pipeline runner, BaseStep ABC
src/gss/steps/s01~s07/ - Each step: step.py + config.py + contracts.py + README.md
src/gss/steps/s03_planargs/ - PlanarGS wrapper (replaces s03+s04+s05)
configs/               - YAML configs (pipeline.yaml, pipeline_planargs.yaml)
data/                  - raw/ → interim/s01~s06/ → processed/
clone/PlanarGS/        - PlanarGS repo (separate conda env)
docs/                  - Research papers, images
```
