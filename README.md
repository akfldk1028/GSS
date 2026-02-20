# GSS — 3DGS to BIM Pipeline

Video → 3D Gaussian Splatting → Surface Reconstruction → BIM (IFC) 자동 변환 파이프라인.

## Pipeline Steps

| # | Step | Description |
|---|------|-------------|
| s01 | extract_frames | 비디오에서 프레임 추출 (OpenCV) |
| s02 | colmap | SfM 카메라 포즈 추정 (hloc + COLMAP) |
| s03 | gaussian_splatting | 2DGS 학습 (gsplat) |
| s04 | depth_render | Depth/Normal 맵 렌더링 |
| s05 | tsdf_fusion | TSDF 볼륨 통합 (Open3D) |
| s06 | plane_extraction | 평면 추출 + 경계 검출 (RANSAC + alphashape) |
| s07 | ifc_export | IFC4 BIM 파일 생성 (IfcOpenShell) |

## Quick Start

```bash
pip install -e .
gss info          # Show pipeline steps
gss run           # Run full pipeline
gss run-step extract_frames -i '{"video_path": "data/raw/video.mp4"}'
```

## Tech Stack

- **SfM**: hloc (SuperPoint + LightGlue) + COLMAP
- **3DGS**: gsplat (2DGS mode)
- **Surface**: Open3D ScalableTSDFVolume
- **BIM**: IfcOpenShell (IFC4)
