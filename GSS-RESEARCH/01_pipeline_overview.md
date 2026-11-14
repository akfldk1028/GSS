# 01. 파이프라인 전체 워크플로우

## 1. 개요

GSS(3D Gaussian Splatting to BIM)는 카메라 영상을 입력으로 받아 건축물의 BIM(IFC) 모델을 자동 생성하는 파이프라인이다. LiDAR 스캐너 없이 일반 카메라만으로 Scan-to-BIM을 자동화하는 것이 핵심 목표이다.

**입력**: 건축물 내/외부 촬영 영상 또는 3DGS 학습 결과물 (PLY 점군)
**출력**: IFC 파일 (BIM), GLB/USD 메시 (디지털 트윈)

## 2. 지원 파이프라인 4종

### 2.1 기존 파이프라인 (Full Pipeline)
```
Video → s01(Frame Extraction) → s02(SfM/COLMAP) → s03(3DGS/gsplat 2DGS)
     → s04(Depth Rendering) → s05(TSDF Mesh) → s06(Plane Extraction)
     → s06b(Regularization) → s07(IFC Export) → s08(Mesh Export)
```
- 영상으로부터 끝까지 자동 처리
- gsplat의 2DGS 모드 사용 (표면 geometry 정확도 우수)

### 2.2 PlanarGS 파이프라인
```
Video → s01 → s02 → s03_planargs(PlanarGS) → s06 → s06b → s07 → s08
```
- NeurIPS 2025 PlanarGS 논문 구현체 활용
- 평면 prior를 3DGS 최적화에 직접 통합 → 평면 품질 대폭 향상
- s03+s04+s05를 하나의 스텝으로 대체

### 2.3 외부 건물 파이프라인 (Exterior)
```
Video → s01 → s02 → s03 → s04 → s05 → s06
     → s06c(Building Extraction) → s06b → s07 → s08
```
- s06c에서 지면 분리, 파사드 감지, 건물 윤곽선 추출, 지붕 구조 분석
- Non-Manhattan 벽 지원 (Atlanta-world 모델)

### 2.4 Import 파이프라인
```
PLY → s00(Import) → s06 → s06b → s07 → s08
```
- 이미 학습된 3DGS 결과물(PLY)을 직접 투입
- 테스트/검증에 주로 사용

## 3. 단계별 상세 설명

### s00: PLY Import
- **입력**: 3DGS 학습 결과 PLY 파일 (xyz + rgb + normals)
- **처리**: SH coefficient → RGB 변환, 표면 포인트 필터링
- **출력**: `surface_points.ply` (COLMAP 좌표계), `metadata.json`
- **핵심 기술**: Open3D 점군 처리, spherical harmonics 변환

### s01: Frame Extraction
- **입력**: 비디오 파일 (MP4/MOV)
- **처리**: FFmpeg으로 프레임 추출 (stride 설정 가능)
- **출력**: `frames/` 폴더에 JPEG 이미지들

### s02: Structure from Motion (SfM)
- **입력**: 프레임 이미지들
- **처리**: hloc + COLMAP으로 카메라 포즈 추정 및 희소 점군 생성
- **출력**: `colmap/sparse/` (카메라 intrinsics/extrinsics, sparse point cloud)
- **참조**: COLMAP (Schönberger & Frahm, CVPR 2016)

### s03: 3DGS Training (gsplat 2DGS)
- **입력**: 프레임 + COLMAP sparse model
- **처리**: gsplat 라이브러리로 2D Gaussian Splatting 학습
- **출력**: 학습된 Gaussian 파라미터 (positions, rotations, scales, SH coefficients)
- **참조**: 3DGS (Kerbl et al., ACM TOG 2023), 2DGS (Huang et al., SIGGRAPH 2024)

### s03_planargs: PlanarGS Training (대안)
- **입력**: 프레임 + COLMAP sparse model
- **처리**: PlanarGS subprocess 실행 (별도 conda 환경)
- **출력**: 평면-정렬된 Gaussian + 직접 평면 정보
- **참조**: PlanarGS (Jin et al., NeurIPS 2025)

### s04: Depth Rendering
- **입력**: 학습된 3DGS 모델
- **처리**: 각 카메라 뷰에서 depth map 렌더링
- **출력**: depth maps (`.npy` 배열)

### s05: TSDF Mesh Reconstruction
- **입력**: depth maps + camera poses
- **처리**: Open3D ScalableTSDFVolume으로 depth fusion → mesh 추출
- **출력**: `surface_mesh.ply` (삼각형 메시)
- **참조**: TSDF (Curless & Levoy, SIGGRAPH 1996)

### s06: Plane Extraction
- **입력**: `surface_points.ply` (또는 TSDF mesh 점군)
- **처리**:
  1. Open3D RANSAC으로 평면 세그먼트 반복 추출
  2. 법선 방향으로 벽(수직)/바닥(수평↓)/천장(수평↑) 분류
  3. Alpha shape로 각 평면의 2D 경계 다각형 계산
- **출력**: `planes.json` (평면 방정식, 분류, 인라이어), `boundaries.json` (경계 다각형)
- **참조**: RANSAC (Fischler & Bolles, 1981), Alpha Shapes (Edelsbrunner, 1983)

### s06c: Building Extraction (외부 건물 전용)
- **입력**: `planes.json`, `boundaries.json`
- **처리**:
  - A) 지면 분리: 최대 수평면 → ground 라벨
  - B) 건물 세그먼테이션: DBSCAN 밀도 클러스터링 (선택)
  - C) 파사드 감지: 법선 클러스터링 → 수직면 그룹핑
  - D) 건물 윤곽선: Alpha shape / concave hull → 2D 평면도
  - E) 지붕 구조: 평면-평면 교차 → 용마루/처마/골 라인
  - F) 층수 감지: 높이 히스토그램 valley detection (선택)
- **출력**: `building_context.json` (footprint, facades, roof_structure, storeys)
- **참조**: GS4Buildings (Zhang et al., ISPRS 2025)

### s06b: Plane Regularization (핵심 단계)
- **입력**: `planes.json`, `boundaries.json`
- **처리** (7개 서브모듈):
  - **A) Normal Snap**: 벽 법선을 Manhattan 축(±X, ±Z)에 정렬
    - `normal_mode: "cluster"` → 데이터 주도 축 탐색 (Non-Manhattan)
  - **B) Height Snap**: 바닥/천장 높이 클러스터링 (DBSCAN)
  - **C) Wall Thickness**: 평행 벽 쌍 감지 → 중심선(center-line) 계산
  - **C2) Wall Closure**: 바닥 경계에서 누락된 벽 합성 (gap filling)
    - Manhattan: AABB, Cluster: ConvexHull 또는 ConcaveHull
  - **D) Intersection Trimming**: 벽 끝점을 교차점에 스냅
    - `_constrained_snap`: 벽 법선축 보존 (X-법선 벽 → X 고정)
  - **E) Space Detection**: 중심선 → Shapely polygonize → 방(room) 경계
  - **F) Opening Detection**: Cloud2BIM 히스토그램 방식 문/창 감지 (선택)
- **출력**: `walls.json`, `spaces.json`, `manhattan_alignment.json`, `stats.json`
- **좌표계**: 내부적으로 Manhattan-aligned space에서 처리, 결과물은 원본 좌표와 Manhattan 좌표 모두 포함
- **참조**: Cloud2BIM (Zbirovský & Nežerka, 2025), AtlasGS (Zhang et al., NeurIPS 2025)

### s07: IFC Export
- **입력**: `walls.json`, `spaces.json`, `planes.json`
- **처리**:
  - IFC 계층 구조 생성: Project → Site → Building → Storey
  - 벽: center_line_2d → IfcArbitraryClosedProfileDef (polyline profile) → IfcExtrudedAreaSolid
  - 슬래브: boundary_2d → IfcSlab (바닥/천장)
  - 공간: boundary_2d → IfcSpace (방 경계)
  - 개구부: IfcOpeningElement → IfcDoor/IfcWindow (IfcRelVoidsElement + IfcRelFillsElement)
- **출력**: `GSS_*.ifc`
- **좌표계**: IFC X,Y = plan view (Manhattan mx/scale, mz/scale), Z = height (my/scale)
- **참조**: IFC4 (ISO 16739-1:2018), IfcOpenShell

### s08: Mesh Export
- **입력**: IFC 파일
- **처리**:
  - ifcopenshell.geom.create_shape() → 삼각형 메시 추출
  - trimesh → GLB (Z-up → Y-up 변환: `(x,y,z)→(x,z,-y)`)
  - pxr UsdGeom.Mesh → USDC/USDZ (PBR 재질 포함)
- **출력**: `GSS_*.glb`, `GSS_*.usdc`, `GSS_*.usdz` (선택)
- **참조**: glTF 2.0, USD (Pixar), NVIDIA IFC-in-USD Initiative

## 4. 좌표계 변환 흐름

```
COLMAP 좌표 (원본)
    │
    │  pts @ R.T  (Manhattan rotation)
    ▼
Manhattan 좌표 (축 정렬, Y-up)
    │  ┌─ X: left-right
    │  ├─ Y: height (floor→ceiling)
    │  └─ Z: front-back
    │
    │  (mx/scale, mz/scale, my/scale)
    ▼
IFC 좌표 (미터 단위)
    │  ┌─ X: plan X
    │  ├─ Y: plan Y
    │  └─ Z: height
    │
    │  (x, z, -y) for GLB/USD
    ▼
Viewer 좌표 (Y-up)
```

## 5. 스케일 추정 전략

3DGS 점군은 임의 스케일이므로 실세계 미터 단위로 변환이 필요:

1. **1순위**: 바닥-천장 Y축 간격 / expected_storey_height (2.7m 기본)
2. **2순위**: 벽의 Y축 범위 / expected_storey_height
3. **3순위**: XZ 평면 범위 기반 폴백

`coordinate_scale` 값이 `spaces.json`에 저장되어 s07 IFC export에서 미터 변환에 사용.

## 6. 핵심 설계 결정

### Manhattan-world 가정
- 대부분의 건축물은 직교 구조 (벽이 90도)
- 벽 법선을 ±X 또는 ±Z로 snap하면 깔끔한 직사각형 벽 생성
- Non-Manhattan 건물은 `normal_mode: "cluster"`로 데이터 주도 축 탐색

### Center-line 기반 벽 모델링
- 감지된 평면 쌍(안/밖)에서 중심선 계산
- IFC 벽은 중심선 + 두께로 정의 (IfcArbitraryClosedProfileDef)
- Cloud2BIM과 동일한 패턴

### 합성 벽 (Synthetic Walls)
- RANSAC이 놓친 벽은 바닥 경계에서 합성
- 완전한 방(room) 경계를 위해 필수
- Shapely polygonize가 닫힌 다각형을 생성하려면 모든 벽이 연결되어야 함
