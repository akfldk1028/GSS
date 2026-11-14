# 02. 단계별 참조 논문 & 기술 근거

각 파이프라인 단계가 어떤 논문/기술을 참조했는지, 왜 해당 방식을 선택했는지 상세히 기술한다.

---

## Phase 1: Scene Capture → 3D Reconstruction (s01~s03)

### 핵심 논문

#### [P1] 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **저자**: Kerbl, Kopanas, Leimkühler, Drettakis
- **발표**: ACM Transactions on Graphics (SIGGRAPH), 2023
- **arxiv**: 2308.04079
- **기여**: 3DGS의 원조 논문. Gaussian primitives로 장면을 표현하여 NeRF 대비 100배 이상 빠른 렌더링 달성.
- **GSS 적용**: 전체 파이프라인의 기반. 3DGS 점군이 모든 후속 단계의 입력.

#### [P2] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **저자**: Huang, Yu, Chen, Geiger, Gao
- **발표**: SIGGRAPH 2024
- **arxiv**: 2403.17888
- **DOI**: 10.1145/3641519.3657428
- **기여**: 3D Gaussian을 2D oriented disk으로 대체하여 표면 geometry 정확도 대폭 향상. 멀티뷰 일관성 문제 해결.
- **GSS 적용**: s03에서 gsplat의 2DGS 모드 사용. 평면 추출 품질에 직접 영향.
- **선택 이유**: 3DGS는 Gaussian이 표면 안팎으로 퍼져서 RANSAC 평면 추출이 어려움. 2DGS는 flat disk이므로 표면에 정렬되어 평면 감지가 훨씬 정확.

#### [P3] PlanarGS: High-Fidelity Indoor 3DGS Guided by Vision-Language Planar Priors
- **저자**: Jin, Jin, Li et al.
- **발표**: NeurIPS 2025
- **arxiv**: 2510.23930
- **기여**: Vision-Language 모델(SAM2+CLIP)을 사용하여 실내 장면의 평면 prior를 추출하고, 이를 3DGS 최적화에 직접 통합. 특히 넓은 low-texture 영역(벽, 바닥)에서 geometry 품질 대폭 향상.
- **GSS 적용**: s03_planargs 스텝이 이 논문의 구현체를 subprocess로 실행. s03+s04+s05를 한 번에 대체.
- **선택 이유**: 실내 건축물 재구성에 특화. 기존 3DGS/2DGS가 실패하는 넓고 텍스처 없는 벽면에서도 정확한 평면 생성.

### 보조 기술

#### COLMAP (Schönberger & Frahm, CVPR 2016)
- SfM(Structure from Motion) 표준 도구
- 카메라 포즈 추정 + 희소 점군 생성
- s02에서 사용

#### gsplat (Ye et al., 2024)
- 3DGS/2DGS 학습을 위한 고성능 CUDA 라이브러리
- PyTorch 네이티브 통합
- s03에서 사용

---

## Phase 2: Surface Extraction (s04~s05)

### 핵심 논문

#### [P4] SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction
- **저자**: Guédon, Lepetit
- **발표**: CVPR 2024
- **arxiv**: 2311.12775
- **기여**: 3DGS에 regularization term을 추가하여 Gaussian을 표면에 정렬시킨 후, Poisson reconstruction으로 mesh 추출. 최초의 3DGS→mesh 방법.
- **GSS 관련**: s04+s05의 대안으로 검토. 현재는 TSDF 방식 사용.
- **비교**: SuGaR는 시각적 품질 우수하나, 평면 추출 목적에는 TSDF가 더 깨끗한 평면 생성.

#### TSDF Fusion (Curless & Levoy, SIGGRAPH 1996)
- **기여**: Depth map들을 volumetric fusion으로 통합하여 watertight mesh 생성
- **GSS 적용**: s05에서 Open3D ScalableTSDFVolume 사용
- **선택 이유**: RANSAC 평면 추출 전 노이즈 제거에 효과적. Depth map의 불확실성을 평균화.

---

## Phase 3: Plane Extraction (s06)

### 핵심 논문

#### RANSAC (Fischler & Bolles, Communications of the ACM, 1981)
- **기여**: 아웃라이어에 강건한 모델 피팅 알고리즘
- **GSS 적용**: s06에서 Open3D의 `segment_plane()` 반복 호출로 점군에서 평면 추출
- **방법**: 가장 많은 인라이어를 가진 평면부터 순차 추출. 추출된 인라이어는 제거 후 다음 평면 탐색.
- **선택 이유**: 구현이 간단하고, 3DGS 점군의 노이즈에 강건. Cloud2BIM도 RANSAC 사용.

#### Alpha Shapes (Edelsbrunner, Kirkpatrick, Seidel, 1983)
- **기여**: 점 집합의 "형태"를 캡처하는 일반화된 convex hull
- **GSS 적용**: s06에서 각 평면 인라이어의 2D 경계 다각형 계산
- **선택 이유**: Convex hull과 달리 L자형 등 오목한 경계도 정확히 표현

#### [P5] GSPlane: Structured Planar Reconstruction via Gaussian Splatting
- **저자**: Gan, Peng, Liu et al.
- **arxiv**: 2510.17095 (2025)
- **기여**: 3DGS 최적화 과정에서 평면 구조를 직접 학습. Gaussian에 평면 라벨을 부여.
- **GSS 관련**: 현재 post-hoc RANSAC 방식 대신, 향후 end-to-end 평면 추출로 전환 시 참고할 핵심 논문.

---

## Phase 4: Building Extraction - 외부 건물 (s06c)

### 핵심 논문

#### [P6] GS4Buildings: Prior-Guided Gaussian Splatting for 3D Building Reconstruction
- **저자**: Zhang, Wysocki, Jutzi
- **발표**: ISPRS 3D GeoInfo 2025
- **arxiv**: 2508.07355
- **기여**: LoD2 geometric prior(건물 = 직육면체 + 지붕)를 2DGS에 통합. 도시 스케일 건물 재구성.
- **GSS 적용**: s06c의 전체 아키텍처 영감. 지면 분리 → 파사드 감지 → 윤곽선 추출 → 지붕 구조 분석의 흐름.
- **선택 이유**: 건물 외부는 내부와 다른 처리 필요. 지면 분리, 파사드 클러스터링 등의 개념 차용.

### 서브모듈별 참조

| 서브모듈 | 방법론 | 참조 |
|----------|--------|------|
| A: 지면 분리 | 최대 수평면 = ground | GS4Buildings의 ground plane detection |
| B: 건물 세그먼테이션 | DBSCAN 밀도 클러스터링 | Ester et al. (1996) |
| C: 파사드 감지 | 법선 방향 클러스터링 → d값 클러스터링 | Cloud2BIM의 wall grouping 확장 |
| D: 건물 윤곽선 | Alpha shape / concave hull | Edelsbrunner (1983) |
| E: 지붕 구조 | 평면-평면 교차 → ridge/eave 라인 | CityGML LoD2 reconstruction 패턴 |
| F: 층수 감지 | 높이 히스토그램 valley detection | Cloud2BIM의 height clustering 확장 |

---

## Phase 5: Plane Regularization (s06b) — 핵심 알고리즘

### 핵심 논문

#### [P7] Cloud2BIM: Open-source Automatic Pipeline for Point Cloud to IFC
- **저자**: Zbirovský, Nežerka
- **발표**: Automation in Construction, Vol. 177, 2025
- **arxiv**: 2503.11498
- **DOI**: 10.1016/j.autcon.2025.106303
- **기여**: 점군 → IFC 전체 자동화. 히스토그램 기반 개구부 감지, 벽 두께 추정, 공간 분할.
- **GSS 적용**: s06b의 전체 아키텍처가 이 논문에서 가장 많은 영향을 받음.
  - 벽 두께 계산 (C): 평행 벽 쌍 감지 → 중심선
  - 개구부 감지 (F): 벽면 projection → 히스토그램 void detection
  - 공간 분할 (E): center-line polygonize → room boundary
- **차이점**: Cloud2BIM은 LiDAR 점군 입력, GSS는 3DGS 점군 입력. GSS는 Manhattan-world 정규화를 추가하여 3DGS의 노이즈 보상.

#### [P8] AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians
- **저자**: Zhang, Bao, Chen et al.
- **발표**: NeurIPS 2025
- **arxiv**: 2510.25129
- **기여**: Manhattan-world를 Atlanta-world로 일반화. 비직교 벽 구조도 처리 가능한 전역 일관성 보장.
- **GSS 적용**: s06b의 `normal_mode: "cluster"` 옵션. Manhattan이 아닌 건물(L자형, 비직교)에서 데이터 주도로 벽 축을 탐색.
- **구현**:
  - `_snap_normals.py`의 `_discover_wall_axes()`: XZ-projected 법선의 greedy 클러스터링
  - `normal_axis` 필드: "x", "z" (Manhattan) 또는 "oblique:<angle>" (Non-Manhattan)
  - `normal_vector` 필드: [nx, nz] (임의 각도 벽)

#### Bassier et al. (2020) — Wall Topology & Polyline Axis
- **기여**: 벽 중심선을 polyline/곡선으로 모델링, 4가지 연결 토폴로지 정의
- **GSS 적용**: s06b의 벽 중심선 모델과 s07의 IfcArbitraryClosedProfileDef 방식에 영향
- **구현**: 현재는 2-point center-line. 향후 N-point polyline 확장 시 참고.

### 서브모듈별 기술 근거

| 서브모듈 | 핵심 알고리즘 | 참조 논문/기술 | 선택 이유 |
|----------|--------------|---------------|-----------|
| A: Normal Snap | Manhattan축 정렬 (10° 이내 → snap) | Manhattan-world 가정 (Coughlan & Yuille, 1999) | 3DGS 점군의 법선 노이즈 보상 |
| B: Height Snap | DBSCAN 클러스터링 (eps=0.3×scale) | DBSCAN (Ester et al., 1996) | 여러 층의 바닥/천장 자동 구분 |
| C: Wall Thickness | 법선축 d값 차이 → 두께 | Cloud2BIM wall pairing | 평행면 쌍에서 중심선 추출 |
| C2: Wall Closure | 바닥 경계 분석 → 합성 벽 | GSS 자체 개발 | RANSAC 누락 벽 보완 |
| D: Intersection Trim | constrained snap (벽 축 보존) | GSS 자체 개발 | 직사각형 벽 정확도 보장 |
| E: Space Detection | Shapely polygonize | Shapely/GEOS 라이브러리 | 중심선 네트워크 → 닫힌 방 경계 |
| F: Opening Detection | projection → histogram gap | Cloud2BIM (2025) | LiDAR 없이 3DGS 점군으로 가능 |

---

## Phase 6: IFC Export (s07)

### 핵심 참조

#### IFC4 Standard (ISO 16739-1:2018)
- **기여**: BIM 데이터 교환 국제 표준
- **GSS 적용**: s07의 전체 출력 형식
- **IFC 계층**: IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey → IfcWall/IfcSlab/IfcSpace

#### IfcOpenShell (open-source IFC toolkit)
- **GSS 적용**: IFC 파일 읽기/쓰기, geometry 생성
- **핵심 패턴**:
  - `IfcArbitraryClosedProfileDef` + `IfcPolyline` → 임의 단면 벽 프로파일
  - `IfcExtrudedAreaSolid` → 프로파일을 높이만큼 돌출
  - `IfcRelVoidsElement` → 벽에 개구부 생성
  - `IfcRelFillsElement` → 개구부에 문/창 삽입

#### Cloud2BIM IFC Generation Pattern
- center-line + thickness → IfcWall
- boundary polygon → IfcSlab (extrusion)
- room polygon → IfcSpace
- GSS의 s07은 이 패턴을 거의 그대로 따름

---

## Phase 7: Mesh Export (s08)

### 핵심 참조

#### glTF 2.0 / GLB (Khronos Group)
- 범용 3D 포맷. 웹 뷰어, 게임 엔진, AR/VR에서 사용
- trimesh 라이브러리로 생성
- Z-up → Y-up 좌표 변환 필요: `(x,y,z) → (x,z,-y)`

#### USD / USDC / USDZ (Pixar/Apple)
- Universal Scene Description. NVIDIA Omniverse, Apple Vision Pro 대응
- usd-core 라이브러리로 생성
- PBR 재질: UsdPreviewSurface (roughness=0.7, metallic=0.0)
- MaterialBindingAPI.Apply() 필수 (Omniverse spec compliance)

#### NVIDIA IFC-in-USD Initiative
- IFC schema를 USD에 네이티브 인코딩하는 프로젝트 (진행 중)
- 완성 시 IFC↔USD 무손실 변환 가능
- 현재 GSS는 geometry-level 변환 (IFC→mesh→USD)

#### [P9] Material-informed Gaussian Splatting for Digital Twin
- **저자**: Huynh, Silva, Caesar et al.
- **발표**: IEEE IV 2026 (accepted)
- **arxiv**: 2511.20348
- **기여**: 3DGS → mesh → material labels → physics simulation. Camera-only 디지털 트윈.
- **GSS 관련**: s08의 자연스러운 확장 방향. 현재는 IFC class별 고정 색상, 향후 semantic material 분류 적용 가능.

---

## 논문 영향도 요약

```
영향도: ★★★ 직접 구현 참조  ★★ 아키텍처 영감  ★ 배경/비교

[P1] 3DGS (Kerbl 2023)          ★★★  전체 파이프라인 기반
[P2] 2DGS (Huang 2024)          ★★★  s03 표면 reconstruction 핵심
[P3] PlanarGS (Jin 2025)        ★★★  s03_planargs 직접 통합
[P7] Cloud2BIM (Zbirovský 2025) ★★★  s06b 전체 + s07 IFC 패턴
[P8] AtlasGS (Zhang 2025)       ★★   s06b Non-Manhattan 지원
[P6] GS4Buildings (Zhang 2025)  ★★   s06c 외부 건물 아키텍처
[P4] SuGaR (Guédon 2024)        ★    mesh 추출 대안 검토
[P5] GSPlane (Gan 2025)         ★    향후 end-to-end 평면 추출 참고
[P9] Material-GS (Huynh 2025)   ★    향후 material 분류 확장 참고
     Bassier et al. (2020)      ★★   벽 polyline/topology 모델링
     COLMAP (Schönberger 2016)  ★★★  s02 SfM 직접 사용
     RANSAC (Fischler 1981)     ★★★  s06 평면 추출 직접 사용
     IFC4 (ISO 16739-1:2018)    ★★★  s07 출력 표준
```
