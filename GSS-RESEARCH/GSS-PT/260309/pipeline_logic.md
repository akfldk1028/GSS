# GSS 파이프라인 알고리즘 상세 설명

## 전체 흐름
동영상 → 프레임 추출 → COLMAP(SfM) → 3D Gaussian Splatting → 평면 추출 → 기하 정규화 → IFC BIM → GLB/USD 메시

## Step 1: 동영상 → 3D Gaussian Splatting (s01~s03)
### 원리
- 동영상에서 프레임을 추출하고, COLMAP의 Structure-from-Motion으로 각 프레임의 카메라 위치를 추정한다.
- 추정된 카메라 포즈를 기반으로 3D Gaussian Splatting(3DGS)을 학습한다.
- 3DGS는 장면을 수백만 개의 3D 가우시안 원소로 표현한다. 각 가우시안은 위치, 크기, 회전, 색상, 투명도를 가진다.
- 학습 완료 후 가우시안의 중심점을 추출하면 건물 구조를 담은 포인트 클라우드가 된다.
### 핵심
- LiDAR 없이 카메라만으로 3D 포인트 클라우드를 생성할 수 있다.
- 벽, 바닥, 천장 표면에 가우시안이 밀집되어 건물 구조가 자연스럽게 드러난다.
- 2DGS(gsplat) 모드: 평평한 디스크 형태 → 표면 품질 향상
- PlanarGS: co-planarity prior 적용 → 평면 품질 대폭 향상 (NeurIPS 2025)

## Step 2: 평면 추출 (s06)
### 원리
- 포인트 클라우드에서 RANSAC 알고리즘으로 평면을 반복 추출한다.
- RANSAC: 무작위로 3개 점을 선택 → 평면 방정식 생성 → 일정 거리 이내의 점(inlier)을 세어 최적 평면 선택
- 추출된 평면의 법선 벡터 방향으로 분류: 수직이면 벽(wall), 수평이면 바닥(floor) 또는 천장(ceiling)
- Manhattan World 가정: 건물의 주축을 찾아 좌표계를 정렬 (벽이 축에 평행하도록 회전)
- Alpha Shape으로 각 평면의 경계 폴리곤을 생성한다.
### 결과
- JH 데이터: 80,000개 포인트 → 8개 평면 (벽 4, 바닥 1, 천장 1, 기타 2)
- 동일 평면(coplanar) 자동 병합으로 중복 제거

## Step 3: 기하 정규화 s06b (10개 서브 모듈)
### A. 법선 스냅 (_snap_normals)
- 벽의 법선 벡터를 가장 가까운 Manhattan 축(±X 또는 ±Z)으로 스냅한다.
- RANSAC으로 추출된 평면은 약간 기울어져 있는데, 이를 직교 좌표계에 정렬한다.

### B. 높이 클러스터링 (_snap_heights)
- 바닥과 천장의 Y좌표를 클러스터링하여 일관된 높이로 통일한다.
- 예: 바닥이 -1.11m, -1.09m, -1.13m → 모두 -1.11m로 통일

### C. 벽 두께 추정 (_wall_thickness)
- 평행한 벽 쌍을 찾아서 (같은 법선, 반대 방향) 두 벽 사이의 거리를 벽 두께로 추정한다.
- 벽 쌍의 중심선(center-line)을 계산한다. 이것이 IFC 벽의 기준선이 된다.

### C2. 벽 보완 (_wall_closure)
- 바닥 경계선에서 벽이 감지되지 않은 부분을 찾아 합성 벽(synthetic wall)을 생성한다.
- Manhattan 모드: AABB 경계, Cluster 모드: Convex/Concave Hull 사용

### D. 교차점 트리밍 (_intersection_trimming)
- 벽의 끝점을 교차점(코너)에 정확히 스냅한다.
- X축 법선 벽은 X좌표만 변경, Z축 법선 벽은 Z좌표만 변경 (축 누출 방지)
- 결과: 완벽한 직사각형 방 형태

### H2. 폴리라인 벽 (_polyline_walls)
- 같은 축 위의 연속된 벽들을 하나의 N-point 폴리라인으로 병합한다.
- L자 건물처럼 복잡한 형태의 벽도 하나의 연속 벽으로 표현 가능

### I. 기둥 감지 (_column_detection)
- 너무 짧은 벽(길이 < 임계값)은 기둥(column)으로 재분류한다.

### E. 공간 감지 (_space_detection)
- 벽 센터라인들로 다각형을 만들고(polygonize), 닫힌 영역을 방(room/space)으로 감지한다.
- L자 건물: 3개의 분리된 공간 자동 감지

### 스케일 추정
- 바닥-천장 높이 차이를 실제 층고(2.7m 기준)와 비교하여 scene_units → meters 변환 비율 계산
- 예: JH 데이터 scale=1.70 (1.70 scene units = 1 meter)

## Step 4: IFC BIM 생성 (s07)
### 원리
- 정규화된 벽/바닥/천장/공간 데이터를 IFC4 표준 BIM 파일로 변환한다.
- IFC 계층: IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey
- IfcWall: center_line_2d를 기반으로 IfcArbitraryClosedProfileDef 프로파일 생성 → 높이만큼 extrude
- IfcSlab: boundary_2d 폴리곤으로 바닥/천장 슬래브 생성
- IfcSpace: 공간 경계 폴리곤을 층고만큼 extrude
- IfcOpeningElement: 문/창 개구부 (Cloud2BIM 히스토그램 방식)
### 좌표 변환
- Manhattan 좌표(axis-aligned) → IFC 좌표(X=plan_x, Y=plan_z, Z=height)
- coordinate_scale로 scene units → meters 변환

## Step 5: 메시 내보내기 (s08)
### GLB (trimesh)
- IFC에서 삼각형 메시 추출 → trimesh.Scene → GLB 바이너리
- Z-up → Y-up 변환: (x,y,z) → (x,z,-y)
- IFC 클래스별 색상 (벽=파랑, 바닥=초록, 천장=노랑)

### USD/USDC (usd-core)
- UsdGeom.Mesh + vertex normals + PBR materials (UsdPreviewSurface)
- MaterialBindingAPI.Apply() 후 Bind() (Omniverse 사양 준수)
- doubleSided=True (얇은 BIM 벽 양면 렌더링)

### USDZ
- USDC를 패키징 → Apple Vision Pro / ARKit 바로 사용 가능

## 검증 결과
### 잘 된 데이터셋
- JH: 4벽 1바닥 1천장 1공간 (완벽한 직사각형 방)
- DrJohnson: 4벽(2 합성) 1바닥 1천장 1공간 (긴 복도형)
- Playroom: 4벽(2 합성) 1바닥 1천장 1공간
- Synthetic_L: 6벽 3바닥 3천장 3공간 (L자 비직교 건물 완벽 감지)

### 3DGS ↔ BIM 좌표 정합성
- JH 데이터 80,000개 포인트를 IFC 좌표로 변환 후 BIM 와이어프레임과 오버레이
- 포인트 클라우드가 BIM 박스 안에 정확히 위치함을 시각적으로 검증

### 365개 유닛 테스트
- s06b 정규화: 94개, s07 IFC: 56개, s08 메시: 23개, s06c 건물추출: 28개, s06d 메시분할: 26개, 개구부: 19개

## 참고 논문
- PlanarGS (NeurIPS 2025): Indoor 3DGS + co-planarity prior
- Cloud2BIM (Automation in Construction, 2025): Point cloud → IFC 자동 변환, 97.6% 벽 감지율
- Material-informed GS (2025): 3DGS → mesh → material labels → physics simulation
- Integrating CAD, BIM, and 3DGS (Nature Scientific Reports, 2026): 3DGS가 as-built 검증의 핵심 도구
- NVIDIA IFC-in-USD Initiative: IFC schema를 USD에 네이티브 인코딩 (진행 중)

## GSS의 차별점
- 카메라만으로 자동 BIM (LiDAR 불필요)
- End-to-End 자동화 (Video → IFC → GLB/USD)
- 크로스 플랫폼: Omniverse(USD), Apple Vision Pro(USDZ), UE5/Unity(GLB)
- 비직교/다각형 건물 지원 (L자, concave hull)
- 365개 테스트로 검증된 견고한 파이프라인
