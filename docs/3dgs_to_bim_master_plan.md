# 3DGS → BIM (최종) 마스터 로드맵 (Claude Code용)  
작성 목적: Claude Code(또는 에이전트)가 “무엇을 어떤 순서로 구현해야 하는지”를 그대로 실행/분해할 수 있도록 **Phase, 입력/출력, DoD, 리스크, 티켓**을 고정해 둔다.

---

## 0) 최종 목표 (North Star)
- 최종 목표: **3D Gaussian Splatting(3DGS) 기반 재구성 결과를 BIM(IFC/Revit 객체)**로 자동 변환
- 범위(최종적으로 “전부”):  
  - LOD200~300: 벽/바닥/천장 + (가능하면) 문/창 + 레벨/룸(부분)  
  - 이후 확장: LOD350~400/MEP, 동적 업데이트, 로봇/캐릭터(충돌/내비)  
- 중요한 원칙(설계 고정):  
  - 3DGS는 “visual twin(보이는 레이어)”에 강함  
  - BIM/로봇은 “geometry+semantics(만질 수 있는 레이어)”가 필요  
  - 따라서 **3DGS → Surface(implicit) → (Plane/Boundary/Openings) → BIM**이 기본 주 흐름  
  - Mesh는 현재 단계에서 만들지 않으며, 필요해지면 TSDF에서 선택적으로 export 가능

---

## 1) 지금 우리가 하는 것 (Phase 1 고정)
- 이번 단계(Phase 1): **3DGS → Surface(implicit: TSDF/SDF 계열)**만 구현
- Mesh는 만들지 않는다(현재 스코프 제외).  
  - 이유: watertight mesh 품질 보정(홀필/리메시/자가교차)은 비용이 크고, BIM 객체화에는 “평면/경계/개구부”가 핵심이라 TSDF/SDF가 MVP에 더 직행임.

---

## 2) 왜 Surface(implicit) 우선이 BIM에 빠르냐
- BIM 객체화(벽/바닥/천장/문/창)는 결국 다음으로 귀결되는 경우가 많음:  
  - 평면(plane) + 경계(polyline) + 개구부(opening) + 레벨(level)  
- TSDF/SDF 기반 surface는:  
  - 거리/법선(∇f) 기반으로 평면 피팅과 경계 추출이 단순  
  - 로봇/충돌/내비까지 확장 가능한 geometry twin으로 재사용  
  - 파라미터/재현성 관리가 수월(실험 관리 쉬움)

---

## 3) Phase 전체 구조(최종까지)
- Phase 0: 비디오 입력 계약 + 좌표/스케일 앵커 고정  
- Phase 1: 3DGS → Surface(implicit)  ← 현재  
- Phase 2: Surface → 구조 프리미티브(벽/바닥/천장) + 경계(polyline)  
- Phase 3: 프리미티브 → BIM 객체화(IFC 우선, 이후 Revit 연계)  
- Phase 4: 문/창(opening) + 룸/레벨 관계(LOD300 강화)  
- Phase 5: LOD350~400 + MEP(난이도 급상승)  
- Phase 6: 동적 업데이트 + 로봇/캐릭터(충돌/내비 + semantics 질의)

---

# Phase 0 — 데이터 계약 + 스케일/좌표계 고정 (필수)
## 목표
- 비디오/프레임이 들어오면 “항상 동일한 월드 프레임(미터)”에서 결과가 재현되도록 한다.

## 입력
- video.mp4 또는 frames/
- (가능하면) AR depth/SLAM, LiDAR, 또는 기준 길이(스케일 바/문 폭 등)

## 출력
- intrinsics K
- extrinsics (R,t) per-frame
- world frame 정의(축, 단위)

## DoD
- 동일 입력으로 2~3회 반복 시 스케일/좌표계가 크게 흔들리지 않음
- 최소 기준 길이 오차가 허용 범위 내(프로젝트 목표에 맞게 기준 설정)

## 리스크/완화
- 텍스처 없는 벽/유리/반사 → 촬영 프로토콜(코너/가구/텍스처 포함), 노출 고정, 프레임 선별

---

# Phase 1 — 3DGS → Surface (Implicit Only)  ✅ (현재 구현 단계)
## 1) 입력/출력(데이터 계약)
### 입력
- video.mp4 또는 frames/
- 카메라 파라미터: intrinsics K, extrinsics (R,t) per-frame  
  - 기본 루트: frames → COLMAP으로 K, (R,t) 추정  
  - (선택) AR depth/SLAM/LiDAR가 있으면 스케일/좌표계 안정화

### 출력
- tsdf/ : TSDF volume 저장(형식은 구현 선택)
- surface_points.ply : 디버깅/검증용 surface point cloud
- metadata.json : 재현용 파라미터 기록(voxel size, truncation, bounds, view sampling, unit)

## 2) 파이프라인(구현 순서)
### Step A) Video → Frames
- 프레임 샘플링 규칙을 코드에 고정(예: fps 또는 이동거리 기반)
- 모션블러/노출변화 최소화 권장

### Step B) Frames → Camera poses (COLMAP 또는 SLAM)
- 산출: 카메라 포즈 + 내부파라미터
- 실패 시(텍스처 부족/유리/반사): 촬영 프로토콜 개선(코너/가구/텍스처 포함), 프레임 선별

### Step C) 3DGS 학습/재구성
- 입력: frames + poses/K
- 산출: gaussians(3DGS 모델)

### Step D) 3DGS → Depth/Normal 렌더 (다중 뷰)
- 학습 카메라 포즈 중 N개 선택(예: 200~800)하여 depth/normal 렌더
- 뷰 샘플링은 “공간 커버리지”가 중요(한 구간에 몰리지 않게)

### Step E) Depth fusion → TSDF volume 통합
- 각 depth + pose를 TSDF에 integrate
- 초기 파라미터 가이드(실내):
  - voxel size: 0.5~2.0 cm에서 탐색
  - truncation: 3~10 voxels
  - bounds: 카메라/포인트 바운딩 박스에서 padding
- 산출:
  - TSDF volume
  - surface_points.ply (0-level 주변 샘플링)

### Step F) 자동 QA(품질 체크)
- hole proxy: 표면 포인트 밀도/빈 영역 비율
- thickness proxy: 벽이 과도하게 두껍게 “이중” 생성되는지
- plane proxy: 바닥/벽 후보에서 평면 피팅 residual

## 3) Definition of Done(Phase 1)
- TSDF(surface)가 생성되고, surface_points.ply로 큰 구조(벽/바닥/천장)가 끊기지 않게 보인다.
- 같은 입력으로 재실행 시 결과가 크게 요동하지 않는다.
- 파라미터/뷰샘플링/스케일 기준이 metadata로 기록되어 재현 가능하다.

---

# Phase 2 — Surface → 구조 프리미티브(벽/바닥/천장) + 경계(polyline)
## 목표
- surface_points에서 plane extraction을 수행해 wall/floor/ceiling을 분해하고, 각 plane의 경계 폴리라인을 만든다.

## 입력/출력
- 입력: surface_points.ply (+ metadata.json)
- 출력:
  - planes.json (각 plane의 normal, d, inlier points, label)
  - boundaries.json (각 plane 경계 polyline)
  - debug visualization(선택)

## DoD
- wall/floor/ceiling 3종 분리가 안정적(최소)
- 주요 경계가 폴리라인으로 생성됨

---

# Phase 3 — 프리미티브 → BIM 객체화(IFC 우선)
## 목표
- plane/boundary/level 정보를 IfcWall/IfcSlab 등으로 변환하고 IFC를 생성한다.

## 입력/출력
- 입력: planes.json + boundaries.json + level info(추정/입력)
- 출력: model.ifc

## DoD
- IFC 뷰어에서 wall/slab가 정상 로드
- 레벨 그룹화(최소)

---

# Phase 4 — 문/창(opening) + 룸/레벨 관계(LOD300 강화)
## 목표
- door/window opening을 추출해 IfcDoor/IfcWindow 생성(호스팅 관계 포함)
- room(IfcSpace) 일부라도 복원(가능한 수준)

---

# Phase 5 — LOD350~400 + MEP (확장 단계)
- 난이도 급상승: 기하보다 시맨틱/규칙/템플릿/학습 데이터가 병목

---

# Phase 6 — 동적 업데이트 + 로봇/캐릭터
- 런타임 권장 레이어:
  - visual: 3DGS
  - interaction: TSDF/SDF(거리/충돌/내비)
  - semantic: BIM(IFC 규칙/관계)

---

## GitHub 레포 / 라이브러리 (URL만)
### 3DGS (학습/기본)
- https://github.com/graphdeco-inria/gaussian-splatting
- https://github.com/nerfstudio-project/gsplat
- https://github.com/graphdeco-inria/diff-gaussian-rasterization

### 3DGS에서 depth/normal(geometry 신호) 뽑기
- https://github.com/HKUST-SAIL/RaDe-GS
- https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting

### 3DGS ↔ SDF 결합(옵션: surface 학습적 강화)
- https://github.com/city-super/GSDF

### TSDF/ESDF fusion (Surface(implicit) 만들기 핵심)
- https://github.com/isl-org/Open3D
- https://github.com/ethz-asl/voxblox
- https://github.com/nvidia-isaac/nvblox

### 영상이면 거의 필수: 포즈(SfM/MVS)
- https://github.com/colmap/colmap

### 전체 논문/레포 모음(탐색용)
- https://github.com/MrNeRF/awesome-3D-gaussian-splatting

---

## Claude Code 작업 티켓(권장 분해)
- T0: video → frames 추출 모듈(샘플링 규칙 포함)
- T1: COLMAP 실행/검증 스크립트(poses/K 산출, 실패 진단 로그)
- T2: 3DGS 학습 자동 실행(gaussian-splatting 또는 gsplat)
- T3: 3DGS depth/normal 렌더 모듈(뷰 샘플링 포함)
- T4: TSDF integration(Open3D로 MVP) + surface_points.ply 출력
- T5: Phase1 QA(holes/thickness/plane residual) 자동 리포트 생성
- T6(다음): plane extraction + boundary polyline(Phase 2) MVP

---

## 현재 결론(방향 고정)
- “3DGS → mesh”가 필수는 아니다.
- BIM으로 빠르게 가려면 **3DGS → TSDF(Surface) → plane/boundary → IFC**가 더 직행이다.
- Mesh는 나중에 엔진/자산/곡면이 필요해질 때 TSDF에서 선택적으로 export하면 된다.
