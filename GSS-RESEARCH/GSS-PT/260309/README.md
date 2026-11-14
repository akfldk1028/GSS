# GSS 프로젝트 발표 자료 (2026.03.09)

## 발표 흐름 & 폴더 구조

### 01_overview/ — 프로젝트 개요 (4장)
- **desktop_screenshot.png**: 개발 환경 전체 화면 (Claude Code + IDE + L자 건물 시각화)
- **desktop_jh.png**: JH 데이터셋 실행 결과 데스크톱
- **desktop_check.png / desktop_check2.png**: 검증 과정 스크린샷
> **설명**: GSS = 3D Gaussian Splatting to BIM 자동화 파이프라인. 동영상/카메라만으로 건물 BIM 모델 자동 생성 (LiDAR 불필요)

### 02_3dgs_input/ — 3DGS 입력 데이터 (2장)
- **planargs_surface.png**: PlanarGS 표면 재구성 결과 (21.4M 포인트, Replica room0)
- **gaussian_vs_planes_comparison.png**: 4개 데이터셋별 원본 가우시안 vs RANSAC 평면 추출 비교
> **설명**: Video → COLMAP(SfM) → 3D Gaussian Splatting → 포인트 클라우드. 2DGS(gsplat) 또는 PlanarGS 두 가지 경로 지원

### 03_plane_extraction/ — 평면 추출 (13장)
- **s00_s06_plane_boundaries.png**: 4개 데이터셋 RANSAC 평면 추출 결과 (바운더리 폴리곤 포함)
- **planes_isometric/topdown/front.png**: JH 데이터 평면 3D/평면/정면 뷰
- **planargs_planes_*.png**: PlanarGS 경로 평면 추출 결과
- **planes_view1~6.png**: 다양한 각도에서 본 평면 추출
- **Figure_1.png**: 추가 평면 분석
> **설명**: Open3D RANSAC + alpha shape로 벽/바닥/천장 평면 자동 감지. 법선 방향으로 Wall/Floor/Ceiling 분류

### 04_regularization/ — 기하 정규화 s06b (5장)
- **s06b_comparison.png**: s06 원본 vs s06b 정규화 비교 (Before/After + 센터라인)
- **s06b_isometric.png**: 정규화된 벽 3D 뷰
- **s06b_normals.png**: 벽 법선 방향 스냅 결과
- **s06b_rectfit.png**: 직사각형 피팅 결과
- **synthetic_L_building.png**: L자 건물 합성 테스트 (6벽, 3공간 완벽 감지)
> **설명**: Manhattan 축 정렬, 높이 클러스터링, 벽 두께 추정, 교차점 트리밍, 공간 감지, 기둥 감지 등 10개 서브모듈

### 05_ifc_bim/ — IFC/BIM 출력 (6장)
- **ifc_isometric/topdown/front.png**: IFC 지오메트리 3뷰 (ifcopenshell.geom 렌더)
- **bim_isometric/topdown/front.png**: BIM 건물 모델 3뷰
> **설명**: IfcWall + IfcSlab + IfcSpace + IfcOpeningElement → IFC4 표준 BIM 파일 출력. 산업 표준 호환

### 06_mesh_export/ — 메시 내보내기 s08 (12장)
- **mesh_GSS_BIM_glb_*.png**: GLB 포맷 (웹/게임엔진용)
- **mesh_GSS_BIM_usdc_*.png**: USD 포맷 Z-up
- **mesh_GSS_BIM_yup_usdc_*.png**: USD 포맷 Y-up (Omniverse/Isaac Sim)
- **mesh_GSS_room/train_glb_*.png**: 다른 데이터셋 GLB 결과
> **설명**: IFC → GLB(trimesh) + USDC(usd-core) + USDZ(Apple Vision Pro). 크로스 플랫폼 디지털 트윈 지원

### 07_validation/ — 3DGS↔BIM 정합성 검증 (2장)
- **3dgs_vs_bim_comparison.png**: 4개 데이터셋 3DGS 포인트 + BIM 와이어프레임 오버레이
- **jh_overlay_closeup.png**: JH 데이터 80K 포인트 + BIM 오버레이 (클로즈업)
> **설명**: 3DGS 포인트 클라우드가 BIM 박스 안에 정확히 위치함을 시각적으로 검증. 좌표 변환 정확성 입증

### 08_multi_dataset/ — 다중 데이터셋 비교 (3장)
- **runs_comparison_isometric.png**: 6개 데이터셋 BIM 결과 비교 (등각뷰 2x3 그리드)
- **runs_comparison_topdown.png**: 평면도 비교
- **runs_comparison_front.png**: 정면도 비교
> **설명**: jh, drjohnson, playroom, bonsai, room, train — 다양한 실내 씬에서 일관된 BIM 추출

### 09_advanced/ — 고급 기능: 건물 추출 s06c (5장)
- **s06c_dataset_comparison.png**: bonsai/bicycle 건물 추출 상세 비교
- **s06c_bicycle_detail/bonsai_detail.png**: 개별 데이터셋 상세
- **s06c_inria_all.png**: INRIA 데이터셋 전체 비교
- **s06c_your_room.png**: 실제 방 데이터 결과
> **설명**: 외부 건물 재구성 — 파사드 감지, 평면도 추출, 지붕 구조화, 층수 감지

### 10_demo_screenshots/ — 추가 데모 (5장)
- **desktop_screenshot2~6.png**: 다양한 실행/검증 과정 스크린샷
> **설명**: 실제 개발/실행 과정 데모

---

## 발표 핵심 포인트

### 1. 문제 정의
- 기존 Scan-to-BIM: LiDAR 장비 + 수동 모델링 (비용↑, 시간↑)
- GSS: **카메라만으로 자동 BIM** (Video → 3DGS → Plane → IFC → GLB/USD)

### 2. 파이프라인 아키텍처
```
Video → Frames(s01) → COLMAP/SfM(s02) → 3DGS(s03) → Depth(s04) → TSDF(s05)
  → Plane Extraction(s06) → Regularization(s06b) → IFC Export(s07) → Mesh Export(s08)
```
- 4가지 파이프라인 모드: 기존/PlanarGS/Exterior/Import

### 3. 핵심 기술
- **3D Gaussian Splatting**: 장면을 가우시안 원소로 표현 → 포인트 클라우드
- **RANSAC 평면 추출**: 벽/바닥/천장 자동 감지
- **Manhattan 정규화**: 축 정렬, 벽 두께, 교차점, 공간 감지
- **IFC 생성**: IfcWall/Slab/Space/Opening → 산업 표준 BIM
- **메시 내보내기**: GLB + USD + USDZ (크로스 플랫폼)

### 4. 검증 결과
- **9개 데이터셋** E2E 테스트 완료
- **365개 유닛 테스트** 통과
- **3DGS↔BIM 좌표 정합성** 시각적 검증 완료
- **L자 다각형 건물** 합성 데이터 검증

### 5. 타깃 플랫폼 & 향후 방향
- NVIDIA Omniverse / Isaac Sim (USD)
- Apple Vision Pro (USDZ)
- UE5 / Unity (GLB)
- 다음: semantic segmentation, material classification

## NotebookLM 사용법
```bash
# 로그인
notebooklm login

# 노트북 생성
notebooklm create "GSS - 3DGS to BIM Pipeline"
notebooklm use <notebook_id>

# 이 README + CLAUDE.md + memory 파일을 소스로 추가
notebooklm source add "./README.md"
notebooklm source add "../../CLAUDE.md"

# 슬라이드 생성
notebooklm generate slides "교수님 발표용, 한국어, GSS 파이프라인 설명"

# 오디오 요약 생성 (선택)
notebooklm generate audio "GSS 프로젝트 개요를 설명해주세요" --wait
```
