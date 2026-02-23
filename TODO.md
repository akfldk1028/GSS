# GSS TODO

## 높은 우선순위

### s06 성능 최적화
- [ ] **Voxel downsampling**: RANSAC 전에 21M → 2M 포인트 다운샘플 (101s → ~30s 예상)
  - Open3D `voxel_down_sample(voxel_size=0.1)` 적용
  - RANSAC inlier index를 원본 포인트에 매핑하는 로직 필요
- [ ] **Normal 추정 가속**: Manhattan alignment의 normal 추정이 ~60s 소요
  - 다운샘플된 cloud에서 normal 추정 후 적용

### s07 IFC Export 재실행
- [ ] s06 개선 결과 (3 walls + 1 floor + 2 ceilings) 기반으로 IFC 재생성
- [ ] IfcLocalPlacement 설정 (현재 미설정)
- [ ] 다층 건물 지원 (현재 단일 층만)

### 시각화 업데이트
- [ ] `docs/images/` 개선 후 시각화 이미지 업데이트
  - plane detection 결과 (3D, top-down, front view)
  - before/after 비교

## 중간 우선순위

### s06 알고리즘 개선
- [ ] **Ceiling 분류**: TSDF mesh 양면 문제 — 천장 외면이 upward-normal로 floor 처리될 수 있음
- [ ] **Manhattan axis snapping**: wall normal을 최근접 Manhattan 축에 snap하여 더 정확한 grouping
- [ ] **Connected component analysis**: 같은 plane이라도 공간적으로 떨어진 영역은 분리
- [ ] **COLMAP 스케일 정규화**: COLMAP 단위 → 미터 변환 (기준 거리 측정 기반)

### 기존 파이프라인 (gsplat) 검증
- [ ] s03(gsplat) → s04 → s05 → s06 → s07 E2E 테스트
- [ ] gsplat 2DGS vs PlanarGS 품질 비교

### 테스트 보강
- [ ] s06 coplanar merge 단위 테스트 (다양한 기하 시나리오)
- [ ] s06 Manhattan alignment 단위 테스트
- [ ] s06 position-based classification 테스트 (edge cases)
- [ ] Integration test: s03_planargs → s06 → s07 전체 파이프라인

## 낮은 우선순위

### BIM 품질
- [ ] Opening 검출 (문/창문) — LOD 300
- [ ] Wall thickness 추정 (현재 고정값 0.2m)
- [ ] 가구 인식 및 IfcFurnishingElement 생성
- [ ] 좌표계 원점을 건물 중심으로 이동

### 성능/확장
- [ ] GPU-accelerated RANSAC (cuML)
- [ ] 대형 point cloud 스트리밍 처리
- [ ] 다중 방(room) 분할
- [ ] 층(storey) 자동 분할

### 문서화
- [ ] `docs/3dgs_to_bim_master_plan.md` 업데이트 (현재 상태 반영)
- [ ] API 문서 (각 step의 config 옵션 설명)
- [ ] 촬영 프로토콜 가이드 (최적 비디오 캡처 방법)
