# 03. End-to-End 검증 결과

## 1. 테스트 데이터셋 개요

총 9개 데이터셋 + 1개 합성 데이터로 파이프라인 검증 수행.

| 데이터셋 | 유형 | 출처 | 파이프라인 |
|----------|------|------|-----------|
| jh | 실내 | Custom scan (한국 주거공간) | Import (s00→s06→s06b→s07→s08) |
| drjohnson | 실내 | Deep Blending | Import |
| playroom | 실내 | Deep Blending | Import |
| room | 실내 | Mip-NeRF 360 | Import |
| bonsai | 실내 (탁상) | Mip-NeRF 360 | Import |
| train | 실외 | Tanks & Temples | Import |
| bicycle | 실외 (비건물) | Mip-NeRF 360 | Import |
| kitchen | 실내 | Mip-NeRF 360 | Import |
| counter | 실내 (탁상) | Mip-NeRF 360 | Import |
| synthetic_L | 합성 | 자체 생성 (L자형 건물) | Import |

## 2. 파이프라인 실행 결과

### 건축물 데이터셋 (Good Results)

| 데이터셋 | 평면 | 벽 | 바닥 | 천장 | 공간 | 메시 | 품질 |
|----------|------|-----|------|------|------|------|------|
| jh | 11 | 4 | 1 | 1 | 1 | 8 | **Best** |
| drjohnson | 6 | 4 | 1 | 1 | 1 | 8 | **Good** |
| playroom | 7 | 4 | 1 | 1 | 1 | 8 | **Good** |
| room | 6 | 4 | 1 | 1 | 1 | 8 | Good |
| train | 7 | 4 | 1 | 1 | 1 | 6 | OK (외부) |

### 비건축물 / 불량 결과

| 데이터셋 | 평면 | 벽 | 바닥 | 천장 | 비고 |
|----------|------|-----|------|------|------|
| bonsai | 5 | 2 | 1 | 0 | 탁상 오브젝트, 건물 아님 |
| kitchen | 3 | 1 | 1 | 0 | 평면 부족 |
| counter | 4 | 0 | 0 | 0 | 탁상 오브젝트, 건물 아님 |
| bicycle | 3 | 2 | 0 | 0 | 야외 장면, 건물 아님 |

### 합성 데이터 (Synthetic L-Shaped Building)

| 항목 | 기대값 | 실측값 | 정확도 |
|------|--------|--------|--------|
| 벽 | 6 | 6 | 100% |
| 바닥 | 1~3 | 3 | OK (층별) |
| 천장 | 1~3 | 3 | OK (층별) |
| 공간 | 1~3 | 3 | OK |
| 전체 형상 | L자형 | L자형 | OK |

## 3. 좌표 정렬 검증

### 방법론
3DGS 원본 점군과 생성된 BIM 모델의 좌표가 정확히 일치하는지 3D isometric overlay로 검증.

### 검증 과정
1. 점군 좌표 변환: `COLMAP → Manhattan → IFC`
   ```
   pts_manhattan = pts_colmap @ R.T    (R = manhattan_rotation)
   pts_ifc = (mx/scale, mz/scale, my/scale)
   ```
2. IFC 파일에서 BIM geometry를 wireframe으로 추출
3. 두 데이터를 동일 3D 좌표에 오버레이하여 시각적으로 확인

### 결과
- **jh 데이터셋**: 80K 점 + BIM wireframe 오버레이 → 점군이 BIM 벽/바닥/천장 내부에 정확히 위치
- BIM bounding box와 점군 범위가 일치
- 벽 wireframe이 점군의 벽면 위치에 정렬됨

### 검증 스크립트
- `scripts/compare_3dgs_vs_bim.py`: 전체 데이터셋 비교 (4패널 isometric)
- `scripts/_jh_closeup.py`: JH 단일 클로즈업 (80K pts, 16x14 figure)

## 4. 시각화 검증

### 비교 Figure 생성 결과

| 시각화 | 스크립트 | 출력 |
|--------|----------|------|
| 6개 데이터셋 isometric 비교 | `visualize_all_runs.py` | `runs_comparison_isometric.png` |
| 6개 데이터셋 top-down 비교 | `visualize_all_runs.py` | `runs_comparison_topdown.png` |
| 6개 데이터셋 front elevation | `visualize_all_runs.py` | `runs_comparison_front.png` |
| 3DGS+BIM overlay (전체) | `compare_3dgs_vs_bim.py` | `3dgs_vs_bim_comparison.png` |
| JH closeup overlay | `_jh_closeup.py` | `jh_overlay_closeup.png` |
| L자형 합성 건물 | `test_polygon_building.py` | `synthetic_L_building.png` |

### BIM 색상 코드

| 색상 | 요소 |
|------|------|
| 파랑 (0.2, 0.4, 1.0) | 감지된 벽 (Detected Wall) |
| 주황 (1.0, 0.3, 0.1) | 합성 벽 (Synthetic Wall) |
| 초록 (0.1, 0.7, 0.1) | 바닥 슬래브 (Floor) |
| 노랑 (0.9, 0.7, 0.0) | 천장 슬래브 (Ceiling) |
| 회색 (0.5, 0.5, 0.5) | 공간 (IfcSpace) |

## 5. 단위 테스트

총 351개 테스트 통과:

| 테스트 모듈 | 테스트 수 | 범위 |
|------------|----------|------|
| test_s06b | 88 | 평면 정규화 전체 (7 서브모듈) |
| test_s07_ifc_export | 22 | IFC 빌더 핵심 기능 |
| test_s07_wall_builder | 16 | 벽 프로파일, polyline 생성 |
| test_s07_element_builders | 18 | 슬래브/공간/개구부 빌더 |
| test_s06c | 28 | 외부 건물 추출 (6 서브모듈) |
| test_s06d | 26 | 메시 세그먼테이션 |
| test_s08 | 15 | GLB/USD 메시 내보내기 |
| test_opening_detection | 19 | 문/창 감지 알고리즘 |
| test_geometry | 6 | 기하학 유틸리티 |
| test_download | ~50+ | 샘플 데이터 다운로드 |

## 6. 성능 지표

### 파이프라인 실행 시간 (Import 파이프라인, s00→s08)

| 데이터셋 | 점군 크기 | 실행 시간 | 비고 |
|----------|----------|----------|------|
| jh | ~400K pts | ~15s | 가장 많은 평면 감지 |
| drjohnson | ~1.6M pts | ~25s | Deep Blending |
| playroom | ~900K pts | ~20s | Deep Blending |
| room | ~1.5M pts | ~25s | Mip-NeRF 360 |
| synthetic_L | ~100K pts | ~5s | 합성 데이터 |

(측정 환경: Windows 11, i7, 32GB RAM, RTX 3060)

## 7. 한계점 및 알려진 이슈

### 데이터 품질 의존성
- **좋은 결과**: 충분한 벽/바닥/천장 평면이 있는 건축물 실내 장면
- **나쁜 결과**: 탁상 오브젝트(bonsai, counter), 야외 비건물(bicycle)
- 3DGS 학습 품질이 낮은 경우 (넓은 텍스처 없는 벽면) → PlanarGS 사용 권장

### 스케일 추정
- 3DGS 점군은 임의 스케일 → storey height 가정(2.7m)으로 추정
- 실세계 스케일 정보(GPS, IMU, known dimension)가 있으면 더 정확

### Non-Manhattan 지원
- `normal_mode: "cluster"`로 비직교 벽 지원하나, 완전 곡면 벽은 미지원
- 합성 L자형 건물로 기본 비직교 처리 검증 완료

### 개구부 감지
- 점군 밀도에 의존 → 3DGS 점군은 LiDAR보다 밀도가 낮아 정확도 제한
- 기본 비활성화 상태 (`enable_opening_detection: false`)
