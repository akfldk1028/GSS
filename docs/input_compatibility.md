# GSS 입력 데이터 호환성 가이드

## 전체 데이터 흐름

```
                         기존 파이프라인 (pipeline.yaml)
                    ┌──────────────────────────────────────────┐
                    │                                          │
Video ─→ s01 Frames ─→ s02 COLMAP ─→ s03 gsplat(2DGS) ─→ s04 Depth ─→ s05 TSDF
                    │                                               │
                    │                                    surface_points.ply
                    │                                               │
                    │          PlanarGS 파이프라인                    │
                    │          (pipeline_planargs.yaml)             │
                    │   s03_planargs ─────────────────────────┐     │
                    │                              surface_points.ply
                    │                                          │    │
                    │                                          ▼    ▼
                    │                                    ┌──────────────┐
                    │                                    │ s06 Plane    │
                    │                                    │ Extraction   │
                    │                                    │ (RANSAC)     │
                    │                                    └──────┬───────┘
                    │                                           │
                    │                              planes.json + boundaries.json
                    │                              manhattan_alignment.json
                    │                                           │
                    │                                    ┌──────┴───────┐
                    │                                    │ s06b Plane   │
                    │                                    │ Regulariz.   │
                    │                                    └──────┬───────┘
                    │                                           │
                    │                              planes.json + walls.json
                    │                              spaces.json + stats.json
                    │                                           │
                    │                                    ┌──────┴───────┐
                    │                                    │ s07 IFC      │
                    │                                    │ Export       │
                    │                                    └──────────────┘
                    │                                           │
                    │                                      output.ifc
                    └──────────────────────────────────────────┘
```

---

## 각 단계별 입력 요구사항

### s06 Plane Extraction — 핵심 입력 지점

s06은 파이프라인에서 **3DGS 결과물을 받는 진입점**입니다.

| 입력 | 형식 | 설명 |
|------|------|------|
| `surface_points.ply` | PLY (Open3D 호환) | XYZ 포인트 클라우드. 법선(normals) 있으면 더 좋음 |
| `metadata.json` | JSON | TSDF 메타데이터 (voxel_size, bounds 등) |

**PLY 요구사항:**
- Open3D `o3d.io.read_point_cloud()`가 읽을 수 있는 모든 PLY
- 최소 필수: vertex XYZ 좌표
- 권장: vertex normals (없으면 s06에서 KDTree로 추정)
- 최소 포인트 수: `min_inliers` (기본 500) 이상, 실제로는 **10,000+ 권장**
- 좌표계: 제한 없음 (Manhattan alignment가 자동으로 축 정렬)

```
ply
format binary_little_endian 1.0
element vertex 1000000
property float x
property float y
property float z
property float nx        ← 권장 (없어도 동작)
property float ny
property float nz
end_header
```

### s06b Plane Regularization — s06 출력을 그대로 받음

| 입력 | 형식 | 설명 |
|------|------|------|
| `planes.json` | JSON | s06이 생성한 평면 리스트 |
| `boundaries.json` | JSON | s06이 생성한 경계 폴리라인 |
| `manhattan_alignment.json` | JSON (선택) | 3x3 회전 행렬. 없으면 원본 좌표에서 처리 |

**planes.json 스키마** (s06b가 기대하는 형식):
```json
[
  {
    "id": 0,
    "normal": [0.98, 0.05, 0.17],
    "d": -4.95,
    "label": "wall",           // "wall" | "floor" | "ceiling" | "other"
    "num_inliers": 1000,
    "boundary_3d": [[5,0,0], [5,3,0], [5,3,4], [5,0,4], [5,0,0]]
  }
]
```

---

## 3DGS 소스별 호환성

### 1. gsplat (2DGS) — 기존 파이프라인

```
s03 gsplat → model.ply (Gaussian params)
  → s04 Depth Render → depth maps (.npy)
    → s05 TSDF Fusion → surface_points.ply ✅
```

- **상태**: 완전 지원
- **좌표계**: COLMAP 스케일 (비미터). s06b auto-scale이 자동 보정
- **포인트 수**: TSDF에서 보통 수백만~수천만
- **config**: `configs/pipeline.yaml` 기본값 사용

### 2. PlanarGS (NeurIPS 2025) — PlanarGS 파이프라인

```
s03_planargs → TSDF fusion 내장 → surface_points.ply ✅
```

- **상태**: 완전 지원
- **좌표계**: COLMAP 스케일
- **포인트 수**: 보통 2천만+
- **config**: `configs/pipeline_planargs.yaml` + s06 config에서 `distance_threshold: 0.3`

### 3. 3D Gaussian Splatting (원본, Kerbl et al.)

```
외부 학습 → point_cloud.ply (Gaussian centers)
  → depth rendering 필요 → TSDF → surface_points.ply
```

- **상태**: 가능하지만 변환 필요
- **방법 A**: `point_cloud.ply`에서 Gaussian center 좌표 추출 → s06 직접 입력
- **방법 B**: depth render + TSDF (s04→s05) 경로 사용
- **주의**: 원본 3DGS의 `.ply`는 Gaussian 파라미터(SH, scale, rotation 등) 포함. s06에는 **XYZ 좌표만** 필요

**Gaussian PLY → 포인트 클라우드 변환:**
```python
import open3d as o3d
import numpy as np

# 원본 3DGS PLY 읽기 (plyfile 사용)
from plyfile import PlyData
ply = PlyData.read("point_cloud.ply")
xyz = np.column_stack([
    ply["vertex"]["x"],
    ply["vertex"]["y"],
    ply["vertex"]["z"],
])

# Open3D 포인트 클라우드로 변환
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals()
o3d.io.write_point_cloud("surface_points.ply", pcd)
```

### 4. Nerfstudio / splatfacto

```
exports/ → point_cloud.ply (Gaussian centers)
```

- **상태**: 변환 필요 (방법 3과 동일)
- **좌표계**: Nerfstudio 내부 좌표 → COLMAP 변환 필요할 수 있음
- **팁**: `ns-export gaussian-splat` 후 위 변환 스크립트 적용

### 5. 외부 포인트 클라우드 (LiDAR, photogrammetry 등)

```
scan.ply → surface_points.ply ✅ (직접 사용 가능)
```

- **상태**: 직접 사용 가능 (3DGS 아닌 소스도 OK)
- **좌표계**: 미터 단위면 `scale_mode: metric`, 아니면 `auto`
- **조건**: Open3D가 읽을 수 있는 PLY + 충분한 포인트 밀도
- **metadata.json**: 수동 생성 필요

```json
{
  "voxel_size": 0.01,
  "sdf_trunc": 0.04,
  "bounds_min": [-10, -10, -10],
  "bounds_max": [10, 10, 10]
}
```

---

## 좌표계와 스케일

| 소스 | 좌표계 | 스케일 | s06b 설정 |
|------|--------|--------|-----------|
| COLMAP (gsplat, PlanarGS) | 임의 | 비미터 (~7 units/m) | `scale_mode: auto` |
| LiDAR / 실측 | 보통 미터 | 1.0 | `scale_mode: metric` |
| Matterport3D | 미터 | 1.0 | `scale_mode: metric` |
| Replica (합성) | 미터 | 1.0 | `scale_mode: metric` |
| 알 수 없음 | ? | ? | `scale_mode: manual`, `coordinate_scale: N` |

**auto-scale 동작**: 모든 벽/바닥/천장의 bounding box에서 중간 차원을 `expected_room_size` (기본 5m)로 나눠서 추정.

---

## 건물 유형별 호환성

| 시나리오 | 동작 | 비고 |
|---------|------|------|
| 직교 방 (Manhattan 구조) | OK | 설계 의도대로 |
| 비직교 건물 (45도 회전된 벽) | DEGRADED | normal snap이 잘못된 축에 snap → `stats.json`에서 `skipped` 확인 |
| L형/T형 방 | DEGRADED | AABB가 직사각형만 생성, 비직사각형 방은 convex hull fallback |
| 복도 / 긴 방 | DEGRADED | auto-scale이 부정확할 수 있음 → `scale_mode: manual` 권장 |
| 바닥 없음 (exterior scan) | DEGRADED | wall closure가 wall AABB fallback 사용, space detection 결과 제한적 |
| 천장 없음 | OK | height snap이 1개 cluster만 생성, wall closure는 ceiling 기본값 사용 |
| 여러 방 (복수 공간) | PARTIAL | space detection이 가장 큰 공간만 잘 감지. 복도 연결부 등은 미지원 |

---

## 빠른 시작: 외부 데이터로 s06→s06b→s07 실행

### 1단계: 데이터 준비

```bash
# 디렉토리 구조 생성
mkdir -p data/interim/s05_tsdf_fusion

# 포인트 클라우드 복사 (이름 맞춰야 함)
cp your_scan.ply data/interim/s05_tsdf_fusion/surface_points.ply

# metadata.json 생성
cat > data/interim/s05_tsdf_fusion/metadata.json << 'EOF'
{
  "voxel_size": 0.01,
  "sdf_trunc": 0.04,
  "bounds_min": [-10, -10, -10],
  "bounds_max": [10, 10, 10]
}
EOF
```

### 2단계: s06 실행

```bash
gss run-step plane_extraction
# 결과: data/interim/s06_planes/planes.json, boundaries.json, manhattan_alignment.json
```

### 3단계: s06b 실행

```bash
gss run-step plane_regularization
# 결과: data/interim/s06b_plane_regularization/planes.json, walls.json, spaces.json, stats.json
```

### 4단계: 결과 확인

```bash
# stats.json으로 품질 진단
cat data/interim/s06b_plane_regularization/stats.json
```

```json
{
  "manhattan_aligned": true,
  "scale": 7.12,
  "normal_snapping": {"snapped_walls": 4, "snapped_horiz": 2, "skipped": 0},
  "height_snapping": {"floor_clusters": 1, "ceiling_clusters": 1},
  "wall_thickness": {"total_walls": 4, "paired": 2, "unpaired": 2},
  "wall_closure": {"synthesized": 0},
  "intersection_trimming": {"snapped_endpoints": 4, "extended_endpoints": 4},
  "space_detection": {"num_spaces": 1}
}
```

**품질 체크리스트:**
- `manhattan_aligned: false` → 결과 품질 저하 가능. 원본 좌표계가 축 정렬되지 않은 경우
- `normal_snapping.skipped` > 전체의 50% → 비직교 건물이거나 Manhattan 감지 실패
- `wall_thickness.unpaired` 많음 → 벽 두께 감지 실패, `max_wall_thickness` 조정 필요
- `wall_closure.synthesized` > 0 → 스캔에서 벽이 누락됨 (정상일 수 있음)
- `space_detection.num_spaces` == 0 → 방 감지 실패, 벽이 너무 적거나 닫히지 않음
