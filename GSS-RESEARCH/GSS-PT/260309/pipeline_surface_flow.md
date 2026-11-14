# Gaussian Splatting → Surface → BIM: 어떻게 가능한가

## 핵심 질문
> 3D Gaussian Splatting PLY 파일에서 어떻게 건물의 벽, 바닥, 천장을 자동으로 찾아내는가?

---

## Step 1: Gaussian Splatting 학습 — 왜 Gaussian이 표면 위에 모이는가

### 학습 과정
```
입력: 사진 100장 + 카메라 위치 (SfM 결과)

반복 (수천 회):
  1. 현재 Gaussian들을 카메라 시점에서 렌더링 → 합성 이미지
  2. 합성 이미지와 실제 사진 비교 → 손실(Loss) 계산
     Loss = (1-λ)·L₁ + λ·L_SSIM
  3. 경사 하강법으로 각 Gaussian의 파라미터 조정:
     - 위치(μ) → 실제 표면 쪽으로 이동
     - 크기/방향(Σ) → 표면에 맞게 납작해짐
     - 불투명도(α) → 표면에 있으면 진해지고, 허공이면 투명해짐
     - 색상(c) → 실제 색과 일치하도록
  4. Adaptive Density Control:
     - 큰 Gaussian → 분할 (세밀한 표면 표현)
     - 작은 Gaussian → 복제 (빈 영역 채움)
     - 투명한 Gaussian → 제거 (허공의 불필요한 것 삭제)
```

### 왜 표면에 모이는가?
```
사진에 보이는 것 = 물체의 표면 (벽, 바닥, 천장, 가구)
사진에 안 보이는 것 = 허공, 물체 내부

→ 사진과 일치하도록 학습하면,
  Gaussian들은 자연스럽게 "사진에 보이는 표면" 위에 밀집하게 된다.
  허공에 있는 Gaussian은 렌더링에 기여하지 못하므로 투명해지고 삭제된다.
```

### 3DGS vs 2DGS: 표면 밀착도 차이
```
3DGS (Kerbl, SIGGRAPH 2023):
  - Gaussian = 3D 타원체 (부피가 있음)
  - 표면 근처에 모이지만, 표면을 약간 관통하거나 떠 있을 수 있음
  - 결과: 표면이 약간 뭉개짐 (noisy surface)

2DGS (Huang, SIGGRAPH 2024):
  - Gaussian = 2D flat disk (두께 = 0, 완전히 납작)
  - 표면에 딱 붙을 수밖에 없음 (디스크가 표면과 정렬)
  - 결과: 깔끔한 surface reconstruction
  - GSS는 이 모드를 사용 (gsplat 2DGS mode)
```

---

## Step 2: PLY 파일 — Gaussian들의 집합

### PLY 파일 구조
학습이 끝나면 PLY 파일이 출력된다. 각 Gaussian은 다음 정보를 가진다:

```
Gaussian #1: 위치(x=3.2, y=-1.1, z=5.7), 크기, 방향, 색상, 불투명도
Gaussian #2: 위치(x=3.2, y=-1.1, z=5.8), 크기, 방향, 색상, 불투명도
Gaussian #3: 위치(x=3.3, y=-1.1, z=5.7), 크기, 방향, 색상, 불투명도
...
(수십만 ~ 수백만 개)
```

### 핵심: 위치(μ)만 뽑으면 = Surface Point Cloud
```python
# 실제 GSS 코드 (s00_import_ply)
plydata = PlyData.read("model.ply")
x = plydata["vertex"]["x"]  # Gaussian 중심 x좌표
y = plydata["vertex"]["y"]  # Gaussian 중심 y좌표
z = plydata["vertex"]["z"]  # Gaussian 중심 z좌표

points = np.stack([x, y, z], axis=1)  # shape: (N, 3)
# → N개의 3D 점 = 표면 위의 점들
```

이 점들이 왜 surface인가?
- 학습 과정에서 Gaussian들이 **실제 표면 위로 이동**했기 때문
- 벽에 수만 개, 바닥에 수만 개, 천장에 수만 개가 밀집해 있음
- 허공에는 점이 거의 없음 (학습 중 삭제됨)

### 실제 데이터 예시 (jh 데이터셋)
```
총 Gaussian 수: 약 80,000개
분포:
  - 벽 표면: ~30,000개 (4개 벽에 분산)
  - 바닥 표면: ~15,000개
  - 천장 표면: ~15,000개
  - 가구/기타 표면: ~20,000개
```

---

## Step 3: RANSAC — Surface Points에서 평면 찾기

### 알고리즘
```
입력: N개의 3D 점 (= surface point cloud)

Sequential RANSAC:
  반복:
    1. 무작위 3점 선택 → 평면 방정식 결정: ax + by + cz + d = 0
    2. 모든 점에서 이 평면까지의 거리 계산
    3. 거리 < ε(threshold)인 점 = inlier
    4. inlier 수가 최대인 평면을 선택
    5. 선택된 평면의 inlier 점들을 제거
    6. 남은 점에서 다시 1번부터 반복

  종료 조건: inlier 수 < min_points (1,000개)
```

### 왜 벽/바닥/천장이 찾아지는가?
```
Surface points 분포:
  - 벽1 표면에 ~8,000개 밀집 (같은 평면 위)
  - 벽2 표면에 ~7,000개 밀집 (같은 평면 위)
  - 바닥 표면에 ~15,000개 밀집 (같은 평면 위)
  - 천장 표면에 ~15,000개 밀집 (같은 평면 위)

RANSAC 1회차: 바닥(15,000개 inlier) → 추출, 제거
RANSAC 2회차: 천장(15,000개 inlier) → 추출, 제거
RANSAC 3회차: 벽1(8,000개 inlier) → 추출, 제거
RANSAC 4회차: 벽2(7,000개 inlier) → 추출, 제거
...
```

### 벽 vs 바닥 분류
```
평면의 법선 벡터 n = (nx, ny, nz)

|ny| > cos(θ)  →  수평면 (바닥 or 천장)
                    ny > 0: 천장
                    ny < 0: 바닥
else            →  수직면 (벽)

※ θ = 허용 각도 (기본 15°)
```

---

## Step 4: 이후 파이프라인 (Surface Planes → BIM)

```
추출된 평면들 (벽4 + 바닥1 + 천장1)
    ↓
Alpha Shape: 각 평면의 경계선(boundary polygon) 추출
    ↓
Manhattan World: 법선을 직교 축에 정렬 (약간 기울어진 벽 → 직각)
    ↓
기하 정규화 6단계:
  1) Normal Snap (법선 직교화)
  2) Height Clustering (바닥/천장 높이 통일)
  3) Wall Thickness (벽 두께 + center-line 추출)
  4) Wall Closure (누락 벽 합성)
  5) Intersection Trim (모서리 스냅)
  6) Space Detection (방 경계 인식)
    ↓
IFC BIM: IfcWall + IfcSlab + IfcSpace (건축 표준 파일)
    ↓
GLB/USD: 디지털 트윈용 3D 메시 파일
```

---

## 전체 한 줄 요약

```
카메라 사진 → 2DGS 학습 → Gaussian들이 표면에 밀착
→ PLY에서 Gaussian 위치 추출 = Surface Point Cloud
→ RANSAC으로 벽/바닥/천장 평면 찾기
→ 정규화 → IFC BIM → GLB/USD 디지털 트윈
```

**Gaussian Splatting이 만든 점들이 이미 surface 위에 있기 때문에,
별도의 surface reconstruction 없이도 바로 건축 평면을 추출할 수 있다.**

---

## 왜 Mesh가 아닌 Surface Point Cloud 기반인가

### 문제: Mesh는 "더러운" 중간 단계다

Gaussian Splatting 결과에서 BIM을 만들려면 **벽/바닥/천장의 평면**을 찾아야 한다.
기존 방법은 먼저 삼각형 mesh를 만들고, 거기서 평면을 추출한다.
하지만 mesh를 만드는 과정 자체가 정보를 오염시킨다.

```
Mesh 생성 = 연속적인 표면을 삼각형으로 강제 분할하는 과정

문제 1: Voxel 양자화
  - TSDF는 공간을 격자(voxel)로 나눔
  - 격자 해상도에 따라 표면 위치가 반올림됨
  - 원래 Gaussian이 있던 정확한 위치 정보가 사라짐

문제 2: Marching Cubes 근사
  - voxel 경계에서 삼각형을 생성하는 알고리즘
  - 실제 표면이 아닌 "voxel 격자에 맞춘 근사 표면"을 만듦
  - 벽이 살짝 울퉁불퉁해짐 (staircase artifact)

문제 3: Over-tessellation
  - 벽 하나 = 수학적으로 평면 1개
  - 그런데 mesh로 만들면 = 삼각형 수천 개
  - 이 수천 개의 삼각형에서 다시 "이게 하나의 평면이었다"를 역추적해야 함
  - 삼각형 normal이 제각각 → 평면 피팅 시 노이즈 발생

문제 4: 불필요한 연산
  - Depth rendering (GPU, view마다 반복)
  - TSDF integration (수 GB 메모리)
  - Marching Cubes (CPU 집약)
  - 이 모든 과정이 BIM에 필요한 "평면 방정식"과 무관
```

### 해결: Gaussian 중심 = 이미 Surface 위의 점

```
2DGS 학습이 끝난 Gaussian들은 이미 건물 표면 위에 있다.
PLY에서 중심 좌표(x,y,z)만 읽으면 = Surface Point Cloud

→ mesh를 만들 필요 없이 바로 RANSAC으로 평면 추출 가능
→ 중간 단계의 오염(양자화, 근사, 삼각형 분할) 없이
   학습이 최적화한 원본 위치에서 직접 평면을 찾는다
```

### Mesh 경로 vs Surface 경로 비교

```
+------------------+-------------------------------+-----------------------------+
|                  | Mesh 경로                     | Surface 경로 (GSS)          |
+------------------+-------------------------------+-----------------------------+
| 파이프라인       | Gaussian → Depth → TSDF       | Gaussian center xyz         |
|                  | → Marching Cubes → Mesh       | → 바로 RANSAC               |
|                  | → mesh에서 평면 피팅          |                             |
+------------------+-------------------------------+-----------------------------+
| 단계 수          | 5단계                         | 2단계                       |
+------------------+-------------------------------+-----------------------------+
| 벽 1개의 표현    | 삼각형 수천 개                | 점 수천 개 (같은 평면 위)   |
|                  | (normal 제각각, 노이즈)       | (RANSAC이 직접 피팅)        |
+------------------+-------------------------------+-----------------------------+
| 정보 손실        | voxel 양자화 (위치 반올림)    | 없음                        |
|                  | + Marching Cubes 근사 오차    | (학습이 최적화한 위치 그대로)|
|                  | + 삼각형 분할 노이즈          |                             |
+------------------+-------------------------------+-----------------------------+
| 평면 추출        | mesh face → region growing    | point cloud → RANSAC        |
|                  | 또는 mesh → point sampling    | (통계적으로 robust,         |
|                  | → 다시 RANSAC (본말전도)      |  outlier에 강함)            |
+------------------+-------------------------------+-----------------------------+
| 처리 시간        | 수십 분 (GPU 필요)            | 수 초 (CPU만)               |
+------------------+-------------------------------+-----------------------------+
| 메모리           | TSDF voxel grid               | Point array만               |
|                  | (512³ = 134M voxels, 수 GB)   | (80K × 3 float = ~2 MB)    |
+------------------+-------------------------------+-----------------------------+
| 외부 의존성      | GPU, CUDA, 대용량 RAM         | numpy, open3d만             |
+------------------+-------------------------------+-----------------------------+
| 기존 PLY 호환    | PLY마다 재처리 필요           | 어떤 3DGS PLY든 즉시 사용   |
+------------------+-------------------------------+-----------------------------+
```

### 핵심 한 줄

```
BIM에 필요한 것 = 평면 방정식 (ax + by + cz + d = 0)
Mesh가 주는 것 = 삼각형 수만 개 (평면을 다시 역추적해야 함)
Surface points가 주는 것 = 같은 평면 위의 점 수천 개 (바로 RANSAC 피팅)

→ Mesh는 평면을 삼각형으로 쪼갠 뒤 다시 합치는 "돌아가는 길"이다.
   Surface points는 평면 위의 점에서 바로 평면을 찾는 "직선 경로"다.
```

### 비유: 종이 위의 점 vs 퍼즐 조각

```
Surface 경로 (GSS):
  벽 위에 점 8,000개가 찍혀 있다.
  → "이 점들이 하나의 평면 위에 있다" = RANSAC 한 번이면 끝

Mesh 경로:
  같은 벽을 삼각형 3,000개로 잘게 쪼갠다.
  → 각 삼각형의 방향이 약간씩 다르다 (양자화 + 근사 오차)
  → "이 삼각형들이 원래 하나의 평면이었다"를 역추적해야 한다
  → 쪼개놓고 다시 맞추는 셈 (불필요한 작업)
```

### GSS 파이프라인에서의 위치

```
기존 경로 (Mesh 포함):
  s03(2DGS) → s04(Depth) → s05(TSDF Mesh) → s06(RANSAC) → ...
                 ↑ 이 구간이 mesh를 만드는 "더러운" 중간 단계
                   GPU 필요, 수십 분 소요, 정보 손실 발생

Import 경로 (Surface 직접, GSS 실험에서 사용):
  s00(PLY import) → s06(RANSAC) → s06b(정규화) → s07(IFC) → s08(GLB/USD)
  ↑ mesh 단계 생략, CPU만으로 수 초, 정보 손실 없음

두 경로 모두 s06 이후는 동일 → BIM 결과도 동일
→ Mesh를 거치든 안 거치든 최종 BIM은 같다
→ 그러면 왜 굳이 mesh를 거치나? → 안 거치는 게 낫다
```

### Mesh는 언제 필요한가?

```
Mesh가 필요한 경우 (BIM 추출이 아닌 다른 용도):
  - 3D 시각화/렌더링 (게임엔진, Three.js)
  - 물리 시뮬레이션 (충돌 감지, 로봇 내비게이션)
  - 3D 프린팅

GSS에서 mesh가 쓰이는 곳:
  - s08 단계: IFC BIM → GLB/USD 변환 시
  - 이때의 mesh는 BIM geometry에서 생성하는 것
    (벽 = 직사각형 → 삼각형 2개, 깔끔한 BIM mesh)
  - 3DGS에서 만든 "더러운" mesh가 아님!

즉: 3DGS mesh (수십만 삼각형, 노이즈) ≠ BIM mesh (건축 요소, 깔끔)
```

---

## References
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance Fields", SIGGRAPH 2024
- Fischler & Bolles, "Random Sample Consensus: A Paradigm for Model Fitting", CACM 1981
