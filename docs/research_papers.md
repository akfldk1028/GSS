# GSS 프로젝트 — 논문 조사 결과

> 조사일: 2026-02-19
> 목적: 3DGS → BIM 파이프라인 기술 스택 검증 및 핵심 참고 논문 정리

---

## 1. 파이프라인 검증 결론

**현재 설계가 논문적으로 탄탄하며, 추가 패키지 없이 기존 스택으로 충분하다.**

```
Video → Frames (opencv-python)        ← pyproject.toml에 있음
  → COLMAP Poses (pycolmap + hloc)     ← 새로 설치 필요
  → 2DGS 학습 (gsplat + torch)         ← 새로 설치 필요
  → Depth/Normal 렌더 (gsplat 내장)     ← 추가 없음
  → TSDF 통합 (open3d)                  ← pyproject.toml에 있음
  → Plane 추출 (open3d RANSAC)          ← 추가 없음
  → Boundary (alphashape + shapely)     ← pyproject.toml에 있음
  → IFC (ifcopenshell)                  ← pyproject.toml에 있음
```

추가 설치 대상: `pycolmap`, `hloc`, `gsplat`, `torch` (4개)

---

## 2. 핵심 참고 논문 (14편)

### 2.1 3D Gaussian Splatting 기초

| # | arXiv | 제목 | 저자 | 게재 | 역할 |
|---|-------|------|------|------|------|
| 1 | [2308.04079](https://arxiv.org/abs/2308.04079) | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl et al. | SIGGRAPH 2023 | 3DGS 원본 논문 |

### 2.2 Surface Reconstruction (핵심)

| # | arXiv | 제목 | 저자 | 게재 | 역할 |
|---|-------|------|------|------|------|
| 2 | [2403.17888](https://arxiv.org/abs/2403.17888) | **2D Gaussian Splatting for Geometrically Accurate Radiance Fields** | Huang et al. | **SIGGRAPH 2024** | **핵심: 2DGS 원본. surfel 기반 surface-aware GS. DTU/TNT SOTA** |
| 3 | [2409.06765](https://arxiv.org/abs/2409.06765) | **gsplat: An Open-Source Library for Gaussian Splatting** | Ye et al. (Nerfstudio) | JMLR MLOSS | **핵심: 우리가 사용하는 라이브러리. 2DGS 모드 내장** |
| 4 | [2412.03428](https://arxiv.org/abs/2412.03428) | **2DGS-Room: Seed-Guided 2DGS with Geometric Constraints for Indoor Reconstruction** | Zhang et al. | 2024 | **실내 특화 2DGS. textureless 벽/바닥 대응** |
| 5 | [2503.06587](https://arxiv.org/abs/2503.06587) | Introducing Unbiased Depth into 2DGS for High-accuracy Surface Reconstruction | Yang et al. | Pacific Graphics 2025 | glossy surface depth bias 해결 |
| 6 | [2406.06521](https://arxiv.org/abs/2406.06521) | **PGSR: Planar-based Gaussian Splatting for Surface Reconstruction** | Chen et al. | **IEEE TVCG** | **평면 우선 정규화. 실내 벽면에 유리** |

### 2.3 대안 방법 (비교 참고용)

| # | arXiv | 제목 | 저자 | 역할 |
|---|-------|------|------|------|
| 7 | [2404.10772](https://arxiv.org/abs/2404.10772) | Gaussian Opacity Fields (GOF) | Yu et al. | opacity → Marching Cubes. 대안 비교용 |
| 8 | [2404.00409](https://arxiv.org/abs/2404.00409) | 3DGSR: Implicit SDF + 3DGS | Lyu et al. | SDF 통합 학습. 미래 확장 참고 |
| 9 | [2311.12775](https://arxiv.org/abs/2311.12775) | SuGaR: Surface-Aligned GS for Mesh | Guédon, Lepetit | GS → mesh 직접 추출. 대안 비교용 |
| 10 | [2506.24096](https://arxiv.org/abs/2506.24096) | MILo: Mesh-In-the-Loop GS | Guédon et al. | 2025 최신 mesh 품질. 비교용 |
| 11 | [2312.00846](https://arxiv.org/abs/2312.00846) | NeuSG: Neural Implicit + GS Guidance | Chen et al. | Neural implicit 하이브리드. 참고용 |

### 2.4 TSDF Fusion

| # | arXiv | 제목 | 저자 | 역할 |
|---|-------|------|------|------|
| 12 | [2105.07468](https://arxiv.org/abs/2105.07468) | TSDF++: Multi-Object Dynamic Tracking+Reconstruction | Grinvald et al. | TSDF 확장 가능성 확인. 우리는 정적 실내이므로 기본 TSDF 충분 |

### 2.5 Point Cloud → BIM/IFC

| # | arXiv | 제목 | 저자 | 게재 | 역할 |
|---|-------|------|------|------|------|
| 13 | [2503.11498](https://arxiv.org/abs/2503.11498) | **Cloud2BIM: Open-source Point Cloud → IFC Pipeline** | Zbirovský, Nežerka | **Automation in Construction 2025** | **핵심: 우리 Phase 2-3과 거의 동일한 파이프라인 (RANSAC plane → boundary → IfcOpenShell → IFC)** |
| 14 | [2411.18898](https://arxiv.org/abs/2411.18898) | Textured As-Is BIM via GIS-informed Point Cloud Segmentation | Alabassy | 2024 | 대규모 건물 Scan-to-BIM |

### 2.6 건물 특화 3DGS

| # | arXiv | 제목 | 저자 | 게재 | 역할 |
|---|-------|------|------|------|------|
| 15 | [2508.07355](https://arxiv.org/abs/2508.07355) | **GS4Buildings: Prior-Guided 2DGS for Building Reconstruction** | Zhang et al. | **ISPRS 2025** | **2DGS 기반 건물 재구성. occlusion 대응** |

---

## 3. 대안 분석: 왜 현재 설계가 최선인가

### 3.1 왜 2DGS인가 (vs 3DGS, SuGaR, GOF)

| 방법 | Surface 품질 | 속도 | 추가 패키지 | 판단 |
|------|------------|------|-----------|------|
| **2DGS (gsplat)** | **SOTA** (DTU Chamfer 0.35mm) | 빠름 | 없음 (gsplat 내장) | **채택** |
| 3DGS 원본 | 낮음 (multi-view inconsistent) | 빠름 | 없음 | Surface 부정확 |
| SuGaR | 중간 | 느림 (Poisson 후처리) | `sugar` 별도 | Mesh 기반, 불필요 |
| GOF | 좋음 | 보통 | 별도 구현 필요 | Marching Cubes 필요 |
| 3DGSR | 좋음 | 느림 (SDF 동시 학습) | 복잡한 코드 | MVP에 과도 |

**결론**: 2DGS가 surface 품질 최고 + gsplat 내장 + depth/normal 직접 렌더 → TSDF에 바로 넣을 수 있어 가장 직행.

### 3.2 왜 TSDF인가 (vs mesh, vs direct SDF)

| 방법 | 장점 | 단점 | 판단 |
|------|------|------|------|
| **Open3D TSDF** | 검증됨, 단순, 빠름, 평면 추출 바로 가능 | 해상도 제한 (voxel) | **채택** |
| Marching Cubes mesh | 시각적으로 좋음 | watertight 보장 어려움, BIM에 불필요 | 불필요 |
| Neural SDF (NeuS 등) | 품질 좋음 | 별도 학습 필요, 느림 | MVP에 과도 |

**결론**: BIM에 필요한 것은 "평면 + 경계"이지 mesh가 아님. TSDF에서 surface points → RANSAC plane이 가장 직행.

### 3.3 왜 Cloud2BIM 방식인가 (Phase 2-3)

Cloud2BIM (2503.11498)이 우리와 거의 동일한 파이프라인:
1. Point Cloud → RANSAC plane detection (Open3D)
2. Normal-based wall/floor/ceiling classification
3. Alpha shape boundary extraction
4. IfcOpenShell로 IFC 생성

**이 논문이 Automation in Construction (IF ~10) 2025에 게재됨** = 학술적으로도 검증된 접근.

---

## 4. 실내 특화 고려사항 (2DGS-Room, PGSR에서 참고)

### 4.1 Textureless 벽/바닥 문제
- **2DGS-Room 해법**: seed-guided initialization + geometric constraints (depth consistency + normal smoothness)
- **PGSR 해법**: planar prior regularization — Gaussian들이 평면에 정렬되도록 강제
- **우리 대응**: 2DGS 학습 시 depth/normal regularization loss 추가 고려

### 4.2 Glossy Surface (유리/반사)
- **Unbiased Depth 2DGS 해법** (2503.06587): reflection discontinuity로 인한 depth bias를 제거
- **우리 대응**: 실내에서 유리가 많으면 해당 기법 적용 검토 (gsplat 커스텀)

### 4.3 Depth Rendering Quality
- 2DGS는 3DGS 대비 **multi-view consistent depth**를 직접 제공
- surfel(2D disk) 표현이 surface와 자연스럽게 정렬 → depth 노이즈 감소
- TSDF integration 시 depth 품질이 핵심 → 2DGS 선택이 올바름

---

## 5. 다음 단계 (구현 우선순위)

### Phase 1 구현 순서 (s02 ~ s05)
1. **s02_colmap** — hloc(SuperPoint+LightGlue) + COLMAP 백엔드
2. **s03_gaussian_splatting** — gsplat 2DGS 모드 학습
3. **s04_depth_render** — gsplat에서 depth/normal map 렌더
4. **s05_tsdf_fusion** — Open3D ScalableTSDFVolume.integrate()

### Phase 2-3 구현 순서 (s06 ~ s07)
5. **s06_plane_extraction** — Open3D iterative RANSAC + normal classification + alphashape boundary
6. **s07_ifc_export** — IfcOpenShell IFC4 생성

### 설치해야 할 새 패키지
```
pip install pycolmap          # COLMAP Python bindings
pip install hloc              # SuperPoint + LightGlue + COLMAP wrapper
pip install gsplat            # 2DGS (Nerfstudio)
pip install torch torchvision # PyTorch (gsplat dependency)
```

---

## 6. 리스크 및 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| textureless 벽 → COLMAP 실패 | 포즈 부정확 | hloc SuperPoint가 SIFT보다 강건; 촬영 프로토콜 개선 |
| 유리/거울 → depth 오류 | TSDF 품질 저하 | Unbiased Depth 기법 적용; 마스킹 |
| 스케일 정확도 (BIM 치수) | IFC 오차 | 기준 거리 측정 (문 높이 등) + scale alignment |
| 2000프레임 속도 | 개발 반복 느림 | 500프레임 서브셋으로 시작; sequential matching |
| GPU VRAM 부족 | gsplat/hloc 실행 불가 | 최소 6GB VRAM 필요; cloud GPU 고려 |
