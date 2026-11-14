# GSS Research Documentation

## 3D Gaussian Splatting to BIM (Scan-to-BIM) 자동화 파이프라인

> **GSS**: 카메라 영상만으로 건축물의 BIM(IFC) 모델을 자동 생성하는 오픈소스 파이프라인

### 문서 구성

| 문서 | 내용 |
|------|------|
| [01_pipeline_overview.md](01_pipeline_overview.md) | 전체 워크플로우 상세 설명 |
| [02_phase_references.md](02_phase_references.md) | 파이프라인 단계별 참조 논문 & 기술 근거 |
| [03_validation_results.md](03_validation_results.md) | End-to-End 테스트 결과 & 검증 방법론 |
| [04_related_work.md](04_related_work.md) | 관련 연구 & 시장/산업 동향 |
| [05_bibliography.md](05_bibliography.md) | 전체 참고문헌 목록 (BibTeX) |

### 핵심 연구 질문

1. **카메라 영상만으로 BIM-ready geometry를 추출할 수 있는가?**
   - LiDAR 없이 3DGS → 평면 추출 → IFC 변환이 가능한지 검증
   - 결론: 실내 건축물에서 벽/바닥/천장 자동 감지 성공 (9개 데이터셋 검증)

2. **3DGS 점군에서 건축 구조를 어떻게 추출하는가?**
   - RANSAC 평면 추출 → Manhattan-world 정규화 → 중심선 기반 벽 모델링
   - Cloud2BIM, PlanarGS, AtlasGS 등의 방법론 통합 적용

3. **비직교(Non-Manhattan) 건물도 처리할 수 있는가?**
   - Atlanta-world 모델 도입 (AtlasGS 영향)
   - L자형/비정형 건물 합성 데이터로 검증 완료

### 파이프라인 요약도

```
Video/Images ──→ SfM (COLMAP) ──→ 3DGS Training ──→ Point Cloud
                                         │
                   ┌─────────────────────┘
                   ▼
            RANSAC Plane Extraction (s06)
                   │
                   ▼
         ┌── Building Extraction (s06c, exterior only)
         │         │
         ▼         ▼
     Plane Regularization (s06b)
         │
         ├── Manhattan Normal Snap (A)
         ├── Height Clustering (B)
         ├── Wall Thickness / Center-lines (C)
         ├── Synthetic Wall Closure (C2)
         ├── Intersection Trimming (D)
         ├── Space Detection / Polygonize (E)
         └── Opening Detection (F, optional)
                   │
                   ▼
          IFC Export (s07) ──→ GSS_*.ifc
                   │
                   ▼
        Mesh Export (s08) ──→ GLB / USD / USDZ
```

### 기여자

- 파이프라인 설계 및 구현: DK
- 코드 생성 지원: Claude (Anthropic)

### 라이선스

연구 목적 사용. 자세한 내용은 프로젝트 루트 참조.
