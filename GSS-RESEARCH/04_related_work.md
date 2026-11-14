# 04. 관련 연구 & 시장/산업 동향

## 1. Scan-to-BIM 자동화 연구 흐름

### 전통적 접근 (LiDAR 기반)
- **입력**: 지상 LiDAR 스캐너 (TLS) 또는 MLS 점군
- **처리**: 수동 또는 반자동 평면/요소 감지 → BIM 모델링
- **도구**: Autodesk ReCap → Revit, FARO SCENE → PointCab
- **한계**: LiDAR 장비 고가 ($10K~$100K), 전문 인력 필요, 수일~수주 소요

### Cloud2BIM (Zbirovský & Nežerka, 2025)
- **핵심 기여**: LiDAR 점군 → IFC 완전 자동화
- **방법**: 밀도 기반 평면 감지 → 히스토그램 개구부 감지 → IFC 생성
- **성능**: 97.6% wall detection accuracy, 기존 대비 7배 빠름
- **GSS와의 관계**: GSS의 s06b→s07 아키텍처가 Cloud2BIM에서 가장 많은 영향을 받음
- **차이점**: Cloud2BIM은 LiDAR 입력, GSS는 카메라 영상 입력

### FloorPP-Net (Wu & Xue, 2021)
- **arxiv**: 2106.10635
- **방법**: 딥러닝 기반 점군 → floor plan 추출 (Point Pillars)
- **의미**: DL-based Scan-to-BIM의 초기 시도
- **GSS와의 비교**: GSS는 rule-based (RANSAC + geometric reasoning), 학습 데이터 불필요

## 2. 3DGS 기반 건축 재구성

### 3DGS → Surface → BIM 경로

| 논문 | 연도 | 접근법 | GSS 관련성 |
|------|------|--------|-----------|
| 3DGS (Kerbl) | 2023 | Gaussian → novel view | 파이프라인 기반 |
| 2DGS (Huang) | 2024 | Flat disk → surface | s03 표면 품질 향상 |
| SuGaR (Guédon) | 2024 | Surface-aligned → mesh | mesh 추출 대안 |
| PlanarGS (Jin) | 2025 | Plane prior → planar GS | s03_planargs 직접 통합 |
| GSPlane (Gan) | 2025 | Structured plane recon | 향후 참고 |
| AtlasGS (Zhang) | 2025 | Atlanta-world surface | Non-Manhattan 지원 |
| GS4Buildings (Zhang) | 2025 | LoD2 prior → building | s06c 외부 건물 |
| Material-GS (Huynh) | 2025 | Material label → DT | 향후 확장 |

### MILo: Mesh-In-the-Loop GS (ACM TOG, 2025)
- 최초로 mesh 추출을 3DGS 최적화 루프에 통합
- train 과정에서 mesh를 반복 추출/정제
- GSS 향후 방향: 3DGS 학습 시 BIM geometry를 직접 제약 조건으로 사용?

### GBM: Gaussian Building Mesh (2025)
- Google Earth 위성영상 + 2DGS → 건물 mesh 자동 추출
- 도시 스케일 건물 재구성
- GSS의 s06c와 유사한 목표 (외부 건물), 다른 입력 (위성 vs 지상)

## 3. 디지털 트윈 & 공간컴퓨팅 동향

### IFC → Digital Twin 변환 경로

| 경로 | 도구 | 비용 | 메타데이터 보존 |
|------|------|------|----------------|
| IFC → Omniverse Connector → USD | NVIDIA | 무료(개인) | BIM 관계 보존 |
| IFC → Blender/Bonsai → USD | Open-source | 무료 | geometry만 |
| IFC → UE5 Datasmith → UE5 | Epic Games | 무료(<$1M) | 계층+속성 |
| IFC → ifcconvert → GLB | IfcOpenShell | 무료 | geometry만 |
| **IFC → GSS s08 → GLB/USD** | GSS | 무료 | geometry+색상 |

### NVIDIA IFC-in-USD Initiative
- IFC schema를 USD에 네이티브 인코딩하는 표준화 작업
- AOUSD(Alliance for OpenUSD) + buildingSMART 공동 추진
- 완성 시 IFC↔USD 무손실 변환 가능 (현재 진행 중)
- GSS의 s08 USD 출력이 이 표준과 자연스럽게 연결

### Apple Vision Pro & Spatial Computing
- Apple + Dassault: 3DExperience → Vision Pro 공간 시각화 (2025)
- USDZ가 Apple 생태계의 표준 3D 포맷
- GSS의 s08에서 USDZ 직접 출력 지원 → Vision Pro 바로 확인 가능

### XGRIDS (상용 경쟁사)
- SLAM + 3DGS → Revit 플러그인 (AI 벽/개구부 인식)
- 상용 제품으로 Scan-to-BIM 자동화 시장 존재 확인
- GSS 대비: XGRIDS는 상용/Revit 종속, GSS는 오픈소스/IFC 표준

## 4. 학계 검증: 3DGS + BIM 방향의 타당성

### Nature Scientific Reports (2026)
- "Integrating CAD, BIM, and 3DGS" — ISO 19650 준수
- 3DGS로 리뷰 사이클 5-7일 → 실시간
- **의미**: 최상위 저널에서 3DGS의 BIM 활용을 인정

### AEC Magazine (2025)
- "Gaussian Splats for AEC" 특집
- Autodesk: 3DGS → ReCap/Revit/Civil 3D 통합 중
- **시사점**: 업계 거대 기업이 3DGS를 BIM 도구로 인정

### NeurIPS 2025 (학회)
- PlanarGS, AtlasGS 등 건축/실내 특화 3DGS 논문 다수 발표
- **의미**: 학계에서 3DGS + 건축 재구성이 주류 연구 주제로 부상

## 5. GSS의 차별점

### vs LiDAR Scan-to-BIM
| 항목 | LiDAR 기반 | GSS (카메라 기반) |
|------|-----------|-----------------|
| 입력 장비 | LiDAR ($10K+) | 카메라/스마트폰 ($0) |
| 점군 밀도 | 높음 (수백만~수천만) | 중간 (수십만~수백만) |
| RGB 정보 | 제한적 | 풍부 |
| 처리 자동화 | 반자동 | 완전 자동 |
| 비용 | 높음 | 낮음 |
| 개구부 감지 | 정확 (높은 밀도) | 제한적 (낮은 밀도) |

### vs 상용 도구 (XGRIDS, Revit)
| 항목 | 상용 도구 | GSS |
|------|----------|-----|
| 가격 | 유료 라이선스 | 오픈소스 |
| 출력 형식 | 특정 소프트웨어 종속 | IFC (표준) + GLB/USD |
| 커스터마이징 | 제한적 | 완전 자유 |
| 확장성 | 벤더 의존 | 연구자 자체 확장 |

### vs DL-based 방법 (FloorPP-Net 등)
| 항목 | DL-based | GSS (Rule-based) |
|------|----------|-----------------|
| 학습 데이터 | 필요 (라벨링 비용) | 불필요 |
| 일반화 | 학습 도메인 의존 | 기하학적 원리 기반 |
| 해석 가능성 | Black box | 투명 (각 단계 검증 가능) |
| 확장성 | 데이터 수집 필요 | 파라미터 조정으로 적응 |

## 6. 향후 연구 방향

### 단기 (현재 기술 기반 확장)
1. **Multi-segment center-line**: 2-point → N-point polyline (Bassier et al. 참고)
2. **Semantic segmentation**: 3DGS 점군에 semantic label 추가 (SAM2 + CLIP)
3. **Material classification**: 텍스처 → 재질 분류 (Material-informed GS 참고)
4. **Multi-storey 자동 분리**: 높이 히스토그램으로 층 자동 감지

### 중기 (아키텍처 개선)
5. **End-to-end plane extraction**: RANSAC → learnable (GSPlane 참고)
6. **Curved wall support**: 원호/곡선 벽 중심선 (Bassier et al. 참고)
7. **Furniture detection**: 3DGS semantic → IfcFurnishingElement
8. **MEP detection**: 배관/덕트/전기 설비 감지 (미래)

### 장기 (플랫폼 연동)
9. **실시간 IoT 연동**: MQTT/WebSocket → USD Live Update (Hiverlab 패턴)
10. **AR/VR 통합**: USDZ → Apple Vision Pro, GLB → WebXR
11. **IFC-in-USD**: NVIDIA 표준화 완료 시 네이티브 BIM↔DT 변환
12. **클라우드 파이프라인**: AWS CDK 기반 자동화 (AWS 3DGS at Scale 참고)
