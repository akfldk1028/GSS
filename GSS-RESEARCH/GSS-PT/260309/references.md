# GSS 관련 핵심 논문 & 레퍼런스

## 1. GSS 파이프라인 직접 관련

### PlanarGS (NeurIPS 2025) — GSS가 이미 통합
- https://arxiv.org/abs/2510.23930
- Indoor 3DGS + vision-language planar prior → 평면 품질 대폭 향상
- GSS의 s03_planargs 스텝이 이 논문 구현체 사용

### Cloud2BIM (Automation in Construction, 2025)
- https://arxiv.org/abs/2503.11498
- https://github.com/VaclavNezerka/Cloud2BIM
- Point cloud → 벽/슬래브/개구부 자동 감지 → IFC
- GSS의 s06→s06b→s07과 동일 패턴, RANSAC 대신 density 기반
- 97.6% wall detection accuracy, 기존 대비 7배 빠름
- **차이점**: Cloud2BIM은 점군 입력, GSS는 영상 입력

### Material-informed Gaussian Splatting for Digital Twin (2025)
- https://arxiv.org/abs/2511.20348
- Camera-only → 3DGS → mesh → material labels → physics simulation
- GSS의 자연스러운 확장 방향 (재질 정보 추가)

## 2. 3DGS + BIM 통합 검증

### Integrating CAD, BIM, and 3DGS (Nature Scientific Reports, 2026)
- https://www.nature.com/articles/s41598-025-32244-y
- ISO 19650 준수, 3DGS로 리뷰 사이클 5-7일 → 실시간
- 3DGS가 as-built 검증의 핵심 도구로 자리잡는 중

### AEC Magazine: Gaussian Splats for AEC (2025)
- https://aecmag.com/technology/introducing-gaussian-splats-for-aec/
- Autodesk: 3DGS → ReCap/Revit/Civil 3D 통합 중
- XGRIDS: SLAM + 3DGS → Revit (AI 벽/개구부 인식)
- 한계: 3DGS는 시각화 도구, geometry 정확도는 LiDAR 앵커 필요

## 3. Digital Twin / Spatial Computing

### NVIDIA IFC-in-USD Initiative
- https://aecmag.com/collaboration/nvidia-looking-to-encode-the-ifc-schema-in-usd/
- IFC schema를 USD에 네이티브 인코딩 (AOUSD + buildingSMART)
- 완성 시 IFC↔USD 무손실 변환 가능 (진행 중, 아직 미완성)

### Apple + Dassault: Vision Pro Digital Twin
- https://www.computerweekly.com/news/366619549/Apple-Dassault-Systemes-join-forces-for-digital-twin-Vision
- 3DExperience → Vision Pro 공간 시각화 (2025 여름 출시)

### Hiverlab: GS Digital Twin Runtime Editor (2025)
- https://hiverlab.com/gaussian-splatting-for-digital-twin-new-feature/
- Multi-splat + 실시간 IoT 데이터 오버레이

### AWS: 3DGS at Scale (2025)
- https://aws.amazon.com/blogs/spatial/3d-gaussian-splatting-performant-3d-scene-reconstruction-at-scale/
- AWS CDK/Terraform 기반 클라우드 3DGS 파이프라인

## 4. Mesh Extraction from 3DGS

### SuGaR (CVPR 2024)
- https://github.com/Anttwo/SuGaR
- Surface-Aligned GS → mesh 추출, GSS s04/s05 대안

### MILo: Mesh-In-the-Loop GS (ACM TOG, 2025)
- https://dl.acm.org/doi/10.1145/3763339
- 최초로 mesh 추출을 3DGS 최적화 루프에 통합

### GBM: Gaussian Building Mesh (2025)
- https://www.sciencedirect.com/science/article/pii/S235293852500360X
- Google Earth + 2DGS → building mesh 자동 추출

## 5. IFC → Digital Twin 변환 경로

| 경로 | 비용 | 메타데이터 보존 |
|------|------|----------------|
| IFC → Omniverse CAD Converter → USD | 무료(개인) | BIM 관계 보존 |
| IFC → Blender/Bonsai → USD | 무료 | geometry만 |
| IFC → UE5 Datasmith | 무료(<$1M) | 계층+속성 |
| IFC → ifcconvert → GLB | 무료 | geometry만 |

## 6. 다음 확장 방향 참고

### 개구부(문/창) 감지
- Cloud2BIM의 opening detection 알고리즘 참고
- 3DGS depth map에서 depth discontinuity → 개구부 후보

### Material/Texture 추출
- Material-informed GS (2511.20348) 방식
- Vision-Language 모델 (SAM2 + CLIP) → 재질 분류

### IoT 센서 연동
- Hiverlab 패턴: 3DGS scene에 실시간 센서 데이터 오버레이
- MQTT/WebSocket → USD Live Update
