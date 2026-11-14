# 05. 참고문헌 (Bibliography)

## 핵심 참조 논문 (GSS 파이프라인에 직접 사용)

### [P1] 3D Gaussian Splatting
```bibtex
@article{kerbl2023_3dgs,
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimkühler, Thomas and Drettakis, George},
  journal   = {ACM Transactions on Graphics},
  volume    = {42},
  number    = {4},
  year      = {2023},
  doi       = {10.1145/3592433},
  note      = {SIGGRAPH 2023}
}
```
- 3DGS 원조 논문. Gaussian primitives 기반 장면 표현 및 실시간 렌더링.
- **GSS 적용**: 전체 파이프라인의 근간. 3DGS 점군이 모든 후속 단계의 입력.

### [P2] 2D Gaussian Splatting
```bibtex
@inproceedings{huang2024_2dgs,
  title     = {2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author    = {Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  booktitle = {ACM SIGGRAPH 2024 Conference Papers},
  year      = {2024},
  doi       = {10.1145/3641519.3657428},
  note      = {arXiv:2403.17888}
}
```
- 3D Gaussian을 2D oriented disk으로 대체. 표면 geometry 정확도 대폭 향상.
- **GSS 적용**: s03에서 gsplat의 2DGS 모드 사용.

### [P3] PlanarGS
```bibtex
@inproceedings{jin2025_planargs,
  title     = {PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors},
  author    = {Jin, Xirui and Jin, Renbiao and Li, Boying and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2510.23930}
}
```
- Vision-Language 평면 prior를 3DGS 최적화에 통합. 실내 저텍스처 영역 해결.
- **GSS 적용**: s03_planargs 스텝에서 직접 통합 (subprocess 실행).

### [P7] Cloud2BIM
```bibtex
@article{zbirovsky2025_cloud2bim,
  title     = {Open-source automatic pipeline for efficient conversion of large-scale point clouds to IFC format},
  author    = {Zbirovský, Slávek and Nežerka, Václav},
  journal   = {Automation in Construction},
  volume    = {177},
  pages     = {106303},
  year      = {2025},
  doi       = {10.1016/j.autcon.2025.106303},
  note      = {arXiv:2503.11498}
}
```
- 점군 → IFC 완전 자동화. 히스토그램 기반 개구부 감지.
- **GSS 적용**: s06b 전체 아키텍처 + s07 IFC 생성 패턴의 주요 참조.

---

## 아키텍처 영감 논문

### [P8] AtlasGS
```bibtex
@inproceedings{zhang2025_atlasgs,
  title     = {AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians},
  author    = {Zhang, Xiyu and Bao, Chong and Chen, Yipeng and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2510.25129}
}
```
- Atlanta-world 모델. Non-Manhattan 건물 재구성.
- **GSS 적용**: s06b `normal_mode: "cluster"` 비직교 벽 지원.

### [P6] GS4Buildings
```bibtex
@inproceedings{zhang2025_gs4buildings,
  title     = {GS4Buildings: Prior-Guided Gaussian Splatting for 3D Building Reconstruction},
  author    = {Zhang, Qilin and Wysocki, Olaf and Jutzi, Boris},
  booktitle = {ISPRS 3D GeoInfo \& Smart Data, Smart Cities},
  year      = {2025},
  note      = {arXiv:2508.07355}
}
```
- LoD2 prior를 2DGS에 통합. 도시 건물 재구성.
- **GSS 적용**: s06c 외부 건물 추출 아키텍처 영감.

### Bassier et al. (2020)
```bibtex
@article{bassier2020_wall_topology,
  title     = {Unsupervised reconstruction of Building Information Modeling wall objects from point cloud data},
  author    = {Bassier, Maarten and Vergauwen, Maarten},
  journal   = {Automation in Construction},
  volume    = {120},
  year      = {2020}
}
```
- Polyline/곡선 벽 중심선 + 4가지 연결 토폴로지.
- **GSS 적용**: 벽 center-line 모델링 및 IFC polyline profile 방식에 영향.

---

## 배경/비교/확장 방향 논문

### [P4] SuGaR
```bibtex
@inproceedings{guedon2024_sugar,
  title     = {SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering},
  author    = {Guédon, Antoine and Lepetit, Vincent},
  booktitle = {CVPR},
  year      = {2024},
  note      = {arXiv:2311.12775}
}
```
- Surface-aligned Gaussian → Poisson mesh. 최초의 3DGS→mesh 방법.

### [P5] GSPlane
```bibtex
@article{gan2025_gsplane,
  title     = {GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation},
  author    = {Gan, Ruitong and Peng, Junran and Liu, Yang and others},
  year      = {2025},
  note      = {arXiv:2510.17095}
}
```
- 3DGS 최적화 과정에서 구조화된 평면 직접 학습.

### [P9] Material-informed Gaussian Splatting for Digital Twin
```bibtex
@inproceedings{huynh2025_material_gs,
  title     = {Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin},
  author    = {Huynh, Andy and Silva, João Malheiro and Caesar, Holger and others},
  booktitle = {IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2026},
  note      = {arXiv:2511.20348, Accepted}
}
```
- Camera-only → 3DGS → mesh → material labels → physics simulation.

### GBM: Gaussian Building Mesh
```bibtex
@article{gbm2025,
  title     = {Gaussian Building Mesh (GBM): Extract Building Meshes by 2D Gaussian Splatting Using Google Earth Data},
  journal   = {Science of Remote Sensing},
  year      = {2025},
  doi       = {10.1016/j.srs.2025.100360X}
}
```

### MILo: Mesh-In-the-Loop GS
```bibtex
@article{milo2025,
  title     = {MILo: Mesh-In-the-Loop for Gaussian Splatting},
  journal   = {ACM Transactions on Graphics},
  year      = {2025},
  doi       = {10.1145/3763339}
}
```

### FloorPP-Net
```bibtex
@inproceedings{wu2021_floorppnet,
  title     = {FloorPP-Net: Reconstructing Floor Plans using Point Pillars for Scan-to-BIM},
  author    = {Wu, Yijie and Xue, Fan},
  year      = {2021},
  note      = {arXiv:2106.10635}
}
```

---

## 산업/표준 참조

### IFC4 Standard
```
ISO 16739-1:2018
Industry Foundation Classes (IFC) for data sharing in the construction and facility management industries
buildingSMART International
```

### COLMAP
```bibtex
@inproceedings{schonberger2016_colmap,
  title     = {Structure-from-Motion Revisited},
  author    = {Schönberger, Johannes L. and Frahm, Jan-Michael},
  booktitle = {CVPR},
  year      = {2016}
}
```

### RANSAC
```bibtex
@article{fischler1981_ransac,
  title     = {Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography},
  author    = {Fischler, Martin A. and Bolles, Robert C.},
  journal   = {Communications of the ACM},
  volume    = {24},
  number    = {6},
  pages     = {381--395},
  year      = {1981}
}
```

### TSDF Fusion
```bibtex
@inproceedings{curless1996_tsdf,
  title     = {A Volumetric Method for Building Complex Models from Range Images},
  author    = {Curless, Brian and Levoy, Marc},
  booktitle = {SIGGRAPH},
  year      = {1996}
}
```

### Manhattan-world
```bibtex
@inproceedings{coughlan1999_manhattan,
  title     = {Manhattan World: Compass Direction from a Single Image by Bayesian Inference},
  author    = {Coughlan, James M. and Yuille, Alan L.},
  booktitle = {ICCV},
  year      = {1999}
}
```

### Alpha Shapes
```bibtex
@article{edelsbrunner1983_alpha,
  title     = {On the Shape of a Set of Points in the Plane},
  author    = {Edelsbrunner, Herbert and Kirkpatrick, David and Seidel, Raimund},
  journal   = {IEEE Transactions on Information Theory},
  volume    = {29},
  number    = {4},
  pages     = {551--559},
  year      = {1983}
}
```

### DBSCAN
```bibtex
@inproceedings{ester1996_dbscan,
  title     = {A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise},
  author    = {Ester, Martin and Kriegel, Hans-Peter and Sander, Jörg and Xu, Xiaowei},
  booktitle = {KDD},
  year      = {1996}
}
```

---

## 산업 동향 출처

| 출처 | URL | 내용 |
|------|-----|------|
| Nature Sci. Reports (2026) | nature.com/articles/s41598-025-32244-y | 3DGS + BIM + CAD 통합 |
| AEC Magazine (2025) | aecmag.com/technology/introducing-gaussian-splats-for-aec/ | AEC 업계 3DGS 도입 |
| NVIDIA IFC-in-USD | aecmag.com/collaboration/nvidia-looking-to-encode-the-ifc-schema-in-usd/ | IFC↔USD 표준화 |
| Apple + Dassault | computerweekly.com | Vision Pro 디지털 트윈 |
| Hiverlab GS DT | hiverlab.com/gaussian-splatting-for-digital-twin-new-feature/ | 실시간 IoT 오버레이 |
| AWS 3DGS at Scale | aws.amazon.com/blogs/spatial/ | 클라우드 3DGS 파이프라인 |
| Cloud2BIM GitHub | github.com/VaclavNezerka/Cloud2BIM | 오픈소스 점군→IFC |
