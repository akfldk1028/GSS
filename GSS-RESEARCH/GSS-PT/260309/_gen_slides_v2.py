"""Generate NotebookLM slide-deck with shorter prompt."""
import subprocess
import sys

# Shorter, focused prompt
prompt = (
    "학술 발표 부록 슬라이드 13장. 한국어. 흰색 배경. "
    "GSS 핵심 알고리즘을 학술 수준으로 설명하되 비전공자도 이해 가능하게. "
    "각 슬라이드에 원 논문 citation 포함. 수식과 pseudocode 적극 활용. "
    "1:표지(GSS 핵심 알고리즘 해설) "
    "2:SfM(Essential Matrix, Bundle Adjustment, COLMAP, Schonberger CVPR2016) "
    "3:3DGS(Gaussian수식, alpha-blending, loss, Kerbl SIGGRAPH2023) "
    "4:PlanarGS(co-planarity prior, plane loss, Jiang NeurIPS2025) "
    "5:RANSAC(pseudocode, iteration공식, sequential RANSAC, Fischler CACM1981) "
    "6:Alpha Shape(Delaunay, concave hull, Edelsbrunner ACM TOG1994) "
    "7:Manhattan World(직교가정, 법선클러스터링, 회전행렬, Coughlan ICCV1999) "
    "8:Geometric Regularization(6단계: normal snap, height cluster, wall thickness, closure, intersection trim, space detection) "
    "9:Scale Estimation(floor-ceiling거리/층고2.7m, 3단계fallback) "
    "10:IFC Export(IfcWall/Slab/Space계층, center-line extrusion, ISO16739) "
    "11:Mesh Export(IFC->GLB/USD/USDZ, Z-up->Y-up변환, PBR material) "
    "12:전체파이프라인 수학적요약(Video->SfM->3DGS->RANSAC->Regularization->IFC->GLB/USD) "
    "13:Related Work비교표(Traditional vs Cloud2BIM vs XGRIDS vs GBM vs GSS)"
)

subprocess.run(["notebooklm", "use", "01581abe"], encoding="utf-8")

result = subprocess.run(
    ["notebooklm", "generate", "slide-deck", prompt, "--wait", "--language", "ko", "--retry", "5"],
    encoding="utf-8",
    timeout=1200,
)
