"""Create pipeline flow comparison: 3DGS -> Planes -> BIM for each best scene.
One row per scene, 3 columns: raw pointcloud | planes (topdown) | BIM (isometric)
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "GSS-RESEARCH" / "GSS-PT" / "260309" / "best_results"
SCENES = ["jh", "drjohnson", "playroom", "synthetic_L"]

fig, axes = plt.subplots(4, 3, figsize=(24, 28))
fig.suptitle("GSS Pipeline: 3DGS Point Cloud → Plane Extraction → BIM Model",
             fontsize=20, fontweight="bold", y=0.98)

col_titles = ["1. 3DGS Point Cloud", "2. Plane Extraction (s06b)", "3. BIM Model (IFC)"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=16, fontweight="bold", pad=15)

for i, scene in enumerate(SCENES):
    # Column 1: Gaussian point cloud
    gauss_img = OUT / f"{scene}_gaussians.png"
    if gauss_img.exists():
        img = mpimg.imread(str(gauss_img))
        axes[i, 0].imshow(img)
    axes[i, 0].set_ylabel(scene, fontsize=14, fontweight="bold", rotation=0, labelpad=60)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    # Column 2: Planes topdown (regularized)
    planes_img = OUT / f"{scene}_planes_isometric.png"
    if planes_img.exists():
        img = mpimg.imread(str(planes_img))
        axes[i, 1].imshow(img)
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])

    # Column 3: BIM isometric (IFC)
    bim_img = OUT / f"{scene}_bim_isometric.png"
    if not bim_img.exists():
        bim_img = OUT / f"{scene}_ifc_isometric.png"
    if bim_img.exists():
        img = mpimg.imread(str(bim_img))
        axes[i, 2].imshow(img)
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])

# Add arrows between columns
for i in range(4):
    for j in range(2):
        # Get position between columns
        pos1 = axes[i, j].get_position()
        pos2 = axes[i, j + 1].get_position()
        arrow_x = (pos1.x1 + pos2.x0) / 2
        arrow_y = (pos1.y0 + pos1.y1) / 2
        fig.text(arrow_x, arrow_y, "→", fontsize=30, ha="center", va="center",
                 fontweight="bold", color="#2196F3")

fig.tight_layout(rect=[0.05, 0, 1, 0.96])
out_path = OUT / "pipeline_flow_comparison.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")

# Also make a single-scene detailed version for jh
fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
fig2.suptitle("GSS Pipeline Flow — JH Dataset (Best Result)",
              fontsize=16, fontweight="bold")

for j, (title, fname) in enumerate([
    ("1. 3DGS Point Cloud\n80,000 gaussians", "jh_gaussians.png"),
    ("2. Regularized Planes\n4 walls, 1 floor, 1 ceiling", "jh_planes_isometric.png"),
    ("3. IFC BIM Model\n4W 1F 1C 1S", "jh_ifc_isometric.png"),
]):
    img_path = OUT / fname
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        axes2[j].imshow(img)
    axes2[j].set_title(title, fontsize=13, fontweight="bold")
    axes2[j].set_xticks([])
    axes2[j].set_yticks([])

fig2.tight_layout()
out2 = OUT / "jh_pipeline_flow.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {out2}")
