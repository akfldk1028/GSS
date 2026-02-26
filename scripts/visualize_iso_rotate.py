"""Generate multiple rotated isometric views of plane extraction results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

LABEL_COLORS = {
    "wall": ("#E74C3C", 0.4),
    "floor": ("#2ECC71", 0.4),
    "ceiling": ("#3498DB", 0.4),
    "other": ("#95A5A6", 0.12),
}
LABEL_EDGE = {
    "wall": "#C0392B",
    "floor": "#27AE60",
    "ceiling": "#2980B9",
    "other": "#7F8C8D",
}


def load_ply(path, n=400_000):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if len(pts) > n:
        pts = pts[np.random.choice(len(pts), n, replace=False)]
    return pts


def render_view(pts, planes, elev, azim, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Point cloud
    ax.scatter(
        pts[:, 0], pts[:, 2], pts[:, 1],
        c="#BDC3C7", s=0.03, alpha=0.1, rasterized=True,
    )

    # Planes
    for plane in planes:
        bnd = np.array(plane["boundary_3d"])
        if len(bnd) < 3:
            continue
        label = plane["label"]
        color, alpha = LABEL_COLORS[label]
        verts = [list(zip(bnd[:, 0], bnd[:, 2], bnd[:, 1]))]
        poly = Poly3DCollection(verts, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor(LABEL_EDGE[label])
        poly.set_linewidth(2.5 if label != "other" else 1.0)
        ax.add_collection3d(poly)

    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Z", fontsize=11)
    ax.set_zlabel("Y", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect
    all_pts = pts.copy()
    for p in planes:
        if p["boundary_3d"]:
            all_pts = np.vstack([all_pts, np.array(p["boundary_3d"])])
    rng = all_pts.max(axis=0) - all_pts.min(axis=0)
    mr = rng.max()
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    ax.set_xlim(mid[0] - mr/2, mid[0] + mr/2)
    ax.set_ylim(mid[2] - mr/2, mid[2] + mr/2)
    ax.set_zlim(mid[1] - mr/2, mid[1] + mr/2)

    # Legend
    counts = {}
    for p in planes:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    legend_items = []
    for lbl in ["wall", "floor", "ceiling", "other"]:
        if lbl in counts:
            c, _ = LABEL_COLORS[lbl]
            legend_items.append(Patch(facecolor=c, edgecolor=LABEL_EDGE[lbl],
                                      label=f"{lbl.capitalize()} ({counts[lbl]})"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ply_path = DATA / "interim" / "s03_planargs" / "surface_points.ply"
    planes_path = DATA / "interim" / "s06_planes" / "planes.json"

    print("Loading...")
    pts = load_ply(ply_path, 400_000)
    with open(planes_path) as f:
        planes = json.load(f)

    out_dir = ROOT / "docs" / "images"

    views = [
        (30, 135, "View 1 — Front-Left (elev=30, azim=135)"),
        (30, 225, "View 2 — Back-Right (elev=30, azim=225)"),
        (20, 315, "View 3 — Front-Right (elev=20, azim=315)"),
        (45, 180, "View 4 — Back Center (elev=45, azim=180)"),
        (10, 90,  "View 5 — Side Low (elev=10, azim=90)"),
        (60, 45,  "View 6 — Top Diagonal (elev=60, azim=45)"),
    ]

    for i, (elev, azim, title) in enumerate(views, 1):
        render_view(pts, planes, elev, azim, title, out_dir / f"planes_view{i}.png")

    print("Done!")


if __name__ == "__main__":
    main()
