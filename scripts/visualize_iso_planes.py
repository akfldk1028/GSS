"""Isometric visualization: surface point cloud + plane boundaries by label.

Shows PlanarGS surface with detected planes colored by classification:
  wall=red, floor=green, ceiling=blue, other=gray

Usage:
    python scripts/visualize_iso_planes.py
    python scripts/visualize_iso_planes.py --subsample 500000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

LABEL_COLORS = {
    "wall": ("#E74C3C", 0.35),     # red
    "floor": ("#2ECC71", 0.35),    # green
    "ceiling": ("#3498DB", 0.35),  # blue
    "other": ("#95A5A6", 0.15),    # gray, more transparent
}

LABEL_EDGE_COLORS = {
    "wall": "#C0392B",
    "floor": "#27AE60",
    "ceiling": "#2980B9",
    "other": "#7F8C8D",
}


def load_ply_points(ply_path: Path, max_points: int = 500_000) -> np.ndarray:
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def load_planes(planes_path: Path) -> list[dict]:
    with open(planes_path) as f:
        return json.load(f)


def make_iso_figure(pts: np.ndarray, planes: list[dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection="3d")

    # --- Point cloud (light gray, background) ---
    ax.scatter(
        pts[:, 0], pts[:, 2], pts[:, 1],
        c="#D5D8DC", s=0.02, alpha=0.08, rasterized=True,
    )

    # --- Plane boundaries as filled polygons ---
    arch_planes = [p for p in planes if p["label"] != "other" and p["boundary_3d"]]
    other_planes = [p for p in planes if p["label"] == "other" and p["boundary_3d"]]

    # Draw other first (behind), then architectural
    for plane in other_planes + arch_planes:
        label = plane["label"]
        bnd = np.array(plane["boundary_3d"])
        if len(bnd) < 3:
            continue

        color, alpha = LABEL_COLORS[label]
        edge_color = LABEL_EDGE_COLORS[label]

        # Swap Y/Z for matplotlib (Y-up → Z-up in plot)
        verts = [[bnd[:, 0], bnd[:, 2], bnd[:, 1]]]
        verts = [list(zip(bnd[:, 0], bnd[:, 2], bnd[:, 1]))]

        poly = Poly3DCollection(verts, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(2.0 if label != "other" else 0.8)
        ax.add_collection3d(poly)

    # --- Labels and view ---
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Z", fontsize=12)
    ax.set_zlabel("Y (height)", fontsize=12)

    # Count by label
    counts = {}
    for p in planes:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    title_parts = []
    for lbl in ["wall", "floor", "ceiling", "other"]:
        if lbl in counts:
            title_parts.append(f"{counts[lbl]} {lbl}s")
    ax.set_title(
        f"Plane Extraction — Isometric View\n{', '.join(title_parts)}",
        fontsize=14, fontweight="bold",
    )

    # True isometric: elev=35.264°, azim=45°
    ax.view_init(elev=35.264, azim=45)

    # Equal aspect ratio
    all_pts = pts.copy()
    for p in planes:
        if p["boundary_3d"]:
            all_pts = np.vstack([all_pts, np.array(p["boundary_3d"])])
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_range = ranges.max()
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    ax.set_zlim(mid[1] - max_range / 2, mid[1] + max_range / 2)

    # Legend
    from matplotlib.patches import Patch
    legend_items = []
    for lbl in ["wall", "floor", "ceiling", "other"]:
        if lbl in counts:
            color, _ = LABEL_COLORS[lbl]
            legend_items.append(Patch(facecolor=color, edgecolor=LABEL_EDGE_COLORS[lbl],
                                      label=f"{lbl.capitalize()} ({counts[lbl]})"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def make_topdown_figure(pts: np.ndarray, planes: list[dict], out_path: Path):
    """Top-down view (floor plan) with plane boundaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(14, 14))

    # Point cloud background (X-Z plane)
    ax.scatter(pts[:, 0], pts[:, 2], c="#D5D8DC", s=0.01, alpha=0.05, rasterized=True)

    # Plane boundaries
    for plane in planes:
        label = plane["label"]
        bnd = np.array(plane["boundary_3d"])
        if len(bnd) < 3:
            continue

        color, alpha = LABEL_COLORS[label]
        edge_color = LABEL_EDGE_COLORS[label]
        # Project to X-Z plane
        poly = Polygon(bnd[:, [0, 2]], closed=True)
        ax.add_patch(poly)
        poly.set_facecolor(color)
        poly.set_alpha(alpha + 0.1)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(2.0 if label != "other" else 0.8)

        # Label text at centroid
        cx, cz = bnd[:, 0].mean(), bnd[:, 2].mean()
        ax.text(cx, cz, f"{label}\n#{plane['id']}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=edge_color)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Z", fontsize=12)
    ax.set_title("Plane Extraction — Top-Down View", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    from matplotlib.patches import Patch
    counts = {}
    for p in planes:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    legend_items = []
    for lbl in ["wall", "floor", "ceiling", "other"]:
        if lbl in counts:
            color, _ = LABEL_COLORS[lbl]
            legend_items.append(Patch(facecolor=color, edgecolor=LABEL_EDGE_COLORS[lbl],
                                      label=f"{lbl.capitalize()} ({counts[lbl]})"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def make_front_figure(pts: np.ndarray, planes: list[dict], out_path: Path):
    """Front elevation view (X-Y) with plane boundaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.scatter(pts[:, 0], pts[:, 1], c="#D5D8DC", s=0.01, alpha=0.05, rasterized=True)

    for plane in planes:
        label = plane["label"]
        bnd = np.array(plane["boundary_3d"])
        if len(bnd) < 3:
            continue

        color, alpha = LABEL_COLORS[label]
        edge_color = LABEL_EDGE_COLORS[label]
        poly = Polygon(bnd[:, [0, 1]], closed=True)
        ax.add_patch(poly)
        poly.set_facecolor(color)
        poly.set_alpha(alpha + 0.1)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(2.0 if label != "other" else 0.8)

        cx, cy = bnd[:, 0].mean(), bnd[:, 1].mean()
        ax.text(cx, cy, f"{label}\n#{plane['id']}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=edge_color)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y (height)", fontsize=12)
    ax.set_title("Plane Extraction — Front Elevation View", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    from matplotlib.patches import Patch
    counts = {}
    for p in planes:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    legend_items = []
    for lbl in ["wall", "floor", "ceiling", "other"]:
        if lbl in counts:
            color, _ = LABEL_COLORS[lbl]
            legend_items.append(Patch(facecolor=color, edgecolor=LABEL_EDGE_COLORS[lbl],
                                      label=f"{lbl.capitalize()} ({counts[lbl]})"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=500_000)
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--planes", type=str, default=None)
    args = parser.parse_args()

    ply_path = Path(args.ply) if args.ply else DATA / "interim" / "s03_planargs" / "surface_points.ply"
    planes_path = Path(args.planes) if args.planes else DATA / "interim" / "s06_planes" / "planes.json"

    if not ply_path.exists():
        print(f"PLY not found: {ply_path}")
        return
    if not planes_path.exists():
        print(f"planes.json not found: {planes_path}")
        return

    print(f"Loading points: {ply_path}")
    pts = load_ply_points(ply_path, max_points=args.subsample)
    print(f"  {len(pts):,} points loaded")

    print(f"Loading planes: {planes_path}")
    planes = load_planes(planes_path)
    counts = {}
    for p in planes:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    print(f"  {len(planes)} planes: {counts}")

    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_iso_figure(pts, planes, out_dir / "planes_iso.png")
    make_topdown_figure(pts, planes, out_dir / "planes_topdown.png")
    make_front_figure(pts, planes, out_dir / "planes_front.png")

    print(f"\nAll saved to {out_dir}/")


if __name__ == "__main__":
    main()