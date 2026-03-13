"""Compare 3DGS point cloud vs BIM — 3D isometric overlay.

BIM rendered as bold wireframe edges so point cloud is visible through it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
IMAGE_DIR = PROJECT_ROOT / "docs" / "images"

SCENES = ["jh", "drjohnson", "playroom", "synthetic_L"]


def load_point_cloud(ply_path, max_points=50000):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def transform_to_ifc(pts, rotation, scale):
    R = np.array(rotation)
    m = pts @ R.T
    return np.column_stack([m[:, 0] / scale, m[:, 2] / scale, m[:, 1] / scale])


def load_ifc_meshes(ifc_path):
    import ifcopenshell
    import ifcopenshell.geom

    COLORS = {
        "wall": [0.2, 0.4, 1.0],
        "wall_synthetic": [1.0, 0.3, 0.1],
        "floor": [0.1, 0.7, 0.1],
        "ceiling": [0.9, 0.7, 0.0],
        "space": [0.5, 0.5, 0.5],
    }
    ifc = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    meshes = []
    for ifc_class in ("IfcWall", "IfcSlab"):  # Skip IfcSpace for clarity
        for element in ifc.by_type(ifc_class):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
            except Exception:
                continue
            v = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            f = np.array(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
            if len(v) < 3:
                continue
            name = element.Name or ""
            cls = element.is_a()
            if cls == "IfcWall":
                label = "wall_synthetic" if "Synthetic" in name else "wall"
            elif cls == "IfcSlab":
                label = "floor" if "Floor" in name else "ceiling"
            else:
                label = "wall"
            meshes.append({"verts": v, "faces": f, "color": COLORS[label], "label": label})
    return meshes


def get_wireframe_edges(meshes):
    """Extract unique edges from mesh triangles for wireframe rendering."""
    edges_by_color = {}
    for m in meshes:
        color = tuple(m["color"])
        if color not in edges_by_color:
            edges_by_color[color] = []
        edge_set = set()
        for tri in m["faces"]:
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edge_set.add(e)
        for e in edge_set:
            p0, p1 = m["verts"][e[0]], m["verts"][e[1]]
            edges_by_color[color].append((p0, p1))
    return edges_by_color


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    np.random.seed(42)

    available = []
    for name in SCENES:
        run_dir = RUNS_DIR / name
        ply_path = run_dir / "interim" / "s00_import_ply" / "surface_points.ply"
        ifc_files = list((run_dir / "processed").glob("*.ifc")) if (run_dir / "processed").exists() else []
        align_path = run_dir / "interim" / "s06b_plane_regularization" / "manhattan_alignment.json"
        spaces_path = run_dir / "interim" / "s06b_plane_regularization" / "spaces.json"
        stats_path = run_dir / "interim" / "s06b_plane_regularization" / "stats.json"

        if ply_path.exists() and ifc_files and align_path.exists():
            scale = 1.0
            if spaces_path.exists():
                with open(spaces_path, encoding="utf-8") as f:
                    scale = json.load(f).get("coordinate_scale", 1.0)
            elif stats_path.exists():
                with open(stats_path, encoding="utf-8") as f:
                    scale = json.load(f).get("scale", 1.0)
            with open(align_path, encoding="utf-8") as f:
                rotation = json.load(f)["manhattan_rotation"]
            available.append({
                "name": name, "ply_path": ply_path, "ifc_path": ifc_files[0],
                "rotation": rotation, "scale": scale,
            })

    n = len(available)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(10 * ncols, 9 * nrows))
    fig.suptitle("3DGS Point Cloud + BIM Wireframe Overlay (Isometric)",
                 fontsize=18, fontweight="bold", y=0.99)

    for idx, scene in enumerate(available):
        name = scene["name"]
        print(f"  Loading {name}...")

        raw_pts = load_point_cloud(scene["ply_path"], max_points=50000)
        ifc_pts = transform_to_ifc(raw_pts, scene["rotation"], scene["scale"])
        meshes = load_ifc_meshes(scene["ifc_path"])
        if not meshes:
            continue

        bim_all = np.vstack([m["verts"] for m in meshes])
        bim_min, bim_max = bim_all.min(axis=0), bim_all.max(axis=0)
        bim_size = bim_max - bim_min
        bim_center = (bim_min + bim_max) / 2
        half = max(bim_size) / 2
        pad = half * 0.5

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        # 1) Point cloud — colored by height
        z_norm = (ifc_pts[:, 2] - ifc_pts[:, 2].min()) / (ifc_pts[:, 2].ptp() + 1e-8)
        ax.scatter(ifc_pts[:, 0], ifc_pts[:, 1], ifc_pts[:, 2],
                   c=z_norm, cmap="coolwarm", s=0.5, alpha=0.4, zorder=1)

        # 2) BIM wireframe — thick colored lines
        edges_by_color = get_wireframe_edges(meshes)
        for color, edges in edges_by_color.items():
            segments = [(list(p0), list(p1)) for p0, p1 in edges]
            lc = Line3DCollection(segments, colors=[color], linewidths=2.0, zorder=5)
            ax.add_collection3d(lc)

        ax.set_xlim(bim_center[0] - half - pad, bim_center[0] + half + pad)
        ax.set_ylim(bim_center[1] - half - pad, bim_center[1] + half + pad)
        ax.set_zlim(bim_center[2] - half - pad, bim_center[2] + half + pad)
        ax.view_init(elev=25, azim=-55)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.set_zlabel("Z (m)", fontsize=9)
        ax.tick_params(labelsize=7)

        n_w = sum(1 for m in meshes if m["label"] in ("wall", "wall_synthetic"))
        n_f = sum(1 for m in meshes if m["label"] == "floor")
        n_c = sum(1 for m in meshes if m["label"] == "ceiling")

        ax.set_title(
            f"{name}  —  {len(ifc_pts):,} pts + BIM({n_w}W {n_f}F {n_c}C)\n"
            f"BIM: {bim_size[0]:.1f}x{bim_size[1]:.1f}x{bim_size[2]:.1f}m  scale={scene['scale']:.2f}",
            fontsize=12, fontweight="bold", pad=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="3DGS Points"),
        Line2D([0], [0], color=[0.2, 0.4, 1.0], lw=3, label="Wall"),
        Line2D([0], [0], color=[1.0, 0.3, 0.1], lw=3, label="Synthetic Wall"),
        Line2D([0], [0], color=[0.1, 0.7, 0.1], lw=3, label="Floor"),
        Line2D([0], [0], color=[0.9, 0.7, 0.0], lw=3, label="Ceiling"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = IMAGE_DIR / "3dgs_vs_bim_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    import os
    os.startfile(str(out))


if __name__ == "__main__":
    main()
