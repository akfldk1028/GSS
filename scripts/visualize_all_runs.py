"""Visualize IFC building geometry from all pipeline runs in a single comparison figure.

Generates a 4x3 grid: (bicycle, bonsai, room, train) x (isometric, top-down, front).
"""

import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RUNS_DIR = DATA_ROOT / "runs"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

DATASETS = ["jh", "drjohnson", "playroom", "bonsai", "room", "train"]

COLORS = {
    "wall": [0.5, 0.7, 1.0],
    "wall_synthetic": [1.0, 0.5, 0.3],
    "floor": [0.4, 0.85, 0.4],
    "ceiling": [1.0, 0.9, 0.4],
    "space": [0.7, 0.7, 0.7],
}


def _classify_element(element, z_avg=None, z_mid=None):
    ifc_class = element.is_a()
    name = element.Name or ""
    if ifc_class == "IfcWall":
        return "wall_synthetic" if "Synthetic" in name else "wall"
    if ifc_class == "IfcSlab":
        if "Floor" in name:
            return "floor"
        if "Ceiling" in name:
            return "ceiling"
        if z_avg is not None and z_mid is not None:
            return "ceiling" if z_avg > z_mid else "floor"
        return "ceiling"
    if ifc_class == "IfcSpace":
        return "space"
    return "wall"


def extract_meshes(ifc_path):
    import ifcopenshell
    import ifcopenshell.geom

    ifc = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    raw = []
    for ifc_class in ("IfcWall", "IfcSlab", "IfcSpace"):
        for element in ifc.by_type(ifc_class):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
            except Exception:
                continue
            v = shape.geometry.verts
            f = shape.geometry.faces
            if len(v) < 9 or len(f) < 3:
                continue
            verts = np.array(v, dtype=np.float64).reshape(-1, 3)
            faces = np.array(f, dtype=np.int32).reshape(-1, 3)
            raw.append((element, verts, faces))

    z_mid = None
    if raw:
        all_z = np.concatenate([v[:, 2] for _, v, _ in raw])
        z_mid = (all_z.min() + all_z.max()) / 2.0

    meshes = []
    for element, verts, faces in raw:
        z_avg = float(verts[:, 2].mean())
        label = _classify_element(element, z_avg=z_avg, z_mid=z_mid)
        meshes.append({
            "verts": verts,
            "faces": faces,
            "color": COLORS[label],
            "label": label,
            "name": element.Name or element.is_a(),
        })
    return meshes


def _get_stats(meshes):
    counts = {"wall": 0, "wall_synthetic": 0, "floor": 0, "ceiling": 0, "space": 0}
    for m in meshes:
        counts[m["label"]] = counts.get(m["label"], 0) + 1
    all_pts = np.vstack([m["verts"] for m in meshes])
    bbox_min, bbox_max = all_pts.min(axis=0), all_pts.max(axis=0)
    size = bbox_max - bbox_min
    n_walls = counts["wall"] + counts["wall_synthetic"]
    return counts, bbox_min, bbox_max, size, n_walls


def main():
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Load all datasets
    all_data = {}
    for name in DATASETS:
        ifc_path = RUNS_DIR / name / "processed" / f"GSS_{name}.ifc"
        if not ifc_path.exists():
            print(f"  Skip {name}: {ifc_path} not found")
            continue
        print(f"  Loading {name}...")
        meshes = extract_meshes(ifc_path)
        if meshes:
            stats_path = RUNS_DIR / name / "interim" / "s06b_plane_regularization" / "stats.json"
            scale = None
            if stats_path.exists():
                with open(stats_path) as f:
                    scale = json.load(f).get("scale")
            all_data[name] = {"meshes": meshes, "scale": scale}

    if not all_data:
        print("No datasets found!")
        sys.exit(1)

    n = len(all_data)
    names = list(all_data.keys())

    # ============================================================
    # Figure 1: Isometric 3D comparison (2-row grid)
    # ============================================================
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows),
                                subplot_kw={"projection": "3d"})
    if n == 1:
        axes1 = np.array([[axes1]])
    axes1 = np.atleast_2d(axes1)
    flat_axes = axes1.flatten()

    fig1.suptitle("GSS Pipeline — IFC Building Geometry (Isometric)",
                  fontsize=20, fontweight="bold", y=0.98)

    for i, name in enumerate(names):
        ax = flat_axes[i]
        meshes = all_data[name]["meshes"]
        scale = all_data[name]["scale"]
        counts, bbox_min, bbox_max, size, n_walls = _get_stats(meshes)

        for m in meshes:
            alpha = 0.1 if m["label"] == "space" else 0.6
            ec_color = "black" if m["label"] != "space" else "gray"
            tris = []
            for tri in m["faces"]:
                pts = m["verts"][tri]
                tris.append(list(zip(pts[:, 0], pts[:, 1], pts[:, 2])))
            ax.add_collection3d(Poly3DCollection(
                tris, fc=m["color"], ec=ec_color, lw=0.3, alpha=alpha))

        pad = max(size) * 0.15
        ax.set_xlim(bbox_min[0] - pad, bbox_max[0] + pad)
        ax.set_ylim(bbox_min[1] - pad, bbox_max[1] + pad)
        ax.set_zlim(bbox_min[2] - pad, bbox_max[2] + pad)
        ax.view_init(elev=25, azim=-60)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.set_zlabel("Z (m)", fontsize=9)
        ax.tick_params(labelsize=7)

        info = f"{name}"
        info += f"\n{n_walls}W  {counts['floor']}F  {counts['ceiling']}C  {counts['space']}S"
        info += f"\n{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m"
        if scale:
            info += f"  (scale={scale:.1f})"
        ax.set_title(info, fontsize=12, fontweight="bold", pad=8)

    # Hide empty subplots
    for j in range(n, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out1 = IMAGE_DIR / "runs_comparison_isometric.png"
    fig1.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {out1}")

    # ============================================================
    # Figure 2: Top-down XY plan comparison
    # ============================================================
    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes2 = [axes2]

    fig2.suptitle("GSS Pipeline — IFC Building Geometry (Top-Down Plan)", fontsize=16)

    for i, name in enumerate(names):
        ax = axes2[i]
        meshes = all_data[name]["meshes"]
        scale = all_data[name]["scale"]
        counts, bbox_min, bbox_max, size, n_walls = _get_stats(meshes)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for m in meshes:
            alpha = 0.15 if m["label"] == "space" else 0.5
            for tri in m["faces"]:
                pts = m["verts"][tri]
                xy = pts[:, :2]
                ax.add_patch(Polygon(xy, closed=True, fc=m["color"], ec="gray",
                                     lw=0.3, alpha=alpha))

        margin = max(size[:2]) * 0.1
        ax.set_xlim(bbox_min[0] - margin, bbox_max[0] + margin)
        ax.set_ylim(bbox_min[1] - margin, bbox_max[1] + margin)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=6)

        info = f"{name} — {n_walls}W {counts['space']}S"
        info += f" | {size[0]:.1f}x{size[1]:.1f}m"
        ax.set_title(info, fontsize=10)

    fig2.tight_layout()
    out2 = IMAGE_DIR / "runs_comparison_topdown.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ============================================================
    # Figure 3: Front elevation XZ
    # ============================================================
    fig3, axes3 = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes3 = [axes3]

    fig3.suptitle("GSS Pipeline — IFC Building Geometry (Front Elevation)", fontsize=16)

    for i, name in enumerate(names):
        ax = axes3[i]
        meshes = all_data[name]["meshes"]
        counts, bbox_min, bbox_max, size, n_walls = _get_stats(meshes)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for m in meshes:
            alpha = 0.1 if m["label"] == "space" else 0.5
            for tri in m["faces"]:
                pts = m["verts"][tri]
                xz = pts[:, [0, 2]]
                ax.add_patch(Polygon(xz, closed=True, fc=m["color"], ec="gray",
                                     lw=0.3, alpha=alpha))

        margin_x = size[0] * 0.1
        margin_z = size[2] * 0.2
        ax.set_xlim(bbox_min[0] - margin_x, bbox_max[0] + margin_x)
        ax.set_ylim(bbox_min[2] - margin_z, bbox_max[2] + margin_z)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Z / Height (m)", fontsize=8)
        ax.tick_params(labelsize=6)

        info = f"{name} — Z: {bbox_min[2]:.2f}~{bbox_max[2]:.2f}m"
        ax.set_title(info, fontsize=10)

    fig3.tight_layout()
    out3 = IMAGE_DIR / "runs_comparison_front.png"
    fig3.savefig(str(out3), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out3}")

    # ============================================================
    # Legend
    # ============================================================
    print("\nLegend:")
    print("  Blue   = detected walls")
    print("  Orange = synthetic walls (gap closure)")
    print("  Green  = floor slab")
    print("  Yellow = ceiling slab")
    print("  Gray   = IfcSpace (room boundary)")


if __name__ == "__main__":
    main()
