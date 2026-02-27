"""Visualize IFC file geometry for round-trip verification.

Modes:
  - Interactive: Open3D window (default)
  - Save: matplotlib renders to PNG (--save)
  - Open: launch system IFC viewer (--open)

Parses actual IFC geometry via ifcopenshell.geom to verify the exported BIM.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_IFC = DATA_ROOT / "processed" / "GSS_BIM.ifc"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

# Colors per IFC class
COLORS = {
    "wall": [0.5, 0.7, 1.0],          # blue
    "wall_synthetic": [1.0, 0.5, 0.3], # orange
    "floor": [0.4, 0.85, 0.4],        # green
    "ceiling": [1.0, 0.9, 0.4],       # yellow
    "space": [0.7, 0.7, 0.7],         # gray
}


def _classify_element(element, z_avg: float | None = None, z_mid: float | None = None) -> str:
    """Classify IFC element into a color category.

    For slabs without Floor/Ceiling in name, uses Z position relative to z_mid.
    """
    ifc_class = element.is_a()
    name = element.Name or ""
    if ifc_class == "IfcWall":
        return "wall_synthetic" if "Synthetic" in name else "wall"
    if ifc_class == "IfcSlab":
        if "Floor" in name:
            return "floor"
        if "Ceiling" in name:
            return "ceiling"
        # Fallback: classify by Z position
        if z_avg is not None and z_mid is not None:
            return "ceiling" if z_avg > z_mid else "floor"
        return "ceiling"
    if ifc_class == "IfcSpace":
        return "space"
    return "wall"


def extract_meshes(ifc_path: Path) -> list[dict]:
    """Extract triangulated meshes from IFC elements.

    Returns list of dicts with keys: verts (Nx3), faces (Mx3), color, label, name.
    """
    import ifcopenshell
    import ifcopenshell.geom

    ifc = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # First pass: extract raw geometry
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

    # Compute Z midpoint for slab classification fallback
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


def _count_elements(meshes: list[dict]) -> dict:
    """Count elements by category."""
    counts = {"wall": 0, "wall_synthetic": 0, "floor": 0, "ceiling": 0, "space": 0}
    for m in meshes:
        counts[m["label"]] = counts.get(m["label"], 0) + 1
    return counts


def _compute_bounds(meshes: list[dict]):
    """Compute overall bounding box across all meshes."""
    all_pts = np.vstack([m["verts"] for m in meshes])
    return all_pts.min(axis=0), all_pts.max(axis=0)


def _build_stats_text(counts: dict, bbox_min, bbox_max) -> list[str]:
    """Build stats lines for overlay."""
    lines = []
    n_walls = counts["wall"] + counts["wall_synthetic"]
    if counts["wall_synthetic"]:
        lines.append(f"IfcWall: {n_walls} ({counts['wall_synthetic']} synthetic)")
    else:
        lines.append(f"IfcWall: {n_walls}")
    lines.append(f"IfcSlab: {counts['floor']} floor + {counts['ceiling']} ceiling")
    if counts["space"]:
        lines.append(f"IfcSpace: {counts['space']}")

    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    dz = bbox_max[2] - bbox_min[2]
    lines.append(f"Bounds: {dx:.1f} x {dy:.1f} x {dz:.1f} m")
    lines.append(f"Z range: {bbox_min[2]:.2f} ~ {bbox_max[2]:.2f} m")
    return lines


# ---------- matplotlib ----------


def save_matplotlib(meshes: list[dict]):
    """Render IFC geometry to 3 PNG files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    counts = _count_elements(meshes)
    bbox_min, bbox_max = _compute_bounds(meshes)
    stats = _build_stats_text(counts, bbox_min, bbox_max)
    stats_text = "\n".join(stats)

    xmin, xmax = bbox_min[0], bbox_max[0]
    ymin, ymax = bbox_min[1], bbox_max[1]
    zmin, zmax = bbox_min[2], bbox_max[2]
    margin = 0.5

    # IFC coords: X,Y = plan, Z = height

    # --- Figure 1: Top-down (XY plan) ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_aspect("equal")
    ax1.set_title("IFC Top-Down (XY Plan)", fontsize=14)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)

    for m in meshes:
        alpha = 0.15 if m["label"] == "space" else 0.5
        for tri in m["faces"]:
            pts = m["verts"][tri]
            xy = pts[:, :2]
            ax1.add_patch(Polygon(xy, closed=True, fc=m["color"], ec="gray",
                                  lw=0.3, alpha=alpha))

    # Dimension annotations
    ax1.annotate(f"{xmax - xmin:.1f}m", xy=((xmin + xmax) / 2, ymin - 0.3),
                 fontsize=11, ha="center", color="red", weight="bold")
    ax1.annotate(f"{ymax - ymin:.1f}m", xy=(xmax + 0.3, (ymin + ymax) / 2),
                 fontsize=11, ha="left", va="center", color="red", weight="bold",
                 rotation=90)

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax1.set_xlim(xmin - margin, xmax + margin)
    ax1.set_ylim(ymin - margin, ymax + margin)
    fig1.tight_layout()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig1.savefig(str(IMAGE_DIR / "ifc_topdown.png"), dpi=150)
    plt.close(fig1)

    # --- Figure 2: Isometric 3D ---
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_title("IFC Geometry — Isometric", fontsize=14)

    for m in meshes:
        alpha = 0.1 if m["label"] == "space" else 0.5
        tris_3d = []
        for tri in m["faces"]:
            pts = m["verts"][tri]
            tris_3d.append(list(zip(pts[:, 0], pts[:, 1], pts[:, 2])))
        ax2.add_collection3d(Poly3DCollection(
            tris_3d, fc=m["color"], ec="gray", lw=0.2, alpha=alpha))

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z / Height (m)")
    ax2.view_init(elev=25, azim=-60)

    pad = 0.5
    ax2.set_xlim(xmin - pad, xmax + pad)
    ax2.set_ylim(ymin - pad, ymax + pad)
    ax2.set_zlim(zmin - pad, zmax + pad)

    fig2.text(0.02, 0.02, stats_text, fontsize=9,
              bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    fig2.tight_layout()
    fig2.savefig(str(IMAGE_DIR / "ifc_isometric.png"), dpi=150)
    plt.close(fig2)

    # --- Figure 3: Front elevation (XZ) ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    ax3.set_aspect("equal")
    ax3.set_title("IFC Front Elevation — X vs Z (Height)", fontsize=14)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z / Height (m)")
    ax3.grid(True, alpha=0.3)

    for m in meshes:
        alpha = 0.1 if m["label"] == "space" else 0.5
        for tri in m["faces"]:
            pts = m["verts"][tri]
            xz = pts[:, [0, 2]]
            ax3.add_patch(Polygon(xz, closed=True, fc=m["color"], ec="gray",
                                  lw=0.3, alpha=alpha))

    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment="top",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax3.set_xlim(xmin - margin, xmax + margin)
    ax3.set_ylim(zmin - 0.5, zmax + 0.5)
    fig3.tight_layout()
    fig3.savefig(str(IMAGE_DIR / "ifc_front.png"), dpi=150)
    plt.close(fig3)

    print(f"\nSaved: {IMAGE_DIR / 'ifc_topdown.png'}")
    print(f"       {IMAGE_DIR / 'ifc_isometric.png'}")
    print(f"       {IMAGE_DIR / 'ifc_front.png'}")


# ---------- Open3D ----------


def show_open3d(meshes: list[dict]):
    """Interactive Open3D visualization of IFC geometry."""
    import open3d as o3d

    geometries = []
    for m in meshes:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(m["verts"])
        mesh.triangles = o3d.utility.Vector3iVector(m["faces"])
        mesh.paint_uniform_color(m["color"])
        mesh.compute_vertex_normals()

        if m["label"] == "space":
            # Make spaces wireframe-only for clarity
            ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            ls.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(ls)
        else:
            geometries.append(mesh)
            # Edge wireframe
            ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            ls.paint_uniform_color([0.1, 0.1, 0.1])
            geometries.append(ls)

    counts = _count_elements(meshes)
    bbox_min, bbox_max = _compute_bounds(meshes)
    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    dz = bbox_max[2] - bbox_min[2]
    n_walls = counts["wall"] + counts["wall_synthetic"]

    title = (f"IFC Viewer — {n_walls} walls, "
             f"{counts['floor']}F/{counts['ceiling']}C slabs, "
             f"{counts['space']} spaces | "
             f"{dx:.1f}x{dy:.1f}x{dz:.1f}m")

    print(f"\nControls: left-drag=rotate, scroll=zoom, middle-drag=pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1400, height=900,
        mesh_show_back_face=True,
    )


# ---------- System viewer ----------


def open_system_viewer(ifc_path: Path):
    """Open IFC file with the system's default application."""
    import os
    import platform

    system = platform.system()
    print(f"\nOpening: {ifc_path}")

    try:
        if system == "Windows":
            os.startfile(str(ifc_path))
        elif system == "Darwin":
            import subprocess
            subprocess.Popen(["open", str(ifc_path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(ifc_path)])
    except OSError:
        print("\nNo application associated with .ifc files.")
        print("Recommended free IFC viewers:")
        print("  - BIM Vision (Windows): https://bimvision.eu/download/")
        print("  - Blender + Bonsai add-on: https://bonsai.community/")
        print("  - IFC.js web viewer: https://ifcjs.io/")


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(description="Visualize IFC file geometry")
    parser.add_argument("--ifc", type=Path, default=DEFAULT_IFC,
                        help="Path to IFC file (default: data/processed/GSS_BIM.ifc)")
    parser.add_argument("--save", action="store_true",
                        help="Save matplotlib PNGs to docs/images/ifc_*.png")
    parser.add_argument("--open", action="store_true",
                        help="Open IFC file with system default application")
    args = parser.parse_args()

    ifc_path = args.ifc.resolve()
    if not ifc_path.exists():
        print(f"Error: IFC file not found: {ifc_path}")
        sys.exit(1)

    if args.open:
        open_system_viewer(ifc_path)
        return

    # Parse IFC geometry
    print(f"Parsing IFC geometry: {ifc_path}")
    meshes = extract_meshes(ifc_path)

    if not meshes:
        print("Error: No geometry found in IFC file")
        sys.exit(1)

    # Print summary
    counts = _count_elements(meshes)
    bbox_min, bbox_max = _compute_bounds(meshes)
    stats = _build_stats_text(counts, bbox_min, bbox_max)

    print(f"\n{'=' * 50}")
    print(f"  IFC Geometry Viewer")
    print(f"{'=' * 50}")
    for line in stats:
        print(f"  {line}")
    print(f"{'=' * 50}")
    print(f"  Blue   = detected walls")
    if counts["wall_synthetic"]:
        print(f"  Orange = synthetic walls")
    print(f"  Green  = floor slab  |  Yellow = ceiling slab")
    if counts["space"]:
        print(f"  Gray   = IfcSpace (wireframe)")
    print(f"{'=' * 50}")

    if args.save:
        save_matplotlib(meshes)
    else:
        show_open3d(meshes)


if __name__ == "__main__":
    main()
