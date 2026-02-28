"""Visualize exported mesh files (GLB / USD) for round-trip verification.

Modes:
  - Interactive: Open3D window (default)
  - Save: matplotlib renders to PNG (--save)

Supports GLB (trimesh) and USDC/USDZ (pxr) file formats.
Validates coordinate transforms (Z-up vs Y-up) visually.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_GLB = DATA_ROOT / "processed" / "GSS_BIM.glb"
DEFAULT_USD = DATA_ROOT / "processed" / "GSS_BIM.usdc"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"


def load_glb(path: Path) -> list[dict]:
    """Load meshes from a GLB file via trimesh."""
    import trimesh

    scene = trimesh.load(str(path))
    meshes = []
    for name, geom in scene.geometry.items():
        verts = np.array(geom.vertices, dtype=np.float64)
        faces = np.array(geom.faces, dtype=np.int32)
        # Extract color from face_colors or visual
        if hasattr(geom.visual, "face_colors") and len(geom.visual.face_colors) > 0:
            fc = geom.visual.face_colors[0][:3] / 255.0
            color = fc.tolist()
        else:
            color = [0.8, 0.8, 0.8]
        meshes.append({
            "verts": verts,
            "faces": faces,
            "color": color,
            "name": name,
        })
    return meshes


def load_usd(path: Path) -> list[dict]:
    """Load meshes from a USDC/USDZ file via pxr."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(path))
    up_axis = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  USD up_axis: {up_axis}, meters_per_unit: {mpu}")

    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()

        if not pts or not fvi:
            continue

        verts = np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float64)

        # Reconstruct triangle faces from face vertex counts/indices
        faces = []
        idx = 0
        for count in fvc:
            if count == 3:
                faces.append([fvi[idx], fvi[idx + 1], fvi[idx + 2]])
            elif count == 4:
                # Triangulate quad
                faces.append([fvi[idx], fvi[idx + 1], fvi[idx + 2]])
                faces.append([fvi[idx], fvi[idx + 2], fvi[idx + 3]])
            idx += count
        faces = np.array(faces, dtype=np.int32)

        # Extract material color
        color = [0.8, 0.8, 0.8]
        binding_api = UsdShade.MaterialBindingAPI(prim)
        mat, _ = binding_api.ComputeBoundMaterial()
        if mat:
            for child in mat.GetPrim().GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)
                    dc = shader.GetInput("diffuseColor")
                    if dc:
                        val = dc.Get()
                        if val:
                            color = [val[0], val[1], val[2]]

        meshes.append({
            "verts": verts,
            "faces": faces,
            "color": color,
            "name": prim.GetName(),
        })
    return meshes


def _compute_bounds(meshes: list[dict]):
    """Compute overall bounding box."""
    all_pts = np.vstack([m["verts"] for m in meshes])
    return all_pts.min(axis=0), all_pts.max(axis=0)


def _build_stats(meshes: list[dict], file_path: Path, bbox_min, bbox_max) -> list[str]:
    """Build stats overlay text."""
    total_verts = sum(len(m["verts"]) for m in meshes)
    total_faces = sum(len(m["faces"]) for m in meshes)
    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    dz = bbox_max[2] - bbox_min[2]

    lines = [
        f"File: {file_path.name}",
        f"Meshes: {len(meshes)} ({total_verts} verts, {total_faces} faces)",
        f"Bounds: {dx:.2f} x {dy:.2f} x {dz:.2f} m",
        f"X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]",
        f"Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]",
        f"Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]",
    ]
    return lines


def save_matplotlib(meshes: list[dict], file_path: Path, suffix: str = ""):
    """Render mesh to PNG files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    bbox_min, bbox_max = _compute_bounds(meshes)
    stats = _build_stats(meshes, file_path, bbox_min, bbox_max)
    stats_text = "\n".join(stats)
    margin = 0.5

    xmin, xmax = bbox_min[0], bbox_max[0]
    ymin, ymax = bbox_min[1], bbox_max[1]
    zmin, zmax = bbox_min[2], bbox_max[2]

    # Include extension in tag to distinguish GLB from USD
    ext = file_path.suffix.lstrip(".").lower()
    name_tag = f"{file_path.stem}_{ext}{suffix}"

    # --- Figure 1: Isometric 3D ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Mesh: {file_path.name} — Isometric", fontsize=14)

    for m in meshes:
        tris_3d = []
        for tri in m["faces"]:
            pts = m["verts"][tri]
            tris_3d.append(list(zip(pts[:, 0], pts[:, 1], pts[:, 2])))
        ax.add_collection3d(Poly3DCollection(
            tris_3d, fc=m["color"], ec="gray", lw=0.2, alpha=0.5))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-60)

    pad = 0.5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_zlim(zmin - pad, zmax + pad)

    fig.text(0.02, 0.02, stats_text, fontsize=9,
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    fig.tight_layout()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = IMAGE_DIR / f"mesh_{name_tag}_isometric.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # --- Figure 2: Top-down (XZ or XY depending on up-axis) ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.set_aspect("equal")
    ax2.set_title(f"Mesh: {file_path.name} — Top-Down", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Determine horizontal plane for top-down view
    # If height is in Y (Y-up): top-down = XZ plane
    # If height is in Z (Z-up): top-down = XY plane
    y_range = ymax - ymin
    z_range = zmax - zmin
    if y_range > z_range:
        # Z-up: height in Z, plan = XY
        h_ax, v_ax = 0, 1
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        h_min, h_max = xmin, xmax
        v_min, v_max = ymin, ymax
    else:
        # Y-up: height in Y, plan = XZ
        h_ax, v_ax = 0, 2
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        h_min, h_max = xmin, xmax
        v_min, v_max = zmin, zmax

    for m in meshes:
        for tri in m["faces"]:
            pts = m["verts"][tri]
            xy = pts[:, [h_ax, v_ax]]
            ax2.add_patch(Polygon(xy, closed=True, fc=m["color"], ec="gray",
                                  lw=0.3, alpha=0.5))

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax2.set_xlim(h_min - margin, h_max + margin)
    ax2.set_ylim(v_min - margin, v_max + margin)
    fig2.tight_layout()

    out2 = IMAGE_DIR / f"mesh_{name_tag}_topdown.png"
    fig2.savefig(str(out2), dpi=150)
    plt.close(fig2)
    print(f"Saved: {out2}")


def show_open3d(meshes: list[dict], file_path: Path):
    """Interactive Open3D visualization."""
    import open3d as o3d

    geometries = []
    for m in meshes:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(m["verts"])
        mesh.triangles = o3d.utility.Vector3iVector(m["faces"])
        mesh.paint_uniform_color(m["color"])
        mesh.compute_vertex_normals()
        geometries.append(mesh)

        ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        ls.paint_uniform_color([0.1, 0.1, 0.1])
        geometries.append(ls)

    bbox_min, bbox_max = _compute_bounds(meshes)
    total_verts = sum(len(m["verts"]) for m in meshes)
    total_faces = sum(len(m["faces"]) for m in meshes)
    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    dz = bbox_max[2] - bbox_min[2]

    title = (f"{file_path.name} — {len(meshes)} meshes, "
             f"{total_verts}v/{total_faces}f | "
             f"{dx:.1f}x{dy:.1f}x{dz:.1f}m")

    print("\nControls: left-drag=rotate, scroll=zoom, middle-drag=pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1400, height=900,
        mesh_show_back_face=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize exported mesh files (GLB / USD)")
    parser.add_argument("file", type=Path, nargs="?",
                        help="Path to .glb / .usdc / .usdz file")
    parser.add_argument("--save", action="store_true",
                        help="Save matplotlib PNGs to docs/images/")
    parser.add_argument("--compare", action="store_true",
                        help="Compare GLB and USD side by side (auto-detect files)")
    args = parser.parse_args()

    if args.compare:
        # Compare mode: load both GLB and USD
        files = []
        for f in [DEFAULT_GLB, DEFAULT_USD]:
            if f.exists():
                files.append(f)
        if not files:
            print("Error: No GLB/USD files found in data/processed/")
            sys.exit(1)

        for f in files:
            print(f"\n{'=' * 50}")
            print(f"  {f.name}")
            print(f"{'=' * 50}")

            if f.suffix == ".glb":
                meshes = load_glb(f)
            else:
                meshes = load_usd(f)

            bbox_min, bbox_max = _compute_bounds(meshes)
            stats = _build_stats(meshes, f, bbox_min, bbox_max)
            for line in stats:
                print(f"  {line}")

            if args.save:
                save_matplotlib(meshes, f)
            else:
                show_open3d(meshes, f)
        return

    # Single file mode
    file_path = args.file
    if file_path is None:
        # Auto-detect: prefer GLB
        if DEFAULT_GLB.exists():
            file_path = DEFAULT_GLB
        elif DEFAULT_USD.exists():
            file_path = DEFAULT_USD
        else:
            print("Error: No mesh file found. Specify path or run s08_mesh_export first.")
            sys.exit(1)

    file_path = file_path.resolve()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".glb":
        meshes = load_glb(file_path)
    elif suffix in (".usdc", ".usd", ".usdz"):
        meshes = load_usd(file_path)
    else:
        print(f"Error: Unsupported format: {suffix}")
        sys.exit(1)

    if not meshes:
        print("Error: No meshes loaded")
        sys.exit(1)

    bbox_min, bbox_max = _compute_bounds(meshes)
    stats = _build_stats(meshes, file_path, bbox_min, bbox_max)

    print(f"\n{'=' * 50}")
    for line in stats:
        print(f"  {line}")
    print(f"{'=' * 50}")

    if args.save:
        save_matplotlib(meshes, file_path)
    else:
        show_open3d(meshes, file_path)


if __name__ == "__main__":
    main()
