"""Visualize s06b regularized planes as 3D surfaces.

Modes:
  - Interactive: Open3D window (default)
  - Save: matplotlib renders to PNG (--save)
  - Cloud overlay: show point cloud with BIM surfaces (--cloud)
  - Coordinate space: --original (COLMAP coords) or default (Manhattan-aligned BIM coords)

Only shows BIM-relevant surfaces: walls, floor, ceiling.
"""

import json
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_planes(stage: str = "s06b_plane_regularization"):
    path = DATA_ROOT / "interim" / stage / "planes.json"
    with open(path) as f:
        return json.load(f)


def load_walls(stage: str = "s06b_plane_regularization"):
    path = DATA_ROOT / "interim" / stage / "walls.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# Colors per label
COLORS = {
    "wall": [0.7, 0.85, 1.0],      # light blue
    "floor": [0.6, 0.9, 0.6],      # green
    "ceiling": [1.0, 0.9, 0.5],    # yellow
}
SYNTH_WALL_COLOR = [1.0, 0.5, 0.3]  # orange for synthetic walls


def _deduplicate_walls(walls):
    """Remove synthetic walls that overlap with detected walls."""
    detected = [w for w in walls if not w.get("synthetic")]
    synthetic = [w for w in walls if w.get("synthetic")]

    result = list(detected)
    for sw in synthetic:
        s_cl = sw.get("center_line_2d")
        if not s_cl or len(s_cl) != 2:
            result.append(sw)
            continue
        s0, s1 = np.array(s_cl[0]), np.array(s_cl[1])

        is_dup = False
        for dw in detected:
            d_cl = dw.get("center_line_2d")
            if not d_cl or len(d_cl) != 2:
                continue
            d0, d1 = np.array(d_cl[0]), np.array(d_cl[1])
            fwd = np.linalg.norm(s0 - d0) + np.linalg.norm(s1 - d1)
            rev = np.linalg.norm(s0 - d1) + np.linalg.norm(s1 - d0)
            if min(fwd, rev) < 0.5:
                is_dup = True
                break
        if not is_dup:
            result.append(sw)
    return result


def _get_wall_quads_manhattan(walls, floor_h, ceil_h):
    """Build 3D wall quads from Manhattan center_line_2d (axis-aligned BIM coords).

    Manhattan space: X, Z = horizontal (room plan), Y = vertical (height).
    center_line_2d = [[x1, z1], [x2, z2]] in Manhattan XZ plane.
    Y axis is preserved by the Manhattan rotation, so floor_h/ceil_h work directly.
    """
    quads = []
    for w in walls:
        cl = w.get("center_line_2d")
        if not cl or len(cl) != 2:
            continue
        p0, p1 = cl[0], cl[1]
        quad = np.array([
            [p0[0], floor_h, p0[1]],
            [p1[0], floor_h, p1[1]],
            [p1[0], ceil_h, p1[1]],
            [p0[0], ceil_h, p0[1]],
        ])
        is_synth = w.get("synthetic", False)
        quads.append((quad, is_synth, w["id"]))
    return quads


def _get_wall_quads_original(walls, floor_h, ceil_h):
    """Build 3D wall quads from original center_line_3d."""
    quads = []
    for w in walls:
        cl3d = w.get("center_line_3d")
        if cl3d and len(cl3d) == 2:
            p0 = np.array(cl3d[0])
            p1 = np.array(cl3d[1])
            quad = np.array([
                [p0[0], floor_h, p0[2]],
                [p1[0], floor_h, p1[2]],
                [p1[0], ceil_h, p1[2]],
                [p0[0], ceil_h, p0[2]],
            ])
            is_synth = w.get("synthetic", False)
            quads.append((quad, is_synth, w["id"]))
    return quads


def _wall_corners_xz(walls, use_manhattan=True):
    """Extract all wall endpoint XZ coordinates."""
    pts = []
    for w in walls:
        if use_manhattan:
            cl = w.get("center_line_2d")
            if cl and len(cl) == 2:
                pts.append(cl[0])
                pts.append(cl[1])
        else:
            cl3d = w.get("center_line_3d")
            if cl3d and len(cl3d) == 2:
                pts.append([cl3d[0][0], cl3d[0][2]])
                pts.append([cl3d[1][0], cl3d[1][2]])
    return np.array(pts) if pts else None


def _floor_ceil_quad(wall_xz, y_val):
    """Build floor/ceiling quad from wall XZ AABB."""
    xmin, zmin = wall_xz.min(axis=0)
    xmax, zmax = wall_xz.max(axis=0)
    return np.array([
        [xmin, y_val, zmin],
        [xmax, y_val, zmin],
        [xmax, y_val, zmax],
        [xmin, y_val, zmax],
    ])


def _build_stats_text(n_walls, n_synth, floor_h, ceil_h, wall_xz):
    """Build stats string for display."""
    lines = []
    lines.append(f"Walls: {n_walls} ({n_synth} synthetic)" if n_synth else f"Walls: {n_walls}")
    lines.append(f"Floor: {floor_h:.2f}m  |  Ceiling: {ceil_h:.2f}m")
    lines.append(f"Room height: {ceil_h - floor_h:.2f}m")
    if wall_xz is not None:
        dx = wall_xz[:, 0].max() - wall_xz[:, 0].min()
        dz = wall_xz[:, 1].max() - wall_xz[:, 1].min()
        lines.append(f"Room size: {dx:.1f} x {dz:.1f} x {ceil_h - floor_h:.1f} m")
    return lines


def save_matplotlib(walls, floor_h, ceil_h, wall_xz, use_manhattan=True):
    """Render BIM surfaces with matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if use_manhattan:
        wall_quads = _get_wall_quads_manhattan(walls, floor_h, ceil_h)
        coord_label = "Manhattan-aligned BIM"
    else:
        wall_quads = _get_wall_quads_original(walls, floor_h, ceil_h)
        coord_label = "Original COLMAP"

    floor_quad = _floor_ceil_quad(wall_xz, floor_h) if wall_xz is not None else None
    ceil_quad = _floor_ceil_quad(wall_xz, ceil_h) if wall_xz is not None else None

    n_walls = len(walls)
    n_synth = sum(1 for w in walls if w.get("synthetic"))
    stats = _build_stats_text(n_walls, n_synth, floor_h, ceil_h, wall_xz)

    xmin, xmax = wall_xz[:, 0].min(), wall_xz[:, 0].max()
    zmin, zmax = wall_xz[:, 1].min(), wall_xz[:, 1].max()

    # --- Figure 1: Top-down 2D floor plan ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_aspect("equal")
    ax1.set_title(f"Top-Down Floor Plan ({coord_label})", fontsize=14)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.grid(True, alpha=0.3)

    # Floor
    if floor_quad is not None:
        floor_xz = floor_quad[:, [0, 2]]
        ax1.add_patch(Polygon(floor_xz, closed=True, fc=COLORS["floor"], ec="black",
                              lw=1.5, alpha=0.3, label="Floor"))

    # Walls as thick lines
    for quad, is_synth, wid in wall_quads:
        color = SYNTH_WALL_COLOR if is_synth else COLORS["wall"]
        x = [quad[0, 0], quad[1, 0]]
        z = [quad[0, 2], quad[1, 2]]
        ax1.plot(x, z, color=color, linewidth=5, solid_capstyle="round")
        mx, mz = (x[0] + x[1]) / 2, (z[0] + z[1]) / 2
        ax1.annotate(f"W{wid}", (mx, mz), fontsize=9, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    # Corner dots
    for quad, _, _ in wall_quads:
        ax1.plot(quad[0, 0], quad[0, 2], "ko", markersize=5)
        ax1.plot(quad[1, 0], quad[1, 2], "ko", markersize=5)

    # Dimensions
    ax1.annotate(f"{xmax - xmin:.1f}m", xy=((xmin + xmax) / 2, zmin - 0.3),
                 fontsize=11, ha="center", color="red", weight="bold")
    ax1.annotate(f"{zmax - zmin:.1f}m", xy=(xmax + 0.3, (zmin + zmax) / 2),
                 fontsize=11, ha="left", va="center", color="red", weight="bold", rotation=90)

    # Stats box
    stats_text = "\n".join(stats)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    margin = 1.0
    ax1.set_xlim(xmin - margin, xmax + margin)
    ax1.set_ylim(zmin - margin, zmax + margin)

    fig1.tight_layout()
    fig1.savefig("docs/images/bim_topdown.png", dpi=150)
    plt.close(fig1)

    # --- Figure 2: 3D isometric view ---
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_title(f"BIM Surfaces — Isometric ({coord_label})", fontsize=14)

    if floor_quad is not None:
        verts = [list(zip(floor_quad[:, 0], floor_quad[:, 2], floor_quad[:, 1]))]
        ax2.add_collection3d(Poly3DCollection(verts, fc=COLORS["floor"], ec="black", lw=1, alpha=0.5))

    if ceil_quad is not None:
        verts = [list(zip(ceil_quad[:, 0], ceil_quad[:, 2], ceil_quad[:, 1]))]
        ax2.add_collection3d(Poly3DCollection(verts, fc=COLORS["ceiling"], ec="black", lw=1, alpha=0.3))

    for quad, is_synth, wid in wall_quads:
        color = SYNTH_WALL_COLOR if is_synth else COLORS["wall"]
        verts = [list(zip(quad[:, 0], quad[:, 2], quad[:, 1]))]
        ax2.add_collection3d(Poly3DCollection(verts, fc=color, ec="black", lw=0.8, alpha=0.6))

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_zlabel("Y / Height (m)")
    ax2.view_init(elev=25, azim=-60)

    pad = 0.5
    ax2.set_xlim(xmin - pad, xmax + pad)
    ax2.set_ylim(zmin - pad, zmax + pad)
    ax2.set_zlim(floor_h - pad, ceil_h + pad)

    # Stats text
    fig2.text(0.02, 0.02, stats_text, fontsize=9,
              bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    fig2.tight_layout()
    fig2.savefig("docs/images/bim_isometric.png", dpi=150)
    plt.close(fig2)

    # --- Figure 3: Front elevation ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    ax3.set_aspect("equal")
    ax3.set_title(f"Front Elevation — X vs Y ({coord_label})", fontsize=14)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y / Height (m)")
    ax3.grid(True, alpha=0.3)

    for quad, is_synth, wid in wall_quads:
        color = SYNTH_WALL_COLOR if is_synth else COLORS["wall"]
        xy = quad[:, [0, 1]]
        ax3.add_patch(Polygon(xy, closed=True, fc=color, ec="black", lw=1, alpha=0.5))

    ax3.axhline(y=floor_h, color="green", linestyle="--", linewidth=2, label="Floor")
    ax3.axhline(y=ceil_h, color="goldenrod", linestyle="--", linewidth=2, label="Ceiling")
    ax3.legend()

    # Stats
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment="top", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    ax3.set_xlim(xmin - margin, xmax + margin)
    ax3.set_ylim(floor_h - 0.5, ceil_h + 0.5)

    fig3.tight_layout()
    fig3.savefig("docs/images/bim_front.png", dpi=150)
    plt.close(fig3)

    print("\nSaved: docs/images/bim_topdown.png, bim_isometric.png, bim_front.png")


def _load_point_cloud(voxel_size=0.05):
    """Load and downsample point cloud from s00 output."""
    import open3d as o3d
    ply_path = DATA_ROOT / "interim" / "s00_import_ply" / "surface_points.ply"
    if not ply_path.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    return pcd


def show_open3d(walls, floor_h, ceil_h, wall_xz, use_manhattan=True, show_cloud=False):
    """Interactive Open3D visualization."""
    import open3d as o3d

    geometries = []

    # Point cloud overlay (always use original coords)
    if show_cloud:
        pcd = _load_point_cloud()
        if pcd is not None:
            geometries.append(pcd)
            print(f"  Point cloud: {len(pcd.points)} points (downsampled)")
            # When showing cloud, force original coords for alignment
            use_manhattan = False

    if use_manhattan:
        wall_quads = _get_wall_quads_manhattan(walls, floor_h, ceil_h)
        wall_xz_for_floor = _wall_corners_xz(walls, use_manhattan=True)
    else:
        wall_quads = _get_wall_quads_original(walls, floor_h, ceil_h)
        wall_xz_for_floor = _wall_corners_xz(walls, use_manhattan=False)

    if wall_xz_for_floor is None:
        wall_xz_for_floor = wall_xz

    def _make_rect_mesh(pts_4, color):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(pts_4, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh

    # Floor
    if wall_xz_for_floor is not None:
        fq = _floor_ceil_quad(wall_xz_for_floor, floor_h)
        mesh = _make_rect_mesh(fq, COLORS["floor"])
        geometries.append(mesh)
        ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        ls.paint_uniform_color([0.1, 0.1, 0.1])
        geometries.append(ls)

    # Ceiling
    if wall_xz_for_floor is not None:
        cq = _floor_ceil_quad(wall_xz_for_floor, ceil_h)
        mesh = _make_rect_mesh(cq, COLORS["ceiling"])
        geometries.append(mesh)
        ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        ls.paint_uniform_color([0.1, 0.1, 0.1])
        geometries.append(ls)

    # Walls (brighter when cloud overlay)
    wall_color = [0.3, 0.5, 1.0] if show_cloud else COLORS["wall"]
    synth_color = [1.0, 0.4, 0.2] if show_cloud else SYNTH_WALL_COLOR
    for quad, is_synth, wid in wall_quads:
        color = synth_color if is_synth else wall_color
        mesh = _make_rect_mesh(quad, color)
        geometries.append(mesh)
        ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        ls.paint_uniform_color([0, 0, 0])
        geometries.append(ls)

    n_walls = len(walls)
    n_synth = sum(1 for w in walls if w.get("synthetic"))
    dx = wall_xz[:, 0].max() - wall_xz[:, 0].min() if wall_xz is not None else 0
    dz = wall_xz[:, 1].max() - wall_xz[:, 1].min() if wall_xz is not None else 0
    cloud_tag = " + PointCloud" if show_cloud else ""
    title = (f"GSS BIM{cloud_tag} — {n_walls} walls | "
             f"{dx:.1f}x{dz:.1f}x{ceil_h - floor_h:.1f}m | "
             f"Floor {floor_h:.1f}m Ceil {ceil_h:.1f}m")

    print(f"\nControls: left-drag=rotate, scroll=zoom, middle-drag=pan")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1400, height=900,
        mesh_show_back_face=True,
    )


def main(save_images=False, use_original=False, show_cloud=False):
    planes = load_planes()
    walls = load_walls()

    walls = _deduplicate_walls(walls)

    # Heights from plane data (Y is preserved by Manhattan rotation)
    floor_h, ceil_h = None, None
    for p in planes:
        b = p.get("boundary_3d", [])
        if not b or len(b) < 3:
            continue
        pts = np.array(b)
        if p["label"] == "floor":
            floor_h = np.mean(pts[:, 1])
        elif p["label"] == "ceiling":
            ceil_h = np.mean(pts[:, 1])

    if floor_h is None:
        floor_h = -1.0
    if ceil_h is None:
        ceil_h = 3.0

    # Cloud mode forces original coords for point cloud alignment
    use_manhattan = not use_original and not show_cloud
    wall_xz = _wall_corners_xz(walls, use_manhattan=use_manhattan)

    # Print info
    n_walls = len(walls)
    n_synth = sum(1 for w in walls if w.get("synthetic"))
    coord_mode = "Manhattan-aligned BIM" if use_manhattan else "Original COLMAP"
    print(f"\n{'=' * 50}")
    print(f"  GSS BIM Visualization ({coord_mode})")
    print(f"{'=' * 50}")
    print(f"  Walls: {n_walls} ({n_synth} synthetic)")
    print(f"  Floor: {floor_h:.2f}m  |  Ceiling: {ceil_h:.2f}m")
    print(f"  Room height: {ceil_h - floor_h:.2f}m")
    if wall_xz is not None:
        dx = wall_xz[:, 0].max() - wall_xz[:, 0].min()
        dz = wall_xz[:, 1].max() - wall_xz[:, 1].min()
        print(f"  Room size: {dx:.1f} x {dz:.1f} x {ceil_h - floor_h:.1f} m")
    print(f"{'=' * 50}")
    print(f"  Blue   = detected walls")
    if n_synth > 0:
        print(f"  Orange = synthetic walls (closure)")
    print(f"  Green  = floor  |  Yellow = ceiling")
    if show_cloud:
        print(f"  Gray   = point cloud")
    print(f"{'=' * 50}")

    if save_images:
        save_matplotlib(walls, floor_h, ceil_h, wall_xz, use_manhattan=use_manhattan)
    else:
        show_open3d(walls, floor_h, ceil_h, wall_xz,
                    use_manhattan=use_manhattan, show_cloud=show_cloud)


if __name__ == "__main__":
    save = "--save" in sys.argv
    use_original = "--original" in sys.argv
    show_cloud = "--cloud" in sys.argv
    main(save_images=save, use_original=use_original, show_cloud=show_cloud)
