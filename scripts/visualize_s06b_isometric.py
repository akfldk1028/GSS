"""Visualize s06b results in isometric 3D view.

Renders walls as solid boxes and room space as floor/ceiling surfaces.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_json(path):
    with open(path) as f:
        return json.load(f)


def wall_box_faces(cl, thickness, height_range, normal_axis):
    """Create 6 faces of a wall box from center-line + thickness + height."""
    p1 = np.array(cl[0])  # [x, z] in Manhattan XZ
    p2 = np.array(cl[1])
    y_min, y_max = height_range

    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return []
    d_unit = direction / length
    perp = np.array([-d_unit[1], d_unit[0]])
    half_t = thickness / 2

    c = [
        p1 - perp * half_t,
        p1 + perp * half_t,
        p2 + perp * half_t,
        p2 - perp * half_t,
    ]

    def to3d(xz, y):
        return [xz[0], y, xz[1]]

    vb = [to3d(ci, y_min) for ci in c]
    vt = [to3d(ci, y_max) for ci in c]

    return [
        [vb[0], vb[1], vb[2], vb[3]],  # bottom
        [vt[0], vt[1], vt[2], vt[3]],  # top
        [vb[0], vb[1], vt[1], vt[0]],  # front
        [vb[2], vb[3], vt[3], vt[2]],  # back
        [vb[0], vb[3], vt[3], vt[0]],  # left
        [vb[1], vb[2], vt[2], vt[1]],  # right
    ]


def main():
    data_root = Path("data")
    s06b_dir = data_root / "interim" / "s06b_plane_regularization"

    walls = load_json(s06b_dir / "walls.json")

    spaces_data = []
    if (s06b_dir / "spaces.json").exists():
        spaces_data = load_json(s06b_dir / "spaces.json").get("spaces", [])

    # --- Figure ---
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection="3d")

    # === Draw walls as solid boxes ===
    wall_colors = {
        False: ("#c0392b", "#922b21", 0.55),  # original: red
        True: ("#2980b9", "#1f618d", 0.40),   # synthetic: blue
    }

    for w in walls:
        is_syn = w.get("synthetic", False)
        faces = wall_box_faces(
            w["center_line_2d"], w["thickness"],
            w["height_range"], w["normal_axis"],
        )
        if not faces:
            continue

        color, edge, alpha = wall_colors[is_syn]

        poly = Poly3DCollection(
            faces, alpha=alpha, facecolor=color,
            edgecolor=edge, linewidth=0.6,
        )
        ax.add_collection3d(poly)

        # Label at wall midpoint
        cl = w["center_line_2d"]
        mx = (cl[0][0] + cl[1][0]) / 2
        mz = (cl[0][1] + cl[1][1]) / 2
        my = (w["height_range"][0] + w["height_range"][1]) / 2
        label = f'W{w["id"]}{"*" if is_syn else ""}\nt={w["thickness"]:.1f}'
        ax.text(mx, my, mz, label, fontsize=8, ha="center", va="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85))

    # === Draw room space as floor + ceiling surfaces ===
    for sp in spaces_data:
        bnd = np.array(sp["boundary_2d"])
        floor_h = sp.get("floor_height", 0.0)
        ceiling_h = sp.get("ceiling_height", 3.0)

        # Floor surface (XZ boundary at floor height)
        floor_verts = [[pt[0], floor_h, pt[1]] for pt in bnd[:4]]
        floor_poly = Poly3DCollection(
            [floor_verts], alpha=0.35, facecolor="#27ae60",
            edgecolor="#1e8449", linewidth=1.5,
        )
        ax.add_collection3d(floor_poly)

        # Ceiling surface (XZ boundary at ceiling height)
        ceil_verts = [[pt[0], ceiling_h, pt[1]] for pt in bnd[:4]]
        ceil_poly = Poly3DCollection(
            [ceil_verts], alpha=0.15, facecolor="#8e44ad",
            edgecolor="#6c3483", linewidth=1.0, linestyle="--",
        )
        ax.add_collection3d(ceil_poly)

        # Room label on floor
        cx = bnd[:4, 0].mean()
        cz = bnd[:4, 1].mean()
        ax.text(cx, floor_h + 1.0, cz,
                f'Room {sp["id"]}\n{sp["area"]:.0f} sq.u.',
                fontsize=11, ha="center", va="bottom", fontweight="bold",
                color="#d35400",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # === Draw corner edges (vertical lines at room corners) ===
    if spaces_data:
        sp = spaces_data[0]
        bnd = np.array(sp["boundary_2d"])
        floor_h = sp.get("floor_height", 0.0)
        ceiling_h = sp.get("ceiling_height", 3.0)
        for pt in bnd[:4]:
            ax.plot([pt[0], pt[0]], [floor_h, ceiling_h], [pt[1], pt[1]],
                    "--", color="#7f8c8d", linewidth=0.8, alpha=0.5)

    # === Axis settings ===
    all_x, all_y, all_z = [], [], []
    for w in walls:
        cl = w["center_line_2d"]
        for pt in cl:
            all_x.append(pt[0])
            all_z.append(pt[1])
        all_y.extend(w["height_range"])

    if all_x:
        pad = 3
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_zlim(min(all_z) - pad, max(all_z) + pad)

    ax.set_xlabel("X (Manhattan)", fontsize=10, labelpad=8)
    ax.set_ylabel("Y (Height)", fontsize=10, labelpad=8)
    ax.set_zlabel("Z (Manhattan)", fontsize=10, labelpad=8)

    # Isometric view angle
    ax.view_init(elev=25, azim=-55)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#c0392b", alpha=0.55, label="Original Wall"),
        Patch(facecolor="#2980b9", alpha=0.40, label="Synthetic Wall"),
        Patch(facecolor="#27ae60", alpha=0.35, label="Floor"),
        Patch(facecolor="#8e44ad", alpha=0.15, label="Ceiling"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9,
              framealpha=0.9)

    # Title
    n_orig = sum(1 for w in walls if not w.get("synthetic"))
    n_syn = sum(1 for w in walls if w.get("synthetic"))
    title = (
        f"s06b BIM Reconstruction — Isometric View\n"
        f"{n_orig} original + {n_syn} synthetic walls, "
        f"{len(spaces_data)} room(s)"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()

    out_path = Path("docs/images/s06b_isometric.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
