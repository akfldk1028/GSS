"""Visualize s06b wall regularization: check rectangular fitting quality.

Shows Manhattan-space XZ top-down view with:
- Floor boundary (green)
- Original wall boundaries from s06 (light red, dashed)
- s06b wall center-lines (red solid, with thickness)
- Synthetic walls (blue)
- Corner endpoint markers and gap annotations
- Space polygon (yellow fill)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    data_root = Path("data")
    s06_dir = data_root / "interim" / "s06_planes"
    s06b_dir = data_root / "interim" / "s06b_plane_regularization"

    s06_planes = load_json(s06_dir / "planes.json")
    s06b_planes = load_json(s06b_dir / "planes.json")
    s06b_walls = load_json(s06b_dir / "walls.json")

    R = None
    if (s06_dir / "manhattan_alignment.json").exists():
        R = np.array(load_json(s06_dir / "manhattan_alignment.json")["manhattan_rotation"])

    spaces_data = []
    if (s06b_dir / "spaces.json").exists():
        spaces_data = load_json(s06b_dir / "spaces.json").get("spaces", [])

    # --- Figure: 2x2 layout ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # === Panel 1: s06 original walls in Manhattan XZ ===
    ax1 = axes[0, 0]
    ax1.set_title("① s06 Original (Manhattan XZ)", fontsize=12, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (scene units)")
    ax1.set_ylabel("Z (scene units)")

    for p in s06_planes:
        bnd = p.get("boundary_3d", [])
        if not bnd:
            continue
        pts = np.array(bnd)
        if R is not None:
            pts = pts @ R.T
        color = {"wall": "red", "floor": "green", "ceiling": "blue"}.get(p["label"], "gray")
        if p["label"] in ("wall", "floor"):
            ax1.fill(pts[:, 0], pts[:, 2], alpha=0.15, color=color)
            ax1.plot(pts[:, 0], pts[:, 2], "-", color=color, linewidth=1.5, alpha=0.8)
            cx, cz = pts[:, 0].mean(), pts[:, 2].mean()
            ax1.text(cx, cz, f'{p["label"][0]}{p["id"]}', fontsize=8, ha="center",
                     va="center", color=color, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax1.grid(True, alpha=0.3)

    # === Panel 2: s06b wall center-lines with thickness ===
    ax2 = axes[0, 1]
    ax2.set_title("② s06b Center-lines + Thickness (Manhattan XZ)", fontsize=12, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (scene units)")
    ax2.set_ylabel("Z (scene units)")

    # Floor boundary background
    for p in s06b_planes:
        if p["label"] == "floor" and p.get("boundary_3d"):
            pts = np.array(p["boundary_3d"])
            if R is not None:
                pts = pts @ R.T
            ax2.fill(pts[:, 0], pts[:, 2], alpha=0.1, color="green")
            ax2.plot(pts[:, 0], pts[:, 2], "--", color="green", linewidth=1, alpha=0.5)

    for w in s06b_walls:
        cl = w["center_line_2d"]
        p1, p2 = np.array(cl[0]), np.array(cl[1])
        is_syn = w.get("synthetic", False)
        color = "dodgerblue" if is_syn else "red"
        lw = 3

        # Draw wall with thickness as rectangle
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        d_unit = direction / length
        perp = np.array([-d_unit[1], d_unit[0]])
        half_t = w["thickness"] / 2

        # Rectangle corners
        corners = np.array([
            p1 - perp * half_t,
            p1 + perp * half_t,
            p2 + perp * half_t,
            p2 - perp * half_t,
            p1 - perp * half_t,
        ])
        ax2.fill(corners[:, 0], corners[:, 1], alpha=0.2, color=color)
        ax2.plot(corners[:, 0], corners[:, 1], "-", color=color, linewidth=0.5, alpha=0.5)

        # Center-line
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color, linewidth=lw, alpha=0.9)

        # Endpoints
        ax2.plot(p1[0], p1[1], "o", color=color, markersize=6)
        ax2.plot(p2[0], p2[1], "o", color=color, markersize=6)

        # Label
        mx, mz = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        label = f'w{w["id"]}{"*" if is_syn else ""}\nt={w["thickness"]:.1f}'
        ax2.text(mx + perp[0] * 2, mz + perp[1] * 2, label, fontsize=7,
                 ha="center", va="center", color=color,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax2.grid(True, alpha=0.3)

    # === Panel 3: Corner/gap analysis ===
    ax3 = axes[1, 0]
    ax3.set_title("③ Corner Analysis (gaps & connections)", fontsize=12, fontweight="bold")
    ax3.set_aspect("equal")
    ax3.set_xlabel("X (scene units)")
    ax3.set_ylabel("Z (scene units)")

    # Draw walls
    for w in s06b_walls:
        cl = w["center_line_2d"]
        is_syn = w.get("synthetic", False)
        color = "dodgerblue" if is_syn else "red"
        ax3.plot([cl[0][0], cl[1][0]], [cl[0][1], cl[1][1]], "-o",
                 color=color, linewidth=2.5, markersize=8)

    # Collect all endpoints
    eps = []
    for w in s06b_walls:
        cl = w["center_line_2d"]
        eps.append((f'w{w["id"]}_s', np.array(cl[0]), w["id"]))
        eps.append((f'w{w["id"]}_e', np.array(cl[1]), w["id"]))

    # Find and annotate close pairs / gaps
    for i, (na, pa, _) in enumerate(eps):
        for j, (nb, pb, _) in enumerate(eps):
            if j <= i:
                continue
            d = np.linalg.norm(pa - pb)
            if d < 10.0:
                mid = (pa + pb) / 2
                if d < 0.5:
                    # Connected corner
                    ax3.plot(mid[0], mid[1], "s", color="green", markersize=12, zorder=5)
                    ax3.text(mid[0] + 1, mid[1] + 1, f"✓ d={d:.1f}",
                             fontsize=7, color="green", fontweight="bold")
                else:
                    # Gap
                    ax3.plot([pa[0], pb[0]], [pa[1], pb[1]], "--",
                             color="orange", linewidth=2)
                    ax3.plot(mid[0], mid[1], "^", color="orange", markersize=10, zorder=5)
                    ax3.text(mid[0] + 1, mid[1] + 1, f"GAP={d:.1f}",
                             fontsize=8, color="red", fontweight="bold",
                             bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))

    # Label endpoints
    for name, pt, wid in eps:
        ax3.text(pt[0], pt[1] - 1.5, name, fontsize=6, ha="center",
                 color="gray", alpha=0.7)

    ax3.grid(True, alpha=0.3)

    # === Panel 4: Space detection result ===
    ax4 = axes[1, 1]
    ax4.set_title("④ Space Detection Result", fontsize=12, fontweight="bold")
    ax4.set_aspect("equal")
    ax4.set_xlabel("X (scene units)")
    ax4.set_ylabel("Z (scene units)")

    # Draw walls
    for w in s06b_walls:
        cl = w["center_line_2d"]
        is_syn = w.get("synthetic", False)
        color = "dodgerblue" if is_syn else "red"
        ax4.plot([cl[0][0], cl[1][0]], [cl[0][1], cl[1][1]], "-",
                 color=color, linewidth=3, alpha=0.8)

    # Draw spaces
    if spaces_data:
        for sp in spaces_data:
            bnd = np.array(sp["boundary_2d"])
            ax4.fill(bnd[:, 0], bnd[:, 1], alpha=0.3, color="gold")
            ax4.plot(bnd[:, 0], bnd[:, 1], "-", color="darkorange", linewidth=2)
            cx = bnd[:, 0].mean()
            cz = bnd[:, 1].mean()
            ax4.text(cx, cz, f'Room {sp["id"]}\n{sp["area"]:.0f} sq.u.',
                     fontsize=10, ha="center", va="center", fontweight="bold",
                     color="darkorange",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    else:
        ax4.text(0.5, 0.5, "No spaces detected", transform=ax4.transAxes,
                 fontsize=14, ha="center", va="center", color="red")

    # Floor boundary for reference
    for p in s06b_planes:
        if p["label"] == "floor" and p.get("boundary_3d"):
            pts = np.array(p["boundary_3d"])
            if R is not None:
                pts = pts @ R.T
            ax4.plot(pts[:, 0], pts[:, 2], "--", color="green", linewidth=1, alpha=0.5,
                     label="floor boundary")

    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # --- Summary text ---
    fig.suptitle("s06b Plane Regularization — Rectangular Fitting Verification", fontsize=14, fontweight="bold")

    summary_lines = [
        f"Walls: {len(s06b_walls)} ({sum(1 for w in s06b_walls if not w.get('synthetic'))} original + {sum(1 for w in s06b_walls if w.get('synthetic'))} synthetic)",
        f"Spaces: {len(spaces_data)}",
    ]
    fig.text(0.5, 0.01, "  |  ".join(summary_lines), fontsize=10,
             ha="center", va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_path = Path("docs/images/s06b_rectfit.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
