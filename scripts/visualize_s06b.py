"""Visualize s06b plane regularization: before (s06) vs after (s06b) comparison."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

COLORS = {"wall": "red", "floor": "green", "ceiling": "blue", "other": "gray"}


def load_planes(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_walls(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_planes_topdown(ax, planes, title, manhattan_R=None):
    """Plot plane boundaries in top-down (XZ) view.

    If manhattan_R is given, transform boundaries to Manhattan space first.
    """
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    for p in planes:
        bnd = p.get("boundary_3d", [])
        if not bnd:
            continue
        pts = np.array(bnd)
        if manhattan_R is not None:
            pts = pts @ manhattan_R.T
        # Top-down: X vs Z
        ax.plot(pts[:, 0], pts[:, 2], "-", color=COLORS.get(p["label"], "gray"),
                linewidth=2, alpha=0.8)
        cx, cz = pts[:, 0].mean(), pts[:, 2].mean()
        ax.text(cx, cz, f'{p["id"]}', fontsize=7, ha="center", va="center",
                color=COLORS.get(p["label"], "gray"), fontweight="bold")

    ax.grid(True, alpha=0.3)
    # Legend
    for label, color in COLORS.items():
        ax.plot([], [], color=color, linewidth=2, label=label)
    ax.legend(fontsize=8, loc="upper right")


def plot_walls_topdown(ax, walls, title):
    """Plot wall center-lines in XZ plane."""
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    for w in walls:
        cl = w["center_line_2d"]
        xs = [cl[0][0], cl[1][0]]
        zs = [cl[0][1], cl[1][1]]
        color = "darkred" if len(w["plane_ids"]) == 2 else "orangered"
        lw = 3 if len(w["plane_ids"]) == 2 else 1.5
        ax.plot(xs, zs, "-o", color=color, linewidth=lw, markersize=4)
        mx, mz = np.mean(xs), np.mean(zs)
        label = f'w{w["id"]} t={w["thickness"]:.2f}'
        ax.text(mx, mz, label, fontsize=6, ha="center", va="bottom",
                color="darkred")

    ax.grid(True, alpha=0.3)


def main():
    data_root = Path("data")
    s06_dir = data_root / "interim" / "s06_planes"
    s06b_dir = data_root / "interim" / "s06b_plane_regularization"

    if not s06_dir.exists() or not s06b_dir.exists():
        print("Missing s06 or s06b output. Run both steps first.")
        return

    s06_planes = load_planes(s06_dir / "planes.json")
    s06b_planes = load_planes(s06b_dir / "planes.json")
    s06b_walls = load_walls(s06b_dir / "walls.json")

    # Load Manhattan rotation if available
    R = None
    manhattan_path = s06_dir / "manhattan_alignment.json"
    if manhattan_path.exists():
        with open(manhattan_path) as f:
            R = np.array(json.load(f)["manhattan_rotation"])

    # --- Figure 1: Top-down before/after ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_planes_topdown(axes[0], s06_planes, "s06 RANSAC (original)", manhattan_R=R)
    plot_planes_topdown(axes[1], s06b_planes, "s06b Regularized (original)", manhattan_R=R)
    plot_walls_topdown(axes[2], s06b_walls, "s06b Wall Center-lines (Manhattan XZ)")

    fig.suptitle("s06 vs s06b: Plane Regularization Comparison", fontsize=13)
    fig.tight_layout()

    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "s06b_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 's06b_comparison.png'}")

    # --- Figure 2: Normal vectors before/after ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    for ax, planes, title in [
        (axes2[0], s06_planes, "s06 Normals"),
        (axes2[1], s06b_planes, "s06b Normals (snapped)"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("nx")
        ax.set_ylabel("nz")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        for p in planes:
            n = np.array(p["normal"])
            if R is not None:
                n = R @ n
            color = COLORS.get(p["label"], "gray")
            if p["label"] in ("wall",):
                ax.arrow(0, 0, n[0] * 0.9, n[2] * 0.9, head_width=0.04,
                         head_length=0.02, fc=color, ec=color, alpha=0.7)
                ax.text(n[0] * 0.95, n[2] * 0.95, str(p["id"]),
                        fontsize=7, color=color, ha="center")
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Wall Normal Vectors (Manhattan XZ plane)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(out_dir / "s06b_normals.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 's06b_normals.png'}")

    # --- Print summary ---
    print("\n=== Summary ===")
    for label in ("wall", "floor", "ceiling", "other"):
        count_before = sum(1 for p in s06_planes if p["label"] == label)
        count_after = sum(1 for p in s06b_planes if p["label"] == label)
        print(f"  {label}: {count_before} → {count_after}")

    print(f"\n  Walls in walls.json: {len(s06b_walls)}")
    paired = sum(1 for w in s06b_walls if len(w["plane_ids"]) == 2)
    print(f"  Paired (has thickness): {paired}")
    print(f"  Unpaired (default thickness): {len(s06b_walls) - paired}")

    plt.show()


if __name__ == "__main__":
    main()
