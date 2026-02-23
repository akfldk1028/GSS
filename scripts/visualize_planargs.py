"""Visualize PlanarGS output: isometric point cloud view.

Usage:
    python scripts/visualize_planargs.py
    python scripts/visualize_planargs.py --subsample 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_ply_points(ply_path: Path, max_points: int = 500_000) -> np.ndarray:
    """Load vertex positions from a PLY file (binary or ASCII)."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def isometric_plot(pts: np.ndarray, title: str, out_path: Path):
    """Create isometric 3D scatter plot and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Color by height (Y axis typically)
    z = pts[:, 1]
    scatter = ax.scatter(
        pts[:, 0], pts[:, 2], pts[:, 1],
        c=z, cmap="viridis", s=0.05, alpha=0.3,
    )
    plt.colorbar(scatter, ax=ax, shrink=0.5, label="Height")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(title)

    # Isometric-ish view
    ax.view_init(elev=25, azim=45)

    # Equal aspect ratio
    ranges = pts.max(axis=0) - pts.min(axis=0)
    max_range = ranges.max()
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    ax.set_zlim(mid[1] - max_range / 2, mid[1] + max_range / 2)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def top_down_plot(pts: np.ndarray, title: str, out_path: Path):
    """Top-down 2D scatter (floor plan view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], cmap="viridis", s=0.02, alpha=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=300_000,
                        help="Max points to render (default 300K)")
    parser.add_argument("--ply", type=str, default=None,
                        help="Path to PLY file (default: PlanarGS surface_points.ply)")
    args = parser.parse_args()

    ply_path = Path(args.ply) if args.ply else DATA / "interim" / "s03_planargs" / "surface_points.ply"
    if not ply_path.exists():
        print(f"PLY not found: {ply_path}")
        return

    print(f"Loading {ply_path}...")
    pts = load_ply_points(ply_path, max_points=args.subsample)
    print(f"Loaded {len(pts):,} points (subsampled from file)")

    # Stats
    print(f"  X range: [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
    print(f"  Y range: [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")
    print(f"  Z range: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")

    out_dir = DATA / "interim" / "visualization"
    out_dir.mkdir(exist_ok=True)

    # Isometric 3D view
    isometric_plot(pts, f"PlanarGS Replica room0 ({len(pts):,} pts)", out_dir / "planargs_isometric.png")

    # Top-down floor plan
    top_down_plot(pts, f"PlanarGS Top-Down View ({len(pts):,} pts)", out_dir / "planargs_topdown.png")

    # Front view (XY)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="coolwarm", s=0.02, alpha=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (height)")
    ax.set_title(f"PlanarGS Front View ({len(pts):,} pts)")
    ax.set_aspect("equal")
    plt.tight_layout()
    front_path = out_dir / "planargs_front.png"
    plt.savefig(str(front_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {front_path}")

    print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
