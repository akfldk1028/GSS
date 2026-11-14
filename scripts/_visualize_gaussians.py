"""Visualize 3DGS point clouds for presentation.
Shows raw gaussian splats with color, proving the pipeline input quality.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

RUNS = Path(__file__).resolve().parent.parent / "data" / "runs"
OUT = Path(__file__).resolve().parent.parent / "GSS-RESEARCH" / "GSS-PT" / "260309" / "best_results"
OUT.mkdir(parents=True, exist_ok=True)

SCENES = ["jh", "drjohnson", "playroom", "synthetic_L"]


def load_ply(path: Path, max_points: int = 80000):
    """Load PLY and return points + colors."""
    if not HAS_O3D:
        raise RuntimeError("open3d required")
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Subsample if too many
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    return pts, colors


def visualize_single(scene: str, pts, colors):
    """Create isometric + topdown views of gaussian point cloud."""
    fig = plt.figure(figsize=(16, 7))

    # Isometric 3D view
    ax1 = fig.add_subplot(121, projection="3d")
    if colors is not None:
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.3, alpha=0.6)
    else:
        z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-8)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=plt.cm.viridis(z_norm), s=0.3, alpha=0.6)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"{scene} — 3DGS Point Cloud (Isometric)\n{len(pts):,} points")
    ax1.view_init(elev=25, azim=-60)

    # Top-down XZ view
    ax2 = fig.add_subplot(122)
    if colors is not None:
        ax2.scatter(pts[:, 0], pts[:, 2], c=colors, s=0.3, alpha=0.6)
    else:
        ax2.scatter(pts[:, 0], pts[:, 2], c="steelblue", s=0.3, alpha=0.3)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title(f"{scene} — Top-Down View\n{len(pts):,} points")
    ax2.set_aspect("equal")

    fig.suptitle(f"3D Gaussian Splatting — {scene}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = OUT / f"{scene}_gaussians.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def visualize_grid(all_data):
    """Create 2x2 grid of all scenes."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("3D Gaussian Splatting — Raw Point Clouds (Best Scenes)",
                 fontsize=16, fontweight="bold")

    for i, (scene, pts, colors) in enumerate(all_data):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        if colors is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.2, alpha=0.5)
        else:
            z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-8)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=plt.cm.viridis(z_norm), s=0.2, alpha=0.5)
        ax.set_title(f"{scene}\n{len(pts):,} points", fontsize=12)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    out_path = OUT / "all_gaussians_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")


if __name__ == "__main__":
    all_data = []

    for scene in SCENES:
        ply_path = RUNS / scene / "interim" / "s00_import_ply" / "surface_points.ply"
        if not ply_path.exists():
            print(f"SKIP {scene}: no surface_points.ply")
            continue

        print(f"=== {scene} ===")
        pts, colors = load_ply(ply_path)
        print(f"  Loaded {len(pts):,} points, colors={'yes' if colors is not None else 'no'}")
        visualize_single(scene, pts, colors)
        all_data.append((scene, pts, colors))

    if all_data:
        print("\n=== Grid view ===")
        visualize_grid(all_data)

    print(f"\nDone! {len(all_data)} scenes visualized.")
