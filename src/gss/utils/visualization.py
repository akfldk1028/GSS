"""Visualization utilities for pipeline debugging."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    title: str = "Point Cloud",
    max_points: int = 50000,
    save_path: Path | None = None,
):
    """Plot 3D point cloud with matplotlib."""
    import matplotlib.pyplot as plt

    if len(points) > max_points:
        indices = np.random.default_rng(42).choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
    return fig


def plot_depth_map(
    depth: np.ndarray,
    title: str = "Depth Map",
    save_path: Path | None = None,
):
    """Plot depth map as heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(depth, cmap="turbo")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Depth (m)")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
    return fig


def plot_planes(
    planes: list[dict],
    boundaries: list[dict],
    title: str = "Detected Planes",
    save_path: Path | None = None,
):
    """Plot detected planes and boundaries in 3D."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    label_colors = {"wall": "blue", "floor": "green", "ceiling": "red", "other": "gray"}

    for boundary in boundaries:
        pts = np.array(boundary.get("boundary_3d", []))
        if len(pts) < 3:
            continue
        label = boundary.get("label", "other")
        color = label_colors.get(label, "gray")
        # Close the polygon
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linewidth=1.5,
                label=label if label not in ax.get_legend_handles_labels()[1] else "")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
    return fig
