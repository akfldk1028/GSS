"""Visualize pipeline results: depth maps, point cloud, planes.

Usage:
    python scripts/visualize_results.py           # Interactive 3D viewer
    python scripts/visualize_results.py --no-3d   # Save screenshots only
    python scripts/visualize_results.py --depth    # Depth maps only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def save_depth_images():
    """Save depth maps as color-mapped PNGs using cv2."""
    depth_dir = DATA / "interim" / "s04_depth_maps" / "depth"
    out_dir = DATA / "interim" / "s04_depth_maps" / "depth_vis"
    out_dir.mkdir(exist_ok=True)

    npy_files = sorted(depth_dir.glob("*.npy"))
    if not npy_files:
        print("No depth maps found")
        return

    # Find global min/max for consistent colormap
    all_depths = [np.load(str(f)) for f in npy_files]
    vmin = min(d.min() for d in all_depths)
    vmax = max(d.max() for d in all_depths)

    for f, depth in zip(npy_files, all_depths):
        # Normalize to 0-255
        norm = ((depth - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        out_path = out_dir / f"{f.stem}.png"
        cv2.imwrite(str(out_path), colored)

    # Create grid image
    h, w = all_depths[0].shape
    cols = min(4, len(npy_files))
    rows = (len(npy_files) + cols - 1) // cols
    # Resize for grid
    thumb_w, thumb_h = 400, int(400 * h / w)
    grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, (f, depth) in enumerate(zip(npy_files, all_depths)):
        r, c = divmod(i, cols)
        norm = ((depth - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        thumb = cv2.resize(colored, (thumb_w, thumb_h))
        # Add label
        cv2.putText(thumb, f.stem, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

    grid_path = out_dir / "_depth_grid.png"
    cv2.imwrite(str(grid_path), grid)

    print(f"Saved {len(npy_files)} depth PNGs to {out_dir}")
    print(f"Grid image: {grid_path}")
    print(f"Depth range: {vmin:.2f} - {vmax:.2f}")
    return grid_path


def visualize_pointcloud_and_planes(interactive: bool = True):
    """Visualize TSDF point cloud with plane overlays using Open3D."""
    import open3d as o3d

    pcd_path = DATA / "interim" / "s05_tsdf" / "surface_points.ply"
    if not pcd_path.exists():
        print(f"Point cloud not found: {pcd_path}")
        return

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    n_pts = len(pcd.points)
    print(f"Loaded point cloud: {n_pts} points")

    # Downsample for faster rendering
    if n_pts > 500_000:
        pcd = pcd.voxel_down_sample(0.05)
        print(f"Downsampled to {len(pcd.points)} points")

    # Estimate normals for better rendering
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geometries = [pcd]

    # Load and overlay planes
    planes_path = DATA / "interim" / "s06_planes" / "planes.json"
    bounds_path = DATA / "interim" / "s06_planes" / "boundaries.json"

    if planes_path.exists() and bounds_path.exists():
        with open(planes_path) as f:
            planes = json.load(f)
        with open(bounds_path) as f:
            boundaries = json.load(f)

        label_colors = {
            "wall": [1.0, 0.2, 0.2],
            "floor": [0.2, 1.0, 0.2],
            "ceiling": [0.2, 0.2, 1.0],
            "other": [1.0, 0.8, 0.0],
        }

        for plane, bound in zip(planes, boundaries):
            verts = bound.get("boundary_3d", [])
            if len(verts) < 3:
                continue

            label = plane.get("label", "other")
            color = label_colors.get(label, [0.5, 0.5, 0.5])
            pts = np.array(verts)

            # Triangle fan mesh
            n_v = len(pts)
            triangles = [[0, i, i + 1] for i in range(1, n_v - 1)]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(pts)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.paint_uniform_color(color)
            mesh.compute_vertex_normals()
            geometries.append(mesh)

            # Boundary wireframe
            lines = [[i, (i + 1) % n_v] for i in range(n_v)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.paint_uniform_color([0.0, 0.0, 0.0])
            geometries.append(ls)

        print(f"Overlaid {len(planes)} planes (wall=red, floor=green, ceiling=blue, other=yellow)")

    if interactive:
        print("\nOpening 3D viewer...")
        print("  Mouse: rotate=left, pan=middle, zoom=scroll")
        print("  Keys: R=reset view, Q=quit")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="GSS: Point Cloud + Planes",
            width=1400, height=900,
        )
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1400, height=900)
        for g in geometries:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = 1.5
        opt.background_color = np.array([0.1, 0.1, 0.1])
        vis.poll_events()
        vis.update_renderer()
        out_dir = DATA / "interim" / "visualization"
        out_dir.mkdir(exist_ok=True)
        screenshot = out_dir / "pointcloud_planes.png"
        vis.capture_screen_image(str(screenshot))
        vis.destroy_window()
        print(f"Saved screenshot: {screenshot}")
        return screenshot


def print_plane_summary():
    """Print plane summary table."""
    planes_path = DATA / "interim" / "s06_planes" / "planes.json"
    if not planes_path.exists():
        return

    with open(planes_path) as f:
        planes = json.load(f)

    print(f"\n{'='*70}")
    print(f"{'ID':>3} {'Label':>8} {'Inliers':>8} {'Verts':>6} {'Normal':>30}")
    print(f"{'-'*70}")
    for p in planes:
        n = p["normal"]
        ns = f"[{n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f}]"
        print(f"{p['id']:>3} {p['label']:>8} {p['num_inliers']:>8} {len(p['boundary_3d']):>6} {ns:>30}")

    labels = [p["label"] for p in planes]
    print(f"{'='*70}")
    print(f"Total: {len(planes)} planes - "
          f"{labels.count('wall')} walls, {labels.count('floor')} floors, "
          f"{labels.count('ceiling')} ceilings, {labels.count('other')} other")


def print_viewable_files():
    """List files that can be opened in external viewers."""
    print("\n=== Files you can open in external viewers ===")
    files = {
        "Point cloud (MeshLab/CloudCompare)": DATA / "interim" / "s05_tsdf" / "surface_points.ply",
        "IFC model (FreeCAD/BIM Viewer)": DATA / "processed" / "GSS_Replica_room0.ifc",
        "Depth grid image": DATA / "interim" / "s04_depth_maps" / "depth_vis" / "_depth_grid.png",
    }
    for desc, path in files.items():
        status = "EXISTS" if path.exists() else "NOT FOUND"
        print(f"  [{status}] {desc}")
        print(f"           {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-3d", action="store_true", help="Save screenshot instead of interactive viewer")
    parser.add_argument("--depth", action="store_true", help="Depth maps only")
    args = parser.parse_args()

    print("=== Depth Map Visualization ===")
    save_depth_images()

    if not args.depth:
        print("\n=== Plane Summary ===")
        print_plane_summary()

        print("\n=== 3D Visualization ===")
        visualize_pointcloud_and_planes(interactive=not args.no_3d)

    print_viewable_files()


if __name__ == "__main__":
    main()
