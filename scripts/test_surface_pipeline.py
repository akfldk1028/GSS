"""Quick surface pipeline test using Open3D sample data or Replica.

Tests: s05 TSDF Fusion → s06 Plane Extraction → s07 IFC Export

Usage:
    python scripts/test_surface_pipeline.py                  # Open3D sample
    python scripts/test_surface_pipeline.py --replica room0  # Replica dataset
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_surface_pipeline")


def prepare_open3d_sample() -> Path:
    """Generate GSS-format test data from Open3D sample + synthetic depth."""
    import open3d as o3d

    logger.info("=== Preparing synthetic test data ===")

    output_dir = PROJECT_ROOT / "data" / "interim" / "s04_depth_maps"
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Create a synthetic room: 4 walls + floor + ceiling
    # Room dimensions: 4m x 3m x 2.5m
    width_px, height_px = 640, 480
    fx = fy = 525.0
    cx, cy = 319.5, 239.5

    intrinsics = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width_px, "height": height_px,
    }

    # Generate depth maps from multiple viewpoints inside a box room
    room_w, room_d, room_h = 4.0, 3.0, 2.5
    views = []

    # Camera positions: standing inside room, looking at different walls
    camera_configs = [
        # (position, look_at, up)
        ([2.0, 1.5, 1.2], [4.0, 1.5, 1.2], [0, 0, 1]),   # +X wall
        ([2.0, 1.5, 1.2], [0.0, 1.5, 1.2], [0, 0, 1]),   # -X wall
        ([2.0, 1.5, 1.2], [2.0, 3.0, 1.2], [0, 0, 1]),   # +Y wall
        ([2.0, 1.5, 1.2], [2.0, 0.0, 1.2], [0, 0, 1]),   # -Y wall
        ([2.0, 1.5, 1.2], [2.0, 1.5, 0.0], [0, -1, 0]),  # floor
        ([2.0, 1.5, 1.2], [2.0, 1.5, 2.5], [0, 1, 0]),   # ceiling
        # Corner views
        ([0.5, 0.5, 1.2], [3.5, 2.5, 1.2], [0, 0, 1]),   # diagonal
        ([3.5, 0.5, 1.2], [0.5, 2.5, 1.2], [0, 0, 1]),   # diagonal
        ([0.5, 2.5, 1.2], [3.5, 0.5, 1.2], [0, 0, 1]),   # diagonal
        ([3.5, 2.5, 1.2], [0.5, 0.5, 1.2], [0, 0, 1]),   # diagonal
        # Additional views for better coverage
        ([1.0, 1.5, 1.2], [4.0, 1.5, 1.2], [0, 0, 1]),
        ([3.0, 1.5, 1.2], [0.0, 1.5, 1.2], [0, 0, 1]),
        ([2.0, 0.5, 1.2], [2.0, 3.0, 1.2], [0, 0, 1]),
        ([2.0, 2.5, 1.2], [2.0, 0.0, 1.2], [0, 0, 1]),
        ([1.0, 0.5, 1.2], [3.5, 2.5, 0.5], [0, 0, 1]),
        ([3.0, 2.5, 1.2], [0.5, 0.5, 2.0], [0, 0, 1]),
    ]

    for idx, (eye, target, up) in enumerate(camera_configs):
        eye = np.array(eye, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
        up = np.array(up, dtype=np.float64)

        # Build camera-to-world matrix
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        # c2w: columns are right, cam_up, -forward (OpenGL convention)
        # But Open3D uses: right, -up, forward (OpenCV convention)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = -cam_up  # OpenCV Y is down
        c2w[:3, 2] = forward   # OpenCV Z is forward
        c2w[:3, 3] = eye

        w2c = np.linalg.inv(c2w)

        # Ray-cast depth for a box room [0,room_w] x [0,room_d] x [0,room_h]
        depth_map = render_box_depth(
            w2c, fx, fy, cx, cy, width_px, height_px,
            room_w, room_d, room_h,
        )

        depth_filename = f"depth_{idx:04d}.npy"
        np.save(str(depth_dir / depth_filename), depth_map.astype(np.float32))

        views.append({
            "index": idx,
            "image_name": f"frame_{idx:04d}.png",
            "depth_file": depth_filename,
            "normal_file": None,
            "matrix_4x4": c2w.flatten().tolist(),
        })

    # Write poses.json
    poses_data = {"intrinsics": intrinsics, "views": views}
    poses_file = output_dir / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_data, f, indent=2)

    logger.info(f"Generated {len(views)} synthetic depth maps in {output_dir}")
    return output_dir


def render_box_depth(
    w2c: np.ndarray, fx, fy, cx, cy, W, H,
    room_w, room_d, room_h,
) -> np.ndarray:
    """Ray-cast depth for an axis-aligned box room."""
    depth = np.full((H, W), np.inf, dtype=np.float64)

    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Camera origin in world coords
    cam_pos = -R.T @ t

    # 6 planes of the box: (normal, d) where n.x + d = 0
    planes = [
        (np.array([1, 0, 0]), 0.0),        # x=0
        (np.array([-1, 0, 0]), room_w),     # x=room_w
        (np.array([0, 1, 0]), 0.0),         # y=0
        (np.array([0, -1, 0]), room_d),     # y=room_d
        (np.array([0, 0, 1]), 0.0),         # z=0  (floor)
        (np.array([0, 0, -1]), room_h),     # z=room_h (ceiling)
    ]

    for v in range(H):
        for u in range(W):
            # Ray direction in camera frame
            ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            # Transform to world
            ray_world = R.T @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            for normal, d in planes:
                denom = normal @ ray_world
                if abs(denom) < 1e-8:
                    continue
                t_hit = -(normal @ cam_pos + d) / denom
                if t_hit <= 0.01:
                    continue
                hit_point = cam_pos + t_hit * ray_world
                # Check if hit is inside box bounds (with small margin)
                margin = 0.001
                if (-margin <= hit_point[0] <= room_w + margin and
                    -margin <= hit_point[1] <= room_d + margin and
                    -margin <= hit_point[2] <= room_h + margin):
                    # Camera-space z depth
                    hit_cam = R @ hit_point + t
                    z_depth = hit_cam[2]
                    if 0 < z_depth < depth[v, u]:
                        depth[v, u] = z_depth

    depth[depth == np.inf] = 0.0
    return depth


def run_tsdf(data_root: Path) -> dict:
    """Run Step 05: TSDF Fusion."""
    from gss.steps.s05_tsdf_fusion import TsdfFusionStep, TsdfFusionInput, TsdfFusionConfig

    logger.info("=== Step 05: TSDF Fusion ===")
    config = TsdfFusionConfig(
        voxel_size=0.02,     # 2cm voxels (coarser for speed)
        sdf_trunc=0.06,
        depth_trunc=5.0,
        depth_scale=1.0,
    )
    step = TsdfFusionStep(config=config, data_root=data_root)
    inputs = TsdfFusionInput(
        depth_dir=data_root / "interim" / "s04_depth_maps" / "depth",
        poses_file=data_root / "interim" / "s04_depth_maps" / "poses.json",
    )
    output = step.execute(inputs)
    logger.info(f"  → {output.num_surface_points} surface points")
    return output.model_dump()


def run_planes(data_root: Path) -> dict:
    """Run Step 06: Plane Extraction."""
    from gss.steps.s06_plane_extraction import PlaneExtractionStep, PlaneExtractionInput, PlaneExtractionConfig

    logger.info("=== Step 06: Plane Extraction ===")
    config = PlaneExtractionConfig(
        max_planes=10,
        distance_threshold=0.03,
        min_inliers=100,
        ransac_iterations=1000,
        angle_threshold=15.0,
        simplify_tolerance=0.1,
    )
    step = PlaneExtractionStep(config=config, data_root=data_root)
    inputs = PlaneExtractionInput(
        surface_points_path=data_root / "interim" / "s05_tsdf" / "surface_points.ply",
        metadata_path=data_root / "interim" / "s05_tsdf" / "metadata.json",
    )
    output = step.execute(inputs)
    logger.info(
        f"  → {output.num_planes} planes: "
        f"{output.num_walls} walls, {output.num_floors} floors, {output.num_ceilings} ceilings"
    )
    return output.model_dump()


def run_ifc(data_root: Path) -> dict:
    """Run Step 07: IFC Export."""
    from gss.steps.s07_ifc_export import IfcExportStep, IfcExportInput, IfcExportConfig

    logger.info("=== Step 07: IFC Export ===")
    config = IfcExportConfig(
        ifc_version="IFC4",
        project_name="GSS_Test",
        building_name="TestBuilding",
        default_wall_thickness=0.2,
        default_slab_thickness=0.3,
    )
    step = IfcExportStep(config=config, data_root=data_root)
    s06b_dir = data_root / "interim" / "s06b_plane_regularization"
    walls_file = s06b_dir / "walls.json"
    spaces_file = s06b_dir / "spaces.json"
    if not walls_file.exists():
        raise FileNotFoundError(f"walls.json not found at {walls_file}; run s06b first")
    inputs = IfcExportInput(
        walls_file=walls_file,
        spaces_file=spaces_file if spaces_file.exists() else None,
        planes_file=data_root / "interim" / "s06_planes" / "planes.json",
        boundaries_file=data_root / "interim" / "s06_planes" / "boundaries.json",
    )
    output = step.execute(inputs)
    logger.info(f"  → IFC: {output.num_walls} walls, {output.num_slabs} slabs → {output.ifc_path}")
    return output.model_dump()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", default=None, help="Replica scene name (e.g., room0)")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "data"

    if args.replica:
        # Use Replica data (must run convert_replica.py first)
        s04_dir = data_root / "interim" / "s04_depth_maps"
        if not (s04_dir / "poses.json").exists():
            logger.error(f"Replica data not found. Run: python scripts/convert_replica.py --scene {args.replica}")
            return
        logger.info(f"Using Replica {args.replica} data from {s04_dir}")
    else:
        # Generate synthetic test data
        prepare_open3d_sample()

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Running Surface Pipeline: s05 → s06 → s07")
    logger.info("=" * 60)
    logger.info("")

    try:
        tsdf_result = run_tsdf(data_root)
    except Exception as e:
        logger.error(f"s05 TSDF failed: {e}", exc_info=True)
        return

    try:
        planes_result = run_planes(data_root)
    except Exception as e:
        logger.error(f"s06 Planes failed: {e}", exc_info=True)
        return

    try:
        ifc_result = run_ifc(data_root)
    except Exception as e:
        logger.error(f"s07 IFC failed: {e}", exc_info=True)
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  TSDF points:  {tsdf_result['num_surface_points']}")
    logger.info(f"  Planes:       {planes_result['num_planes']}")
    logger.info(f"  Walls:        {planes_result['num_walls']}")
    logger.info(f"  Floors:       {planes_result['num_floors']}")
    logger.info(f"  Ceilings:     {planes_result['num_ceilings']}")
    logger.info(f"  IFC file:     {ifc_result['ifc_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
