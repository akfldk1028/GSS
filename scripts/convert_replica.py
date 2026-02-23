"""Convert Replica (NICE-SLAM version) to GSS s04 output format.

Usage:
    python scripts/convert_replica.py --scene room0
    python scripts/convert_replica.py --scene room0 --max-frames 200 --stride 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Replica NICE-SLAM intrinsics (shared across all scenes)
REPLICA_INTRINSICS = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 599.5,
    "cy": 339.5,
    "width": 1200,
    "height": 680,
}


def read_pose(path: Path) -> np.ndarray:
    """Read a 4x4 camera-to-world matrix from a NICE-SLAM pose .txt file."""
    return np.loadtxt(str(path)).reshape(4, 4)


def convert_depth(depth_png_path: Path) -> np.ndarray:
    """Convert Replica depth PNG to meters.

    Replica NICE-SLAM depth: 16-bit PNG, value / 6553.5 = meters.
    """
    depth_raw = cv2.imread(str(depth_png_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read: {depth_png_path}")
    return (depth_raw.astype(np.float32) / 6553.5)


def main():
    parser = argparse.ArgumentParser(description="Convert Replica to GSS format")
    parser.add_argument("--scene", default="room0", help="Scene name (e.g., room0, office0)")
    parser.add_argument("--replica-root", default=None, help="Path to extracted Replica dir")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to convert")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride (1=all, 10=every 10th)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    replica_root = Path(args.replica_root) if args.replica_root else project_root / "data" / "test" / "replica" / "Replica"
    scene_dir = replica_root / args.scene

    if not scene_dir.exists():
        logger.error(f"Scene not found: {scene_dir}")
        return

    results_dir = scene_dir / "results"
    if not results_dir.exists():
        logger.error(f"Results dir not found: {results_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "interim" / "s04_depth_maps"
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Find all frame indices
    depth_files = sorted(results_dir.glob("depth*.png"))
    pose_files = sorted(results_dir.glob("frame*.txt"))

    if not depth_files:
        # Try alternative naming: just numbered files
        depth_files = sorted(results_dir.glob("depth/*.png"))

    # Detect naming convention
    # NICE-SLAM Replica: results/frame000000.jpg, results/depth000000.png, results/frame000000.txt (pose)
    # Or: results/ with depth{N}.png and {N}.txt
    logger.info(f"Found {len(depth_files)} depth files, {len(pose_files)} pose files")

    if not depth_files or not pose_files:
        # Try NICE-SLAM traj.txt format
        traj_file = scene_dir / "traj.txt"
        if traj_file.exists():
            logger.info("Found traj.txt, using trajectory file format")
            return convert_with_traj(scene_dir, traj_file, output_dir, depth_dir, args)

        logger.error("No depth/pose files found. Check directory structure.")
        logger.info(f"Contents of {results_dir}:")
        for p in sorted(results_dir.iterdir())[:20]:
            logger.info(f"  {p.name}")
        return

    # Select frames with stride
    all_indices = list(range(0, len(depth_files), args.stride))[:args.max_frames]
    logger.info(f"Converting {len(all_indices)} frames (stride={args.stride})")

    views = []
    for count, idx in enumerate(all_indices):
        depth_src = depth_files[idx]
        # Find matching pose file
        frame_num = int("".join(filter(str.isdigit, depth_src.stem)))
        pose_candidates = [
            results_dir / f"frame{frame_num:06d}.txt",
            results_dir / f"{frame_num}.txt",
        ]
        pose_src = None
        for pc in pose_candidates:
            if pc.exists():
                pose_src = pc
                break

        if pose_src is None:
            continue

        # Convert depth
        depth_meters = convert_depth(depth_src)
        depth_filename = f"depth_{count:04d}.npy"
        np.save(str(depth_dir / depth_filename), depth_meters)

        # Read pose (c2w)
        c2w = read_pose(pose_src)

        views.append({
            "index": count,
            "image_name": f"frame_{count:04d}.png",
            "depth_file": depth_filename,
            "normal_file": None,
            "matrix_4x4": c2w.flatten().tolist(),
        })

    write_poses_json(output_dir, views)
    logger.info(f"Done: {len(views)} frames → {output_dir}")


def convert_with_traj(scene_dir: Path, traj_file: Path, output_dir: Path, depth_dir: Path, args):
    """Convert using traj.txt format.

    NICE-SLAM Replica: each line is a full 4x4 c2w matrix (16 floats space-separated).
    """
    lines = traj_file.read_text().strip().split("\n")
    poses = []
    for line in lines:
        vals = list(map(float, line.split()))
        if len(vals) == 16:
            # One line = one 4x4 matrix (NICE-SLAM format)
            poses.append(np.array(vals).reshape(4, 4))
        elif len(vals) == 4 and len(poses) > 0:
            # Skip: this is part of a multi-line format handled below
            pass

    # Fallback: try 4-lines-per-matrix format
    if not poses and len(lines) >= 4:
        for i in range(0, len(lines), 4):
            if i + 4 > len(lines):
                break
            rows = [list(map(float, lines[i + j].split())) for j in range(4)]
            poses.append(np.array(rows))

    logger.info(f"Read {len(poses)} poses from traj.txt")

    # Find depth images
    results_dir = scene_dir / "results"
    depth_files = sorted(results_dir.glob("depth*.png"))

    if not depth_files:
        # Alternative: numbered in results/
        depth_files = sorted(results_dir.glob("*.png"))
        depth_files = [f for f in depth_files if "depth" in f.stem]

    n_frames = min(len(poses), len(depth_files))
    all_indices = list(range(0, n_frames, args.stride))[:args.max_frames]
    logger.info(f"Converting {len(all_indices)} frames (stride={args.stride}, total={n_frames})")

    views = []
    for count, idx in enumerate(all_indices):
        depth_meters = convert_depth(depth_files[idx])
        depth_filename = f"depth_{count:04d}.npy"
        np.save(str(depth_dir / depth_filename), depth_meters)

        c2w = poses[idx]
        views.append({
            "index": count,
            "image_name": f"frame_{count:04d}.png",
            "depth_file": depth_filename,
            "normal_file": None,
            "matrix_4x4": c2w.flatten().tolist(),
        })

    write_poses_json(output_dir, views)
    logger.info(f"Done: {len(views)} frames → {output_dir}")


def write_poses_json(output_dir: Path, views: list[dict]):
    """Write poses.json in GSS s04 output format."""
    poses_data = {
        "intrinsics": REPLICA_INTRINSICS,
        "views": views,
    }
    poses_file = output_dir / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_data, f, indent=2)
    logger.info(f"Wrote {poses_file} ({len(views)} views)")


if __name__ == "__main__":
    main()
