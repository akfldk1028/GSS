"""Shared I/O and transform utilities for plane-based steps (s06b, s06c, s06d).

Extracted from duplicated code in s06b/s06c/s06d step.py files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_planes(planes_file: Path) -> list[dict]:
    """Load planes.json and convert arrays to numpy."""
    with open(planes_file, encoding="utf-8") as f:
        data = json.load(f)
    for p in data:
        p["normal"] = np.asarray(p["normal"], dtype=float)
        if p.get("boundary_3d"):
            p["boundary_3d"] = np.asarray(p["boundary_3d"], dtype=float)
        else:
            p["boundary_3d"] = np.empty((0, 3))
    return data


def load_manhattan_rotation(s06_dir: Path) -> np.ndarray | None:
    """Load Manhattan rotation matrix from s06 output directory."""
    path = s06_dir / "manhattan_alignment.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        R = np.asarray(data["manhattan_rotation"], dtype=float)
        if R.shape != (3, 3):
            logger.warning(f"Invalid manhattan_rotation shape {R.shape}, ignoring")
            return None
        return R
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(
            f"Failed to load manhattan_alignment.json: {e}; "
            "proceeding without Manhattan rotation"
        )
        return None


def serialize_planes(planes: list[dict]) -> list[dict]:
    """Convert planes back to JSON-serializable dicts."""
    result = []
    for p in planes:
        entry = {
            "id": p["id"],
            "normal": (
                p["normal"].tolist()
                if isinstance(p["normal"], np.ndarray)
                else p["normal"]
            ),
            "d": float(p["d"]),
            "label": p["label"],
            "num_inliers": p.get("num_inliers", 0),
            "boundary_3d": (
                p["boundary_3d"].tolist()
                if isinstance(p["boundary_3d"], np.ndarray)
                and len(p["boundary_3d"]) > 0
                else []
            ),
        }
        result.append(entry)
    return result


def transform_to_manhattan(planes: list[dict], R: np.ndarray) -> None:
    """Transform planes from original to Manhattan-aligned coordinates (in-place).

    R maps original -> Manhattan: p_m = R @ p_o, n_m = R @ n_o.
    """
    for p in planes:
        p["normal"] = R @ p["normal"]
        if len(p["boundary_3d"]) > 0:
            p["boundary_3d"] = p["boundary_3d"] @ R.T


def transform_from_manhattan(planes: list[dict], R: np.ndarray) -> None:
    """Transform planes back from Manhattan to original coordinates (in-place)."""
    R_inv = R.T
    for p in planes:
        p["normal"] = R_inv @ p["normal"]
        if len(p["boundary_3d"]) > 0:
            p["boundary_3d"] = p["boundary_3d"] @ R_inv.T


def estimate_scale(
    planes: list[dict],
    expected_storey_height: float = 2.7,
    expected_room_size: float = 5.0,
) -> float:
    """Estimate coordinate scale using architectural height priors.

    In Manhattan-aligned space (Y-up), uses three strategies in priority order:
    1. Floor-ceiling height difference
    2. Wall Y-extent (proxy for room height)
    3. XZ bounding box fallback

    Returns scale factor (scene_units / meter).
    """
    floor_heights: list[float] = []
    ceiling_heights: list[float] = []
    wall_y_extents: list[float] = []
    arch_pts: list[np.ndarray] = []

    for p in planes:
        label = p.get("label", "other")
        bnd = p.get("boundary_3d")
        has_bnd = bnd is not None and len(bnd) > 0

        if label in ("floor", "ceiling", "ground"):
            normal = p.get("normal")
            d_val = p.get("d")
            ny = (
                normal[1]
                if normal is not None
                and hasattr(normal, "__len__")
                and len(normal) > 1
                else 0
            )
            if abs(ny) > 0.3 and d_val is not None:
                h = -d_val / ny
            elif has_bnd:
                h = float(np.asarray(bnd)[:, 1].mean())
            else:
                continue

            if label in ("floor", "ground"):
                floor_heights.append(h)
            else:
                ceiling_heights.append(h)

        if label in ("wall", "facade") and has_bnd:
            pts = np.asarray(bnd)
            if pts.ndim == 2 and pts.shape[0] >= 2:
                y_ext = pts[:, 1].max() - pts[:, 1].min()
                if y_ext > 0.1:
                    wall_y_extents.append(y_ext)

        if label in ("wall", "floor", "ceiling", "facade", "ground") and has_bnd:
            pts = np.asarray(bnd)
            if pts.ndim == 2:
                arch_pts.append(pts)

    # Strategy 1: floor-ceiling distance
    if floor_heights and ceiling_heights:
        floor_h = np.median(floor_heights)
        ceil_h = np.median(ceiling_heights)
        fc_diff = abs(ceil_h - floor_h)
        if fc_diff > 0.1:
            scale = fc_diff / expected_storey_height
            scale = max(0.1, min(scale, 100.0))
            logger.info(
                f"Scale from floor-ceiling: {scale:.2f} "
                f"(fc_diff={fc_diff:.2f}, expected_h={expected_storey_height}m)"
            )
            return scale

    # Strategy 2: wall Y-extent
    if wall_y_extents:
        median_wall_h = float(np.median(wall_y_extents))
        if median_wall_h > 0.1:
            scale = median_wall_h / expected_storey_height
            scale = max(0.1, min(scale, 100.0))
            logger.info(
                f"Scale from wall height: {scale:.2f} "
                f"(median_wall_h={median_wall_h:.2f}, expected_h={expected_storey_height}m)"
            )
            return scale

    # Strategy 3: XZ bounding box fallback
    if arch_pts:
        combined = np.vstack(arch_pts)
        extents = combined.max(axis=0) - combined.min(axis=0)
        xz_extents = [extents[0], extents[2]]
        median_xz = float(np.median(xz_extents))
        if median_xz > 0.1:
            scale = median_xz / expected_room_size
            scale = max(0.1, min(scale, 100.0))
            logger.info(
                f"Scale from XZ extent (fallback): {scale:.2f} "
                f"(median_xz={median_xz:.2f}, expected_room={expected_room_size}m)"
            )
            return scale

    logger.warning("Scale estimation: no usable planes, defaulting to 1.0")
    return 1.0
