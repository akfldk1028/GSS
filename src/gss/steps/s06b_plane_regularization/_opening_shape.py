"""Module J: Opening shape refinement.

Analyzes the actual void boundary of detected rectangular openings
to reclassify them as arched or circular when appropriate.

Shape classification:
- rectangular: default (no change)
- arched: flat bottom + arc top (e.g., Romanesque window)
- circular: round void (e.g., porthole, rose window)

Requires surface_points.ply for void boundary tracing.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def _fit_arc_to_top(
    points_2d: np.ndarray,
    rect_top: float,
    rect_left: float,
    rect_right: float,
    tolerance: float = 0.05,
) -> tuple[bool, float]:
    """Check if the top portion of the void follows an arc.

    Examines points near the top of the rectangular opening.
    If they form a curve (points below the rectangle top by a
    consistent arc amount), returns True and the arc radius.

    Args:
        points_2d: Void boundary points in wall-local 2D (u, v).
        rect_top: Top of rectangular opening in v coordinate.
        rect_left: Left edge in u coordinate.
        rect_right: Right edge in u coordinate.

    Returns:
        (is_arched, radius) where radius is the arc radius.
    """
    width = rect_right - rect_left
    if width < 0.1:
        return False, 0.0

    # Select points in the top 30% of the opening height
    v_threshold = rect_top - 0.3 * width
    top_mask = points_2d[:, 1] > v_threshold
    top_pts = points_2d[top_mask]

    if len(top_pts) < 5:
        return False, 0.0

    # Center u coordinate
    center_u = (rect_left + rect_right) / 2.0
    half_width = width / 2.0

    # For each point, compute expected arc height
    # Arc: v = rect_top - R + sqrt(R^2 - (u - center_u)^2)
    # where R is the radius. For a semicircle, R = half_width.

    # Fit radius by least squares
    # Simplification: check if points form a semicircle
    u_normalized = (top_pts[:, 0] - center_u) / max(half_width, 1e-6)
    v_normalized = (top_pts[:, 1] - rect_top) / max(half_width, 1e-6)

    # For a semicircle: v = sqrt(1 - u^2) - 1 (in normalized coords)
    valid = np.abs(u_normalized) <= 1.0
    if np.sum(valid) < 3:
        return False, 0.0

    u_valid = u_normalized[valid]
    v_valid = v_normalized[valid]

    expected_v = np.sqrt(np.clip(1.0 - u_valid**2, 0, None)) - 1.0
    residual = np.mean(np.abs(v_valid - expected_v))

    if residual < tolerance:
        radius = half_width
        return True, float(radius)

    return False, 0.0


def _is_circular_void(
    points_2d: np.ndarray,
    center_u: float,
    center_v: float,
    tolerance: float = 0.1,
) -> tuple[bool, float]:
    """Check if void boundary points form a circle.

    Args:
        points_2d: Void boundary points in wall-local 2D (u, v).
        center_u, center_v: Estimated center of the opening.
        tolerance: Relative tolerance for circularity.

    Returns:
        (is_circular, radius).
    """
    if len(points_2d) < 8:
        return False, 0.0

    # Compute distances from center
    distances = np.sqrt(
        (points_2d[:, 0] - center_u) ** 2
        + (points_2d[:, 1] - center_v) ** 2
    )

    if len(distances) < 3:
        return False, 0.0

    mean_r = float(np.mean(distances))
    if mean_r < 0.05:
        return False, 0.0

    # Check circularity: std / mean < tolerance
    std_r = float(np.std(distances))
    if std_r / mean_r < tolerance:
        return True, mean_r

    return False, 0.0


def refine_opening_shapes(
    walls: list[dict],
    surface_points: np.ndarray | None = None,
    scale: float = 1.0,
    manhattan_rotation: np.ndarray | None = None,
    *,
    arch_segments: int = 12,
    inlier_distance: float = 0.1,
) -> dict:
    """Refine the shapes of detected openings.

    For each opening in walls, analyzes void boundary points to determine
    if the opening is arched or circular (rather than rectangular).

    Args:
        walls: List of wall dicts (modified in-place with shape field).
        surface_points: Nx3 point cloud (COLMAP coordinates).
        scale: Scene units / meter.
        manhattan_rotation: 3x3 rotation matrix (COLMAP → Manhattan).
        arch_segments: Number of segments to approximate an arch.
        inlier_distance: Max distance from wall plane for inlier points (meters).

    Returns:
        Stats dict with counts per shape type.
    """
    stats = {"rectangular": 0, "arched": 0, "circular": 0}

    if surface_points is None:
        # Without surface points, just ensure all openings have a shape field
        for w in walls:
            for opening in w.get("openings", []):
                if "shape" not in opening:
                    opening["shape"] = "rectangular"
                    stats["rectangular"] += 1
        return stats

    # Transform surface points to Manhattan space if rotation available
    if manhattan_rotation is not None:
        pts_manhattan = (surface_points @ manhattan_rotation.T)
    else:
        pts_manhattan = surface_points

    eff_inlier_dist = inlier_distance * scale

    for w in walls:
        openings = w.get("openings", [])
        if not openings:
            continue

        cl = w["center_line_2d"]
        if len(cl) < 2:
            continue

        p1 = np.array(cl[0], dtype=float)
        p2 = np.array(cl[-1], dtype=float)
        wall_dir = p2 - p1
        wall_len = np.linalg.norm(wall_dir)
        if wall_len < 1e-6:
            continue
        wall_dir = wall_dir / wall_len

        # Wall normal in XZ plane
        wall_normal = np.array([-wall_dir[1], wall_dir[0]])
        wall_d = np.dot(wall_normal, p1)

        # Find points near this wall plane (in XZ)
        pts_xz = pts_manhattan[:, [0, 2]]
        pts_y = pts_manhattan[:, 1]
        dists_to_plane = np.abs(pts_xz @ wall_normal - wall_d)
        thickness = w.get("thickness", 0.2 * scale)
        inlier_mask = dists_to_plane < max(thickness, eff_inlier_dist)

        wall_pts_xz = pts_xz[inlier_mask]
        wall_pts_y = pts_y[inlier_mask]

        if len(wall_pts_xz) < 20:
            for opening in openings:
                opening["shape"] = "rectangular"
                stats["rectangular"] += 1
            continue

        # Project to wall-local 2D (u, v)
        u_coords = (wall_pts_xz - p1) @ wall_dir
        v_coords = wall_pts_y

        for opening in openings:
            pos = opening.get("position_along_wall", [0, 0])
            h_range = opening.get("height_range", [0, 0])

            u_start, u_end = pos[0], pos[1]
            v_start, v_end = h_range[0], h_range[1]

            # Find void boundary points (near opening edges)
            margin_u = (u_end - u_start) * 0.1
            margin_v = (v_end - v_start) * 0.1

            in_opening = (
                (u_coords > u_start - margin_u)
                & (u_coords < u_end + margin_u)
                & (v_coords > v_start - margin_v)
                & (v_coords < v_end + margin_v)
            )
            opening_pts = np.column_stack([u_coords[in_opening], v_coords[in_opening]])

            if len(opening_pts) < 10:
                opening["shape"] = "rectangular"
                stats["rectangular"] += 1
                continue

            # Check circular first (higher specificity)
            center_u = (u_start + u_end) / 2.0
            center_v = (v_start + v_end) / 2.0
            is_circ, radius = _is_circular_void(
                opening_pts, center_u, center_v,
            )
            if is_circ:
                opening["shape"] = "circular"
                opening["radius"] = float(radius / scale)
                stats["circular"] += 1
                continue

            # Check arched
            is_arch, arch_radius = _fit_arc_to_top(
                opening_pts, v_end, u_start, u_end,
            )
            if is_arch:
                opening["shape"] = "arched"
                opening["arch_radius"] = float(arch_radius / scale)
                opening["arch_segments"] = arch_segments
                stats["arched"] += 1
                continue

            opening["shape"] = "rectangular"
            stats["rectangular"] += 1

    logger.info(
        f"Opening shape refinement: "
        f"{stats['rectangular']} rectangular, "
        f"{stats['arched']} arched, "
        f"{stats['circular']} circular"
    )
    return stats
