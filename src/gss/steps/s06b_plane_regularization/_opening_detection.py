"""Module F: Opening detection in walls (Cloud2BIM histogram void detection).

Detect doors and windows by analyzing density gaps in wall surface points.

Algorithm (Cloud2BIM pattern):
1. For each wall plane, extract nearby inlier points from surface point cloud
2. Project points onto wall-local 2D frame (u=along wall, v=height)
3. Horizontal histogram → find low-density gaps (candidate openings)
4. Vertical histogram per gap → determine opening height bounds
5. Classify: sill < 0.1m + height > 1.8m → door, else → window
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OpeningConfig:
    """Configuration for opening detection."""

    histogram_resolution: float = 0.05  # bin size in scene units
    histogram_threshold: float = 0.7  # fraction of peak density to consider "gap"
    min_opening_width: float = 0.3  # meters
    min_opening_height: float = 0.3  # meters
    door_sill_max: float = 0.1  # meters: sill below this → door
    door_min_height: float = 1.8  # meters: minimum height for door classification
    min_points_for_analysis: int = 100  # skip walls with fewer points


@dataclass
class Opening:
    """Detected opening in a wall."""

    type: str  # "door" or "window"
    u_start: float  # position along wall (scene units)
    u_end: float
    v_start: float  # height position (scene units)
    v_end: float
    width: float  # in scene units
    height: float  # in scene units


def _extract_wall_inliers(
    points: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    distance_threshold: float = 0.1,
) -> np.ndarray:
    """Extract points near a plane.

    Args:
        points: (N, 3) point cloud.
        plane_normal: Plane normal (unit vector).
        plane_d: Plane equation d (n·p + d = 0).
        distance_threshold: Max point-to-plane distance.

    Returns:
        Inlier points (M, 3).
    """
    distances = np.abs(points @ plane_normal + plane_d)
    mask = distances < distance_threshold
    return points[mask]


def _project_to_wall_frame(
    points: np.ndarray,
    normal_axis: str,
    center_line_2d: list[list[float]],
    height_range: list[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project 3D points to wall-local 2D frame.

    Manhattan-aligned walls:
    - normal_axis="x" → wall runs along Z, height along Y
    - normal_axis="z" → wall runs along X, height along Y

    u is measured from p1 in the p1→p2 direction so that the
    coordinate matches the opening builder's ``p1 + dir * u_mid``
    convention regardless of whether p1 > p2 or p1 < p2.

    Args:
        points: (M, 3) inlier points in Manhattan space.
        normal_axis: "x" or "z".
        center_line_2d: [[x1, z1], [x2, z2]] in Manhattan XZ.
        height_range: [y_min, y_max] in Manhattan Y.

    Returns:
        (u, v) arrays where u is along wall and v is height.
        Returns None if insufficient data.
    """
    if len(points) == 0:
        return None

    p1 = np.array(center_line_2d[0])
    p2 = np.array(center_line_2d[1])
    y_min, y_max = height_range

    # Filter points within wall height range (with margin)
    margin = 0.1 * (y_max - y_min)
    y_mask = (points[:, 1] >= y_min - margin) & (points[:, 1] <= y_max + margin)
    pts = points[y_mask]

    if len(pts) == 0:
        return None

    # u coordinate: projection along wall direction in XZ plane
    # Always measured from p1 in p1→p2 direction for consistency
    # with the opening builder which uses p1 + dir * u_mid.
    if normal_axis == "x":
        # Wall runs along Z axis; u-axis index in center_line_2d is [1]
        wall_x = (p1[0] + p2[0]) / 2.0
        x_mask = np.abs(pts[:, 0] - wall_x) < 0.5 * (y_max - y_min)
        pts = pts[x_mask]
        if len(pts) == 0:
            return None
        # Clip to wall extent
        z_min = min(p1[1], p2[1])
        z_max = max(p1[1], p2[1])
        extent_mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
        pts = pts[extent_mask]
        if len(pts) == 0:
            return None
        # u from p1 along p1→p2 direction
        u_dir = 1.0 if p2[1] >= p1[1] else -1.0
        u = (pts[:, 2] - p1[1]) * u_dir
    elif normal_axis == "z":
        # Wall runs along X axis; u-axis index in center_line_2d is [0]
        wall_z = (p1[1] + p2[1]) / 2.0
        z_mask = np.abs(pts[:, 2] - wall_z) < 0.5 * (y_max - y_min)
        pts = pts[z_mask]
        if len(pts) == 0:
            return None
        # Clip to wall extent
        x_min = min(p1[0], p2[0])
        x_max = max(p1[0], p2[0])
        extent_mask = (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max)
        pts = pts[extent_mask]
        if len(pts) == 0:
            return None
        # u from p1 along p1→p2 direction
        u_dir = 1.0 if p2[0] >= p1[0] else -1.0
        u = (pts[:, 0] - p1[0]) * u_dir
    else:
        return None

    if len(pts) == 0:
        return None

    # v coordinate: height (Y in Manhattan)
    v = pts[:, 1] - y_min

    return u, v


def _find_gaps_1d(
    values: np.ndarray,
    bin_size: float,
    threshold_ratio: float,
    min_gap_size: float,
) -> list[tuple[float, float]]:
    """Find low-density gaps in a 1D distribution.

    Args:
        values: 1D array of positions (already in target coordinate system).
        bin_size: Histogram bin size.
        threshold_ratio: Fraction of robust peak to use as gap threshold.
        min_gap_size: Minimum gap width to report.

    Returns:
        List of (start, end) gap intervals in the same coordinate system
        as the input values (absolute, not offset).
    """
    if len(values) == 0:
        return []

    v_min, v_max = float(values.min()), float(values.max())
    extent = v_max - v_min
    if extent < min_gap_size:
        return []

    n_bins = max(1, int(np.ceil(extent / bin_size)))
    hist, bin_edges = np.histogram(values, bins=n_bins, range=(v_min, v_max))

    if hist.max() == 0:
        return []

    # Robust peak: use 90th percentile of non-zero bins to avoid outlier spikes
    nonzero = hist[hist > 0]
    if len(nonzero) < 3:
        return []
    peak = np.percentile(nonzero, 90)
    threshold = peak * threshold_ratio

    # Find contiguous low-density regions
    gaps: list[tuple[float, float]] = []
    in_gap = False
    gap_start = 0.0

    for i, count in enumerate(hist):
        if count < threshold:
            if not in_gap:
                in_gap = True
                gap_start = bin_edges[i]
        else:
            if in_gap:
                gap_end = bin_edges[i]
                if (gap_end - gap_start) >= min_gap_size:
                    gaps.append((float(gap_start), float(gap_end)))
                in_gap = False

    # Close final gap
    if in_gap:
        gap_end = bin_edges[-1]
        if (gap_end - gap_start) >= min_gap_size:
            gaps.append((float(gap_start), float(gap_end)))

    return gaps


def _detect_openings_in_wall(
    u: np.ndarray,
    v: np.ndarray,
    wall_height: float,
    cfg: OpeningConfig,
) -> list[Opening]:
    """Detect openings in a single wall from projected u,v coordinates.

    Args:
        u: Along-wall positions.
        v: Height positions (0 = floor).
        wall_height: Total wall height.
        cfg: Detection configuration.

    Returns:
        List of detected openings.
    """
    # Step 1: Horizontal histogram → find u-gaps
    u_gaps = _find_gaps_1d(
        u, cfg.histogram_resolution, cfg.histogram_threshold, cfg.min_opening_width,
    )

    if not u_gaps:
        return []

    openings: list[Opening] = []

    # Step 2: For each u-gap, analyze vertical distribution
    for u_start, u_end in u_gaps:
        # Select points within this u-gap
        gap_mask = (u >= u_start) & (u <= u_end)
        v_gap = v[gap_mask]

        if len(v_gap) < 5:
            # Very few points in gap → likely a real void
            # Estimate opening height from surrounding points
            v_min_est = 0.0
            v_max_est = wall_height
        else:
            # Find vertical extent of the gap
            v_gaps = _find_gaps_1d(
                v_gap, cfg.histogram_resolution, cfg.histogram_threshold, cfg.min_opening_height,
            )
            if not v_gaps:
                # No clear vertical gap → the horizontal gap might be noise
                # Use the full v range where point density drops
                v_min_est = 0.0
                v_max_est = wall_height * 0.8  # conservative estimate
            else:
                # Use the largest vertical gap
                v_gaps_sorted = sorted(v_gaps, key=lambda g: g[1] - g[0], reverse=True)
                v_min_est, v_max_est = v_gaps_sorted[0]

        width = u_end - u_start
        height = v_max_est - v_min_est

        if height < cfg.min_opening_height:
            continue

        # Step 3: Classify
        sill_height = v_min_est  # distance from floor
        if sill_height < cfg.door_sill_max and height >= cfg.door_min_height:
            opening_type = "door"
            # Snap door bottom to floor
            v_min_est = 0.0
        else:
            opening_type = "window"

        openings.append(Opening(
            type=opening_type,
            u_start=u_start,
            u_end=u_end,
            v_start=v_min_est,
            v_end=v_max_est,
            width=width,
            height=v_max_est - v_min_est,
        ))

    return openings


def detect_openings(
    planes: list[dict],
    walls: list[dict],
    *,
    surface_points_path: str | Path | None = None,
    config: OpeningConfig | None = None,
    scale: float = 1.0,
    manhattan_rotation: np.ndarray | None = None,
) -> list[dict]:
    """Detect openings (doors, windows) in wall planes.

    Uses Cloud2BIM histogram void detection on the surface point cloud.

    Args:
        planes: Plane dicts in Manhattan space.
        walls: Wall info dicts from wall_thickness module.
        surface_points_path: Path to surface point cloud (.ply).
            If None, attempts to load from data/interim/s06_planes/.
        config: Detection configuration. Uses defaults if None.
        scale: Coordinate scale (scene_units / meter).
        manhattan_rotation: 3×3 rotation matrix (original → Manhattan).
            When provided, surface points are rotated into Manhattan space
            before inlier extraction.

    Returns:
        List of opening dicts, also mutates walls in-place to add "openings" key.
    """
    if config is None:
        config = OpeningConfig()

    # --- Load surface point cloud ---
    if surface_points_path is None:
        logger.info("Opening detection: no surface_points_path provided, skipping")
        return []

    surface_points_path = Path(surface_points_path)
    if not surface_points_path.exists():
        logger.warning(f"Surface points not found: {surface_points_path}, skipping opening detection")
        return []

    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(surface_points_path))
        all_points = np.asarray(pcd.points)
    except ImportError:
        logger.warning("open3d not available, skipping opening detection")
        return []

    if len(all_points) == 0:
        logger.warning("Empty point cloud, skipping opening detection")
        return []

    # Rotate surface points into Manhattan space to match planes/walls
    if manhattan_rotation is not None:
        all_points = all_points @ manhattan_rotation.T
        logger.info("Rotated surface points into Manhattan space")

    logger.info(f"Opening detection: loaded {len(all_points)} surface points")

    # Scale config thresholds to scene units
    scaled_cfg = OpeningConfig(
        histogram_resolution=config.histogram_resolution * scale,
        histogram_threshold=config.histogram_threshold,
        min_opening_width=config.min_opening_width * scale,
        min_opening_height=config.min_opening_height * scale,
        door_sill_max=config.door_sill_max * scale,
        door_min_height=config.door_min_height * scale,
        min_points_for_analysis=config.min_points_for_analysis,
    )

    total_openings = 0

    # Build plane lookup
    plane_by_id = {p["id"]: p for p in planes}

    for wall in walls:
        wall_openings: list[dict] = []
        wall_id = wall.get("id", "?")
        normal_axis = wall.get("normal_axis", "")
        center_line = wall.get("center_line_2d")
        height_range = wall.get("height_range")

        if not center_line or not height_range or not normal_axis:
            continue

        # Get plane for distance-based point selection
        plane_ids = wall.get("plane_ids", [])
        if not plane_ids:
            continue

        plane = plane_by_id.get(plane_ids[0])
        if plane is None:
            continue

        normal = np.asarray(plane["normal"], dtype=float)
        d = float(plane["d"])

        # Extract inlier points near this wall's plane
        thickness = wall.get("thickness", 0.2 * scale)
        inliers = _extract_wall_inliers(all_points, normal, d, distance_threshold=thickness)

        if len(inliers) < scaled_cfg.min_points_for_analysis:
            logger.debug(
                f"Wall {wall_id}: only {len(inliers)} inliers "
                f"(need {scaled_cfg.min_points_for_analysis}), skipping"
            )
            continue

        # Project to wall-local frame
        result = _project_to_wall_frame(inliers, normal_axis, center_line, height_range)
        if result is None:
            continue

        u, v = result
        if len(u) < scaled_cfg.min_points_for_analysis:
            continue

        wall_height = height_range[1] - height_range[0]

        # Detect openings
        openings = _detect_openings_in_wall(u, v, wall_height, scaled_cfg)

        for op in openings:
            wall_openings.append({
                "type": op.type,
                "position_along_wall": [float(op.u_start), float(op.u_end)],
                "height_range": [float(op.v_start), float(op.v_end)],
                "width": float(op.width),
                "height": float(op.height),
            })

        if wall_openings:
            wall["openings"] = wall_openings
            total_openings += len(wall_openings)
            logger.info(
                f"Wall {wall_id}: detected {len(wall_openings)} openings "
                f"({sum(1 for o in wall_openings if o['type'] == 'door')} doors, "
                f"{sum(1 for o in wall_openings if o['type'] == 'window')} windows)"
            )

    logger.info(f"Opening detection complete: {total_openings} openings in {len(walls)} walls")
    return [
        op
        for wall in walls
        if "openings" in wall
        for op in wall["openings"]
    ]
