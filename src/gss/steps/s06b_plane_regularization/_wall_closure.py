"""Module C2: Synthesize missing walls from floor/ceiling boundary.

When scan data produces an incomplete set of walls (e.g., interior-only scan),
use the floor boundary to estimate the room outline and fill gaps.

Algorithm:
1. Collect floor boundary points → project to XZ → compute convex hull / oriented bbox
2. For each edge of the estimated outline, check if an existing wall covers it
3. For uncovered edges, synthesize a new wall (plane + wall object)
4. For partial walls (shorter than the corresponding edge), extend them
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _floor_boundary_xz(planes: list[dict]) -> np.ndarray | None:
    """Extract floor boundary points projected to XZ plane."""
    floor_pts = []
    for p in planes:
        if p["label"] == "floor":
            bnd = p.get("boundary_3d")
            if bnd is not None and len(bnd) > 0:
                pts = np.asarray(bnd)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    floor_pts.append(pts[:, [0, 2]])  # XZ projection
    if not floor_pts:
        return None
    return np.vstack(floor_pts)


def _oriented_bbox_edges(pts_2d: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute axis-aligned bounding box of convex hull and return its 4 edges.

    Uses AABB which works well for Manhattan-aligned rooms (post-rotation).
    Returns list of (p1, p2) edge endpoint pairs in XZ.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
    except (ImportError, Exception):
        hull_pts = pts_2d

    # Simple approach: use axis-aligned bounding box of hull points
    # (works well for Manhattan-aligned rooms)
    xmin, zmin = hull_pts.min(axis=0)
    xmax, zmax = hull_pts.max(axis=0)

    corners = [
        np.array([xmin, zmin]),
        np.array([xmax, zmin]),
        np.array([xmax, zmax]),
        np.array([xmin, zmax]),
    ]
    edges = []
    for i in range(4):
        edges.append((corners[i], corners[(i + 1) % 4]))
    return edges


def _edge_axis(p1: np.ndarray, p2: np.ndarray) -> str | None:
    """Determine if an edge is X-aligned or Z-aligned."""
    dx = abs(p2[0] - p1[0])
    dz = abs(p2[1] - p1[1])
    if dx < 1e-6 and dz > 1e-6:
        return "x"  # vertical line in XZ → wall normal along X
    if dz < 1e-6 and dx > 1e-6:
        return "z"  # horizontal line in XZ → wall normal along Z
    return None


def _wall_covers_edge(
    wall: dict, edge_p1: np.ndarray, edge_p2: np.ndarray, tolerance: float,
) -> float:
    """Check how much of an edge is covered by a wall's center-line.

    Returns coverage fraction [0, 1].
    """
    cl = wall["center_line_2d"]
    wp1 = np.array(cl[0])
    wp2 = np.array(cl[1])

    axis = _edge_axis(edge_p1, edge_p2)
    if axis is None:
        return 0.0

    wall_axis = wall.get("normal_axis")
    if wall_axis != axis:
        return 0.0

    # Check if wall is near the edge position
    if axis == "x":
        # Wall normal along X: wall position is X, edge position is X
        edge_x = edge_p1[0]  # both endpoints have same X
        wall_x = wp1[0]  # center-line X (approximately same for both endpoints)
        if abs(wall_x - edge_x) > tolerance:
            return 0.0
        # Coverage along Z
        edge_min, edge_max = sorted([edge_p1[1], edge_p2[1]])
        wall_min, wall_max = sorted([wp1[1], wp2[1]])
    else:  # z
        edge_z = edge_p1[1]
        wall_z = wp1[1]
        if abs(wall_z - edge_z) > tolerance:
            return 0.0
        # Coverage along X
        edge_min, edge_max = sorted([edge_p1[0], edge_p2[0]])
        wall_min, wall_max = sorted([wp1[0], wp2[0]])

    edge_len = edge_max - edge_min
    if edge_len < 1e-6:
        return 1.0

    overlap = max(0.0, min(edge_max, wall_max) - max(edge_min, wall_min))
    return overlap / edge_len


def synthesize_missing_walls(
    walls: list[dict],
    planes: list[dict],
    floor_heights: list[float],
    ceiling_heights: list[float],
    scale: float = 1.0,
    max_gap_ratio: float = 0.3,
    use_floor_ceiling_hints: bool = True,
    default_thickness: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Synthesize missing walls based on floor boundary outline.

    Args:
        walls: existing wall objects (modified in-place to extend partials).
        planes: all plane dicts.
        floor_heights: detected floor heights.
        ceiling_heights: detected ceiling heights.
        scale: coordinate scale (scene_units/meter).
        max_gap_ratio: max ratio of uncovered edge to consider filling.
        use_floor_ceiling_hints: whether to use floor boundary for outline.
        default_thickness: thickness for synthesized walls.

    Returns:
        (updated_walls, new_planes): walls list (possibly with new entries),
        and list of new synthetic plane dicts.
    """
    if not use_floor_ceiling_hints:
        return walls, []

    floor_pts = _floor_boundary_xz(planes)
    if floor_pts is None or len(floor_pts) < 3:
        logger.info("Wall closure: no floor boundary available, skipping")
        return walls, []

    edges = _oriented_bbox_edges(floor_pts)

    floor_h = min(floor_heights) if floor_heights else 0.0
    ceiling_h = max(ceiling_heights) if ceiling_heights else floor_h + 3.0 * scale

    # Position tolerance: allow wall to be within this distance of edge
    pos_tolerance = 2.0 * scale

    new_walls = []
    new_planes = []
    next_wall_id = max((w["id"] for w in walls), default=-1) + 1
    next_plane_id = max((p["id"] for p in planes), default=-1) + 1

    for edge_p1, edge_p2 in edges:
        axis = _edge_axis(edge_p1, edge_p2)
        if axis is None:
            continue

        # Check coverage by existing walls (sum of all covering walls, capped at 1.0)
        total_coverage = 0.0
        for w in walls:
            cov = _wall_covers_edge(w, edge_p1, edge_p2, pos_tolerance)
            total_coverage += cov
        total_coverage = min(total_coverage, 1.0)

        gap_ratio = 1.0 - total_coverage
        if gap_ratio <= max_gap_ratio:
            continue

        # Synthesize a new wall for this edge
        edge_len = float(np.linalg.norm(edge_p2 - edge_p1))
        if edge_len < 1e-3:
            continue

        # Create center-line
        cl = [edge_p1.tolist(), edge_p2.tolist()]

        # Determine wall normal — outward-facing from room center
        # Compute room center from all wall center-line midpoints
        all_midpoints = []
        for w in walls:
            wc = w["center_line_2d"]
            mid = [(wc[0][0] + wc[1][0]) / 2.0, (wc[0][1] + wc[1][1]) / 2.0]
            all_midpoints.append(mid)
        room_center = np.mean(all_midpoints, axis=0) if all_midpoints else np.array([0.0, 0.0])

        if axis == "x":
            mid_x = edge_p1[0]
            # Normal points away from room center
            normal = np.array([1.0, 0.0, 0.0]) if mid_x > room_center[0] else np.array([-1.0, 0.0, 0.0])
            d = float(-normal[0] * mid_x)
        else:
            mid_z = edge_p1[1]
            normal = np.array([0.0, 0.0, 1.0]) if mid_z > room_center[1] else np.array([0.0, 0.0, -1.0])
            d = float(-normal[2] * mid_z)

        # Create synthetic plane
        if axis == "x":
            pos = edge_p1[0]
            bnd = np.array([
                [pos, floor_h, edge_p1[1]],
                [pos, floor_h, edge_p2[1]],
                [pos, ceiling_h, edge_p2[1]],
                [pos, ceiling_h, edge_p1[1]],
                [pos, floor_h, edge_p1[1]],
            ])
        else:
            pos = edge_p1[1]
            bnd = np.array([
                [edge_p1[0], floor_h, pos],
                [edge_p2[0], floor_h, pos],
                [edge_p2[0], ceiling_h, pos],
                [edge_p1[0], ceiling_h, pos],
                [edge_p1[0], floor_h, pos],
            ])

        new_plane = {
            "id": next_plane_id,
            "normal": normal,
            "d": d,
            "label": "wall",
            "num_inliers": 0,
            "boundary_3d": bnd,
            "synthetic": True,
        }
        new_planes.append(new_plane)

        new_wall = {
            "id": next_wall_id,
            "plane_ids": [next_plane_id],
            "center_line_2d": cl,
            "thickness": float(default_thickness),
            "height_range": [float(floor_h), float(ceiling_h)],
            "normal_axis": axis,
            "synthetic": True,
        }
        new_walls.append(new_wall)

        logger.info(
            f"Wall closure: synthesized wall {next_wall_id} "
            f"(axis={axis}, edge_len={edge_len:.1f}, gap={gap_ratio:.0%})"
        )
        next_wall_id += 1
        next_plane_id += 1

    walls.extend(new_walls)
    logger.info(
        f"Wall closure: {len(new_walls)} walls synthesized, "
        f"{len(walls)} total walls"
    )
    return walls, new_planes
