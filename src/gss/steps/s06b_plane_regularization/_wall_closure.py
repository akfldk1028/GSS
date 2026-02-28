"""Module C2: Synthesize missing walls from floor/ceiling boundary.

When scan data produces an incomplete set of walls (e.g., interior-only scan),
use the floor boundary to estimate the room outline and fill gaps.

Supports two modes:
- Manhattan: AABB-based outline (original, for axis-aligned rooms)
- General: ConvexHull edge-based outline (for arbitrary-angle buildings)

Algorithm:
1. Collect floor boundary points → project to XZ → compute outline
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


def _wall_centerline_aabb_xz(walls: list[dict]) -> np.ndarray | None:
    """Compute AABB from wall center-line endpoints projected to XZ.

    Fallback when floor boundary is unavailable: uses the bounding box of
    all wall center-line endpoints as a rough room outline estimate.
    """
    pts = []
    for w in walls:
        cl = w.get("center_line_2d")
        if cl and len(cl) == 2:
            pts.append(cl[0])
            pts.append(cl[1])
    if len(pts) < 3:
        return None
    pts_arr = np.array(pts)
    xmin, zmin = pts_arr.min(axis=0)
    xmax, zmax = pts_arr.max(axis=0)
    # Return 4 corners of AABB (enough for convex hull / oriented bbox)
    return np.array([
        [xmin, zmin], [xmax, zmin], [xmax, zmax], [xmin, zmax],
    ])


def _aabb_edges(pts_2d: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
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


def _convex_hull_edges(pts_2d: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute convex hull and return edges for arbitrary-angle outline.

    Returns list of (p1, p2) edge endpoint pairs in XZ.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_2d)
        vertices = hull.vertices
        hull_pts = pts_2d[vertices]
        edges = []
        n = len(hull_pts)
        for i in range(n):
            edges.append((hull_pts[i].copy(), hull_pts[(i + 1) % n].copy()))
        return edges
    except (ImportError, Exception):
        # Fallback to AABB
        logger.warning("ConvexHull failed, falling back to AABB")
        return _aabb_edges(pts_2d)


def _edge_axis(p1: np.ndarray, p2: np.ndarray) -> str | None:
    """Determine if an edge is X-aligned or Z-aligned (Manhattan)."""
    dx = abs(p2[0] - p1[0])
    dz = abs(p2[1] - p1[1])
    if dx < 1e-6 and dz > 1e-6:
        return "x"  # vertical line in XZ → wall normal along X
    if dz < 1e-6 and dx > 1e-6:
        return "z"  # horizontal line in XZ → wall normal along Z
    return None


def _edge_normal_vector(p1: np.ndarray, p2: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Compute outward-facing normal for an edge relative to a center point.

    Returns 3D unit normal vector (Y component = 0).
    """
    edge_dir = p2 - p1
    # 2D perpendicular (rotate 90°)
    perp = np.array([edge_dir[1], -edge_dir[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    perp = perp / perp_norm

    # Ensure outward-facing (away from center)
    mid = (p1 + p2) / 2.0
    to_center = center - mid
    if np.dot(perp, to_center) > 0:
        perp = -perp

    return np.array([perp[0], 0.0, perp[1]])


def _wall_covers_edge_manhattan(
    wall: dict, edge_p1: np.ndarray, edge_p2: np.ndarray, tolerance: float,
) -> float:
    """Check how much of a Manhattan-aligned edge is covered by a wall.

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

    if axis == "x":
        edge_x = edge_p1[0]
        wall_x = wp1[0]
        if abs(wall_x - edge_x) > tolerance:
            return 0.0
        edge_min, edge_max = sorted([edge_p1[1], edge_p2[1]])
        wall_min, wall_max = sorted([wp1[1], wp2[1]])
    else:
        edge_z = edge_p1[1]
        wall_z = wp1[1]
        if abs(wall_z - edge_z) > tolerance:
            return 0.0
        edge_min, edge_max = sorted([edge_p1[0], edge_p2[0]])
        wall_min, wall_max = sorted([wp1[0], wp2[0]])

    edge_len = edge_max - edge_min
    if edge_len < 1e-6:
        return 1.0

    overlap = max(0.0, min(edge_max, wall_max) - max(edge_min, wall_min))
    return overlap / edge_len


def _wall_covers_edge_general(
    wall: dict, edge_p1: np.ndarray, edge_p2: np.ndarray, tolerance: float,
) -> float:
    """Check how much of an arbitrary-angle edge is covered by a wall.

    Uses projection along the edge direction to check overlap,
    and distance perpendicular to the edge for proximity.

    Returns coverage fraction [0, 1].
    """
    cl = wall["center_line_2d"]
    wp1 = np.array(cl[0])
    wp2 = np.array(cl[1])

    edge_dir = edge_p2 - edge_p1
    edge_len = np.linalg.norm(edge_dir)
    if edge_len < 1e-6:
        return 1.0
    edge_unit = edge_dir / edge_len

    # Edge perpendicular
    edge_perp = np.array([-edge_unit[1], edge_unit[0]])

    # Check perpendicular distance from wall midpoint to edge line
    wall_mid = (wp1 + wp2) / 2.0
    edge_mid = (edge_p1 + edge_p2) / 2.0
    perp_dist = abs(np.dot(wall_mid - edge_mid, edge_perp))
    if perp_dist > tolerance:
        return 0.0

    # Check if wall direction is roughly parallel to edge
    wall_dir = wp2 - wp1
    wall_len = np.linalg.norm(wall_dir)
    if wall_len < 1e-6:
        return 0.0
    wall_unit = wall_dir / wall_len
    if abs(np.dot(wall_unit, edge_unit)) < 0.7:  # Not parallel enough
        return 0.0

    # Project wall endpoints onto edge direction
    w_proj1 = np.dot(wp1 - edge_p1, edge_unit)
    w_proj2 = np.dot(wp2 - edge_p1, edge_unit)
    w_min, w_max = sorted([w_proj1, w_proj2])

    overlap = max(0.0, min(edge_len, w_max) - max(0.0, w_min))
    return overlap / edge_len


def _wall_covers_edge(
    wall: dict, edge_p1: np.ndarray, edge_p2: np.ndarray, tolerance: float,
) -> float:
    """Check coverage, dispatching to Manhattan or general method."""
    axis = _edge_axis(edge_p1, edge_p2)
    wall_axis = wall.get("normal_axis", "")

    # Use Manhattan method if edge is axis-aligned AND wall is Manhattan
    if axis is not None and wall_axis in ("x", "z"):
        return _wall_covers_edge_manhattan(wall, edge_p1, edge_p2, tolerance)

    # General method for everything else
    return _wall_covers_edge_general(wall, edge_p1, edge_p2, tolerance)


def synthesize_missing_walls(
    walls: list[dict],
    planes: list[dict],
    floor_heights: list[float],
    ceiling_heights: list[float],
    scale: float = 1.0,
    max_gap_ratio: float = 0.3,
    use_floor_ceiling_hints: bool = True,
    default_thickness: float = 0.2,
    normal_mode: str = "manhattan",
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
        normal_mode: "manhattan" uses AABB, "cluster" uses ConvexHull.

    Returns:
        (updated_walls, new_planes): walls list (possibly with new entries),
        and list of new synthetic plane dicts.
    """
    if not use_floor_ceiling_hints:
        return walls, []

    floor_pts = _floor_boundary_xz(planes)
    if floor_pts is None or len(floor_pts) < 3:
        # Fallback: use wall center-line AABB as rough room outline
        floor_pts = _wall_centerline_aabb_xz(walls)
        if floor_pts is None or len(floor_pts) < 3:
            logger.info("Wall closure: no floor boundary and insufficient walls for AABB, skipping")
            return walls, []
        logger.info("Wall closure: no floor boundary — using wall center-line AABB as fallback")

    # Choose outline method based on normal_mode
    if normal_mode == "cluster":
        edges = _convex_hull_edges(floor_pts)
    else:
        edges = _aabb_edges(floor_pts)

    floor_h = min(floor_heights) if floor_heights else 0.0
    ceiling_h = max(ceiling_heights) if ceiling_heights else floor_h + 3.0 * scale

    # Position tolerance: allow wall to be within this distance of edge
    pos_tolerance = 2.0 * scale

    new_walls = []
    new_planes = []
    next_wall_id = max((w["id"] for w in walls), default=-1) + 1
    next_plane_id = max((p["id"] for p in planes), default=-1) + 1

    # Room center for outward normal computation
    all_midpoints = []
    for w in walls:
        wc = w["center_line_2d"]
        mid = [(wc[0][0] + wc[1][0]) / 2.0, (wc[0][1] + wc[1][1]) / 2.0]
        all_midpoints.append(mid)
    room_center = np.mean(all_midpoints, axis=0) if all_midpoints else np.array([0.0, 0.0])

    for edge_p1, edge_p2 in edges:
        edge_len = float(np.linalg.norm(edge_p2 - edge_p1))
        if edge_len < 1e-3:
            continue

        # Check coverage by existing walls
        total_coverage = 0.0
        for w in walls:
            cov = _wall_covers_edge(w, edge_p1, edge_p2, pos_tolerance)
            total_coverage += cov
        total_coverage = min(total_coverage, 1.0)

        gap_ratio = 1.0 - total_coverage
        if gap_ratio <= max_gap_ratio:
            continue

        # Determine wall axis and normal
        axis = _edge_axis(edge_p1, edge_p2)
        cl = [edge_p1.tolist(), edge_p2.tolist()]

        if axis is not None:
            # Manhattan edge
            normal = _compute_manhattan_normal(axis, edge_p1, room_center)
            d = _compute_plane_d(normal, axis, edge_p1)
            bnd = _compute_manhattan_boundary(axis, edge_p1, edge_p2, floor_h, ceiling_h)
            normal_axis = axis
            wall_extra = {}
        else:
            # Arbitrary-angle edge
            normal = _edge_normal_vector(edge_p1, edge_p2, room_center)
            mid = (edge_p1 + edge_p2) / 2.0
            d = float(-np.dot(normal, np.array([mid[0], 0.0, mid[1]])))
            bnd = _compute_general_boundary(edge_p1, edge_p2, floor_h, ceiling_h)
            nv_2d = np.array([normal[0], normal[2]])
            nv_norm = np.linalg.norm(nv_2d)
            if nv_norm > 1e-6:
                nv_2d = nv_2d / nv_norm
            angle_deg = float(np.degrees(np.arctan2(nv_2d[1], nv_2d[0])))
            normal_axis = f"oblique:{angle_deg:.0f}"
            wall_extra = {"normal_vector": nv_2d.tolist()}

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
            "normal_axis": normal_axis,
            "synthetic": True,
            **wall_extra,
        }
        new_walls.append(new_wall)

        logger.info(
            f"Wall closure: synthesized wall {next_wall_id} "
            f"(axis={normal_axis}, edge_len={edge_len:.1f}, gap={gap_ratio:.0%})"
        )
        next_wall_id += 1
        next_plane_id += 1

    walls.extend(new_walls)
    logger.info(
        f"Wall closure: {len(new_walls)} walls synthesized, "
        f"{len(walls)} total walls"
    )
    return walls, new_planes


def _compute_manhattan_normal(axis: str, edge_p1: np.ndarray, room_center: np.ndarray) -> np.ndarray:
    """Compute outward-facing normal for a Manhattan edge."""
    if axis == "x":
        mid_x = edge_p1[0]
        return np.array([1.0, 0.0, 0.0]) if mid_x > room_center[0] else np.array([-1.0, 0.0, 0.0])
    else:
        mid_z = edge_p1[1]
        return np.array([0.0, 0.0, 1.0]) if mid_z > room_center[1] else np.array([0.0, 0.0, -1.0])


def _compute_plane_d(normal: np.ndarray, axis: str, edge_p1: np.ndarray) -> float:
    """Compute plane d for a Manhattan wall."""
    if axis == "x":
        return float(-normal[0] * edge_p1[0])
    else:
        return float(-normal[2] * edge_p1[1])


def _compute_manhattan_boundary(
    axis: str, edge_p1: np.ndarray, edge_p2: np.ndarray,
    floor_h: float, ceiling_h: float,
) -> np.ndarray:
    """Compute 3D boundary for a Manhattan-aligned synthetic wall."""
    if axis == "x":
        pos = edge_p1[0]
        return np.array([
            [pos, floor_h, edge_p1[1]],
            [pos, floor_h, edge_p2[1]],
            [pos, ceiling_h, edge_p2[1]],
            [pos, ceiling_h, edge_p1[1]],
            [pos, floor_h, edge_p1[1]],
        ])
    else:
        pos = edge_p1[1]
        return np.array([
            [edge_p1[0], floor_h, pos],
            [edge_p2[0], floor_h, pos],
            [edge_p2[0], ceiling_h, pos],
            [edge_p1[0], ceiling_h, pos],
            [edge_p1[0], floor_h, pos],
        ])


def _compute_general_boundary(
    edge_p1: np.ndarray, edge_p2: np.ndarray,
    floor_h: float, ceiling_h: float,
) -> np.ndarray:
    """Compute 3D boundary for an arbitrary-angle synthetic wall."""
    return np.array([
        [edge_p1[0], floor_h, edge_p1[1]],
        [edge_p2[0], floor_h, edge_p2[1]],
        [edge_p2[0], ceiling_h, edge_p2[1]],
        [edge_p1[0], ceiling_h, edge_p1[1]],
        [edge_p1[0], floor_h, edge_p1[1]],
    ])
