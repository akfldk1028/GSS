"""Module C: Detect parallel wall pairs and compute thickness + center-lines.

Supports both Manhattan-aligned (±X/±Z) and arbitrary-angle wall normals.
Two walls are paired if they have parallel normals, close d-values,
and overlap sufficiently in the perpendicular extent.

Ref: Cloud2BIM group_segments / calculate_wall_axis pattern.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _wall_axis(plane: dict) -> str | None:
    """Determine which Manhattan axis a wall is aligned to ('x' or 'z').

    Returns None if the wall is not aligned to a Manhattan axis.
    """
    n = np.asarray(plane["normal"])
    if abs(n[0]) > 0.9:
        return "x"
    if abs(n[2]) > 0.9:
        return "z"
    return None


def _wall_axis_vector(plane: dict) -> np.ndarray | None:
    """Return the wall's normal projected to XZ plane (unit vector).

    Returns None if the wall is nearly horizontal (floor/ceiling).
    """
    n = np.asarray(plane["normal"])
    xz = np.array([n[0], 0.0, n[2]])
    norm = np.linalg.norm(xz)
    if norm < 0.1:
        return None
    return xz / norm


def _are_parallel(n1: np.ndarray, n2: np.ndarray, threshold: float = 0.98) -> bool:
    """Check if two normal vectors are parallel (|dot| > threshold)."""
    return abs(float(np.dot(n1, n2))) > threshold


def _wall_position_general(plane: dict, axis_vec: np.ndarray) -> float:
    """Get the wall's position along a given axis direction.

    Projects the plane's position point onto the axis direction.
    For n·p + d = 0 → closest point to origin is -d*n, position = dot(-d*n, axis).
    """
    n = np.asarray(plane["normal"])
    return float(-plane["d"] * np.dot(n, axis_vec) / (np.dot(n, n) + 1e-12))


def _wall_position(plane: dict) -> float:
    """Get the wall's position along its normal axis (Manhattan shortcut).

    For normal [1,0,0], d: plane eq x + d = 0 → position = -d.
    For normal [-1,0,0], d: plane eq -x + d = 0 → position = d.
    """
    n = np.asarray(plane["normal"])
    axis = _wall_axis(plane)
    if axis == "x":
        return -plane["d"] * np.sign(n[0])
    elif axis == "z":
        return -plane["d"] * np.sign(n[2])
    return -plane["d"]


def _wall_extents(plane: dict, axis: str) -> tuple[float, float, float, float]:
    """Get wall extents in the perpendicular horizontal axis and Y (height).

    Returns (perp_min, perp_max, y_min, y_max).
    """
    bnd = plane.get("boundary_3d")
    if bnd is None or len(bnd) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    pts = np.asarray(bnd)
    if pts.ndim != 2:
        return (0.0, 0.0, 0.0, 0.0)

    # Perpendicular horizontal axis: if wall is X-aligned, perp is Z; vice versa
    perp_idx = 2 if axis == "x" else 0
    return (
        float(pts[:, perp_idx].min()),
        float(pts[:, perp_idx].max()),
        float(pts[:, 1].min()),
        float(pts[:, 1].max()),
    )


def _wall_extents_general(
    plane: dict, normal_vec: np.ndarray,
) -> tuple[float, float, float, float]:
    """Get wall extents along the wall's direction and Y (height).

    For arbitrary-angle walls: project boundary onto the wall's tangent direction.
    Returns (along_min, along_max, y_min, y_max).
    """
    bnd = plane.get("boundary_3d")
    if bnd is None or len(bnd) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    pts = np.asarray(bnd)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return (0.0, 0.0, 0.0, 0.0)

    # Wall tangent: perpendicular to normal in XZ plane
    tangent = np.array([-normal_vec[2], 0.0, normal_vec[0]])
    # Project XZ components onto tangent
    pts_xz = pts[:, [0, 2]]
    tangent_2d = np.array([tangent[0], tangent[2]])
    projections = pts_xz @ tangent_2d

    return (
        float(projections.min()),
        float(projections.max()),
        float(pts[:, 1].min()),
        float(pts[:, 1].max()),
    )


def _overlap_fraction(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """Compute overlap fraction between two 1D intervals."""
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    len_a = a_max - a_min
    len_b = b_max - b_min
    if len_a < 1e-6 or len_b < 1e-6:
        return 0.0
    return overlap / min(len_a, len_b)


def _compute_center_line_manhattan(
    axis: str, center_pos: float, perp_min: float, perp_max: float,
) -> list[list[float]]:
    """Compute center-line for Manhattan-aligned walls."""
    if axis == "x":
        return [[center_pos, perp_min], [center_pos, perp_max]]
    else:
        return [[perp_min, center_pos], [perp_max, center_pos]]


def _compute_center_line_general(
    normal_vec: np.ndarray,
    center_d: float,
    along_min: float,
    along_max: float,
) -> list[list[float]]:
    """Compute center-line for arbitrary-angle walls.

    The wall tangent direction is perpendicular to the normal in XZ plane.
    Center-line runs from along_min to along_max in the tangent direction,
    positioned at center_d along the normal.
    """
    # Wall tangent (in 3D, Y=0)
    tangent = np.array([-normal_vec[2], 0.0, normal_vec[0]])

    # Center point on the normal axis
    center_point_xz = center_d * np.array([normal_vec[0], normal_vec[2]])

    tangent_2d = np.array([tangent[0], tangent[2]])

    p1_xz = center_point_xz + along_min * tangent_2d
    p2_xz = center_point_xz + along_max * tangent_2d

    return [p1_xz.tolist(), p2_xz.tolist()]


def _normal_axis_label(plane: dict) -> str:
    """Return 'x', 'z', or 'oblique' depending on wall alignment."""
    axis = _wall_axis(plane)
    return axis if axis is not None else "oblique"


def _group_by_normal(
    wall_planes: list[tuple[int, dict]], parallel_threshold: float = 0.98,
) -> list[list[tuple[int, dict]]]:
    """Group wall planes by parallel normals.

    Returns list of groups, each group contains planes with parallel normals.
    Works for both Manhattan and arbitrary angles.
    """
    groups: list[list[tuple[int, dict]]] = []
    group_normals: list[np.ndarray] = []

    for item in wall_planes:
        _, p = item
        n_vec = _wall_axis_vector(p)
        if n_vec is None:
            continue

        assigned = False
        for gi, g_normal in enumerate(group_normals):
            if _are_parallel(n_vec, g_normal, parallel_threshold):
                groups[gi].append(item)
                assigned = True
                break

        if not assigned:
            groups.append([item])
            group_normals.append(n_vec)

    return groups


def compute_wall_thickness(
    planes: list[dict],
    max_wall_thickness: float = 1.0,
    default_wall_thickness: float = 0.2,
    min_parallel_overlap: float = 0.3,
) -> list[dict]:
    """Detect parallel wall pairs and compute thickness + center-lines.

    Supports both Manhattan (±X/±Z) and arbitrary-angle wall normals.

    Args:
        planes: plane dicts in Manhattan space.
        max_wall_thickness: max distance between parallel planes to pair.
        default_wall_thickness: thickness for unpaired walls.
        min_parallel_overlap: min overlap fraction in perpendicular direction.

    Returns:
        list of wall info dicts with keys:
        {id, plane_ids, center_line_2d, thickness, height_range, normal_axis}
        normal_axis is "x", "z", or "oblique:<angle_deg>" for non-Manhattan.
    """
    wall_planes = [(i, p) for i, p in enumerate(planes) if p["label"] == "wall"]

    # Group by parallel normals (works for any angle)
    normal_groups = _group_by_normal(wall_planes)

    walls: list[dict] = []
    paired: set[int] = set()
    wall_id = 0

    for group in normal_groups:
        if not group:
            continue

        # Determine if this is a Manhattan-aligned group
        sample_axis = _wall_axis(group[0][1])
        is_manhattan = sample_axis is not None
        group_normal = _wall_axis_vector(group[0][1])
        if group_normal is None:
            continue

        if is_manhattan:
            # Manhattan path: use existing position/extents logic
            members_sorted = sorted(group, key=lambda x: _wall_position(x[1]))

            for i in range(len(members_sorted)):
                idx_i, pi = members_sorted[i]
                if idx_i in paired:
                    continue
                pos_i = _wall_position(pi)
                ext_i = _wall_extents(pi, sample_axis)

                best_j = None
                best_dist = max_wall_thickness + 1

                for j in range(i + 1, len(members_sorted)):
                    idx_j, pj = members_sorted[j]
                    if idx_j in paired:
                        continue
                    pos_j = _wall_position(pj)
                    dist = abs(pos_j - pos_i)

                    if dist > max_wall_thickness:
                        break

                    ext_j = _wall_extents(pj, sample_axis)
                    overlap = _overlap_fraction(ext_i[0], ext_i[1], ext_j[0], ext_j[1])

                    if overlap >= min_parallel_overlap and dist < best_dist:
                        best_j = j
                        best_dist = dist

                if best_j is not None:
                    idx_j, pj = members_sorted[best_j]
                    pos_j = _wall_position(pj)
                    ext_j = _wall_extents(pj, sample_axis)

                    thickness = abs(pos_j - pos_i)
                    center_pos = (pos_i + pos_j) / 2.0

                    perp_min = min(ext_i[0], ext_j[0])
                    perp_max = max(ext_i[1], ext_j[1])
                    y_min = min(ext_i[2], ext_j[2])
                    y_max = max(ext_i[3], ext_j[3])

                    cl = _compute_center_line_manhattan(sample_axis, center_pos, perp_min, perp_max)

                    walls.append({
                        "id": wall_id,
                        "plane_ids": [pi["id"], pj["id"]],
                        "center_line_2d": cl,
                        "thickness": float(thickness),
                        "height_range": [float(y_min), float(y_max)],
                        "normal_axis": sample_axis,
                    })
                    paired.add(idx_i)
                    paired.add(idx_j)
                    wall_id += 1
                    logger.info(
                        f"Wall pair: planes [{pi['id']}, {pj['id']}], "
                        f"thickness={thickness:.3f}, axis={sample_axis}"
                    )
        else:
            # General path: use normal vector for projection
            canon = group_normal if group_normal[0] > 1e-6 or (abs(group_normal[0]) < 1e-6 and group_normal[2] > 1e-6) else -group_normal
            members_sorted = sorted(
                group, key=lambda x: _wall_position_general(x[1], canon),
            )

            for i in range(len(members_sorted)):
                idx_i, pi = members_sorted[i]
                if idx_i in paired:
                    continue
                pos_i = _wall_position_general(pi, canon)
                ext_i = _wall_extents_general(pi, canon)

                best_j = None
                best_dist = max_wall_thickness + 1

                for j in range(i + 1, len(members_sorted)):
                    idx_j, pj = members_sorted[j]
                    if idx_j in paired:
                        continue
                    pos_j = _wall_position_general(pj, canon)
                    dist = abs(pos_j - pos_i)

                    if dist > max_wall_thickness:
                        break

                    ext_j = _wall_extents_general(pj, canon)
                    overlap = _overlap_fraction(ext_i[0], ext_i[1], ext_j[0], ext_j[1])

                    if overlap >= min_parallel_overlap and dist < best_dist:
                        best_j = j
                        best_dist = dist

                if best_j is not None:
                    idx_j, pj = members_sorted[best_j]
                    pos_j = _wall_position_general(pj, canon)
                    ext_j = _wall_extents_general(pj, canon)

                    thickness = abs(pos_j - pos_i)
                    center_d = (pos_i + pos_j) / 2.0

                    along_min = min(ext_i[0], ext_j[0])
                    along_max = max(ext_i[1], ext_j[1])
                    y_min = min(ext_i[2], ext_j[2])
                    y_max = max(ext_i[3], ext_j[3])

                    cl = _compute_center_line_general(canon, center_d, along_min, along_max)

                    # Compute angle for label
                    angle_deg = float(np.degrees(np.arctan2(canon[2], canon[0])))
                    axis_label = f"oblique:{angle_deg:.0f}"

                    walls.append({
                        "id": wall_id,
                        "plane_ids": [pi["id"], pj["id"]],
                        "center_line_2d": cl,
                        "thickness": float(thickness),
                        "height_range": [float(y_min), float(y_max)],
                        "normal_axis": axis_label,
                        "normal_vector": canon[[0, 2]].tolist(),
                    })
                    paired.add(idx_i)
                    paired.add(idx_j)
                    wall_id += 1
                    logger.info(
                        f"Wall pair: planes [{pi['id']}, {pj['id']}], "
                        f"thickness={thickness:.3f}, axis={axis_label}"
                    )

    # Unpaired walls get default thickness
    for idx, p in wall_planes:
        if idx in paired:
            continue
        axis = _wall_axis(p)
        n_vec = _wall_axis_vector(p)
        if n_vec is None:
            continue

        if axis is not None:
            # Manhattan path
            pos = _wall_position(p)
            ext = _wall_extents(p, axis)
            cl = _compute_center_line_manhattan(axis, pos, ext[0], ext[1])
            walls.append({
                "id": wall_id,
                "plane_ids": [p["id"]],
                "center_line_2d": cl,
                "thickness": float(default_wall_thickness),
                "height_range": [float(ext[2]), float(ext[3])],
                "normal_axis": axis,
            })
        else:
            # General path
            canon = n_vec if n_vec[0] > 1e-6 or (abs(n_vec[0]) < 1e-6 and n_vec[2] > 1e-6) else -n_vec
            pos = _wall_position_general(p, canon)
            ext = _wall_extents_general(p, canon)
            cl = _compute_center_line_general(canon, pos, ext[0], ext[1])
            angle_deg = float(np.degrees(np.arctan2(canon[2], canon[0])))
            axis_label = f"oblique:{angle_deg:.0f}"
            walls.append({
                "id": wall_id,
                "plane_ids": [p["id"]],
                "center_line_2d": cl,
                "thickness": float(default_wall_thickness),
                "height_range": [float(ext[2]), float(ext[3])],
                "normal_axis": axis_label,
                "normal_vector": canon[[0, 2]].tolist(),
            })
        wall_id += 1

    logger.info(
        f"Wall thickness: {sum(1 for w in walls if len(w['plane_ids']) == 2)} paired, "
        f"{sum(1 for w in walls if len(w['plane_ids']) == 1)} unpaired, "
        f"{len(walls)} total"
    )
    return walls
