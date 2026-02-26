"""Module C: Detect parallel wall pairs and compute thickness + center-lines.

In Manhattan Y-up space, walls have axis-aligned normals (±X or ±Z).
Two walls are paired if they share the same axis, have close d-values,
and overlap sufficiently in the perpendicular extent.

Ref: Cloud2BIM group_segments / calculate_wall_axis pattern.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _wall_axis(plane: dict) -> str | None:
    """Determine which Manhattan axis a wall is aligned to ('x' or 'z')."""
    n = np.asarray(plane["normal"])
    if abs(n[0]) > 0.9:
        return "x"
    if abs(n[2]) > 0.9:
        return "z"
    return None


def _wall_position(plane: dict) -> float:
    """Get the wall's position along its normal axis.

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


def _overlap_fraction(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """Compute overlap fraction between two 1D intervals."""
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    len_a = a_max - a_min
    len_b = b_max - b_min
    if len_a < 1e-6 or len_b < 1e-6:
        return 0.0
    return overlap / min(len_a, len_b)


def compute_wall_thickness(
    planes: list[dict],
    max_wall_thickness: float = 1.0,
    default_wall_thickness: float = 0.2,
    min_parallel_overlap: float = 0.3,
) -> list[dict]:
    """Detect parallel wall pairs and compute thickness + center-lines.

    Args:
        planes: plane dicts in Manhattan space.
        max_wall_thickness: max distance between parallel planes to pair.
        default_wall_thickness: thickness for unpaired walls.
        min_parallel_overlap: min overlap fraction in perpendicular direction.

    Returns:
        list of wall info dicts with keys:
        {id, plane_ids, center_line_2d, thickness, height_range, normal_axis}
    """
    # Group wall planes by axis
    wall_planes = [(i, p) for i, p in enumerate(planes) if p["label"] == "wall"]
    groups: dict[str, list[tuple[int, dict]]] = {"x": [], "z": []}
    for i, p in wall_planes:
        axis = _wall_axis(p)
        if axis is not None:
            groups[axis].append((i, p))

    walls: list[dict] = []
    paired: set[int] = set()
    wall_id = 0

    for axis, members in groups.items():
        # Sort by position along the normal axis
        members_sorted = sorted(members, key=lambda x: _wall_position(x[1]))

        # Greedy pairing: check adjacent members for proximity
        for i in range(len(members_sorted)):
            idx_i, pi = members_sorted[i]
            if idx_i in paired:
                continue
            pos_i = _wall_position(pi)
            ext_i = _wall_extents(pi, axis)

            best_j = None
            best_dist = max_wall_thickness + 1

            for j in range(i + 1, len(members_sorted)):
                idx_j, pj = members_sorted[j]
                if idx_j in paired:
                    continue
                pos_j = _wall_position(pj)
                dist = abs(pos_j - pos_i)

                if dist > max_wall_thickness:
                    break  # sorted, so all further are farther

                ext_j = _wall_extents(pj, axis)
                overlap = _overlap_fraction(ext_i[0], ext_i[1], ext_j[0], ext_j[1])

                if overlap >= min_parallel_overlap and dist < best_dist:
                    best_j = j
                    best_dist = dist

            if best_j is not None:
                idx_j, pj = members_sorted[best_j]
                pos_j = _wall_position(pj)
                ext_j = _wall_extents(pj, axis)

                thickness = abs(pos_j - pos_i)
                center_pos = (pos_i + pos_j) / 2.0

                # Perpendicular extent: union of both walls
                perp_min = min(ext_i[0], ext_j[0])
                perp_max = max(ext_i[1], ext_j[1])
                y_min = min(ext_i[2], ext_j[2])
                y_max = max(ext_i[3], ext_j[3])

                # Center-line in XZ plane
                if axis == "x":
                    cl = [[center_pos, perp_min], [center_pos, perp_max]]
                else:
                    cl = [[perp_min, center_pos], [perp_max, center_pos]]

                walls.append({
                    "id": wall_id,
                    "plane_ids": [pi["id"], pj["id"]],
                    "center_line_2d": cl,
                    "thickness": float(thickness),
                    "height_range": [float(y_min), float(y_max)],
                    "normal_axis": axis,
                })
                paired.add(idx_i)
                paired.add(idx_j)
                wall_id += 1
                logger.info(
                    f"Wall pair: planes [{pi['id']}, {pj['id']}], "
                    f"thickness={thickness:.3f}, axis={axis}"
                )

    # Unpaired walls get default thickness
    for idx, p in wall_planes:
        if idx in paired:
            continue
        axis = _wall_axis(p) or "x"
        pos = _wall_position(p)
        ext = _wall_extents(p, axis)

        if axis == "x":
            cl = [[pos, ext[0]], [pos, ext[1]]]
        else:
            cl = [[ext[0], pos], [ext[1], pos]]

        walls.append({
            "id": wall_id,
            "plane_ids": [p["id"]],
            "center_line_2d": cl,
            "thickness": float(default_wall_thickness),
            "height_range": [float(ext[2]), float(ext[3])],
            "normal_axis": axis,
        })
        wall_id += 1

    logger.info(
        f"Wall thickness: {sum(1 for w in walls if len(w['plane_ids']) == 2)} paired, "
        f"{sum(1 for w in walls if len(w['plane_ids']) == 1)} unpaired, "
        f"{len(walls)} total"
    )
    return walls
