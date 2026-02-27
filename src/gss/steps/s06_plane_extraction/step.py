"""Step 06: Plane extraction and boundary polyline generation."""

from __future__ import annotations

import json
import logging
import math
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from .config import PlaneExtractionConfig
from .contracts import DetectedPlane, PlaneExtractionInput, PlaneExtractionOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C. Manhattan World Alignment
# ---------------------------------------------------------------------------

def _estimate_manhattan_frame(pcd) -> np.ndarray | None:
    """Detect dominant orthogonal axes from point cloud normals.

    Returns 3x3 rotation matrix R such that R @ point aligns the cloud
    to Manhattan axes, or None if detection fails.
    """
    import open3d as o3d

    # Estimate normals if missing
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.5, max_nn=30,
        ))

    normals = np.asarray(pcd.normals)
    if len(normals) < 100:
        return None

    # Make all normals point to the same hemisphere (consistent sign)
    # We'll work with absolute directions, so map n -> n if n[max_component] > 0
    abs_normals = normals.copy()
    # For each normal, flip so the largest-magnitude component is positive
    for i in range(len(abs_normals)):
        max_idx = np.argmax(np.abs(abs_normals[i]))
        if abs_normals[i, max_idx] < 0:
            abs_normals[i] *= -1

    # --- Step 1: Find gravity (up) direction ---
    # Gravity direction = the axis with most normals close to it
    # Use histogram of |cos(angle with each axis candidate)|
    # Instead, use SVD-based approach: cluster normals near vertical
    # Simple: find the axis that most normals are nearly parallel to

    # Compute angle of each normal with Y and Z axes
    # Pick the one that has the strongest peak (most normals within 15°)
    candidates = []
    for axis_vec in [np.array([0, 1, 0]), np.array([0, 0, 1])]:
        cos_angles = np.abs(abs_normals @ axis_vec)
        near_vertical = np.sum(cos_angles > np.cos(np.radians(15)))
        candidates.append((near_vertical, axis_vec))

    # Also try PCA: the direction with most normals clustered
    # Use normal histogram on unit sphere
    # Bin normals by elevation angle to find dominant vertical
    # Simple approach: SVD on normals that are "nearly vertical"
    best_count, best_up_hint = max(candidates, key=lambda x: x[0])

    # Refine: collect normals within 15° of best_up_hint, SVD to get precise up
    cos_thresh = np.cos(np.radians(15))
    cos_angles = np.abs(abs_normals @ best_up_hint)
    vertical_mask = cos_angles > cos_thresh
    vertical_normals = abs_normals[vertical_mask]

    if len(vertical_normals) < 10:
        logger.warning("Manhattan alignment: too few vertical normals, skipping")
        return None

    # SVD to find dominant direction among vertical normals
    _, _, Vt = np.linalg.svd(vertical_normals, full_matrices=False)
    up = Vt[0]  # first principal component
    if up[np.argmax(np.abs(up))] < 0:
        up = -up
    up /= np.linalg.norm(up)

    # --- Step 2: Find dominant horizontal direction ---
    # Horizontal normals = normals nearly perpendicular to up (within 15° of 90°)
    cos_with_up = abs_normals @ up
    horizontal_mask = np.abs(cos_with_up) < np.sin(np.radians(15))
    horizontal_normals = abs_normals[horizontal_mask]

    if len(horizontal_normals) < 10:
        logger.warning("Manhattan alignment: too few horizontal normals, skipping")
        return None

    # Project horizontal normals onto the plane perpendicular to up
    horiz_proj = horizontal_normals - np.outer(horizontal_normals @ up, up)
    norms = np.linalg.norm(horiz_proj, axis=1, keepdims=True)
    valid = norms.flatten() > 1e-6
    horiz_proj = horiz_proj[valid]
    norms = norms[valid]
    horiz_proj /= norms

    # SVD to find dominant horizontal direction
    _, _, Vt_h = np.linalg.svd(horiz_proj, full_matrices=False)
    forward = Vt_h[0]
    forward -= forward @ up * up  # ensure orthogonal to up
    forward /= np.linalg.norm(forward)

    # --- Step 3: Build rotation matrix ---
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)  # re-orthogonalize

    # R maps: up->Y, forward->X, right->Z (Y-up convention)
    # Columns of R^T are the new axes expressed in old coords
    R = np.array([forward, up, right])  # rows = new axes

    logger.info(
        f"Manhattan frame detected: up={up.round(3)}, "
        f"forward={forward.round(3)}, vertical_normals={vertical_mask.sum()}, "
        f"horizontal_normals={horizontal_mask.sum()}"
    )
    return R


# ---------------------------------------------------------------------------
# Plane classification
# ---------------------------------------------------------------------------

def _classify_plane(normal: np.ndarray, angle_threshold: float, up_axis: str = "z") -> str:
    """Classify a plane as wall/floor/ceiling based on its normal vector."""
    if up_axis == "y":
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
    cos_angle = abs(np.dot(normal, up))
    angle_deg = math.degrees(math.acos(np.clip(cos_angle, 0, 1)))

    if angle_deg < angle_threshold:
        return "horizontal"  # refined to floor/ceiling/other by position later
    elif angle_deg > (90 - angle_threshold):
        return "wall"
    else:
        return "other"


def _refine_horizontal_labels(
    planes: list[dict],
    up_axis: str = "y",
    height_margin: float = 0.25,
) -> None:
    """Reclassify 'horizontal' planes to floor/ceiling/other by centroid position.

    Horizontal planes in the bottom *height_margin* fraction of the bounding box
    are 'floor', top fraction are 'ceiling', and everything in between is 'other'
    (furniture surfaces).
    """
    up_idx = 1 if up_axis == "y" else 2

    # Collect centroid heights for all horizontal planes
    horiz = [(i, p["inlier_points"][:, up_idx].mean()) for i, p in enumerate(planes)
             if p["label"] == "horizontal"]
    if not horiz:
        return

    all_heights = [h for _, h in horiz]
    h_min, h_max = min(all_heights), max(all_heights)
    h_range = h_max - h_min

    if h_range < 1e-3:
        # All at same height → floor
        for i, _ in horiz:
            planes[i]["label"] = "floor"
        return

    floor_thresh = h_min + h_range * height_margin
    ceil_thresh = h_max - h_range * height_margin

    for i, h in horiz:
        if h <= floor_thresh:
            planes[i]["label"] = "floor"
        elif h >= ceil_thresh:
            planes[i]["label"] = "ceiling"
        else:
            planes[i]["label"] = "other"
            logger.info(f"Reclassified plane {planes[i].get('_ransac_id', i)} "
                        f"from horizontal to other (furniture, height={h:.1f})")


# ---------------------------------------------------------------------------
# A. Coplanar Merging
# ---------------------------------------------------------------------------

def _merge_coplanar_planes(
    planes: list[dict],
    angle_thresh: float = 10.0,
    dist_thresh: float = 1.5,
) -> list[dict]:
    """Merge planes with same label, similar normals, and small centroid separation.

    Uses Union-Find for transitive merging: if A merges with B and B with C,
    all three become one plane.

    Coplanarity test (both must pass):
      1. Normal angle < angle_thresh
      2. Centroid-to-plane distance < dist_thresh (symmetric: check both directions)
    """
    if len(planes) <= 1:
        return planes

    n = len(planes)
    cos_thresh = np.cos(np.radians(angle_thresh))

    # Precompute centroids
    centroids = [p["inlier_points"].mean(axis=0) for p in planes]

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairwise coplanarity check
    for i in range(n):
        for j in range(i + 1, n):
            if planes[i]["label"] != planes[j]["label"]:
                continue
            # Normal similarity (handles flipped normals via abs)
            cos_angle = abs(np.dot(planes[i]["normal"], planes[j]["normal"]))
            if cos_angle < cos_thresh:
                continue
            # Centroid-to-plane distance (symmetric: check both directions)
            # dist_i = distance from centroid_j to plane_i
            dist_i = abs(np.dot(centroids[j] - centroids[i], planes[i]["normal"]))
            # dist_j = distance from centroid_i to plane_j
            dist_j = abs(np.dot(centroids[i] - centroids[j], planes[j]["normal"]))
            if max(dist_i, dist_j) > dist_thresh:
                continue
            union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Merge each group
    result: list[dict] = []
    for root, members in groups.items():
        all_points = np.concatenate([planes[k]["inlier_points"] for k in members])
        # SVD refit
        centroid = all_points.mean(axis=0)
        centered = all_points - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        new_normal = Vt[2]  # smallest singular value = plane normal
        new_normal /= np.linalg.norm(new_normal)
        # Ensure consistent orientation with the largest plane in the group
        ref_idx = max(members, key=lambda k: len(planes[k]["inlier_points"]))
        if np.dot(new_normal, planes[ref_idx]["normal"]) < 0:
            new_normal = -new_normal
        new_d = -np.dot(new_normal, centroid)

        if len(members) > 1:
            logger.info(
                f"Merged {len(members)} '{planes[root]['label']}' planes "
                f"(ids={members}, {len(all_points)} total inliers)"
            )

        result.append({
            "normal": new_normal,
            "d": new_d,
            "label": planes[root]["label"],
            "inlier_points": all_points,
        })

    return result


# ---------------------------------------------------------------------------
# B. Boundary extraction (improved)
# ---------------------------------------------------------------------------

def _extract_boundary(
    inlier_points: np.ndarray,
    normal: np.ndarray,
    simplify_tolerance: float,
    label: str = "other",
) -> list[list[float]]:
    """Project plane inliers to 2D, compute boundary, reproject to 3D.

    For wall/floor/ceiling: use minimum_rotated_rectangle (clean rectangles).
    For other: use alpha shape (arbitrary geometry).
    """
    from shapely.geometry import MultiPoint, Polygon

    if len(inlier_points) < 4:
        return []

    # Build a local 2D coordinate frame on the plane
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    centroid = inlier_points.mean(axis=0)
    local = inlier_points - centroid
    coords_2d = np.column_stack([local @ u, local @ v])

    if label in ("wall", "floor", "ceiling"):
        # B. Use minimum rotated rectangle for architectural elements
        mp = MultiPoint(coords_2d)
        rect = mp.minimum_rotated_rectangle
        if rect.is_empty or not hasattr(rect, "exterior"):
            return []
        boundary_2d = np.array(rect.exterior.coords)
    else:
        # Downsample for alpha shape performance (>10K points is slow)
        if len(coords_2d) > 10000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(coords_2d), 10000, replace=False)
            coords_2d_sample = coords_2d[idx]
        else:
            coords_2d_sample = coords_2d
        try:
            import alphashape
            alpha = alphashape.optimizealpha(coords_2d_sample, max_iterations=25)
            shape = alphashape.alphashape(coords_2d_sample, alpha)
        except Exception:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords_2d_sample)
            hull_pts = coords_2d_sample[hull.vertices]
            shape = Polygon(hull_pts)

        if shape.is_empty or not hasattr(shape, "exterior"):
            return []

        simplified = shape.simplify(simplify_tolerance, preserve_topology=True)
        if simplified.is_empty:
            return []

        if hasattr(simplified, "exterior"):
            boundary_2d = np.array(simplified.exterior.coords)
        else:
            return []

    # Reproject to 3D
    boundary_3d = centroid + boundary_2d[:, 0:1] * u + boundary_2d[:, 1:2] * v
    return boundary_3d.tolist()


# ---------------------------------------------------------------------------
# Step class
# ---------------------------------------------------------------------------

class PlaneExtractionStep(
    BaseStep[PlaneExtractionInput, PlaneExtractionOutput, PlaneExtractionConfig]
):
    name: ClassVar[str] = "plane_extraction"
    input_type: ClassVar = PlaneExtractionInput
    output_type: ClassVar = PlaneExtractionOutput
    config_type: ClassVar = PlaneExtractionConfig

    def validate_inputs(self, inputs: PlaneExtractionInput) -> bool:
        return inputs.surface_points_path.exists()

    def run(self, inputs: PlaneExtractionInput) -> PlaneExtractionOutput:
        import open3d as o3d

        output_dir = self.data_root / "interim" / "s06_planes"
        output_dir.mkdir(parents=True, exist_ok=True)

        pcd = o3d.io.read_point_cloud(str(inputs.surface_points_path))

        # --- C. Manhattan World Alignment ---
        R_manhattan = None
        if self.config.normalize_coords:
            R_manhattan = _estimate_manhattan_frame(pcd)
            if R_manhattan is not None:
                points = np.asarray(pcd.points)
                pcd.points = o3d.utility.Vector3dVector(points @ R_manhattan.T)
                if pcd.has_normals():
                    normals = np.asarray(pcd.normals)
                    pcd.normals = o3d.utility.Vector3dVector(normals @ R_manhattan.T)
                logger.info("Applied Manhattan World alignment")
                # After alignment, coordinate system is Y-up
                effective_up_axis = "y"
            else:
                effective_up_axis = self.config.up_axis
        else:
            effective_up_axis = self.config.up_axis

        # --- RANSAC plane segmentation ---
        remaining = pcd
        raw_planes: list[dict] = []

        for plane_id in range(self.config.max_planes):
            if len(remaining.points) < self.config.min_inliers:
                break

            plane_model, inlier_indices = remaining.segment_plane(
                distance_threshold=self.config.distance_threshold,
                ransac_n=3,
                num_iterations=self.config.ransac_iterations,
            )

            if len(inlier_indices) < self.config.min_inliers:
                break

            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal_len = np.linalg.norm(normal)
            if normal_len < 1e-6:
                break
            normal /= normal_len
            d /= normal_len

            label = _classify_plane(normal, self.config.angle_threshold, effective_up_axis)

            inlier_pcd = remaining.select_by_index(inlier_indices)
            inlier_points = np.asarray(inlier_pcd.points).copy()

            raw_planes.append({
                "normal": normal,
                "d": float(d),
                "label": label,
                "inlier_points": inlier_points,
                "_ransac_id": plane_id,
            })

            remaining = remaining.select_by_index(inlier_indices, invert=True)
            logger.info(
                f"RANSAC plane {plane_id}: {label}, {len(inlier_indices)} inliers"
            )

        # --- Refine horizontal plane labels by position ---
        _refine_horizontal_labels(raw_planes, effective_up_axis)

        # --- A. Coplanar Merging ---
        if self.config.merge_coplanar and len(raw_planes) > 1:
            before_count = len(raw_planes)
            raw_planes = _merge_coplanar_planes(
                raw_planes,
                angle_thresh=self.config.merge_angle_threshold,
                dist_thresh=self.config.merge_distance_threshold,
            )
            logger.info(f"Coplanar merge: {before_count} → {len(raw_planes)} planes")

        # --- Filter small architectural planes as furniture ---
        for arch_label in ("wall", "ceiling"):
            counts = [len(p["inlier_points"]) for p in raw_planes if p["label"] == arch_label]
            if not counts:
                continue
            thresh = max(counts) * self.config.wall_min_ratio
            for p in raw_planes:
                if p["label"] == arch_label and len(p["inlier_points"]) < thresh:
                    logger.info(
                        f"Reclassified {arch_label} ({len(p['inlier_points'])} inliers) "
                        f"to other (below {self.config.wall_min_ratio:.0%} of max {arch_label})"
                    )
                    p["label"] = "other"

        # --- Build final DetectedPlane list ---
        planes: list[DetectedPlane] = []
        for idx, rp in enumerate(raw_planes):
            inlier_points = rp["inlier_points"]

            boundary_3d = _extract_boundary(
                inlier_points, rp["normal"],
                self.config.simplify_tolerance, rp["label"],
            )

            if R_manhattan is not None and boundary_3d:
                # Inverse transform: R^T (since R is orthogonal, R^-1 = R^T)
                R_inv = R_manhattan.T
                boundary_3d = (np.array(boundary_3d) @ R_inv.T).tolist()
                # Also transform normal back
                original_normal = (R_inv @ rp["normal"]).tolist()
                original_d = float(-np.dot(
                    R_inv @ rp["normal"],
                    R_inv @ inlier_points.mean(axis=0),
                ))
            else:
                original_normal = rp["normal"].tolist()
                original_d = float(rp["d"])

            planes.append(
                DetectedPlane(
                    id=idx,
                    normal=original_normal,
                    d=original_d,
                    label=rp["label"],
                    num_inliers=len(inlier_points),
                    boundary_3d=boundary_3d,
                )
            )
            logger.info(
                f"Plane {idx}: {rp['label']}, {len(inlier_points)} inliers, "
                f"{len(boundary_3d)} boundary verts"
            )

        # --- Save outputs ---
        planes_data = [p.model_dump() for p in planes]
        planes_file = output_dir / "planes.json"
        with open(planes_file, "w") as f:
            json.dump(planes_data, f, indent=2)

        boundaries_data = [
            {"id": p.id, "label": p.label, "boundary_3d": p.boundary_3d}
            for p in planes
        ]
        boundaries_file = output_dir / "boundaries.json"
        with open(boundaries_file, "w") as f:
            json.dump(boundaries_data, f, indent=2)

        # Save Manhattan rotation if used (for downstream reference)
        if R_manhattan is not None:
            meta = {"manhattan_rotation": R_manhattan.tolist()}
            with open(output_dir / "manhattan_alignment.json", "w") as f:
                json.dump(meta, f, indent=2)

        num_walls = sum(1 for p in planes if p.label == "wall")
        num_floors = sum(1 for p in planes if p.label == "floor")
        num_ceilings = sum(1 for p in planes if p.label == "ceiling")

        logger.info(
            f"Extracted {len(planes)} planes: "
            f"{num_walls} walls, {num_floors} floors, {num_ceilings} ceilings"
        )

        return PlaneExtractionOutput(
            planes_file=planes_file,
            boundaries_file=boundaries_file,
            num_planes=len(planes),
            num_walls=num_walls,
            num_floors=num_floors,
            num_ceilings=num_ceilings,
        )
