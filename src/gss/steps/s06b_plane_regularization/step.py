"""Step 06b: Plane regularization — geometric cleanup between RANSAC and IFC.

Orchestrates sub-modules A→B→C→C2→D→E→G→(F) in order:
  A. Normal snapping (manhattan: ±X/±Z, cluster: data-driven axes)
  B. Height snapping (cluster floor/ceiling d-values)
  C. Wall thickness (parallel pair detection + center-lines)
  C2. Wall closure (synthesize missing walls from floor boundary)
  D. Intersection trimming (snap center-line endpoints to corners)
  E. Space detection (polygonize center-lines → room boundaries)
  G. Exterior classification (interior/exterior wall labeling, disabled by default)
  F. Opening detection (disabled by default)

All processing in Manhattan-aligned Y-up space. Results transformed back
to original coordinates for backward-compatible output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from .config import PlaneRegularizationConfig
from .contracts import PlaneRegularizationInput, PlaneRegularizationOutput

logger = logging.getLogger(__name__)


def _load_planes(planes_file: Path) -> list[dict]:
    """Load planes.json and convert arrays to numpy."""
    with open(planes_file) as f:
        data = json.load(f)
    for p in data:
        p["normal"] = np.asarray(p["normal"], dtype=float)
        if p.get("boundary_3d"):
            p["boundary_3d"] = np.asarray(p["boundary_3d"], dtype=float)
        else:
            p["boundary_3d"] = np.empty((0, 3))
    return data


def _load_manhattan_rotation(s06_dir: Path) -> np.ndarray | None:
    """Load Manhattan rotation matrix from s06 output."""
    path = s06_dir / "manhattan_alignment.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        R = np.asarray(data["manhattan_rotation"], dtype=float)
        if R.shape != (3, 3):
            logger.warning(f"Invalid manhattan_rotation shape {R.shape}, ignoring")
            return None
        return R
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to load manhattan_alignment.json: {e}; proceeding without Manhattan rotation")
        return None


def _transform_to_manhattan(planes: list[dict], R: np.ndarray) -> None:
    """Transform planes from original to Manhattan-aligned coordinates (in-place).

    R maps original → Manhattan: p_m = R @ p_o, n_m = R @ n_o.
    d is invariant under orthogonal rotation.
    """
    for p in planes:
        p["normal"] = R @ p["normal"]
        if len(p["boundary_3d"]) > 0:
            p["boundary_3d"] = (p["boundary_3d"] @ R.T)


def _transform_from_manhattan(planes: list[dict], R: np.ndarray) -> None:
    """Transform planes back from Manhattan to original coordinates (in-place).

    Inverse: R^T (R is orthogonal).
    """
    R_inv = R.T
    for p in planes:
        p["normal"] = R_inv @ p["normal"]
        if len(p["boundary_3d"]) > 0:
            p["boundary_3d"] = (p["boundary_3d"] @ R_inv.T)


def _reproject_boundary(plane: dict) -> None:
    """Project boundary_3d points onto the (possibly snapped) plane.

    After normal/d changes, boundary points may be slightly off-plane.
    Project each point onto the new plane to maintain consistency.
    """
    bnd = plane["boundary_3d"]
    if len(bnd) == 0:
        return
    n = plane["normal"]
    d = plane["d"]
    # For each point p, project: p' = p - (n·p + d) * n
    dots = bnd @ n + d
    plane["boundary_3d"] = bnd - np.outer(dots, n)


def _rebuild_wall_boundary(plane: dict, wall: dict) -> None:
    """Rebuild a wall's boundary_3d from its center-line + thickness + height.

    Creates a 3D rectangle from the wall's center-line (XZ) and height (Y).
    """
    cl = wall["center_line_2d"]
    p1 = np.array(cl[0])
    p2 = np.array(cl[1])
    y_min, y_max = wall["height_range"]

    # 3D rectangle: 4 corners + closing point
    plane["boundary_3d"] = np.array([
        [p1[0], y_min, p1[1]],
        [p2[0], y_min, p2[1]],
        [p2[0], y_max, p2[1]],
        [p1[0], y_max, p1[1]],
        [p1[0], y_min, p1[1]],
    ])


def _serialize_planes(planes: list[dict]) -> list[dict]:
    """Convert planes back to JSON-serializable dicts."""
    result = []
    for p in planes:
        entry = {
            "id": p["id"],
            "normal": p["normal"].tolist() if isinstance(p["normal"], np.ndarray) else p["normal"],
            "d": float(p["d"]),
            "label": p["label"],
            "num_inliers": p.get("num_inliers", 0),
            "boundary_3d": (
                p["boundary_3d"].tolist()
                if isinstance(p["boundary_3d"], np.ndarray) and len(p["boundary_3d"]) > 0
                else []
            ),
        }
        result.append(entry)
    return result


def _estimate_scale(
    planes: list[dict],
    expected_storey_height: float = 2.7,
    expected_room_size: float = 5.0,
) -> float:
    """Estimate coordinate scale using architectural height priors.

    In Manhattan-aligned space (Y-up), uses three strategies in priority order:

    1. **Floor-ceiling height difference**: Most reliable. If both floor and
       ceiling planes exist, their Y-distance ≈ storey height (~2.7m).
    2. **Wall Y-extent**: If no floor/ceiling, wall boundary Y-range
       approximates room height (walls span floor to ceiling).
    3. **XZ bounding box fallback**: Last resort, uses median of XZ extents
       divided by expected_room_size (least reliable).

    Returns scale factor (scene_units / meter).
    """
    # --- Collect plane heights and wall Y-extents ---
    floor_heights: list[float] = []
    ceiling_heights: list[float] = []
    wall_y_extents: list[float] = []
    arch_pts: list[np.ndarray] = []  # for fallback

    for p in planes:
        label = p.get("label", "other")
        bnd = p.get("boundary_3d")
        has_bnd = bnd is not None and len(bnd) > 0

        if label in ("floor", "ceiling"):
            # Height from plane equation: n·p + d = 0 → y = -d/ny
            normal = p.get("normal")
            d_val = p.get("d")
            ny = normal[1] if normal is not None and hasattr(normal, "__len__") and len(normal) > 1 else 0
            if abs(ny) > 0.3 and d_val is not None:
                h = -d_val / ny
            elif has_bnd:
                h = float(np.asarray(bnd)[:, 1].mean())
            else:
                continue

            if label == "floor":
                floor_heights.append(h)
            else:
                ceiling_heights.append(h)

        if label == "wall" and has_bnd:
            pts = np.asarray(bnd)
            if pts.ndim == 2 and pts.shape[0] >= 2:
                y_ext = pts[:, 1].max() - pts[:, 1].min()
                if y_ext > 0.1:
                    wall_y_extents.append(y_ext)

        if label in ("wall", "floor", "ceiling") and has_bnd:
            pts = np.asarray(bnd)
            if pts.ndim == 2:
                arch_pts.append(pts)

    # --- Strategy 1: Floor-ceiling distance ---
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

    # --- Strategy 2: Wall Y-extent (proxy for room height) ---
    if wall_y_extents:
        # Use median wall height to be robust to partial walls
        median_wall_h = float(np.median(wall_y_extents))
        if median_wall_h > 0.1:
            scale = median_wall_h / expected_storey_height
            scale = max(0.1, min(scale, 100.0))
            logger.info(
                f"Scale from wall height: {scale:.2f} "
                f"(median_wall_h={median_wall_h:.2f}, expected_h={expected_storey_height}m)"
            )
            return scale

    # --- Strategy 3: XZ bounding box fallback ---
    if arch_pts:
        combined = np.vstack(arch_pts)
        extents = combined.max(axis=0) - combined.min(axis=0)
        # In Manhattan Y-up space, XZ are the horizontal dims
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


def _transform_walls_from_manhattan(
    walls: list[dict], R: np.ndarray,
) -> list[dict]:
    """Transform walls.json center-lines from Manhattan to original coordinates.

    Walls have center_line_2d in XZ plane of Manhattan space.
    Convert to 3D, apply inverse rotation, return with center_line_3d.
    """
    R_inv = R.T
    result = []
    for w in walls:
        cl = w["center_line_2d"]
        p1_xz, p2_xz = np.array(cl[0]), np.array(cl[1])
        y_min, y_max = w["height_range"]
        y_mid = (y_min + y_max) / 2.0

        # 3D points in Manhattan space (Y-up): [x, y, z]
        p1_3d = np.array([p1_xz[0], y_mid, p1_xz[1]])
        p2_3d = np.array([p2_xz[0], y_mid, p2_xz[1]])

        # Transform to original coordinates
        p1_orig = R_inv @ p1_3d
        p2_orig = R_inv @ p2_3d

        entry = dict(w)
        entry["center_line_3d"] = [p1_orig.tolist(), p2_orig.tolist()]
        result.append(entry)
    return result


class PlaneRegularizationStep(
    BaseStep[PlaneRegularizationInput, PlaneRegularizationOutput, PlaneRegularizationConfig]
):
    name: ClassVar[str] = "plane_regularization"
    input_type: ClassVar = PlaneRegularizationInput
    output_type: ClassVar = PlaneRegularizationOutput
    config_type: ClassVar = PlaneRegularizationConfig

    def validate_inputs(self, inputs: PlaneRegularizationInput) -> bool:
        if not inputs.planes_file.exists():
            logger.error(f"planes_file not found: {inputs.planes_file}")
            return False
        if not inputs.boundaries_file.exists():
            logger.error(f"boundaries_file not found: {inputs.boundaries_file}")
            return False
        return True

    def run(self, inputs: PlaneRegularizationInput) -> PlaneRegularizationOutput:
        output_dir = self.data_root / "interim" / "s06b_plane_regularization"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Load data ---
        planes = _load_planes(inputs.planes_file)
        logger.info(f"Loaded {len(planes)} planes from {inputs.planes_file}")

        s06_dir = self.data_root / "interim" / "s06_planes"
        R = _load_manhattan_rotation(s06_dir)

        # --- Transform to Manhattan space ---
        manhattan_aligned = R is not None
        if manhattan_aligned:
            _transform_to_manhattan(planes, R)
            logger.info("Transformed to Manhattan-aligned coordinates")
        else:
            logger.warning(
                "No Manhattan rotation available — processing in original coordinates. "
                "Affected modules: normal snapping (may snap to wrong axes), "
                "wall thickness pairing (may miss parallel pairs), "
                "wall closure (AABB will be axis-misaligned), "
                "space detection (polygonization may fail). "
                "Results may be significantly degraded for non-axis-aligned scenes."
            )

        # --- Scale estimation ---
        if self.config.scale_mode == "auto":
            scale = _estimate_scale(
                planes,
                expected_storey_height=self.config.expected_storey_height,
                expected_room_size=self.config.expected_room_size,
            )
            logger.info(f"Auto scale estimation: {scale:.2f} scene_units/meter")
        elif self.config.scale_mode == "manual":
            scale = self.config.coordinate_scale
            logger.info(f"Manual scale: {scale:.2f} scene_units/meter")
        else:  # metric
            scale = 1.0

        # Compute effective tolerances (scale distance thresholds, keep angles)
        eff_height_tol = self.config.height_cluster_tolerance * scale
        eff_snap_tol = self.config.snap_tolerance * scale
        eff_min_area = self.config.min_space_area * scale * scale
        eff_default_thickness = self.config.default_wall_thickness * scale
        # max_wall_thickness scaled but capped at 5*scale to prevent cross-room pairing
        eff_max_wall_thickness = min(
            self.config.max_wall_thickness * scale,
            5.0 * scale,  # 5m equivalent — no wall is thicker than this
        )

        logger.info(
            f"Effective tolerances (scale={scale:.2f}): "
            f"height_tol={eff_height_tol:.2f}, snap_tol={eff_snap_tol:.2f}, "
            f"min_area={eff_min_area:.2f}, default_thickness={eff_default_thickness:.2f}"
        )

        # --- Stats collection ---
        stats: dict = {"manhattan_aligned": manhattan_aligned, "scale": float(scale)}

        # --- A. Normal Snapping ---
        normal_stats: dict = {"snapped_walls": 0, "snapped_horiz": 0, "skipped": 0}
        if self.config.enable_normal_snapping:
            from ._snap_normals import snap_normals
            normal_stats = snap_normals(
                planes,
                threshold_deg=self.config.normal_snap_threshold,
                normal_mode=self.config.normal_mode,
                cluster_angle_tolerance=self.config.cluster_angle_tolerance,
            )
            # Reproject boundaries onto snapped planes
            for p in planes:
                _reproject_boundary(p)
        stats["normal_snapping"] = normal_stats

        # --- B. Height Snapping ---
        height_stats: dict = {"floor_heights": [], "ceiling_heights": []}
        if self.config.enable_height_snapping:
            from ._snap_heights import snap_heights
            height_stats = snap_heights(
                planes, tolerance=eff_height_tol, scale=scale,
            )
            for p in planes:
                if p["label"] in ("floor", "ceiling"):
                    _reproject_boundary(p)
        stats["height_snapping"] = {
            "floor_clusters": len(height_stats.get("floor_heights", [])),
            "ceiling_clusters": len(height_stats.get("ceiling_heights", [])),
        }

        # --- C. Wall Thickness ---
        walls: list[dict] = []
        if self.config.enable_wall_thickness:
            from ._wall_thickness import compute_wall_thickness
            walls = compute_wall_thickness(
                planes,
                max_wall_thickness=eff_max_wall_thickness,
                default_wall_thickness=eff_default_thickness,
                min_parallel_overlap=self.config.min_parallel_overlap,
            )
        paired_count = sum(1 for w in walls if len(w.get("plane_ids", [])) == 2)
        stats["wall_thickness"] = {
            "total_walls": len(walls),
            "paired": paired_count,
            "unpaired": len(walls) - paired_count,
        }

        # --- C2. Wall Closure ---
        walls_before_closure = len(walls)
        if self.config.enable_wall_closure and walls:
            from ._wall_closure import synthesize_missing_walls
            walls, new_planes = synthesize_missing_walls(
                walls,
                planes,
                floor_heights=height_stats.get("floor_heights", []),
                ceiling_heights=height_stats.get("ceiling_heights", []),
                scale=scale,
                max_gap_ratio=self.config.max_closure_gap_ratio,
                use_floor_ceiling_hints=self.config.use_floor_ceiling_hints,
                default_thickness=eff_default_thickness,
                normal_mode=self.config.normal_mode,
                wall_closure_mode=self.config.wall_closure_mode,
            )
            planes.extend(new_planes)
        stats["wall_closure"] = {"synthesized": len(walls) - walls_before_closure}

        # --- D. Intersection Trimming ---
        trim_stats: dict = {"snapped_endpoints": 0, "extended_endpoints": 0}
        if self.config.enable_intersection_trimming and walls:
            from ._intersection_trimming import trim_intersections
            trim_stats = trim_intersections(walls, snap_tolerance=eff_snap_tol)

            # Rebuild wall boundaries from trimmed center-lines
            plane_by_id = {p["id"]: p for p in planes}
            for w in walls:
                for pid in w["plane_ids"]:
                    if pid in plane_by_id:
                        _rebuild_wall_boundary(plane_by_id[pid], w)
        stats["intersection_trimming"] = trim_stats

        # --- H2. Polyline Merging (after D, before E) ---
        polyline_stats: dict = {"merged_pairs": 0}
        if self.config.enable_polyline_merging and walls:
            from ._polyline_walls import merge_collinear_walls
            walls_before = len(walls)
            walls = merge_collinear_walls(
                walls,
                angle_tolerance_deg=self.config.polyline_merge_angle_tolerance,
            )
            polyline_stats["merged_pairs"] = walls_before - len(walls)
        stats["polyline_merging"] = polyline_stats

        # --- I. Column Detection (after D, before E) ---
        columns: list[dict] = []
        if self.config.enable_column_detection and walls:
            from ._column_detection import detect_columns
            columns, walls = detect_columns(
                walls,
                scale=scale,
                max_column_width=self.config.max_column_width,
                column_aspect_ratio=self.config.column_aspect_ratio,
            )
        stats["column_detection"] = {"num_columns": len(columns)}

        # --- E. Space Detection ---
        spaces: list[dict] = []
        if self.config.enable_space_detection and walls:
            from ._space_detection import detect_spaces
            spaces = detect_spaces(
                walls,
                floor_heights=height_stats.get("floor_heights", []),
                ceiling_heights=height_stats.get("ceiling_heights", []),
                min_area=eff_min_area,
                snap_tolerance=eff_snap_tol,
                scale=scale,
            )
        stats["space_detection"] = {"num_spaces": len(spaces)}

        # --- G. Exterior Classification ---
        exterior_stats: dict = {"exterior": 0, "interior": 0}
        if self.config.enable_exterior_classification and walls:
            from ._exterior_classification import classify_walls as _classify_walls
            exterior_stats = _classify_walls(walls)
        stats["exterior_classification"] = exterior_stats

        # --- G2. Building Footprint ---
        building_footprint: list[list[float]] | None = None
        if self.config.enable_exterior_classification and walls:
            from ._exterior_classification import extract_building_footprint
            building_footprint = extract_building_footprint(walls)
            if building_footprint:
                logger.info(f"Building footprint: {len(building_footprint) - 1} vertices")

        # --- H. Roof Detection ---
        roof_stats: dict = {"num_roof_planes": 0, "roof_type": "none"}
        if self.config.enable_roof_detection:
            from ._roof_detection import detect_roof_planes
            roof_stats = detect_roof_planes(
                planes,
                ceiling_heights=height_stats.get("ceiling_heights", []),
            )
        stats["roof_detection"] = roof_stats

        # --- F. Opening Detection ---
        opening_stats: dict = {"num_openings": 0, "num_doors": 0, "num_windows": 0}
        if self.config.enable_opening_detection and walls:
            from ._opening_detection import detect_openings, OpeningConfig

            # Find surface point cloud from s06 or TSDF output
            surface_points_path = None
            for candidate in [
                self.data_root / "interim" / "s06_planes" / "surface_points.ply",
                self.data_root / "interim" / "s05_tsdf_fusion" / "surface_points.ply",
            ]:
                if candidate.exists():
                    surface_points_path = candidate
                    break

            if surface_points_path:
                opening_cfg = OpeningConfig(
                    histogram_resolution=self.config.opening_histogram_resolution,
                    histogram_threshold=self.config.opening_histogram_threshold,
                    min_opening_width=self.config.opening_min_width,
                    min_opening_height=self.config.opening_min_height,
                    door_sill_max=self.config.opening_door_sill_max,
                    door_min_height=self.config.opening_door_min_height,
                    min_points_for_analysis=self.config.opening_min_points,
                )
                all_openings = detect_openings(
                    planes, walls,
                    surface_points_path=surface_points_path,
                    config=opening_cfg,
                    scale=scale,
                    manhattan_rotation=R,
                )
                opening_stats["num_openings"] = len(all_openings)
                opening_stats["num_doors"] = sum(
                    1 for o in all_openings if o.get("type") == "door"
                )
                opening_stats["num_windows"] = sum(
                    1 for o in all_openings if o.get("type") == "window"
                )
            else:
                logger.warning(
                    "Opening detection enabled but no surface_points.ply found"
                )
        stats["opening_detection"] = opening_stats

        # --- Transform back to original coordinates ---
        if R is not None:
            _transform_from_manhattan(planes, R)
            logger.info("Transformed back to original coordinates")

        # --- Transform walls to original coordinates ---
        if R is not None:
            walls_output = _transform_walls_from_manhattan(walls, R)
        else:
            # No rotation: center_line_3d is just XZ → 3D with Y=mid
            walls_output = []
            for w in walls:
                entry = dict(w)
                cl = w["center_line_2d"]
                y_mid = sum(w["height_range"]) / 2.0
                entry["center_line_3d"] = [
                    [cl[0][0], y_mid, cl[0][1]],
                    [cl[1][0], y_mid, cl[1][1]],
                ]
                walls_output.append(entry)

        # --- Save outputs ---
        # 1. Regularized planes.json (same schema as s06 output)
        planes_data = _serialize_planes(planes)
        planes_file = output_dir / "planes.json"
        with open(planes_file, "w") as f:
            json.dump(planes_data, f, indent=2)

        # 2. Regularized boundaries.json
        boundaries_data = [
            {"id": p["id"], "label": p["label"], "boundary_3d": p["boundary_3d"]}
            for p in planes_data
        ]
        boundaries_file = output_dir / "boundaries.json"
        with open(boundaries_file, "w") as f:
            json.dump(boundaries_data, f, indent=2)

        # 3. Walls.json (with both Manhattan center_line_2d and original center_line_3d)
        walls_file = output_dir / "walls.json"
        with open(walls_file, "w") as f:
            json.dump(walls_output, f, indent=2)

        # 4. Spaces.json (room polygons in Manhattan XZ plane)
        spaces_file = None
        if spaces:
            spaces_output: dict = {
                "manhattan_rotation": R.tolist() if R is not None else None,
                "coordinate_scale": float(scale),
                "spaces": spaces,
            }
            # Include storey definitions if available
            storey_defs = height_stats.get("storeys", [])
            if storey_defs:
                spaces_output["storeys"] = storey_defs
                logger.info(f"Including {len(storey_defs)} storey definitions in spaces.json")
            # Include building footprint if available
            if building_footprint:
                spaces_output["building_footprint"] = building_footprint
            spaces_file = output_dir / "spaces.json"
            with open(spaces_file, "w") as f:
                json.dump(spaces_output, f, indent=2)

        # 5. Columns.json (if column detection is enabled)
        columns_file = None
        if columns:
            columns_file = output_dir / "columns.json"
            with open(columns_file, "w") as f:
                json.dump(columns, f, indent=2)

        # 6. Copy manhattan_alignment.json for reference
        if R is not None:
            with open(output_dir / "manhattan_alignment.json", "w") as f:
                json.dump({"manhattan_rotation": R.tolist()}, f, indent=2)

        # 7. Diagnostic stats
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        num_walls = sum(1 for p in planes if p["label"] == "wall")
        num_spaces = len(spaces)

        logger.info(
            f"Plane regularization complete: {num_walls} walls, "
            f"{len(walls)} wall objects, {num_spaces} spaces, "
            f"{len(columns)} columns (scale={scale:.2f})"
        )

        return PlaneRegularizationOutput(
            planes_file=planes_file,
            boundaries_file=boundaries_file,
            walls_file=walls_file,
            spaces_file=spaces_file,
            columns_file=columns_file,
            num_walls=num_walls,
            num_spaces=num_spaces,
            num_columns=len(columns),
        )
