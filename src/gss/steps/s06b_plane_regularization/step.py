"""Step 06b: Plane regularization — geometric cleanup between RANSAC and IFC.

Orchestrates sub-modules A→B→C→D→E→(F) in order:
  A. Normal snapping (walls → ±X/±Z, floors/ceilings → ±Y)
  B. Height snapping (cluster floor/ceiling d-values)
  C. Wall thickness (parallel pair detection + center-lines)
  D. Intersection trimming (snap center-line endpoints to corners)
  E. Space detection (polygonize center-lines → room boundaries)
  F. Opening detection (Phase 2, disabled by default)

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
    with open(path) as f:
        data = json.load(f)
    R = np.asarray(data["manhattan_rotation"], dtype=float)
    if R.shape != (3, 3):
        logger.warning(f"Invalid manhattan_rotation shape {R.shape}, ignoring")
        return None
    return R


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
        if R is not None:
            _transform_to_manhattan(planes, R)
            logger.info("Transformed to Manhattan-aligned coordinates")
        else:
            logger.warning(
                "No manhattan_alignment.json found; "
                "processing in original coordinates (results may be less accurate)"
            )

        # --- A. Normal Snapping ---
        if self.config.enable_normal_snapping:
            from ._snap_normals import snap_normals
            snap_normals(planes, threshold_deg=self.config.normal_snap_threshold)
            # Reproject boundaries onto snapped planes
            for p in planes:
                _reproject_boundary(p)

        # --- B. Height Snapping ---
        height_stats: dict = {"floor_heights": [], "ceiling_heights": []}
        if self.config.enable_height_snapping:
            from ._snap_heights import snap_heights
            height_stats = snap_heights(
                planes, tolerance=self.config.height_cluster_tolerance,
            )
            for p in planes:
                if p["label"] in ("floor", "ceiling"):
                    _reproject_boundary(p)

        # --- C. Wall Thickness ---
        walls: list[dict] = []
        if self.config.enable_wall_thickness:
            from ._wall_thickness import compute_wall_thickness
            walls = compute_wall_thickness(
                planes,
                max_wall_thickness=self.config.max_wall_thickness,
                default_wall_thickness=self.config.default_wall_thickness,
                min_parallel_overlap=self.config.min_parallel_overlap,
            )

        # --- D. Intersection Trimming ---
        if self.config.enable_intersection_trimming and walls:
            from ._intersection_trimming import trim_intersections
            trim_intersections(walls, snap_tolerance=self.config.snap_tolerance)

            # Rebuild wall boundaries from trimmed center-lines
            plane_by_id = {p["id"]: p for p in planes}
            for w in walls:
                for pid in w["plane_ids"]:
                    if pid in plane_by_id:
                        _rebuild_wall_boundary(plane_by_id[pid], w)

        # --- E. Space Detection ---
        spaces: list[dict] = []
        if self.config.enable_space_detection and walls:
            from ._space_detection import detect_spaces
            spaces = detect_spaces(
                walls,
                floor_heights=height_stats.get("floor_heights", []),
                ceiling_heights=height_stats.get("ceiling_heights", []),
                min_area=self.config.min_space_area,
            )

        # --- F. Opening Detection (Phase 2) ---
        if self.config.enable_opening_detection and walls:
            from ._opening_detection import detect_openings
            detect_openings(planes, walls)

        # --- Transform back to original coordinates ---
        if R is not None:
            _transform_from_manhattan(planes, R)
            logger.info("Transformed back to original coordinates")

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

        # 3. Walls.json (center-lines + thickness, in Manhattan space)
        walls_file = output_dir / "walls.json"
        with open(walls_file, "w") as f:
            json.dump(walls, f, indent=2)

        # 4. Spaces.json (room polygons in Manhattan XZ plane)
        spaces_file = None
        if spaces:
            spaces_output = {
                "manhattan_rotation": R.tolist() if R is not None else None,
                "spaces": spaces,
            }
            spaces_file = output_dir / "spaces.json"
            with open(spaces_file, "w") as f:
                json.dump(spaces_output, f, indent=2)

        # 5. Copy manhattan_alignment.json for reference
        if R is not None:
            with open(output_dir / "manhattan_alignment.json", "w") as f:
                json.dump({"manhattan_rotation": R.tolist()}, f, indent=2)

        num_walls = sum(1 for p in planes if p["label"] == "wall")
        num_spaces = len(spaces)

        logger.info(
            f"Plane regularization complete: {num_walls} walls, "
            f"{len(walls)} wall objects, {num_spaces} spaces"
        )

        return PlaneRegularizationOutput(
            planes_file=planes_file,
            boundaries_file=boundaries_file,
            walls_file=walls_file,
            spaces_file=spaces_file,
            num_walls=num_walls,
            num_spaces=num_spaces,
        )
