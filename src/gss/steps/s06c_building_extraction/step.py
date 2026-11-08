"""Step 06c: Building extraction — exterior building reconstruction.

Orchestrates sub-modules A→B→C→D→E→F:
  A. Ground separation (detect and label ground plane)
  B. Building segmentation (building vs vegetation, optional)
  C. Facade detection (group vertical planes into building faces)
  D. Footprint extraction (concave hull → 2D building outline)
  E. Roof structuring (ridge/eave/valley reconstruction)
  F. Storey detection (exterior facade histogram, optional)

All processing in Manhattan-aligned Y-up space (if available).
Output: building_context.json with footprint, facades, roof, storeys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from gss.utils.plane_io import (
    load_planes as _load_planes,
    load_manhattan_rotation as _load_manhattan_rotation,
    serialize_planes as _serialize_planes,
    transform_to_manhattan as _transform_to_manhattan,
    transform_from_manhattan as _transform_from_manhattan,
    estimate_scale as _estimate_scale_base,
)
from .config import BuildingExtractionConfig
from .contracts import BuildingExtractionInput, BuildingExtractionOutput

logger = logging.getLogger(__name__)


def _estimate_scale(
    planes: list[dict],
    expected_storey_height: float = 3.0,
    expected_building_size: float = 12.0,
) -> float:
    """Thin wrapper for s06c-specific defaults (exterior buildings)."""
    return _estimate_scale_base(
        planes,
        expected_storey_height=expected_storey_height,
        expected_room_size=expected_building_size,
    )


def _load_surface_points(path: Path | None) -> np.ndarray | None:
    """Load surface_points.ply as numpy array."""
    if path is None or not path.exists():
        return None
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return None
        return pts
    except (ImportError, Exception) as e:
        logger.warning(f"Could not load surface points: {e}")
        return None


class BuildingExtractionStep(
    BaseStep[BuildingExtractionInput, BuildingExtractionOutput, BuildingExtractionConfig]
):
    name: ClassVar[str] = "building_extraction"
    input_type: ClassVar = BuildingExtractionInput
    output_type: ClassVar = BuildingExtractionOutput
    config_type: ClassVar = BuildingExtractionConfig

    def validate_inputs(self, inputs: BuildingExtractionInput) -> bool:
        if not inputs.planes_file.exists():
            logger.error(f"planes_file not found: {inputs.planes_file}")
            return False
        if not inputs.boundaries_file.exists():
            logger.error(f"boundaries_file not found: {inputs.boundaries_file}")
            return False
        return True

    def run(self, inputs: BuildingExtractionInput) -> BuildingExtractionOutput:
        output_dir = self.data_root / "interim" / "s06c_building_extraction"
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

        # --- Scale estimation ---
        if self.config.scale_mode == "auto":
            scale = _estimate_scale(
                planes,
                expected_storey_height=self.config.expected_storey_height,
                expected_building_size=self.config.expected_building_size,
            )
            logger.info(f"Auto scale estimation: {scale:.2f}")
        elif self.config.scale_mode == "manual":
            scale = self.config.coordinate_scale
        else:
            scale = 1.0

        # --- Load optional surface points ---
        surface_points = _load_surface_points(inputs.surface_points_file)
        if surface_points is not None and R is not None:
            surface_points = surface_points @ R.T  # transform to Manhattan

        # --- Stats ---
        stats: dict = {"scale": float(scale)}
        building_context: dict = {"coordinate_scale": float(scale)}

        # --- A. Ground Separation ---
        ground_plane = None
        if self.config.enable_ground_separation:
            from ._ground_separation import detect_ground_plane
            ground_plane = detect_ground_plane(
                planes,
                normal_threshold=self.config.ground_normal_threshold,
                min_ground_extent=self.config.min_ground_extent,
                scale=scale,
            )
            if ground_plane:
                building_context["ground_plane"] = {
                    "normal": ground_plane["normal"].tolist()
                    if isinstance(ground_plane["normal"], np.ndarray)
                    else ground_plane["normal"],
                    "d": float(ground_plane["d"]),
                    "elevation": float(
                        -ground_plane["d"] / (abs(ground_plane["normal"][1]) + 1e-12) / scale
                    ),
                }
        stats["ground_detected"] = ground_plane is not None

        # --- B. Building Segmentation ---
        building_points = surface_points
        building_points_file = None
        if self.config.enable_building_segmentation and surface_points is not None:
            from ._building_segmentation import segment_building_points
            seg_labels = segment_building_points(
                surface_points,
                planes=planes,
                ground_plane=ground_plane,
                max_building_height=self.config.max_building_height,
                dbscan_eps=self.config.dbscan_eps,
                min_cluster_size=self.config.min_cluster_size,
                scale=scale,
            )
            building_mask = seg_labels == 0
            if np.any(building_mask):
                building_points = surface_points[building_mask]
                # Save segmented points
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pts_out = building_points
                    if R is not None:
                        pts_out = pts_out @ R  # back to original
                    pcd.points = o3d.utility.Vector3dVector(pts_out)
                    out_path = output_dir / "building_points.ply"
                    o3d.io.write_point_cloud(str(out_path), pcd)
                    building_points_file = out_path
                except ImportError:
                    pass
            stats["building_segmentation"] = {
                "building": int(np.sum(seg_labels == 0)),
                "vegetation": int(np.sum(seg_labels == 1)),
                "other": int(np.sum(seg_labels == 2)),
            }

        # --- C. Facade Detection ---
        facades: list[dict] = []
        if self.config.enable_facade_detection:
            from ._facade_detection import detect_facades
            facades = detect_facades(
                planes,
                coplanar_dist_threshold=self.config.coplanar_dist_threshold,
                coplanar_angle_threshold=self.config.coplanar_angle_threshold,
                min_facade_area=self.config.min_facade_area,
                scale=scale,
            )
            building_context["facades"] = facades
        stats["num_facades"] = len(facades)

        # --- D. Footprint Extraction ---
        footprint = None
        if self.config.enable_footprint_extraction:
            from ._footprint_extraction import extract_footprint
            footprint = extract_footprint(
                building_points=building_points,
                facades=facades,
                planes=planes,
                alpha=self.config.footprint_alpha,
                simplify_tolerance=self.config.footprint_simplify_tolerance,
                scale=scale,
            )
            if footprint:
                building_context["footprint"] = footprint
        stats["footprint_extracted"] = footprint is not None

        # --- E. Roof Structuring ---
        roof_structure: dict = {"roof_type": "none", "faces": [], "ridges": [], "eaves": [], "valleys": []}
        if self.config.enable_roof_structuring:
            from ._roof_structuring import structure_roof

            # Get ceiling heights from planes
            ceiling_heights = []
            for p in planes:
                if p.get("label") == "ceiling":
                    bnd = p.get("boundary_3d")
                    if bnd is not None and len(bnd) > 0:
                        pts = np.asarray(bnd)
                        if pts.ndim == 2:
                            ceiling_heights.append(float(pts[:, 1].mean()))

            roof_structure = structure_roof(
                planes,
                footprint=footprint,
                ceiling_heights=ceiling_heights,
                ridge_snap_tolerance=self.config.ridge_snap_tolerance,
                min_roof_tilt=self.config.min_roof_tilt,
                max_roof_tilt=self.config.max_roof_tilt,
                scale=scale,
            )
            building_context["roof_structure"] = roof_structure
        stats["roof_type"] = roof_structure.get("roof_type", "none")
        stats["num_roof_faces"] = len(roof_structure.get("faces", []))

        # --- F. Storey Detection ---
        storeys: list[dict] = []
        if self.config.enable_storey_detection and facades:
            from ._storey_detection import detect_storeys_from_exterior
            storeys = detect_storeys_from_exterior(
                facades,
                planes,
                building_points=building_points,
                min_storey_height=self.config.min_storey_height,
                max_storey_height=self.config.max_storey_height,
                scale=scale,
            )
            building_context["storeys_exterior"] = storeys
        stats["num_storeys"] = len(storeys)

        # --- Transform back to original coordinates ---
        if R is not None:
            _transform_from_manhattan(planes, R)

        # --- Save outputs ---
        # 1. Updated planes.json
        planes_data = _serialize_planes(planes)
        planes_file = output_dir / "planes.json"
        with open(planes_file, "w", encoding="utf-8") as f:
            json.dump(planes_data, f, indent=2)

        # 2. Updated boundaries.json
        boundaries_data = [
            {"id": p["id"], "label": p["label"], "boundary_3d": p["boundary_3d"]}
            for p in planes_data
        ]
        boundaries_file = output_dir / "boundaries.json"
        with open(boundaries_file, "w", encoding="utf-8") as f:
            json.dump(boundaries_data, f, indent=2)

        # 3. building_context.json
        building_context_file = output_dir / "building_context.json"
        with open(building_context_file, "w", encoding="utf-8") as f:
            json.dump(building_context, f, indent=2)

        # 4. Stats
        with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        # 5. Copy manhattan_alignment.json
        if R is not None:
            with open(output_dir / "manhattan_alignment.json", "w", encoding="utf-8") as f:
                json.dump({"manhattan_rotation": R.tolist()}, f, indent=2)

        num_facades = len(facades)
        num_roof_faces = len(roof_structure.get("faces", []))
        num_storeys = len(storeys)

        logger.info(
            f"Building extraction complete: {num_facades} facades, "
            f"{num_roof_faces} roof faces ({roof_structure.get('roof_type', 'none')}), "
            f"{num_storeys} storeys (scale={scale:.2f})"
        )

        return BuildingExtractionOutput(
            planes_file=planes_file,
            boundaries_file=boundaries_file,
            building_context_file=building_context_file,
            building_points_file=building_points_file,
            num_facades=num_facades,
            num_roof_faces=num_roof_faces,
            num_storeys=num_storeys,
        )
