"""Configuration for Step 06b: Plane regularization."""

from typing import Literal

from pydantic import BaseModel, Field


class PlaneRegularizationConfig(BaseModel):
    # Sub-module toggles
    enable_normal_snapping: bool = Field(True, description="Snap wall normals to Manhattan axes")
    enable_height_snapping: bool = Field(True, description="Cluster floor/ceiling heights")
    enable_wall_thickness: bool = Field(True, description="Detect parallel wall pairs and compute thickness")
    enable_intersection_trimming: bool = Field(True, description="Snap wall endpoints to corners")
    enable_space_detection: bool = Field(True, description="Detect room boundaries from wall layout")
    enable_opening_detection: bool = Field(False, description="Detect openings in walls (Phase 2)")
    enable_wall_closure: bool = Field(True, description="Synthesize missing walls from floor boundary")

    # Scale
    scale_mode: Literal["auto", "metric", "manual"] = Field(
        "auto",
        description="'auto': estimate scale from bounding box, 'metric': assume metric units, 'manual': use coordinate_scale",
    )
    coordinate_scale: float = Field(
        1.0,
        description="Manual scale factor (scene_units / meter). Only used when scale_mode='manual'.",
    )
    expected_room_size: float = Field(
        5.0,
        gt=0,
        description="Expected typical room dimension in meters (fallback for auto scale).",
    )
    expected_storey_height: float = Field(
        2.7,
        gt=0,
        description="Expected floor-to-ceiling height in meters (primary for auto scale).",
    )

    # A. Normal snapping
    normal_mode: Literal["manhattan", "cluster"] = Field(
        "manhattan",
        description="Wall normal discovery mode: 'manhattan' snaps to ±X/±Z, "
        "'cluster' discovers dominant directions from data",
    )
    normal_snap_threshold: float = Field(20.0, description="Max angle (degrees) to snap normal to axis")
    cluster_angle_tolerance: float = Field(15.0, description="Max angle (degrees) between normals to cluster together")

    # B. Height snapping
    height_cluster_tolerance: float = Field(0.5, description="Max height diff to cluster floor/ceiling planes (meters)")

    # C. Wall thickness
    max_wall_thickness: float = Field(1.0, description="Max distance between parallel planes to pair as wall faces (meters)")
    default_wall_thickness: float = Field(0.2, gt=0, description="Default thickness for unpaired walls (meters)")
    min_parallel_overlap: float = Field(0.3, description="Min overlap fraction between parallel walls to pair")

    # C2. Wall closure
    wall_closure_mode: Literal["auto", "manhattan", "convex", "concave"] = Field(
        "auto",
        description="Outline method for wall closure: 'auto' dispatches based on normal_mode "
        "(manhattan→AABB, cluster→ConvexHull), 'concave' uses concave hull for L-shaped buildings",
    )
    max_closure_gap_ratio: float = Field(0.3, description="Max gap ratio to fill when synthesizing walls")
    use_floor_ceiling_hints: bool = Field(True, description="Use floor/ceiling boundary for wall closure")

    # D. Intersection trimming
    snap_tolerance: float = Field(0.5, description="Max distance to snap wall endpoint to intersection (meters)")

    # E. Space detection
    min_space_area: float = Field(1.0, description="Min area (sq meters) for a valid room polygon")

    # G. Exterior classification
    enable_exterior_classification: bool = Field(
        False, description="Classify walls as interior/exterior based on convex hull"
    )

    # H. Roof detection
    enable_roof_detection: bool = Field(
        False, description="Detect roof planes above ceiling (exterior scans)"
    )

    # H2. Polyline merging
    enable_polyline_merging: bool = Field(
        False, description="Merge collinear walls into multi-segment polyline walls"
    )
    polyline_merge_angle_tolerance: float = Field(
        10.0, description="Max angle (degrees) between wall directions to merge"
    )

    # I. Column detection
    enable_column_detection: bool = Field(
        False, description="Reclassify narrow walls as columns"
    )
    max_column_width: float = Field(
        1.0, description="Max extent in both axes to classify as column (meters)"
    )
    column_aspect_ratio: float = Field(
        0.3, description="Max min(length,thickness)/height for column classification"
    )

    # J. Opening shape refinement
    enable_opening_shape_refinement: bool = Field(
        False, description="Refine opening shapes (arched/circular detection)"
    )
    opening_arch_segments: int = Field(12, description="Number of segments to approximate an arch")

    # F. Opening detection
    opening_histogram_resolution: float = Field(0.05, description="Histogram bin size (meters)")
    opening_histogram_threshold: float = Field(0.7, description="Fraction of peak density for gap detection")
    opening_min_width: float = Field(0.3, description="Min opening width (meters)")
    opening_min_height: float = Field(0.3, description="Min opening height (meters)")
    opening_door_sill_max: float = Field(0.1, description="Max sill height to classify as door (meters)")
    opening_door_min_height: float = Field(1.8, description="Min height for door classification (meters)")
    opening_min_points: int = Field(100, description="Min inlier points per wall for analysis")
