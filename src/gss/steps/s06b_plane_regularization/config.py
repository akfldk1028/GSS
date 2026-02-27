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
        description="Expected typical room dimension in meters (used for auto scale estimation).",
    )

    # A. Normal snapping
    normal_snap_threshold: float = Field(20.0, description="Max angle (degrees) to snap normal to axis")

    # B. Height snapping
    height_cluster_tolerance: float = Field(0.5, description="Max height diff to cluster floor/ceiling planes (meters)")

    # C. Wall thickness
    max_wall_thickness: float = Field(1.0, description="Max distance between parallel planes to pair as wall faces (meters)")
    default_wall_thickness: float = Field(0.2, description="Default thickness for unpaired walls (meters)")
    min_parallel_overlap: float = Field(0.3, description="Min overlap fraction between parallel walls to pair")

    # C2. Wall closure
    max_closure_gap_ratio: float = Field(0.3, description="Max gap ratio to fill when synthesizing walls")
    use_floor_ceiling_hints: bool = Field(True, description="Use floor/ceiling boundary for wall closure")

    # D. Intersection trimming
    snap_tolerance: float = Field(0.5, description="Max distance to snap wall endpoint to intersection (meters)")

    # E. Space detection
    min_space_area: float = Field(1.0, description="Min area (sq meters) for a valid room polygon")
