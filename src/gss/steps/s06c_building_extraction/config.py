"""Configuration for Step 06c: Building extraction."""

from typing import Literal

from pydantic import BaseModel, Field


class BuildingExtractionConfig(BaseModel):
    # Sub-module toggles
    enable_ground_separation: bool = Field(True, description="Detect and label ground plane")
    enable_building_segmentation: bool = Field(False, description="Segment building vs vegetation (requires surface_points.ply)")
    enable_facade_detection: bool = Field(True, description="Group vertical planes into facades")
    enable_footprint_extraction: bool = Field(True, description="Extract 2D building footprint")
    enable_roof_structuring: bool = Field(True, description="Reconstruct roof structure (ridges, eaves)")
    enable_storey_detection: bool = Field(False, description="Detect storeys from facade patterns")

    # Scale
    scale_mode: Literal["auto", "metric", "manual"] = Field(
        "auto",
        description="'auto': estimate from bounding box, 'metric': assume meters, 'manual': use coordinate_scale",
    )
    coordinate_scale: float = Field(1.0, description="Manual scale (scene_units / meter)")
    expected_building_size: float = Field(
        12.0, gt=0, description="Expected building dimension in meters (fallback for auto scale)"
    )
    expected_storey_height: float = Field(
        3.0, gt=0, description="Expected storey height in meters (primary for auto scale)"
    )

    # A. Ground separation
    ground_normal_threshold: float = Field(
        0.8, description="Min |ny| to consider a plane as horizontal"
    )
    min_ground_extent: float = Field(
        10.0, description="Min XZ extent (meters) for ground plane"
    )
    ground_tolerance: float = Field(
        0.3, description="Height tolerance (meters) for ground point separation"
    )

    # B. Building segmentation
    max_building_height: float = Field(30.0, description="Max height (meters) above ground for building points")
    dbscan_eps: float = Field(0.5, description="DBSCAN epsilon (meters)")
    min_cluster_size: int = Field(100, description="Min points per DBSCAN cluster")

    # C. Facade detection
    coplanar_dist_threshold: float = Field(0.5, description="Max distance (meters) to merge coplanar facade planes")
    coplanar_angle_threshold: float = Field(15.0, description="Max angle (degrees) for coplanar merging")
    min_facade_area: float = Field(2.0, description="Min facade area (sq meters)")

    # D. Footprint extraction
    footprint_alpha: float | None = Field(None, description="Alpha shape parameter (None = auto)")
    footprint_simplify_tolerance: float = Field(0.1, description="Douglas-Peucker simplification (meters)")

    # E. Roof structuring
    ridge_snap_tolerance: float = Field(0.3, description="Snap tolerance for ridge endpoints (meters)")
    min_roof_tilt: float = Field(0.15, description="Min |ny| for roof plane (vs pure wall)")
    max_roof_tilt: float = Field(0.85, description="Max |ny| for inclined roof (above = flat)")

    # F. Storey detection
    min_storey_height: float = Field(2.5, description="Min storey height (meters)")
    max_storey_height: float = Field(5.0, description="Max storey height (meters)")
