"""Configuration for Step 06: Plane extraction."""

from pydantic import BaseModel, Field


class PlaneExtractionConfig(BaseModel):
    max_planes: int = Field(30, description="Maximum number of planes to extract")
    distance_threshold: float = Field(0.02, description="RANSAC inlier distance (meters)")
    min_inliers: int = Field(500, description="Minimum inlier points per plane")
    ransac_iterations: int = Field(1000, description="RANSAC iterations per plane")
    angle_threshold: float = Field(15.0, description="Degrees from vertical/horizontal for classification")
    simplify_tolerance: float = Field(0.05, description="Douglas-Peucker simplification tolerance (meters)")
    up_axis: str = Field("z", description="Up axis: 'z' (default/IFC) or 'y' (COLMAP/PlanarGS)")

    # Coplanar merging (A)
    merge_coplanar: bool = Field(True, description="Merge coplanar planes with same label")
    merge_angle_threshold: float = Field(10.0, description="Max angle (degrees) between normals to consider coplanar")
    merge_distance_threshold: float = Field(1.5, description="Max plane offset difference (COLMAP units) for merging")

    # Architectural filtering
    wall_min_ratio: float = Field(0.1, description="Min inlier ratio vs largest wall to keep (smaller → other)")

    # Manhattan World alignment (C)
    normalize_coords: bool = Field(True, description="Auto-detect Manhattan axes and align point cloud")
