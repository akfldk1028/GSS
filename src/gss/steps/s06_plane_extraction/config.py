"""Configuration for Step 06: Plane extraction."""

from pydantic import BaseModel, Field


class PlaneExtractionConfig(BaseModel):
    max_planes: int = Field(30, description="Maximum number of planes to extract")
    distance_threshold: float = Field(0.02, description="RANSAC inlier distance (meters)")
    min_inliers: int = Field(500, description="Minimum inlier points per plane")
    ransac_iterations: int = Field(1000, description="RANSAC iterations per plane")
    angle_threshold: float = Field(15.0, description="Degrees from vertical/horizontal for classification")
    simplify_tolerance: float = Field(0.05, description="Douglas-Peucker simplification tolerance (meters)")
