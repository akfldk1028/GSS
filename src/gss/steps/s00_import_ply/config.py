"""Configuration for Step 00: Import external PLY file."""

from pydantic import BaseModel, Field


class ImportPlyConfig(BaseModel):
    min_opacity: float = Field(
        0.5, description="Gaussian opacity threshold (after sigmoid). Only used for Gaussian PLY."
    )
    remove_outliers: bool = Field(True, description="Apply statistical outlier removal")
    outlier_nb_neighbors: int = Field(20, description="Number of neighbors for outlier detection")
    outlier_std_ratio: float = Field(2.0, description="Standard deviation ratio for outlier detection")
    estimate_normals: bool = Field(True, description="Estimate normals via KDTree if missing")
    voxel_downsample: float = Field(
        0.0, ge=0, description="Voxel downsampling size in scene units (0 = disabled)"
    )
