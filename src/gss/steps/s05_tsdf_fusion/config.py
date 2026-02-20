"""Configuration for Step 05: TSDF fusion."""

from pydantic import BaseModel, Field


class TsdfFusionConfig(BaseModel):
    voxel_size: float = Field(0.006, description="Voxel size in meters (6mm default)")
    sdf_trunc: float = Field(0.04, description="Truncation distance in meters")
    depth_trunc: float = Field(5.0, description="Max depth to integrate (meters)")
    depth_scale: float = Field(1.0, description="Depth scale (1.0 if meters, 1000.0 if mm)")
    use_gpu: bool = Field(False, description="Use Open3D tensor GPU pipeline")
