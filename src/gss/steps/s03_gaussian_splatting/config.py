"""Configuration for Step 03: 3D Gaussian Splatting."""

from pydantic import BaseModel, Field


class GaussianSplattingConfig(BaseModel):
    method: str = Field("2dgs", description="Method: 2dgs|3dgs (via gsplat)")
    iterations: int = Field(30000, description="Training iterations")
    learning_rate: float = Field(1.6e-4, description="Initial learning rate for positions")
    densify_until: int = Field(15000, description="Densification stops at this iteration")
    sh_degree: int = Field(3, description="Spherical harmonics degree")
