"""Configuration for Step 03: 3D Gaussian Splatting."""

from pydantic import BaseModel, Field


class GaussianSplattingConfig(BaseModel):
    method: str = Field("2dgs", description="Method: 2dgs|3dgs (via gsplat)")
    iterations: int = Field(30000, description="Training iterations")
    learning_rate: float = Field(1.6e-4, description="Initial learning rate for positions")
    lr_final: float = Field(1.6e-6, description="Final learning rate for positions")

    # Densification
    densify_from: int = Field(500, description="Start densification at this iteration")
    densify_until: int = Field(15000, description="Stop densification at this iteration")
    densify_interval: int = Field(100, description="Densify every N iterations")
    densify_grad_threshold: float = Field(0.0002, description="Positional gradient threshold for densification")
    densify_scale_threshold: float = Field(0.01, description="Scale threshold: split if larger, clone if smaller")
    max_gaussians: int = Field(500_000, description="Cap on total number of Gaussians")

    # Pruning
    prune_opacity_threshold: float = Field(0.005, description="Remove Gaussians with opacity below this")
    opacity_reset_interval: int = Field(3000, description="Reset opacities every N iterations (0 to disable)")
