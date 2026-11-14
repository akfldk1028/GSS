"""Configuration for Step 02-alt: GS-SLAM subprocess wrapper."""

from pathlib import Path

from pydantic import BaseModel, Field


class SlamConfig(BaseModel):
    backend: str = Field(
        "mast3r_slam",
        description="SLAM backend: 'mast3r_slam', 'monogs', or 'slam3r'",
    )
    repo_path: Path = Field(
        Path("clone/MASt3R-SLAM"),
        description="Path to SLAM repository (auto-resolved per backend if default)",
    )
    conda_env: str = Field(
        "mast3r_slam",
        description="Conda environment name for SLAM backend",
    )
    voxel_downsample: float = Field(
        0.02,
        description="Voxel size for Open3D downsampling (0 to skip)",
    )
    estimate_normals: bool = Field(
        True,
        description="Estimate normals via Open3D (required if SLAM output lacks normals)",
    )
    normal_radius: float = Field(
        0.1,
        description="Radius for Open3D normal estimation",
    )
    normal_max_nn: int = Field(
        30,
        description="Max neighbors for Open3D normal estimation",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional CLI arguments passed to SLAM subprocess",
    )
    timeout: int = Field(
        7200,
        description="Maximum runtime in seconds (default 2h)",
    )
