"""Realtime pipeline configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    device: int = Field(0, description="Camera device index")
    width: int = Field(1280, description="Capture width")
    height: int = Field(720, description="Capture height")
    blur_threshold: float = Field(20.0, description="Min Laplacian variance")
    motion_threshold: float = Field(1.0, description="Min frame diff for diversity")
    max_buffer_size: int = Field(50, description="Max frames in buffer before flush")
    use_depth: bool = Field(False, description="Use Azure Kinect depth camera")
    depth_voxel_size: float = Field(0.01, description="Voxel downsample size in meters")


class TrainerConfig(BaseModel):
    # COLMAP SfM
    colmap_max_features: int = Field(2048, description="SfM feature limit (lower=faster)")
    colmap_matcher: str = Field("sequential", description="sequential|exhaustive|vocab_tree")
    reuse_poses: bool = Field(True, description="Reuse previous COLMAP DB for incremental runs")
    # 3DGRUT
    threedgrut_repo: Path = Field(Path("clone/3dgrut"), description="3DGRUT repo path")
    threedgrut_conda_env: str = Field("3dgrut", description="3DGRUT conda env name")
    n_iterations: int = Field(3000, description="3DGRUT training iterations (lower=faster)")


class ExporterConfig(BaseModel):
    threedgrut_repo: Path = Field(Path("clone/3dgrut"), description="3DGRUT repo path")
    conda_env: str = Field("3dgrut", description="3DGRUT conda env name")
    up_axis: str = Field("Y", description="USD up axis")
    generate_manifest: bool = Field(True, description="Generate NuRec manifest")


class RealtimeConfig(BaseModel):
    data_root: Path = Field(Path("data/realtime"), description="Realtime session data root")
    cycle_interval_sec: float = Field(30.0, description="Seconds between train-export cycles")
    min_frames_per_cycle: int = Field(10, description="Min frames needed to trigger training")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    notify_isaac: bool = Field(True, description="Signal Isaac Sim to reload scene")
    isaac_usdz_path: Path | None = Field(None, description="Target USDZ path for Isaac Sim")
