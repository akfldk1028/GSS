"""Common Pydantic models shared across pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StepMeta(BaseModel):
    """Metadata attached to every step output for reproducibility."""

    step_name: str
    elapsed_seconds: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)


class CameraIntrinsics(BaseModel):
    """Camera intrinsic parameters (pinhole model)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class CameraPose(BaseModel):
    """Camera extrinsic: 4x4 camera-to-world matrix stored as flat list (row-major)."""

    image_name: str
    matrix_4x4: list[float] = Field(..., min_length=16, max_length=16)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration loaded from pipeline.yaml."""

    project_name: str = "gss_project"
    data_root: Path = Path("./data")
    steps: list[StepEntry] = Field(default_factory=list)


class StepEntry(BaseModel):
    """One entry in the pipeline step list."""

    name: str
    module: str
    config_file: str
    depends_on: list[str] = Field(default_factory=list)
    enabled: bool = True


# Fix forward reference
PipelineConfig.model_rebuild()
