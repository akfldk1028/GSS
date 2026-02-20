"""Configuration for Step 01: Video to Frames."""

from pydantic import BaseModel, Field


class ExtractFramesConfig(BaseModel):
    target_fps: float = Field(2.0, description="Target frames per second to extract")
    max_frames: int = Field(2000, description="Maximum number of frames")
    output_format: str = Field("png", description="Frame image format")
    blur_threshold: float = Field(100.0, description="Laplacian variance threshold for blur")
    resize_width: int | None = Field(None, description="Resize width (None = keep original)")
