"""I/O contracts for Step 01: Video to Frames extraction."""

from pathlib import Path
from pydantic import BaseModel, Field


class ExtractFramesInput(BaseModel):
    video_path: Path = Field(..., description="Path to input video file (.mp4)")


class ExtractFramesOutput(BaseModel):
    frames_dir: Path = Field(..., description="Directory containing extracted frames")
    frame_count: int = Field(..., description="Number of frames extracted")
    fps_used: float = Field(..., description="Effective FPS used for extraction")
    frame_list: list[str] = Field(default_factory=list, description="List of frame filenames")
