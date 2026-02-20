"""Configuration for Step 02: COLMAP SfM."""

from pydantic import BaseModel, Field


class ColmapConfig(BaseModel):
    use_gpu: bool = Field(True, description="Use GPU for feature extraction")
    matcher: str = Field("sequential", description="Matching strategy: sequential|exhaustive|vocab_tree")
    match_window: int = Field(10, description="Sequential matcher overlap window")
    single_camera: bool = Field(True, description="Share intrinsics across all images")
    camera_model: str = Field("OPENCV", description="Camera model: PINHOLE|OPENCV|RADIAL")
    max_num_features: int = Field(8192, description="Max features per image")
