"""Configuration for Step 04: Depth/Normal rendering."""

from pydantic import BaseModel, Field


class DepthRenderConfig(BaseModel):
    num_views: int = Field(400, description="Number of views to render (subset of training views)")
    render_normals: bool = Field(True, description="Also render normal maps")
    render_resolution_scale: float = Field(1.0, description="Resolution scale factor")
    view_selection: str = Field("uniform", description="View selection: uniform|coverage|all")
