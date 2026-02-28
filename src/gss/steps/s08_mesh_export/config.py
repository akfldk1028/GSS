"""Configuration for Step 08: Mesh export."""

from typing import Literal

from pydantic import BaseModel, Field


class MeshExportConfig(BaseModel):
    export_glb: bool = Field(True, description="Export GLB file (universal 3D format)")
    export_usd: bool = Field(True, description="Export USDC file (Omniverse/Isaac Sim)")
    export_usdz: bool = Field(False, description="Export USDZ package (Apple Vision Pro)")

    color_scheme: Literal["by_class", "uniform"] = Field(
        "by_class",
        description="Color scheme: 'by_class' colors by IFC type, 'uniform' uses single color",
    )
    include_spaces: bool = Field(False, description="Include IfcSpace geometry in export")

    # Color mapping (RGBA 0-1) for by_class scheme
    color_wall: list[float] = Field(
        default=[0.85, 0.85, 0.85, 1.0], description="Wall color RGBA"
    )
    color_slab: list[float] = Field(
        default=[0.7, 0.7, 0.7, 1.0], description="Slab color RGBA"
    )
    color_door: list[float] = Field(
        default=[0.55, 0.35, 0.2, 1.0], description="Door color RGBA"
    )
    color_window: list[float] = Field(
        default=[0.6, 0.8, 1.0, 0.7], description="Window color RGBA"
    )
    color_space: list[float] = Field(
        default=[0.9, 0.95, 1.0, 0.3], description="Space color RGBA"
    )
    color_default: list[float] = Field(
        default=[0.8, 0.8, 0.8, 1.0], description="Default color RGBA"
    )

    # USD-specific
    usd_up_axis: Literal["Y", "Z"] = Field("Z", description="USD stage up axis")
    usd_meters_per_unit: float = Field(1.0, description="USD meters per unit")
