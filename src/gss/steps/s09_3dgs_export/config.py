"""Configuration for Step 09: 3DGS → USDZ export."""

from pathlib import Path

from pydantic import BaseModel, Field


class ThreeDGSExportConfig(BaseModel):
    # 3DGRUT repository (git cloned separately)
    threedgrut_repo: Path = Field(
        Path("clone/3dgrut"),
        description="Path to 3DGRUT repository (nv-tlabs/3dgrut)",
    )
    conda_env: str = Field(
        "3dgrut",
        description="Conda environment name for 3DGRUT",
    )

    # ply_to_usd options
    up_axis: str = Field("Y", description="USD up axis (Y or Z)")
    meters_per_unit: float = Field(1.0, description="USD meters per unit")

    # Scene metadata for Isaac Sim
    generate_manifest: bool = Field(
        True, description="Generate scene_manifest_nurec.json for Isaac Sim"
    )
