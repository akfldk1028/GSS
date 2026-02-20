"""Configuration for Step 07: IFC export."""

from pydantic import BaseModel, Field


class IfcExportConfig(BaseModel):
    ifc_version: str = Field("IFC4", description="IFC schema version")
    project_name: str = Field("GSS_BIM", description="IFC project name")
    building_name: str = Field("Building", description="Building name in IFC")
    default_wall_thickness: float = Field(0.2, description="Default wall thickness (meters)")
    default_slab_thickness: float = Field(0.3, description="Default slab thickness (meters)")
