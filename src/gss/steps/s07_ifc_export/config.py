"""Configuration for Step 07: IFC export."""

from pydantic import BaseModel, Field


class IfcExportConfig(BaseModel):
    ifc_version: str = Field("IFC4", description="IFC schema version")
    project_name: str = Field("GSS_BIM", description="IFC project name")
    building_name: str = Field("Building", description="Building name in IFC")
    storey_name: str = Field("Ground Floor", description="Building storey name")
    default_wall_thickness: float = Field(0.2, description="Default wall thickness (meters)")
    default_slab_thickness: float = Field(0.3, description="Default slab thickness (meters)")

    # Author / organization
    author_name: str = Field("GSS", description="Author name for IfcOwnerHistory")
    organization_name: str = Field("GSS Pipeline", description="Organization for IfcOwnerHistory")

    # Material names
    wall_material_name: str = Field("Concrete", description="Wall material name")
    slab_material_name: str = Field("Concrete", description="Slab material name")

    # Feature toggles
    create_wall_types: bool = Field(True, description="Create IfcWallType + IfcRelDefinesByType")
    create_material_layers: bool = Field(True, description="Create IfcMaterialLayerSet")
    create_property_sets: bool = Field(True, description="Create PropertySets (IsExternal, Synthetic)")
    create_axis_representation: bool = Field(True, description="Create Axis (center-line) representation")
    create_spaces: bool = Field(True, description="Create IfcSpace from spaces.json")
    create_slabs: bool = Field(True, description="Create floor/ceiling IfcSlab")
    create_roof: bool = Field(True, description="Create IfcRoof from roof planes")
    create_roof_annotations: bool = Field(True, description="Create IfcAnnotation for ridge/eave/valley lines")
    create_columns: bool = Field(True, description="Create IfcColumn from columns.json")
    create_tessellated: bool = Field(True, description="Create tessellated elements from mesh_elements.json")
    tessellation_max_faces: int = Field(50000, description="Max faces per tessellated element")
    include_synthetic_walls: bool = Field(True, description="Include synthetic walls in IFC export")

    # Scale override (None = use coordinate_scale from spaces.json)
    scale_override: float | None = Field(None, description="Override coordinate_scale (None = auto)")
