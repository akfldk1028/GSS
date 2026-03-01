"""Configuration for Step 06d: Mesh segmentation."""

from pydantic import BaseModel, Field


class MeshSegmentationConfig(BaseModel):
    # Face classification
    plane_distance_threshold: float = Field(
        0.03, description="Max face centroid → plane distance (meters)"
    )
    plane_normal_threshold: float = Field(
        0.85, description="Min cos(angle) between face normal and plane normal"
    )

    # Cluster filtering
    min_cluster_faces: int = Field(50, description="Minimum faces per cluster")
    min_cluster_area: float = Field(0.01, description="Minimum cluster area (m²)")

    # Simplification
    enable_simplification: bool = Field(True, description="Decimate large clusters")
    target_face_ratio: float = Field(0.5, description="Decimate to this fraction of faces")
    max_faces_per_element: int = Field(
        50000, description="Max faces per output element (matches s07 tessellation_max_faces)"
    )

    # Output
    default_ifc_class: str = Field(
        "IfcBuildingElementProxy", description="IFC class for residual mesh elements"
    )
    color_by_cluster: bool = Field(True, description="Assign distinct color per cluster")
