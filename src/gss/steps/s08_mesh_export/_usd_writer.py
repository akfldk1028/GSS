"""USD writer — export meshes to USDC/USDZ format.

Uses pxr (usd-core) to create USD stages with UsdGeom.Mesh and
PBR materials. Compatible with NVIDIA Omniverse, Isaac Sim, and
Apple Vision Pro (USDZ).

Cross-platform notes:
- All internal asset paths use POSIX separators (/) for Linux compatibility
- Default up_axis=Y matches Isaac Sim / Omniverse convention
- Vertex normals written explicitly for proper shading in Isaac Sim
- doubleSided=True for thin BIM geometry (walls viewed from both sides)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._ifc_to_mesh import MeshData

logger = logging.getLogger(__name__)


def _has_pxr() -> bool:
    """Check if pxr (usd-core) is available."""
    try:
        from pxr import Usd  # noqa: F401
        return True
    except (ImportError, OSError) as e:
        if isinstance(e, OSError):
            logger.warning(
                f"pxr (usd-core) import failed with OS error: {e}. "
                "This often happens with usd-core >= 26.3 on Windows. "
                "Try: pip install usd-core==24.11"
            )
        return False


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a USD prim path.

    USD prim paths must start with a letter or underscore and contain
    only alphanumeric characters and underscores.
    """
    sanitized = ""
    for ch in name:
        if ch.isalnum() or ch == "_":
            sanitized += ch
        else:
            sanitized += "_"
    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_unnamed"


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from triangle mesh via area-weighted averaging.

    Returns (N, 3) float64 array of unit normals, one per vertex.
    """
    normals = np.zeros_like(vertices)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Face normals (area-weighted — cross product magnitude = 2 * triangle area)
    face_normals = np.cross(v1 - v0, v2 - v0)

    # Accumulate face normals to each vertex
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)  # avoid division by zero
    normals /= lengths

    return normals


def write_usd(
    meshes: list[MeshData],
    output_path: Path,
    *,
    up_axis: str = "Y",
    meters_per_unit: float = 1.0,
    double_sided: bool = True,
) -> Path:
    """Write meshes to a USDC (binary USD) file.

    Args:
        meshes: List of MeshData extracted from IFC.
        output_path: Output .usdc file path.
        up_axis: Stage up axis ("Y" or "Z"). Y-up is Isaac Sim / Omniverse standard.
        meters_per_unit: Scale factor for the stage.
        double_sided: Mark meshes as double-sided (important for thin BIM walls).

    Returns:
        Path to the written USDC file.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

    # Root xform
    root = UsdGeom.Xform.Define(stage, "/Building")
    stage.SetDefaultPrim(root.GetPrim())

    # Material cache to avoid duplicates
    material_cache: dict[str, UsdShade.Material] = {}

    for mesh_data in meshes:
        prim_name = _sanitize_name(mesh_data.name)
        mesh_path = f"/Building/{prim_name}"

        # Ensure unique path
        counter = 1
        while stage.GetPrimAtPath(mesh_path).IsValid():
            mesh_path = f"/Building/{prim_name}_{counter}"
            counter += 1

        mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)

        # Vertices — apply Z-up → Y-up transform when requested
        verts = mesh_data.vertices
        if up_axis == "Y":
            # IFC is Z-up; convert (x,y,z) → (x,z,-y) same as GLB writer
            verts = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])
        points = [Gf.Vec3f(*v) for v in verts.tolist()]
        mesh_prim.GetPointsAttr().Set(points)

        # Faces (all triangles)
        n_faces = len(mesh_data.faces)
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * n_faces)
        mesh_prim.GetFaceVertexIndicesAttr().Set(mesh_data.faces.flatten().tolist())

        # Subdivision scheme = none (polygon mesh, not subdivision surface)
        mesh_prim.GetSubdivisionSchemeAttr().Set("none")

        # Vertex normals for proper shading in Isaac Sim / Omniverse
        normals = _compute_normals(verts, mesh_data.faces)
        normal_vecs = [Gf.Vec3f(*n) for n in normals.tolist()]
        mesh_prim.GetNormalsAttr().Set(normal_vecs)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Double-sided rendering (critical for thin BIM walls seen from both sides)
        if double_sided:
            mesh_prim.GetDoubleSidedAttr().Set(True)

        # Display name for Omniverse UI
        mesh_prim.GetPrim().SetMetadata("displayName", mesh_data.name)

        # Material (cache key includes alpha to avoid transparency collisions)
        alpha = mesh_data.color[3] if len(mesh_data.color) > 3 else 1.0
        color_key = f"{mesh_data.color[0]:.2f}_{mesh_data.color[1]:.2f}_{mesh_data.color[2]:.2f}_{alpha:.2f}"
        if color_key not in material_cache:
            mat_name = _sanitize_name(f"Mat_{mesh_data.ifc_class}_{color_key}")
            mat_path = f"/Building/Materials/{mat_name}"

            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(mesh_data.color[0], mesh_data.color[1], mesh_data.color[2])
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if len(mesh_data.color) > 3 and mesh_data.color[3] < 1.0:
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
                    mesh_data.color[3]
                )
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            material_cache[color_key] = material

        mat_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        mat_api.Bind(material_cache[color_key])

    stage.Save()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"USD exported: {output_path} ({size_mb:.2f} MB, {len(meshes)} meshes)")
    return output_path


def write_usdz(usdc_path: Path, output_path: Path) -> Path:
    """Package a USDC file as USDZ (Apple Vision Pro compatible).

    Uses POSIX paths for cross-platform archive compatibility.

    Args:
        usdc_path: Path to the input .usdc file.
        output_path: Path for the output .usdz file.

    Returns:
        Path to the written USDZ file.
    """
    from pxr import Sdf, UsdUtils

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use POSIX-style path (forward slashes) for the asset reference.
    # Windows backslashes in Sdf.AssetPath would break on Linux/macOS.
    success = UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(usdc_path.as_posix()),
        str(output_path),
    )
    if not success:
        raise RuntimeError(f"Failed to create USDZ package: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"USDZ exported: {output_path} ({size_mb:.2f} MB)")
    return output_path
