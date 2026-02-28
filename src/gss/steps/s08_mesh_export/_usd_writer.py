"""USD writer — export meshes to USDC/USDZ format.

Uses pxr (usd-core) to create USD stages with UsdGeom.Mesh and
PBR materials. Compatible with NVIDIA Omniverse, Isaac Sim, and
Apple Vision Pro (USDZ).
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
    except ImportError:
        return False


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a USD prim path.

    USD prim paths must start with a letter or underscore and contain
    only alphanumeric characters and underscores.
    """
    sanitized = ""
    for i, ch in enumerate(name):
        if ch.isalnum() or ch == "_":
            sanitized += ch
        else:
            sanitized += "_"
    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_unnamed"


def write_usd(
    meshes: list[MeshData],
    output_path: Path,
    *,
    up_axis: str = "Z",
    meters_per_unit: float = 1.0,
) -> Path:
    """Write meshes to a USDC (binary USD) file.

    Args:
        meshes: List of MeshData extracted from IFC.
        output_path: Output .usdc file path.
        up_axis: Stage up axis ("Y" or "Z"). IFC uses Z-up.
        meters_per_unit: Scale factor for the stage.

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

    Args:
        usdc_path: Path to the input .usdc file.
        output_path: Path for the output .usdz file.

    Returns:
        Path to the written USDZ file.
    """
    from pxr import Sdf, UsdUtils

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(usdc_path)),
        str(output_path),
    )
    if not success:
        raise RuntimeError(f"Failed to create USDZ package: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"USDZ exported: {output_path} ({size_mb:.2f} MB)")
    return output_path
