"""Isaac Sim USD writer — export meshes with physics for Isaac Sim/Lab.

Fork of _usd_writer.py with additions:
- UsdPhysics.CollisionAPI on each mesh prim
- UsdPhysics.MeshCollisionAPI with configurable approximation
- UsdPhysics.RigidBodyAPI (kinematic=True) on /Building root
- PhysicsScene prim (/World/PhysicsScene)
- Optional ground plane (/World/GroundPlane)
- Optional collision mesh from TSDF surface_mesh.ply (decimated)
- Scene manifest (bbox, spawn points, IFC class prim list)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._ifc_to_mesh import MeshData
from ._usd_writer import _compute_normals, _has_pxr, _sanitize_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collision mesh helpers
# ---------------------------------------------------------------------------

@dataclass
class CollisionMeshConfig:
    """Configuration for collision mesh optimization."""

    collision_mesh_path: Path | None = None
    face_target: int = 20_000
    default_approximation: str = "meshSimplification"
    # Per-IFC-class overrides
    class_approximation: dict[str, str] = field(default_factory=lambda: {
        "IfcWall": "meshSimplification",
        "IfcSlab": "meshSimplification",
        "IfcDoor": "convexDecomposition",
        "IfcWindow": "convexDecomposition",
    })


def _load_and_decimate_mesh(
    ply_path: Path,
    target_faces: int = 20_000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a PLY mesh and decimate it for collision use.

    Uses Open3D (already a GSS dependency).

    Returns:
        (vertices, faces) tuple or None if loading fails.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("open3d not available — skipping collision mesh decimation")
        return None

    if not ply_path.exists():
        logger.warning(f"Collision mesh not found: {ply_path}")
        return None

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if mesh.is_empty():
        logger.warning(f"Empty mesh: {ply_path}")
        return None

    n_faces = len(mesh.triangles)
    logger.info(f"Collision mesh loaded: {n_faces} faces from {ply_path.name}")

    if n_faces > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        logger.info(f"Decimated: {n_faces} → {len(mesh.triangles)} faces")

    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


# ---------------------------------------------------------------------------
# Scene manifest
# ---------------------------------------------------------------------------

def _deduplicate_prim_paths(meshes: list[MeshData]) -> list[str]:
    """Compute deduplicated prim paths matching USD write order.

    Must mirror the deduplication logic in write_isaac_usd() exactly.
    """
    used: set[str] = set()
    paths: list[str] = []
    for m in meshes:
        prim_name = _sanitize_name(m.name)
        mesh_path = f"/Building/{prim_name}"
        counter = 1
        while mesh_path in used:
            mesh_path = f"/Building/{prim_name}_{counter}"
            counter += 1
        used.add(mesh_path)
        paths.append(mesh_path)
    return paths


def _compute_scene_manifest(
    meshes: list[MeshData],
    up_axis: str = "Y",
) -> dict:
    """Compute scene manifest with bbox, spawn points, and prim listing.

    The manifest is consumed by soarm_stack's scene_loader to determine
    robot placement within the reconstructed environment.
    """
    all_verts = []
    prim_by_class: dict[str, list[str]] = {}
    prim_paths = _deduplicate_prim_paths(meshes)

    for m, prim_path in zip(meshes, prim_paths):
        verts = m.vertices
        if up_axis == "Y":
            verts = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])
        all_verts.append(verts)

        prim_by_class.setdefault(m.ifc_class, []).append(prim_path)

    if not all_verts:
        return {"bbox": {}, "spawn_points": {}, "prims_by_class": {}}

    combined = np.vstack(all_verts)
    bbox_min = combined.min(axis=0).tolist()
    bbox_max = combined.max(axis=0).tolist()
    center = ((combined.min(axis=0) + combined.max(axis=0)) / 2).tolist()

    # Default spawn points — center of floor + offset
    # Ground level = min Y (or min Z if Z-up)
    if up_axis == "Y":
        ground_y = bbox_min[1]
        spawn_center = [center[0], ground_y, center[2]]
    else:
        ground_z = bbox_min[2]
        spawn_center = [center[0], center[1], ground_z]

    return {
        "up_axis": up_axis,
        "bbox": {
            "min": bbox_min,
            "max": bbox_max,
            "center": center,
        },
        "spawn_points": {
            "center": {
                "position": spawn_center,
                "orientation": [1.0, 0.0, 0.0, 0.0],  # wxyz quaternion (identity)
                "description": "Center of floor bounding box",
            },
        },
        "prims_by_class": prim_by_class,
        "num_meshes": len(meshes),
    }


def write_scene_manifest(
    meshes: list[MeshData],
    output_path: Path,
    *,
    up_axis: str = "Y",
) -> Path:
    """Write scene_manifest.json for soarm_stack consumption.

    Args:
        meshes: Mesh data from IFC extraction.
        output_path: Path for the JSON file.
        up_axis: Stage up axis.

    Returns:
        Path to written manifest.
    """
    manifest = _compute_scene_manifest(meshes, up_axis=up_axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Scene manifest written: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Isaac USD writer
# ---------------------------------------------------------------------------

def write_isaac_usd(
    meshes: list[MeshData],
    output_path: Path,
    *,
    up_axis: str = "Y",
    meters_per_unit: float = 1.0,
    double_sided: bool = True,
    collision_approximation: str = "meshSimplification",
    include_ground_plane: bool = True,
    collision_config: CollisionMeshConfig | None = None,
) -> Path:
    """Write meshes to Isaac Sim-compatible USDC with physics.

    Extends the base USD writer with:
    - PhysicsScene for simulation
    - CollisionAPI + MeshCollisionAPI on each mesh
    - RigidBodyAPI (kinematic) on /Building root
    - Optional ground plane with collision
    - Optional decimated collision mesh from TSDF

    Args:
        meshes: List of MeshData from IFC.
        output_path: Output .isaac.usdc path.
        up_axis: Stage up axis (Y recommended for Isaac Sim).
        meters_per_unit: Scale factor.
        double_sided: Double-sided rendering for thin walls.
        collision_approximation: Default collision approximation method.
        include_ground_plane: Add a ground plane with collision.
        collision_config: Optional collision mesh configuration.

    Returns:
        Path to the written USDC file.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y,
    )
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

    # --- PhysicsScene ---
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(
        Gf.Vec3f(0.0, -1.0, 0.0) if up_axis == "Y" else Gf.Vec3f(0.0, 0.0, -1.0)
    )
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # --- Ground plane (optional) ---
    if include_ground_plane:
        UsdGeom.Xform.Define(stage, "/World/GroundPlane")
        ground_mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")

        half = 50.0  # 100m x 100m ground
        if up_axis == "Y":
            pts = [
                Gf.Vec3f(-half, 0, -half),
                Gf.Vec3f(half, 0, -half),
                Gf.Vec3f(half, 0, half),
                Gf.Vec3f(-half, 0, half),
            ]
        else:
            pts = [
                Gf.Vec3f(-half, -half, 0),
                Gf.Vec3f(half, -half, 0),
                Gf.Vec3f(half, half, 0),
                Gf.Vec3f(-half, half, 0),
            ]
        ground_mesh.GetPointsAttr().Set(pts)
        ground_mesh.GetFaceVertexCountsAttr().Set([4])
        ground_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        ground_mesh.GetSubdivisionSchemeAttr().Set("none")

        # Ground collision
        UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())
        ground_mesh.GetPrim().SetMetadata("displayName", "GroundPlane")

        # Ground material (grey)
        ground_mat = UsdShade.Material.Define(stage, "/World/GroundPlane/Material")
        ground_shader = UsdShade.Shader.Define(
            stage, "/World/GroundPlane/Material/PBRShader"
        )
        ground_shader.CreateIdAttr("UsdPreviewSurface")
        ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.5, 0.5, 0.5)
        )
        ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        ground_mat.CreateSurfaceOutput().ConnectToSource(
            ground_shader.ConnectableAPI(), "surface"
        )
        mat_api = UsdShade.MaterialBindingAPI.Apply(ground_mesh.GetPrim())
        mat_api.Bind(ground_mat)

    # --- Building root (kinematic rigid body) ---
    root = UsdGeom.Xform.Define(stage, "/Building")
    stage.SetDefaultPrim(root.GetPrim())

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_api.CreateKinematicEnabledAttr().Set(True)

    # --- Material cache ---
    material_cache: dict[str, UsdShade.Material] = {}

    # Resolve per-class approximation map
    class_approx = {}
    if collision_config:
        class_approx = collision_config.class_approximation

    for mesh_data in meshes:
        prim_name = _sanitize_name(mesh_data.name)
        mesh_path = f"/Building/{prim_name}"

        counter = 1
        while stage.GetPrimAtPath(mesh_path).IsValid():
            mesh_path = f"/Building/{prim_name}_{counter}"
            counter += 1

        mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)

        # Vertices with coordinate transform
        verts = mesh_data.vertices
        if up_axis == "Y":
            verts = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])
        points = [Gf.Vec3f(*v) for v in verts.tolist()]
        mesh_prim.GetPointsAttr().Set(points)

        # Faces
        n_faces = len(mesh_data.faces)
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * n_faces)
        mesh_prim.GetFaceVertexIndicesAttr().Set(
            mesh_data.faces.flatten().tolist()
        )
        mesh_prim.GetSubdivisionSchemeAttr().Set("none")

        # Normals
        normals = _compute_normals(verts, mesh_data.faces)
        normal_vecs = [Gf.Vec3f(*n) for n in normals.tolist()]
        mesh_prim.GetNormalsAttr().Set(normal_vecs)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        if double_sided:
            mesh_prim.GetDoubleSidedAttr().Set(True)

        mesh_prim.GetPrim().SetMetadata("displayName", mesh_data.name)

        # --- Physics: Collision ---
        UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
        approx = class_approx.get(
            mesh_data.ifc_class, collision_approximation
        )
        mesh_collision.CreateApproximationAttr().Set(approx)

        # --- Material (same logic as _usd_writer) ---
        alpha = mesh_data.color[3] if len(mesh_data.color) > 3 else 1.0
        color_key = (
            f"{mesh_data.color[0]:.2f}_{mesh_data.color[1]:.2f}_"
            f"{mesh_data.color[2]:.2f}_{alpha:.2f}"
        )
        if color_key not in material_cache:
            mat_name = _sanitize_name(
                f"Mat_{mesh_data.ifc_class}_{color_key}"
            )
            mat_path = f"/Building/Materials/{mat_name}"

            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).Set(
                Gf.Vec3f(
                    mesh_data.color[0],
                    mesh_data.color[1],
                    mesh_data.color[2],
                )
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if len(mesh_data.color) > 3 and mesh_data.color[3] < 1.0:
                shader.CreateInput(
                    "opacity", Sdf.ValueTypeNames.Float
                ).Set(mesh_data.color[3])
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            material_cache[color_key] = material

        mat_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        mat_api.Bind(material_cache[color_key])

    # --- Optional: TSDF collision mesh (decimated, invisible) ---
    if collision_config and collision_config.collision_mesh_path:
        result = _load_and_decimate_mesh(
            collision_config.collision_mesh_path,
            target_faces=collision_config.face_target,
        )
        if result is not None:
            col_verts, col_faces = result
            if up_axis == "Y":
                col_verts = np.column_stack(
                    [col_verts[:, 0], col_verts[:, 2], -col_verts[:, 1]]
                )

            col_mesh = UsdGeom.Mesh.Define(stage, "/Building/CollisionMesh")
            points = [Gf.Vec3f(*v) for v in col_verts.tolist()]
            col_mesh.GetPointsAttr().Set(points)
            col_mesh.GetFaceVertexCountsAttr().Set(
                [3] * len(col_faces)
            )
            col_mesh.GetFaceVertexIndicesAttr().Set(
                col_faces.flatten().tolist()
            )
            col_mesh.GetSubdivisionSchemeAttr().Set("none")

            # Invisible (collision only)
            col_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

            # Collision APIs
            UsdPhysics.CollisionAPI.Apply(col_mesh.GetPrim())
            mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(col_mesh.GetPrim())
            mesh_col_api.CreateApproximationAttr().Set("meshSimplification")

            col_mesh.GetPrim().SetMetadata(
                "displayName", "TSDF_CollisionMesh"
            )
            logger.info(
                f"TSDF collision mesh added: {len(col_faces)} faces"
            )

    stage.Save()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Isaac USD exported: {output_path} "
        f"({size_mb:.2f} MB, {len(meshes)} meshes)"
    )
    return output_path
