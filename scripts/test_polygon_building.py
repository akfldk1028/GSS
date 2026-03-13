"""Test GSS pipeline with synthetic L-shaped building point cloud.

Generates a synthetic L-shaped room from point samples on walls/floor/ceiling,
saves as PLY, runs the import pipeline, and visualizes the result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_ROOT = PROJECT_ROOT / "data" / "runs" / "synthetic_L"
IMAGE_DIR = PROJECT_ROOT / "docs" / "images"


def sample_plane(p0, u_vec, v_vec, u_len, v_len, density=500):
    """Sample points on a rectangular plane patch."""
    n = int(u_len * v_len * density)
    us = np.random.uniform(0, u_len, n)
    vs = np.random.uniform(0, v_len, n)
    pts = p0 + np.outer(us, u_vec) + np.outer(vs, v_vec)
    normal = np.cross(u_vec, v_vec)
    normal = normal / np.linalg.norm(normal)
    normals = np.tile(normal, (n, 1))
    return pts, normals


def generate_l_shaped_building():
    """Generate L-shaped building point cloud.

    L-shape footprint (top-down XZ view):

        Z
        ^
      6 |####|
      5 |####|
      4 |####|_______
      3 |############|
      2 |############|
      1 |############|
      0 +------------> X
        0  2  4  6  8

    Two rectangles:
    - Left block:  X=[0,2], Z=[0,6], Y=[0,3]
    - Right block: X=[2,8], Z=[0,4], Y=[0,3]
    """
    floor_y = 0.0
    ceil_y = 3.0
    density = 800

    all_pts = []
    all_normals = []

    def add_plane(p0, u, v, ul, vl):
        pts, norms = sample_plane(np.array(p0), np.array(u), np.array(v), ul, vl, density)
        all_pts.append(pts)
        all_normals.append(norms)

    # === Floor (Y=0) ===
    # Left block floor
    add_plane([0, 0, 0], [1, 0, 0], [0, 0, 1], 2, 6)
    # Right block floor
    add_plane([2, 0, 0], [1, 0, 0], [0, 0, 1], 6, 4)

    # === Ceiling (Y=3) ===
    # Left block ceiling
    add_plane([0, 3, 0], [1, 0, 0], [0, 0, 1], 2, 6)
    # Right block ceiling
    add_plane([2, 3, 0], [1, 0, 0], [0, 0, 1], 6, 4)

    # === Outer walls ===
    # West wall (X=0, Z=0~6)
    add_plane([0, 0, 0], [0, 0, 1], [0, 1, 0], 6, 3)
    # South wall (Z=0, X=0~8)
    add_plane([0, 0, 0], [1, 0, 0], [0, 1, 0], 8, 3)
    # East wall bottom (X=8, Z=0~4)
    add_plane([8, 0, 0], [0, 0, 1], [0, 1, 0], 4, 3)
    # North wall right (Z=4, X=2~8)
    add_plane([2, 0, 4], [1, 0, 0], [0, 1, 0], 6, 3)
    # North wall left (Z=6, X=0~2)
    add_plane([0, 0, 6], [1, 0, 0], [0, 1, 0], 2, 3)
    # Inner corner vertical (X=2, Z=4~6)
    add_plane([2, 0, 4], [0, 0, 1], [0, 1, 0], 2, 3)

    pts = np.vstack(all_pts)
    normals = np.vstack(all_normals)

    # Add small noise for realism
    pts += np.random.normal(0, 0.02, pts.shape)

    return pts, normals


def save_ply(pts, normals, path):
    """Save point cloud as PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"Saved {len(pts):,} points -> {path}")


def run_pipeline(ply_path):
    """Run import pipeline: s00 → s06 → s06b → s07 → s08."""
    import yaml

    from gss.steps.s00_import_ply.step import ImportPlyStep
    from gss.steps.s00_import_ply.contracts import ImportPlyInput
    from gss.steps.s00_import_ply.config import ImportPlyConfig
    from gss.steps.s06_plane_extraction.step import PlaneExtractionStep
    from gss.steps.s06_plane_extraction.contracts import PlaneExtractionInput
    from gss.steps.s06_plane_extraction.config import PlaneExtractionConfig
    from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep
    from gss.steps.s06b_plane_regularization.contracts import PlaneRegularizationInput
    from gss.steps.s06b_plane_regularization.config import PlaneRegularizationConfig
    from gss.steps.s07_ifc_export.step import IfcExportStep
    from gss.steps.s07_ifc_export.contracts import IfcExportInput
    from gss.steps.s07_ifc_export.config import IfcExportConfig
    from gss.steps.s08_mesh_export.step import MeshExportStep
    from gss.steps.s08_mesh_export.contracts import MeshExportInput
    from gss.steps.s08_mesh_export.config import MeshExportConfig

    def load_cfg(path, cls):
        with open(path, encoding="utf-8") as f:
            return cls(**(yaml.safe_load(f) or {}))

    cfg_dir = PROJECT_ROOT / "configs" / "steps"

    # s00
    print("\n=== s00: Import PLY ===")
    s00_cfg = load_cfg(cfg_dir / "s00_import_ply.yaml", ImportPlyConfig)
    s00 = ImportPlyStep(config=s00_cfg, data_root=DATA_ROOT)
    s00_out = s00.execute(ImportPlyInput(ply_path=ply_path))
    print(f"  {s00_out.num_surface_points:,} points")

    # s06
    print("\n=== s06: Plane Extraction ===")
    s06_cfg = load_cfg(cfg_dir / "s06_plane_extraction.yaml", PlaneExtractionConfig)
    s06 = PlaneExtractionStep(config=s06_cfg, data_root=DATA_ROOT)
    s06_out = s06.execute(PlaneExtractionInput(
        surface_points_path=s00_out.surface_points_path,
        metadata_path=s00_out.metadata_path,
    ))
    print(f"  {s06_out.num_planes} planes (W={s06_out.num_walls}, F={s06_out.num_floors}, C={s06_out.num_ceilings})")

    # s06b
    print("\n=== s06b: Plane Regularization ===")
    s06b_cfg = load_cfg(cfg_dir / "s06b_plane_regularization.yaml", PlaneRegularizationConfig)
    s06b = PlaneRegularizationStep(config=s06b_cfg, data_root=DATA_ROOT)
    s06b_out = s06b.execute(PlaneRegularizationInput(
        planes_file=s06_out.planes_file,
        boundaries_file=s06_out.boundaries_file,
    ))
    print(f"  walls={s06b_out.num_walls}, spaces={s06b_out.num_spaces}")

    # s07
    print("\n=== s07: IFC Export ===")
    s07_cfg = load_cfg(cfg_dir / "s07_ifc_export.yaml", IfcExportConfig)
    s07_cfg = s07_cfg.model_copy(update={
        "project_name": "GSS_synthetic_L",
        "building_name": "L-Shaped Building",
    })
    s07 = IfcExportStep(config=s07_cfg, data_root=DATA_ROOT)
    s07_out = s07.execute(IfcExportInput(
        walls_file=s06b_out.walls_file,
        spaces_file=s06b_out.spaces_file,
        planes_file=s06b_out.planes_file,
    ))
    print(f"  walls={s07_out.num_walls}, slabs={s07_out.num_slabs}, spaces={s07_out.num_spaces}")
    print(f"  IFC: {s07_out.ifc_path}")

    # s08
    print("\n=== s08: Mesh Export ===")
    s08_cfg = load_cfg(cfg_dir / "s08_mesh_export.yaml", MeshExportConfig)
    s08 = MeshExportStep(config=s08_cfg, data_root=DATA_ROOT)
    s08_out = s08.execute(MeshExportInput(ifc_path=s07_out.ifc_path))
    print(f"  meshes={s08_out.num_meshes}, verts={s08_out.num_vertices}, faces={s08_out.num_faces}")
    if s08_out.glb_path:
        print(f"  GLB: {s08_out.glb_path}")

    return s07_out.ifc_path


def visualize_isometric(ifc_path):
    """Render isometric view of the IFC file."""
    import ifcopenshell
    import ifcopenshell.geom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    COLORS = {
        "wall": [0.5, 0.7, 1.0],
        "wall_synthetic": [1.0, 0.5, 0.3],
        "floor": [0.4, 0.85, 0.4],
        "ceiling": [1.0, 0.9, 0.4],
        "space": [0.7, 0.7, 0.7],
    }

    ifc = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    meshes = []
    for ifc_class in ("IfcWall", "IfcSlab", "IfcSpace"):
        for element in ifc.by_type(ifc_class):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
            except Exception:
                continue
            v = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            f = np.array(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
            if len(v) < 3:
                continue
            name = element.Name or ""
            cls = element.is_a()
            if cls == "IfcWall":
                label = "wall_synthetic" if "Synthetic" in name else "wall"
            elif cls == "IfcSlab":
                label = "floor" if "Floor" in name else "ceiling"
            elif cls == "IfcSpace":
                label = "space"
            else:
                label = "wall"
            meshes.append({"verts": v, "faces": f, "color": COLORS[label], "label": label, "name": name})

    if not meshes:
        print("No meshes found!")
        return

    all_pts = np.vstack([m["verts"] for m in meshes])
    bbox_min, bbox_max = all_pts.min(axis=0), all_pts.max(axis=0)
    size = bbox_max - bbox_min

    # Two views: isometric + top-down
    fig = plt.figure(figsize=(18, 8))

    # Isometric
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("L-Shaped Building — Isometric", fontsize=14, fontweight="bold")
    for m in meshes:
        alpha = 0.15 if m["label"] == "space" else 0.6
        tris = [list(zip(m["verts"][t, 0], m["verts"][t, 1], m["verts"][t, 2])) for t in m["faces"]]
        ax1.add_collection3d(Poly3DCollection(tris, fc=m["color"], ec="black", lw=0.3, alpha=alpha))
    pad = max(size) * 0.15
    ax1.set_xlim(bbox_min[0] - pad, bbox_max[0] + pad)
    ax1.set_ylim(bbox_min[1] - pad, bbox_max[1] + pad)
    ax1.set_zlim(bbox_min[2] - pad, bbox_max[2] + pad)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")

    # Count elements
    n_walls = sum(1 for m in meshes if m["label"] in ("wall", "wall_synthetic"))
    n_syn = sum(1 for m in meshes if m["label"] == "wall_synthetic")
    n_floor = sum(1 for m in meshes if m["label"] == "floor")
    n_ceil = sum(1 for m in meshes if m["label"] == "ceiling")
    n_space = sum(1 for m in meshes if m["label"] == "space")

    stats = (f"Walls: {n_walls} ({n_syn} synthetic)\n"
             f"Floor: {n_floor}, Ceiling: {n_ceil}\n"
             f"Spaces: {n_space}\n"
             f"Size: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m")
    ax1.text2D(0.02, 0.98, stats, transform=ax1.transAxes, fontsize=10,
               verticalalignment="top",
               bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # Top-down
    from matplotlib.patches import Polygon as MplPolygon
    ax2 = fig.add_subplot(122)
    ax2.set_aspect("equal")
    ax2.set_title("L-Shaped Building — Top-Down Plan", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    for m in meshes:
        alpha = 0.2 if m["label"] == "space" else 0.5
        for tri in m["faces"]:
            pts = m["verts"][tri]
            ax2.add_patch(MplPolygon(pts[:, :2], closed=True, fc=m["color"], ec="gray", lw=0.3, alpha=alpha))
    margin = max(size[:2]) * 0.1
    ax2.set_xlim(bbox_min[0] - margin, bbox_max[0] + margin)
    ax2.set_ylim(bbox_min[1] - margin, bbox_max[1] + margin)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(fc=COLORS["wall"], ec="black", label="Detected Wall"),
        Patch(fc=COLORS["wall_synthetic"], ec="black", label="Synthetic Wall"),
        Patch(fc=COLORS["floor"], ec="black", label="Floor"),
        Patch(fc=COLORS["ceiling"], ec="black", label="Ceiling"),
        Patch(fc=COLORS["space"], ec="black", alpha=0.3, label="Space"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = IMAGE_DIR / "synthetic_L_building.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    np.random.seed(42)

    # 1. Generate synthetic L-shaped point cloud
    print("Generating L-shaped building point cloud...")
    pts, normals = generate_l_shaped_building()
    ply_path = DATA_ROOT / "raw" / "synthetic_L.ply"
    save_ply(pts, normals, ply_path)

    # 2. Run pipeline
    t0 = time.time()
    ifc_path = run_pipeline(ply_path)
    print(f"\nPipeline done in {time.time() - t0:.1f}s")

    # 3. Visualize
    out = visualize_isometric(ifc_path)

    # 4. Open figure
    import os
    os.startfile(str(out))


if __name__ == "__main__":
    main()
