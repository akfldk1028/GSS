"""Tests for S06d: Mesh Segmentation step."""

import json
from pathlib import Path

import numpy as np
import pytest

from gss.steps.s06d_mesh_segmentation.config import MeshSegmentationConfig
from gss.steps.s06d_mesh_segmentation.contracts import (
    MeshSegmentationInput,
    MeshSegmentationOutput,
)
from gss.steps.s06d_mesh_segmentation._face_classification import classify_faces
from gss.steps.s06d_mesh_segmentation._cluster_extraction import (
    extract_clusters,
    _union_find_components,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box_mesh(
    center: np.ndarray, size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple box (8 vertices, 12 triangles)."""
    hs = size / 2.0
    cx, cy, cz = center
    verts = np.array([
        [cx - hs, cy - hs, cz - hs],
        [cx + hs, cy - hs, cz - hs],
        [cx + hs, cy + hs, cz - hs],
        [cx - hs, cy + hs, cz - hs],
        [cx - hs, cy - hs, cz + hs],
        [cx + hs, cy - hs, cz + hs],
        [cx + hs, cy + hs, cz + hs],
        [cx - hs, cy + hs, cz + hs],
    ])
    faces = np.array([
        # -Z face
        [0, 1, 2], [0, 2, 3],
        # +Z face
        [4, 6, 5], [4, 7, 6],
        # -Y face
        [0, 5, 1], [0, 4, 5],
        # +Y face
        [2, 7, 3], [2, 6, 7],
        # -X face
        [0, 3, 7], [0, 7, 4],
        # +X face
        [1, 5, 6], [1, 6, 2],
    ])
    return verts, faces


def _make_plane_mesh(
    normal: np.ndarray, d: float, extent: float = 5.0, n_grid: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a flat mesh lying on the plane n·x + d = 0.

    Generates a grid of triangles on the plane surface.
    """
    normal = normal / np.linalg.norm(normal)
    # Find two orthogonal vectors on the plane
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Center point on the plane
    center = -d * normal

    verts = []
    for i in range(n_grid):
        for j in range(n_grid):
            s = (i / (n_grid - 1) - 0.5) * extent
            t = (j / (n_grid - 1) - 0.5) * extent
            verts.append(center + s * u + t * v)
    verts = np.array(verts)

    faces = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            idx = i * n_grid + j
            faces.append([idx, idx + 1, idx + n_grid])
            faces.append([idx + 1, idx + n_grid + 1, idx + n_grid])
    faces = np.array(faces)
    return verts, faces


def _combine_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple (verts, faces) into a single mesh."""
    all_verts = []
    all_faces = []
    offset = 0
    for verts, faces in meshes:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)
    return np.vstack(all_verts), np.vstack(all_faces)


# ---------------------------------------------------------------------------
# A. Face Classification
# ---------------------------------------------------------------------------

class TestFaceClassification:
    def test_faces_on_plane_are_classified(self):
        """Faces lying on a known plane should be labeled with that plane's ID."""
        verts, faces = _make_plane_mesh(
            np.array([1.0, 0.0, 0.0]), d=-2.0, extent=4.0, n_grid=5,
        )
        planes = [
            {"normal": [1.0, 0.0, 0.0], "d": -2.0, "label": "wall"},
        ]
        labels = classify_faces(verts, faces, planes, distance_thresh=0.05, normal_thresh=0.8)
        assert np.all(labels == 0)

    def test_faces_off_plane_are_residual(self):
        """Faces far from any plane should be labeled -1."""
        verts, faces = _make_box_mesh(np.array([10.0, 10.0, 10.0]), size=1.0)
        planes = [
            {"normal": [1.0, 0.0, 0.0], "d": 0.0, "label": "wall"},
        ]
        labels = classify_faces(verts, faces, planes, distance_thresh=0.05, normal_thresh=0.8)
        # The box at (10,10,10) is far from the plane x=0
        assert np.all(labels == -1)

    def test_mixed_planar_and_residual(self):
        """Mesh with both on-plane and off-plane faces."""
        plane_verts, plane_faces = _make_plane_mesh(
            np.array([0.0, 1.0, 0.0]), d=0.0, extent=4.0, n_grid=4,
        )
        box_verts, box_faces = _make_box_mesh(np.array([0.0, 5.0, 0.0]), size=0.5)
        verts, faces = _combine_meshes([(plane_verts, plane_faces), (box_verts, box_faces)])

        planes = [{"normal": [0.0, 1.0, 0.0], "d": 0.0, "label": "floor"}]
        labels = classify_faces(verts, faces, planes, distance_thresh=0.05, normal_thresh=0.8)

        n_plane_faces = len(plane_faces)
        # Plane mesh faces should match
        assert np.sum(labels[:n_plane_faces] == 0) == n_plane_faces
        # Some box faces should be residual (not all align with y=0 plane)
        assert np.sum(labels[n_plane_faces:] == -1) > 0

    def test_empty_faces(self):
        """Empty mesh should return empty labels."""
        labels = classify_faces(
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int),
            [{"normal": [1, 0, 0], "d": 0}],
        )
        assert len(labels) == 0

    def test_multiple_planes_closest_wins(self):
        """When a face matches multiple planes, the closest plane wins."""
        verts, faces = _make_plane_mesh(
            np.array([1.0, 0.0, 0.0]), d=-2.0, extent=2.0, n_grid=3,
        )
        planes = [
            {"normal": [1.0, 0.0, 0.0], "d": -2.05},  # plane at x=2.05
            {"normal": [1.0, 0.0, 0.0], "d": -2.0},    # plane at x=2.0 (closer)
        ]
        labels = classify_faces(verts, faces, planes, distance_thresh=0.1, normal_thresh=0.8)
        # All should match plane 1 (closer)
        assert np.all(labels == 1)

    def test_normal_misalignment_rejects(self):
        """Faces with misaligned normals should be rejected even if close."""
        # Create faces on the x=2 plane (normal = [1,0,0])
        verts, faces = _make_plane_mesh(
            np.array([1.0, 0.0, 0.0]), d=-2.0, extent=2.0, n_grid=3,
        )
        # But define a plane with perpendicular normal
        planes = [
            {"normal": [0.0, 1.0, 0.0], "d": -2.0},
        ]
        labels = classify_faces(verts, faces, planes, distance_thresh=10.0, normal_thresh=0.8)
        # Should all be residual (normals don't align)
        assert np.all(labels == -1)


# ---------------------------------------------------------------------------
# B. Cluster Extraction
# ---------------------------------------------------------------------------

class TestClusterExtraction:
    def test_single_connected_component(self):
        """A single box should yield one cluster."""
        verts, faces = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=1.0)
        labels = np.full(len(faces), -1, dtype=np.intp)  # all residual

        elements, discarded = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=0.0,
        )
        assert len(elements) == 1
        assert len(elements[0]["faces"]) == 12
        assert discarded == 0

    def test_two_separated_components(self):
        """Two separated boxes should yield two clusters."""
        box1_v, box1_f = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=1.0)
        box2_v, box2_f = _make_box_mesh(np.array([10.0, 10.0, 10.0]), size=1.0)
        verts, faces = _combine_meshes([(box1_v, box1_f), (box2_v, box2_f)])
        labels = np.full(len(faces), -1, dtype=np.intp)

        elements, discarded = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=0.0,
        )
        assert len(elements) == 2
        assert discarded == 0

    def test_min_faces_filter(self):
        """Clusters below min_faces should be discarded."""
        verts, faces = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=1.0)
        labels = np.full(len(faces), -1, dtype=np.intp)

        elements, discarded = extract_clusters(
            verts, faces, labels, min_faces=100, min_area=0.0,
        )
        assert len(elements) == 0
        assert discarded == 12

    def test_min_area_filter(self):
        """Clusters below min_area should be discarded."""
        verts, faces = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=0.001)
        labels = np.full(len(faces), -1, dtype=np.intp)

        elements, discarded = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=1.0,
        )
        assert len(elements) == 0
        assert discarded == 12

    def test_planar_faces_excluded(self):
        """Faces labeled as planar (>=0) should not appear in clusters."""
        verts, faces = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=1.0)
        labels = np.zeros(len(faces), dtype=np.intp)  # all planar (plane 0)

        elements, discarded = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=0.0,
        )
        assert len(elements) == 0
        assert discarded == 0

    def test_no_residual_faces(self):
        """All planar → empty result."""
        verts, faces = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=1.0)
        labels = np.zeros(len(faces), dtype=np.intp)

        elements, discarded = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=0.0,
        )
        assert len(elements) == 0

    def test_vertex_reindexing(self):
        """Output faces should reference 0-based indices into output vertices."""
        verts, faces = _make_box_mesh(np.array([0.0, 0.0, 0.0]), size=1.0)
        labels = np.full(len(faces), -1, dtype=np.intp)

        elements, _ = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=0.0,
        )
        for elem in elements:
            elem_faces = np.array(elem["faces"])
            num_verts = len(elem["vertices"])
            assert elem_faces.max() < num_verts
            assert elem_faces.min() >= 0

    def test_color_assignment(self):
        """Each cluster should get a distinct color when color_by_cluster=True."""
        box1_v, box1_f = _make_box_mesh(np.array([0.0, 0.0, 0.0]))
        box2_v, box2_f = _make_box_mesh(np.array([10.0, 10.0, 10.0]))
        verts, faces = _combine_meshes([(box1_v, box1_f), (box2_v, box2_f)])
        labels = np.full(len(faces), -1, dtype=np.intp)

        elements, _ = extract_clusters(
            verts, faces, labels, min_faces=1, min_area=0.0,
            color_by_cluster=True,
        )
        assert len(elements) == 2
        # Colors should be different
        assert elements[0]["color"] != elements[1]["color"]


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_no_edges(self):
        """Each node is its own component."""
        components = _union_find_components(5, [])
        assert len(components) == 5

    def test_single_component(self):
        """All connected → one component."""
        components = _union_find_components(4, [(0, 1), (1, 2), (2, 3)])
        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2, 3]

    def test_two_components(self):
        """Two disconnected groups."""
        components = _union_find_components(4, [(0, 1), (2, 3)])
        assert len(components) == 2
        sizes = sorted(len(c) for c in components)
        assert sizes == [2, 2]


# ---------------------------------------------------------------------------
# Config / Contracts
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = MeshSegmentationConfig()
        assert cfg.plane_distance_threshold == 0.03
        assert cfg.plane_normal_threshold == 0.85
        assert cfg.min_cluster_faces == 50
        assert cfg.enable_simplification is True
        assert cfg.default_ifc_class == "IfcBuildingElementProxy"

    def test_custom_values(self):
        cfg = MeshSegmentationConfig(
            plane_distance_threshold=0.05,
            min_cluster_faces=100,
            enable_simplification=False,
        )
        assert cfg.plane_distance_threshold == 0.05
        assert cfg.min_cluster_faces == 100
        assert cfg.enable_simplification is False


class TestContracts:
    def test_input_optional_mesh(self):
        inp = MeshSegmentationInput(planes_file=Path("/tmp/planes.json"))
        assert inp.surface_mesh_path is None

    def test_input_with_mesh(self):
        inp = MeshSegmentationInput(
            surface_mesh_path=Path("/tmp/mesh.ply"),
            planes_file=Path("/tmp/planes.json"),
        )
        assert inp.surface_mesh_path == Path("/tmp/mesh.ply")

    def test_output_defaults(self):
        out = MeshSegmentationOutput()
        assert out.mesh_elements_file is None
        assert out.num_elements == 0
        assert out.num_faces_planar == 0


# ---------------------------------------------------------------------------
# Integration: Step
# ---------------------------------------------------------------------------

def _has_open3d() -> bool:
    try:
        import open3d  # noqa: F401
        return True
    except ImportError:
        return False


needs_o3d = pytest.mark.skipif(not _has_open3d(), reason="open3d not installed")


class TestMeshSegmentationStep:
    @needs_o3d
    def test_graceful_skip_no_mesh(self, data_root: Path):
        """Step should skip gracefully when surface_mesh_path is None."""
        from gss.steps.s06d_mesh_segmentation.step import MeshSegmentationStep

        planes_file = data_root / "interim" / "s06_planes" / "planes.json"
        planes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(planes_file, "w") as f:
            json.dump([], f)

        step = MeshSegmentationStep(
            config=MeshSegmentationConfig(),
            data_root=data_root,
        )
        inp = MeshSegmentationInput(
            surface_mesh_path=None,
            planes_file=planes_file,
        )
        assert step.validate_inputs(inp)
        out = step.run(inp)
        assert out.mesh_elements_file is None
        assert out.num_elements == 0

    @needs_o3d
    def test_full_pipeline_with_mesh(self, data_root: Path):
        """Full step with a synthetic mesh: plane + off-plane box."""
        import open3d as o3d

        # Create planes.json with one floor plane at y=0
        planes = [
            {
                "id": 0, "normal": [0.0, 1.0, 0.0], "d": 0.0,
                "label": "floor", "num_inliers": 500,
                "boundary_3d": [[-5, 0, -5], [5, 0, -5], [5, 0, 5], [-5, 0, 5]],
            },
        ]
        planes_file = data_root / "interim" / "s06_planes" / "planes.json"
        with open(planes_file, "w") as f:
            json.dump(planes, f)

        # Create mesh: floor plane + elevated box
        floor_v, floor_f = _make_plane_mesh(
            np.array([0.0, 1.0, 0.0]), d=0.0, extent=8.0, n_grid=10,
        )
        box_v, box_f = _make_box_mesh(np.array([0.0, 3.0, 0.0]), size=1.0)
        all_v, all_f = _combine_meshes([(floor_v, floor_f), (box_v, box_f)])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_v)
        mesh.triangles = o3d.utility.Vector3iVector(all_f)

        mesh_path = data_root / "interim" / "s05_tsdf" / "surface_mesh.ply"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        # Run step
        from gss.steps.s06d_mesh_segmentation.step import MeshSegmentationStep

        step = MeshSegmentationStep(
            config=MeshSegmentationConfig(
                min_cluster_faces=1,
                min_cluster_area=0.0,
                enable_simplification=False,
            ),
            data_root=data_root,
        )
        inp = MeshSegmentationInput(
            surface_mesh_path=mesh_path,
            planes_file=planes_file,
        )
        assert step.validate_inputs(inp)
        out = step.run(inp)

        assert out.mesh_elements_file is not None
        assert out.mesh_elements_file.exists()
        assert out.num_faces_planar > 0
        assert out.num_elements >= 1

        # Verify output schema is s07-compatible
        with open(out.mesh_elements_file) as f:
            elements = json.load(f)
        assert isinstance(elements, list)
        for elem in elements:
            assert "name" in elem
            assert "ifc_class" in elem
            assert "vertices" in elem
            assert "faces" in elem
            assert len(elem["vertices"]) >= 3
            assert len(elem["faces"]) >= 1

    @needs_o3d
    def test_all_faces_planar(self, data_root: Path):
        """When all faces match planes, output should be empty list."""
        import open3d as o3d

        planes = [
            {
                "id": 0, "normal": [0.0, 1.0, 0.0], "d": 0.0,
                "label": "floor", "num_inliers": 500,
                "boundary_3d": [[-5, 0, -5], [5, 0, -5], [5, 0, 5], [-5, 0, 5]],
            },
        ]
        planes_file = data_root / "interim" / "s06_planes" / "planes.json"
        with open(planes_file, "w") as f:
            json.dump(planes, f)

        # Create mesh: only the floor plane (all faces should match)
        floor_v, floor_f = _make_plane_mesh(
            np.array([0.0, 1.0, 0.0]), d=0.0, extent=4.0, n_grid=5,
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(floor_v)
        mesh.triangles = o3d.utility.Vector3iVector(floor_f)

        mesh_path = data_root / "interim" / "s05_tsdf" / "surface_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        from gss.steps.s06d_mesh_segmentation.step import MeshSegmentationStep

        step = MeshSegmentationStep(
            config=MeshSegmentationConfig(min_cluster_faces=1, min_cluster_area=0.0),
            data_root=data_root,
        )
        out = step.run(MeshSegmentationInput(
            surface_mesh_path=mesh_path, planes_file=planes_file,
        ))

        assert out.mesh_elements_file is not None
        with open(out.mesh_elements_file) as f:
            elements = json.load(f)
        assert elements == []
        assert out.num_faces_planar > 0
        assert out.num_elements == 0

    @needs_o3d
    def test_with_manhattan_rotation(self, data_root: Path):
        """Step should apply Manhattan rotation when available."""
        import open3d as o3d

        # Create a 90-degree rotation around Y axis
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        alignment_path = data_root / "interim" / "s06_planes" / "manhattan_alignment.json"
        with open(alignment_path, "w") as f:
            json.dump({"manhattan_rotation": R.tolist()}, f)

        # Plane normal in COLMAP: [0, 1, 0] → Manhattan: R @ [0,1,0] = [0,1,0]
        planes = [
            {
                "id": 0, "normal": [0.0, 1.0, 0.0], "d": 0.0,
                "label": "floor", "num_inliers": 500,
                "boundary_3d": [[-5, 0, -5], [5, 0, -5], [5, 0, 5], [-5, 0, 5]],
            },
        ]
        planes_file = data_root / "interim" / "s06_planes" / "planes.json"
        with open(planes_file, "w") as f:
            json.dump(planes, f)

        # Mesh in COLMAP coords (same as planes)
        floor_v, floor_f = _make_plane_mesh(
            np.array([0.0, 1.0, 0.0]), d=0.0, extent=4.0, n_grid=5,
        )
        box_v, box_f = _make_box_mesh(np.array([0.0, 3.0, 0.0]), size=1.0)
        all_v, all_f = _combine_meshes([(floor_v, floor_f), (box_v, box_f)])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_v)
        mesh.triangles = o3d.utility.Vector3iVector(all_f)

        mesh_path = data_root / "interim" / "s05_tsdf" / "surface_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        from gss.steps.s06d_mesh_segmentation.step import MeshSegmentationStep

        step = MeshSegmentationStep(
            config=MeshSegmentationConfig(
                min_cluster_faces=1, min_cluster_area=0.0,
                enable_simplification=False,
            ),
            data_root=data_root,
        )
        out = step.run(MeshSegmentationInput(
            surface_mesh_path=mesh_path, planes_file=planes_file,
        ))

        # Should still work with rotation
        assert out.num_faces_planar > 0
        assert out.num_elements >= 1

        # Output mesh_elements should be in Manhattan coordinates
        with open(out.mesh_elements_file) as f:
            elements = json.load(f)
        assert len(elements) >= 1
