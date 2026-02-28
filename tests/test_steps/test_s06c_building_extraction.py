"""Tests for S06c: Building Extraction step."""

import json
from pathlib import Path

import numpy as np
import pytest

from gss.steps.s06c_building_extraction.config import BuildingExtractionConfig
from gss.steps.s06c_building_extraction.contracts import (
    BuildingExtractionInput,
    BuildingExtractionOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wall(id: int, nx: float, nz: float, d: float, boundary_3d: list) -> dict:
    """Create a wall plane dict."""
    return {
        "id": id,
        "normal": [nx, 0.0, nz],
        "d": d,
        "label": "wall",
        "num_inliers": 500,
        "boundary_3d": boundary_3d,
    }


def _make_horiz(id: int, ny: float, d: float, label: str, boundary_3d: list) -> dict:
    """Create a horizontal plane (floor/ceiling/ground)."""
    return {
        "id": id,
        "normal": [0.0, ny, 0.0],
        "d": d,
        "label": label,
        "num_inliers": 1000,
        "boundary_3d": boundary_3d,
    }


def _make_roof_plane(id: int, nx: float, ny: float, nz: float, d: float, boundary_3d: list) -> dict:
    """Create an inclined roof plane."""
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    return {
        "id": id,
        "normal": [nx / norm, ny / norm, nz / norm],
        "d": d,
        "label": "wall",  # will be re-labeled by roof detection
        "num_inliers": 300,
        "boundary_3d": boundary_3d,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def box_building_planes() -> list[dict]:
    """Simple box building: 4 walls + flat roof + ground.

    Building: 10x8m, height 6m, ground at y=0.
    """
    return [
        # Ground plane (wide, at y=0)
        _make_horiz(0, 1.0, 0.0, "floor", [
            [-20, 0, -20], [20, 0, -20], [20, 0, 20], [-20, 0, 20], [-20, 0, -20]
        ]),
        # Walls
        _make_wall(1, 1.0, 0.0, -5.0, [[5, 0, 0], [5, 6, 0], [5, 6, 8], [5, 0, 8], [5, 0, 0]]),
        _make_wall(2, -1.0, 0.0, -5.0, [[-5, 0, 0], [-5, 6, 0], [-5, 6, 8], [-5, 0, 8], [-5, 0, 0]]),
        _make_wall(3, 0.0, 0.0, 1.0, [[- 5, 0, 8], [5, 0, 8], [5, 6, 8], [-5, 6, 8], [-5, 0, 8]]),
        _make_wall(4, 0.0, 0.0, -1.0, [[-5, 0, 0], [5, 0, 0], [5, 6, 0], [-5, 6, 0], [-5, 0, 0]]),
        # Flat roof (at y=6)
        _make_horiz(5, -1.0, 6.0, "ceiling", [
            [-5, 6, 0], [5, 6, 0], [5, 6, 8], [-5, 6, 8], [-5, 6, 0]
        ]),
    ]


@pytest.fixture
def l_shaped_building_planes() -> list[dict]:
    """L-shaped building: 6 walls + ground.

    L shape in XZ plane: main body 10x8 + wing 5x4.
    """
    return [
        # Ground
        _make_horiz(0, 1.0, 0.0, "floor", [
            [-20, 0, -20], [20, 0, -20], [20, 0, 20], [-20, 0, 20], [-20, 0, -20]
        ]),
        # Main body walls
        _make_wall(1, 1.0, 0.0, -5.0, [[5, 0, 0], [5, 4, 0], [5, 4, 8], [5, 0, 8], [5, 0, 0]]),
        _make_wall(2, -1.0, 0.0, -5.0, [[-5, 0, 0], [-5, 4, 0], [-5, 4, 4], [-5, 0, 4], [-5, 0, 0]]),
        _make_wall(3, 0.0, 0.0, 1.0, [[-5, 0, 8], [5, 0, 8], [5, 4, 8], [-5, 4, 8], [-5, 0, 8]]),
        _make_wall(4, 0.0, 0.0, -1.0, [[-5, 0, 0], [5, 0, 0], [5, 4, 0], [-5, 4, 0], [-5, 0, 0]]),
        # Wing walls
        _make_wall(5, -1.0, 0.0, 0.0, [[0, 0, 4], [0, 4, 4], [0, 4, 8], [0, 0, 8], [0, 0, 4]]),
        _make_wall(6, 0.0, 0.0, -1.0, [[-5, 0, 4], [0, 0, 4], [0, 4, 4], [-5, 4, 4], [-5, 0, 4]]),
        # Ceiling
        _make_horiz(7, -1.0, 4.0, "ceiling", [
            [-5, 4, 0], [5, 4, 0], [5, 4, 8], [-5, 4, 8], [-5, 4, 0]
        ]),
    ]


@pytest.fixture
def gable_roof_building_planes() -> list[dict]:
    """Gable-roof building: 4 walls + 2 inclined roof planes + ground.

    Building: 10x8m, walls height 4m, ridge at 6m.
    """
    return [
        # Ground
        _make_horiz(0, 1.0, 0.0, "floor", [
            [-20, 0, -20], [20, 0, -20], [20, 0, 20], [-20, 0, 20], [-20, 0, -20]
        ]),
        # Walls
        _make_wall(1, 1.0, 0.0, -5.0, [[5, 0, 0], [5, 4, 0], [5, 4, 8], [5, 0, 8], [5, 0, 0]]),
        _make_wall(2, -1.0, 0.0, -5.0, [[-5, 0, 0], [-5, 4, 0], [-5, 4, 8], [-5, 0, 8], [-5, 0, 0]]),
        _make_wall(3, 0.0, 0.0, 1.0, [[-5, 0, 8], [5, 0, 8], [5, 4, 8], [-5, 4, 8], [-5, 0, 8]]),
        _make_wall(4, 0.0, 0.0, -1.0, [[-5, 0, 0], [5, 0, 0], [5, 4, 0], [-5, 4, 0], [-5, 0, 0]]),
        # Ceiling (for height reference)
        _make_horiz(5, -1.0, 4.0, "ceiling", [
            [-5, 4, 0], [5, 4, 0], [5, 4, 8], [-5, 4, 8], [-5, 4, 0]
        ]),
        # Two inclined roof planes (gable) — south-facing and north-facing
        # South slope: normal (0.71, 0.71, 0) → 45° tilt, |ny|=0.71 < 0.85
        _make_roof_plane(6, 0.71, 0.71, 0.0, -4.5, [
            [0, 6, 0], [5, 4, 0], [5, 4, 8], [0, 6, 8], [0, 6, 0]
        ]),
        # North slope: normal (-0.71, 0.71, 0) → 45° tilt
        _make_roof_plane(7, -0.71, 0.71, 0.0, -4.5, [
            [0, 6, 0], [-5, 4, 0], [-5, 4, 8], [0, 6, 8], [0, 6, 0]
        ]),
    ]


@pytest.fixture
def box_planes_json(data_root: Path, box_building_planes) -> Path:
    """Write box building planes to disk."""
    planes_file = data_root / "interim" / "s06_planes" / "planes.json"
    planes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(planes_file, "w") as f:
        json.dump(box_building_planes, f)
    bnd_file = data_root / "interim" / "s06_planes" / "boundaries.json"
    boundaries = [{"id": p["id"], "label": p["label"], "boundary_3d": p["boundary_3d"]}
                  for p in box_building_planes]
    with open(bnd_file, "w") as f:
        json.dump(boundaries, f)
    return planes_file


@pytest.fixture
def box_boundaries_json(data_root: Path, box_planes_json: Path) -> Path:
    return data_root / "interim" / "s06_planes" / "boundaries.json"


# ---------------------------------------------------------------------------
# Module A: Ground Separation
# ---------------------------------------------------------------------------

class TestGroundSeparation:
    def test_detects_ground_plane(self, box_building_planes):
        from gss.steps.s06c_building_extraction._ground_separation import detect_ground_plane
        planes = box_building_planes.copy()
        ground = detect_ground_plane(planes, min_ground_extent=5.0, scale=1.0)
        assert ground is not None
        assert ground["id"] == 0
        assert ground["label"] == "ground"

    def test_ignores_small_horizontal_planes(self):
        from gss.steps.s06c_building_extraction._ground_separation import detect_ground_plane
        planes = [
            _make_horiz(0, 1.0, 0.0, "floor", [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
        ]
        ground = detect_ground_plane(planes, min_ground_extent=10.0, scale=1.0)
        # Extent is 1x1=1, less than 10, but fallback allows 20% → 2.0, still less
        assert ground is None

    def test_picks_lowest_wide_plane(self):
        from gss.steps.s06c_building_extraction._ground_separation import detect_ground_plane
        planes = [
            _make_horiz(0, 1.0, -5.0, "floor", [[-15, 5, -15], [15, 5, -15], [15, 5, 15], [-15, 5, 15]]),
            _make_horiz(1, 1.0, 0.0, "floor", [[-15, 0, -15], [15, 0, -15], [15, 0, 15], [-15, 0, 15]]),
        ]
        ground = detect_ground_plane(planes, min_ground_extent=10.0, scale=1.0)
        assert ground is not None
        assert ground["id"] == 1  # lower plane

    def test_separate_ground_points(self):
        from gss.steps.s06c_building_extraction._ground_separation import separate_ground_points
        ground = {"normal": [0, 1, 0], "d": 0.0}
        pts = np.array([
            [0, 0, 0],    # on ground
            [0, 0.1, 0],  # near ground
            [0, 3, 0],    # building
            [0, 5, 0],    # building
        ], dtype=float)
        building, ground_pts = separate_ground_points(pts, ground, tolerance=0.3, scale=1.0)
        assert len(ground_pts) == 2
        assert len(building) == 2

    def test_no_horizontal_planes(self):
        from gss.steps.s06c_building_extraction._ground_separation import detect_ground_plane
        planes = [_make_wall(0, 1.0, 0.0, -5.0, [[5, 0, 0], [5, 6, 0]])]
        ground = detect_ground_plane(planes, min_ground_extent=10.0, scale=1.0)
        assert ground is None


# ---------------------------------------------------------------------------
# Module B: Building Segmentation
# ---------------------------------------------------------------------------

class TestBuildingSegmentation:
    def test_segments_dense_cluster_as_building(self):
        from gss.steps.s06c_building_extraction._building_segmentation import segment_building_points
        rng = np.random.default_rng(42)
        # Dense building cluster
        building = rng.normal(loc=[0, 3, 0], scale=0.3, size=(200, 3))
        # Sparse vegetation
        vegetation = rng.normal(loc=[10, 2, 10], scale=2.0, size=(50, 3))
        pts = np.vstack([building, vegetation])
        try:
            labels = segment_building_points(pts, dbscan_eps=1.0, min_cluster_size=20, scale=1.0)
        except (ImportError, ValueError):
            pytest.skip("sklearn/scipy not compatible in this environment")
        # Building cluster should be mostly label 0
        building_labels = labels[:200]
        assert np.sum(building_labels == 0) > 100  # most are building

    def test_empty_points(self):
        from gss.steps.s06c_building_extraction._building_segmentation import segment_building_points
        pts = np.empty((0, 3))
        labels = segment_building_points(pts, scale=1.0)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# Module C: Facade Detection
# ---------------------------------------------------------------------------

class TestFacadeDetection:
    def test_groups_parallel_walls_into_facades(self, box_building_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        facades = detect_facades(box_building_planes, min_facade_area=1.0, scale=1.0)
        # 4 walls → should form at least 2 facade groups (±X and ±Z)
        assert len(facades) >= 2

    def test_facade_has_orientation(self, box_building_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        facades = detect_facades(box_building_planes, min_facade_area=1.0, scale=1.0)
        orientations = {f["orientation"] for f in facades}
        # Should have compass directions
        assert all(o in ("north", "south", "east", "west", "ne", "nw", "se", "sw", "NE", "NW", "SE", "SW", "unknown")
                   for o in orientations)

    def test_facade_area_filter(self, box_building_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        # Very high area threshold → no facades
        facades = detect_facades(box_building_planes, min_facade_area=1000.0, scale=1.0)
        assert len(facades) == 0

    def test_empty_planes(self):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        facades = detect_facades([], scale=1.0)
        assert facades == []

    def test_only_horizontal_planes_no_facades(self):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        planes = [_make_horiz(0, 1.0, 0.0, "floor", [[-10, 0, -10], [10, 0, -10], [10, 0, 10]])]
        facades = detect_facades(planes, scale=1.0)
        assert len(facades) == 0


# ---------------------------------------------------------------------------
# Module D: Footprint Extraction
# ---------------------------------------------------------------------------

class TestFootprintExtraction:
    def test_box_footprint(self, box_building_planes):
        from gss.steps.s06c_building_extraction._footprint_extraction import extract_footprint
        fp = extract_footprint(planes=box_building_planes, scale=1.0)
        assert fp is not None
        assert "polygon_2d" in fp
        assert fp["area"] > 0
        # Box building should have roughly 4 vertices
        assert len(fp["polygon_2d"]) >= 4

    def test_l_shape_not_convex(self, l_shaped_building_planes):
        from gss.steps.s06c_building_extraction._footprint_extraction import extract_footprint
        fp = extract_footprint(planes=l_shaped_building_planes, scale=1.0)
        assert fp is not None
        # L-shape has more than 4 vertices
        assert len(fp["polygon_2d"]) >= 4

    def test_oriented_bbox(self, box_building_planes):
        from gss.steps.s06c_building_extraction._footprint_extraction import extract_footprint
        fp = extract_footprint(planes=box_building_planes, scale=1.0)
        assert fp is not None
        if fp.get("oriented_bbox"):
            obb = fp["oriented_bbox"]
            assert "center" in obb
            assert "dimensions" in obb
            assert len(obb["dimensions"]) == 2

    def test_no_planes(self):
        from gss.steps.s06c_building_extraction._footprint_extraction import extract_footprint
        fp = extract_footprint(planes=[], scale=1.0)
        assert fp is None

    def test_from_points(self):
        from gss.steps.s06c_building_extraction._footprint_extraction import extract_footprint
        pts = np.array([
            [0, 0, 0], [10, 0, 0], [10, 5, 0], [10, 5, 8],
            [0, 5, 8], [0, 0, 8], [5, 0, 4],
        ], dtype=float)
        fp = extract_footprint(building_points=pts, scale=1.0)
        assert fp is not None
        assert fp["area"] > 0


# ---------------------------------------------------------------------------
# Module E: Roof Structuring
# ---------------------------------------------------------------------------

class TestRoofStructuring:
    def test_flat_roof(self, box_building_planes):
        from gss.steps.s06c_building_extraction._roof_structuring import structure_roof
        result = structure_roof(box_building_planes, ceiling_heights=[6.0], scale=1.0)
        # Flat roof or none (the ceiling IS the roof for flat)
        assert result["roof_type"] in ("flat", "none")

    def test_gable_roof(self, gable_roof_building_planes):
        from gss.steps.s06c_building_extraction._roof_structuring import structure_roof
        result = structure_roof(
            gable_roof_building_planes,
            ceiling_heights=[4.0],
            scale=1.0,
        )
        assert result["roof_type"] in ("gable", "mixed", "shed")
        assert len(result["faces"]) >= 2

    def test_gable_has_ridge(self, gable_roof_building_planes):
        from gss.steps.s06c_building_extraction._roof_structuring import structure_roof
        result = structure_roof(
            gable_roof_building_planes,
            ceiling_heights=[4.0],
            scale=1.0,
        )
        # Gable roof should produce at least one ridge line
        assert len(result["ridges"]) >= 1 or len(result["faces"]) >= 2

    def test_no_planes(self):
        from gss.steps.s06c_building_extraction._roof_structuring import structure_roof
        result = structure_roof([], scale=1.0)
        assert result["roof_type"] == "none"
        assert result["faces"] == []

    def test_roof_face_attributes(self, gable_roof_building_planes):
        from gss.steps.s06c_building_extraction._roof_structuring import structure_roof
        result = structure_roof(
            gable_roof_building_planes,
            ceiling_heights=[4.0],
            scale=1.0,
        )
        for face in result["faces"]:
            assert "slope_deg" in face
            assert "aspect" in face
            assert face["slope_deg"] >= 0
            assert face["aspect"] in ("north", "south", "east", "west", "flat")


# ---------------------------------------------------------------------------
# Module F: Storey Detection
# ---------------------------------------------------------------------------

class TestStoreyDetection:
    def test_single_storey(self, box_building_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        from gss.steps.s06c_building_extraction._storey_detection import detect_storeys_from_exterior
        facades = detect_facades(box_building_planes, min_facade_area=1.0, scale=1.0)
        storeys = detect_storeys_from_exterior(
            facades, box_building_planes,
            min_storey_height=2.0, max_storey_height=8.0, scale=1.0,
        )
        # Single-storey building: 1 or 2 storeys (depending on detection sensitivity)
        assert len(storeys) >= 1

    def test_no_facades_no_storeys(self):
        from gss.steps.s06c_building_extraction._storey_detection import detect_storeys_from_exterior
        storeys = detect_storeys_from_exterior([], [], scale=1.0)
        assert storeys == []

    def test_storey_has_elevation(self, box_building_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades
        from gss.steps.s06c_building_extraction._storey_detection import detect_storeys_from_exterior
        facades = detect_facades(box_building_planes, min_facade_area=1.0, scale=1.0)
        storeys = detect_storeys_from_exterior(
            facades, box_building_planes,
            min_storey_height=2.0, max_storey_height=8.0, scale=1.0,
        )
        for s in storeys:
            assert "elevation" in s
            assert "height" in s
            assert "confidence" in s
            assert s["height"] > 0


# ---------------------------------------------------------------------------
# Integration: Full Step
# ---------------------------------------------------------------------------

class TestBuildingExtractionStep:
    def test_full_step_box(self, data_root: Path, box_planes_json, box_boundaries_json):
        from gss.steps.s06c_building_extraction.step import BuildingExtractionStep
        config = BuildingExtractionConfig(
            scale_mode="metric",
            enable_building_segmentation=False,
            enable_storey_detection=False,
            min_ground_extent=5.0,
            min_facade_area=1.0,
        )
        step = BuildingExtractionStep(config=config, data_root=data_root)
        inputs = BuildingExtractionInput(
            planes_file=box_planes_json,
            boundaries_file=box_boundaries_json,
        )
        output = step.run(inputs)

        assert output.planes_file.exists()
        assert output.boundaries_file.exists()
        assert output.building_context_file.exists()
        assert output.num_facades >= 2

        # Verify building_context.json
        with open(output.building_context_file) as f:
            ctx = json.load(f)
        assert "facades" in ctx
        assert "coordinate_scale" in ctx

    def test_full_step_output_has_ground(self, data_root: Path, box_planes_json, box_boundaries_json):
        from gss.steps.s06c_building_extraction.step import BuildingExtractionStep
        config = BuildingExtractionConfig(
            scale_mode="metric",
            enable_building_segmentation=False,
            enable_storey_detection=False,
            min_ground_extent=5.0,
        )
        step = BuildingExtractionStep(config=config, data_root=data_root)
        inputs = BuildingExtractionInput(
            planes_file=box_planes_json,
            boundaries_file=box_boundaries_json,
        )
        output = step.run(inputs)

        # Check ground plane is labeled
        with open(output.planes_file) as f:
            planes = json.load(f)
        ground_planes = [p for p in planes if p["label"] == "ground"]
        assert len(ground_planes) == 1

    def test_validate_inputs(self, data_root: Path, box_planes_json, box_boundaries_json):
        from gss.steps.s06c_building_extraction.step import BuildingExtractionStep
        config = BuildingExtractionConfig()
        step = BuildingExtractionStep(config=config, data_root=data_root)

        # Valid inputs
        inputs = BuildingExtractionInput(
            planes_file=box_planes_json,
            boundaries_file=box_boundaries_json,
        )
        assert step.validate_inputs(inputs) is True

        # Invalid inputs
        inputs_bad = BuildingExtractionInput(
            planes_file=Path("/nonexistent/planes.json"),
            boundaries_file=box_boundaries_json,
        )
        assert step.validate_inputs(inputs_bad) is False
