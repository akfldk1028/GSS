"""Integration tests: run s06c on real s06 pipeline output.

These tests are skipped if real data doesn't exist at data/interim/s06_planes/.
They validate that s06c handles actual noisy RANSAC planes correctly.

Run: pytest tests/integration/test_s06c_real_data.py -v
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from .conftest import requires_real_data, REAL_S06_DIR


@requires_real_data
class TestRealDataGroundSeparation:
    """Module A on real data."""

    def test_detects_ground_or_floor(self, real_planes):
        from gss.steps.s06c_building_extraction._ground_separation import detect_ground_plane

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        ground = detect_ground_plane(planes, min_ground_extent=1.0, scale=1.0)
        # Interior room should have a floor detectable as ground
        if ground is not None:
            assert ground["label"] == "ground"
            # Normal should be roughly vertical (|ny| > 0.8)
            ny = abs(np.asarray(ground["normal"])[1])
            assert ny > 0.8, f"Ground normal not vertical: ny={ny}"

    def test_ground_not_ceiling(self, real_planes):
        from gss.steps.s06c_building_extraction._ground_separation import detect_ground_plane

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        ground = detect_ground_plane(planes, min_ground_extent=1.0, scale=1.0)
        if ground is not None:
            # Ground should be the LOWEST horizontal plane, not ceiling
            ceiling_planes = [p for p in real_planes if p.get("label") == "ceiling"]
            if ceiling_planes:
                # Get centroid Y of ground
                g_bnd = ground.get("boundary_3d")
                if g_bnd is not None and len(g_bnd) > 0:
                    g_y = float(np.asarray(g_bnd)[:, 1].mean())
                    for cp in ceiling_planes:
                        c_bnd = cp.get("boundary_3d")
                        if c_bnd and len(c_bnd) > 0:
                            c_y = float(np.asarray(c_bnd)[:, 1].mean())
                            assert g_y < c_y, f"Ground ({g_y}) not below ceiling ({c_y})"


@requires_real_data
class TestRealDataFacadeDetection:
    """Module C on real data."""

    def test_detects_facades(self, real_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        facades = detect_facades(planes, min_facade_area=0.5, scale=1.0)
        # Real room has walls → should detect at least 1 facade
        wall_count = sum(1 for p in real_planes if p.get("label") == "wall")
        if wall_count > 0:
            assert len(facades) >= 1, f"Expected facades from {wall_count} walls"

    def test_facade_orientations_valid(self, real_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        facades = detect_facades(planes, min_facade_area=0.5, scale=1.0)
        valid_orientations = {"north", "south", "east", "west", "ne", "nw", "se", "sw", "unknown"}
        for f in facades:
            assert f["orientation"] in valid_orientations, f"Invalid orientation: {f['orientation']}"
            assert f["area"] > 0
            assert len(f["plane_ids"]) >= 1

    def test_no_degenerate_normals(self, real_planes):
        from gss.steps.s06c_building_extraction._facade_detection import detect_facades

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        facades = detect_facades(planes, min_facade_area=0.5, scale=1.0)
        for f in facades:
            n = np.asarray(f["normal"])
            norm = np.linalg.norm(n)
            assert norm > 0.5, f"Degenerate facade normal: {n} (norm={norm})"


@requires_real_data
class TestRealDataFootprint:
    """Module D on real data."""

    def test_extracts_footprint(self, real_planes):
        from gss.steps.s06c_building_extraction._footprint_extraction import extract_footprint

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        fp = extract_footprint(planes=planes, scale=1.0)
        if fp is not None:
            assert fp["area"] > 0, "Footprint area should be positive"
            assert len(fp["polygon_2d"]) >= 3, "Footprint needs at least 3 vertices"


@requires_real_data
class TestRealDataRoof:
    """Module E on real data."""

    def test_roof_structure(self, real_planes):
        from gss.steps.s06c_building_extraction._roof_structuring import structure_roof

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        result = structure_roof(planes, scale=1.0)
        # Interior room: should be flat or none
        assert result["roof_type"] in ("none", "flat", "shed", "gable", "hip", "mixed")
        assert isinstance(result["faces"], list)
        assert isinstance(result["ridges"], list)


@requires_real_data
class TestRealDataFullPipeline:
    """Full s06c step on real data."""

    def test_full_step_real_data(self, tmp_path):
        from gss.steps.s06c_building_extraction.step import BuildingExtractionStep
        from gss.steps.s06c_building_extraction.config import BuildingExtractionConfig
        from gss.steps.s06c_building_extraction.contracts import BuildingExtractionInput

        # Set up data_root pointing to tmp but with real s06 data copied in
        data_root = tmp_path
        s06_dir = data_root / "interim" / "s06_planes"
        s06_dir.mkdir(parents=True, exist_ok=True)
        s06c_dir = data_root / "interim" / "s06c_building_extraction"
        s06c_dir.mkdir(parents=True, exist_ok=True)

        # Copy real data
        shutil.copy(REAL_S06_DIR / "planes.json", s06_dir / "planes.json")
        shutil.copy(REAL_S06_DIR / "boundaries.json", s06_dir / "boundaries.json")
        if (REAL_S06_DIR / "manhattan_alignment.json").exists():
            shutil.copy(REAL_S06_DIR / "manhattan_alignment.json", s06_dir / "manhattan_alignment.json")

        config = BuildingExtractionConfig(
            scale_mode="auto",
            enable_building_segmentation=False,
            enable_storey_detection=True,
            min_ground_extent=1.0,
            min_facade_area=0.5,
        )
        step = BuildingExtractionStep(config=config, data_root=data_root)
        inputs = BuildingExtractionInput(
            planes_file=s06_dir / "planes.json",
            boundaries_file=s06_dir / "boundaries.json",
        )

        output = step.run(inputs)

        # Basic output checks
        assert output.planes_file.exists()
        assert output.boundaries_file.exists()
        assert output.building_context_file.exists()

        # Load and validate building_context.json
        with open(output.building_context_file, encoding="utf-8") as f:
            ctx = json.load(f)

        assert "coordinate_scale" in ctx
        assert ctx["coordinate_scale"] > 0

        # Should have facades (real data has walls)
        if "facades" in ctx:
            for facade in ctx["facades"]:
                assert "id" in facade
                assert "normal" in facade
                assert "orientation" in facade
                assert "area" in facade
                assert facade["area"] > 0

        # Validate planes.json has proper structure
        with open(output.planes_file, encoding="utf-8") as f:
            planes = json.load(f)
        assert len(planes) > 0
        for p in planes:
            assert "id" in p
            assert "normal" in p
            assert "d" in p
            assert "label" in p
            assert len(p["normal"]) == 3

        # Stats file should exist
        stats_file = data_root / "interim" / "s06c_building_extraction" / "stats.json"
        assert stats_file.exists()
        with open(stats_file, encoding="utf-8") as f:
            stats = json.load(f)
        assert "scale" in stats

    def test_manhattan_transform_roundtrip(self, real_planes, real_manhattan_rotation):
        """Verify Manhattan transform preserves data after roundtrip."""
        if real_manhattan_rotation is None:
            pytest.skip("No manhattan_alignment.json")

        from gss.steps.s06c_building_extraction.step import (
            _transform_to_manhattan, _transform_from_manhattan,
        )

        R = real_manhattan_rotation
        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]

        # Save originals
        original_normals = [p["normal"].copy() for p in planes]
        original_boundaries = [p["boundary_3d"].copy() for p in planes]

        # Forward + back
        _transform_to_manhattan(planes, R)
        _transform_from_manhattan(planes, R)

        # Should match originals (within floating point)
        for i, p in enumerate(planes):
            np.testing.assert_allclose(
                p["normal"], original_normals[i], atol=1e-10,
                err_msg=f"Normal roundtrip failed for plane {p['id']}",
            )
            if len(p["boundary_3d"]) > 0:
                np.testing.assert_allclose(
                    p["boundary_3d"], original_boundaries[i], atol=1e-10,
                    err_msg=f"Boundary roundtrip failed for plane {p['id']}",
                )

    def test_scale_estimation_reasonable(self, real_planes):
        """Auto-scale should produce a reasonable value for the real room."""
        from gss.steps.s06c_building_extraction.step import _estimate_scale

        planes = [
            {**p, "normal": np.asarray(p["normal"], dtype=float),
             "boundary_3d": np.asarray(p["boundary_3d"], dtype=float) if p.get("boundary_3d") else np.empty((0, 3))}
            for p in real_planes
        ]
        scale = _estimate_scale(planes, expected_building_size=12.0)
        # Real room is ~6.9m wide, expected 12m → scale ≈ 0.58
        # Should be in reasonable range (0.1 to 10)
        assert 0.1 <= scale <= 10.0, f"Scale {scale} out of reasonable range"

    def test_output_json_serializable(self, tmp_path):
        """Ensure all outputs are valid JSON (no numpy types)."""
        from gss.steps.s06c_building_extraction.step import BuildingExtractionStep
        from gss.steps.s06c_building_extraction.config import BuildingExtractionConfig
        from gss.steps.s06c_building_extraction.contracts import BuildingExtractionInput

        data_root = tmp_path
        s06_dir = data_root / "interim" / "s06_planes"
        s06_dir.mkdir(parents=True, exist_ok=True)
        (data_root / "interim" / "s06c_building_extraction").mkdir(parents=True, exist_ok=True)

        shutil.copy(REAL_S06_DIR / "planes.json", s06_dir / "planes.json")
        shutil.copy(REAL_S06_DIR / "boundaries.json", s06_dir / "boundaries.json")
        if (REAL_S06_DIR / "manhattan_alignment.json").exists():
            shutil.copy(REAL_S06_DIR / "manhattan_alignment.json", s06_dir / "manhattan_alignment.json")

        config = BuildingExtractionConfig(
            scale_mode="auto",
            enable_building_segmentation=False,
            enable_storey_detection=False,
            min_ground_extent=1.0,
            min_facade_area=0.5,
        )
        step = BuildingExtractionStep(config=config, data_root=data_root)
        inputs = BuildingExtractionInput(
            planes_file=s06_dir / "planes.json",
            boundaries_file=s06_dir / "boundaries.json",
        )
        output = step.run(inputs)

        # All JSON files should round-trip cleanly
        for json_file in [output.planes_file, output.boundaries_file, output.building_context_file]:
            with open(json_file, encoding="utf-8") as f:
                text = f.read()
            data = json.loads(text)
            # Re-serialize should work (no numpy types)
            text2 = json.dumps(data)
            assert len(text2) > 2  # not empty
