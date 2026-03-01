"""Unit tests for download_sample_data script (refactored API)."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from scripts.download_sample_data import (
    DATASETS,
    _download_with_progress,
    _extract_zip,
    _find_ply_files,
    _validate_ply,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATASETS configuration
# ---------------------------------------------------------------------------

class TestDatasetsConfig:
    """Test that DATASETS registry is properly structured."""

    def test_datasets_not_empty(self):
        assert len(DATASETS) > 0

    def test_required_keys(self):
        required = {"name", "url", "target_dir", "download_type"}
        for key, config in DATASETS.items():
            missing = required - set(config.keys())
            assert not missing, f"Dataset '{key}' missing keys: {missing}"

    def test_known_datasets_exist(self):
        for name in ["bonsai", "bicycle", "stump", "lego", "tandt_db", "mipnerf360"]:
            assert name in DATASETS, f"Expected dataset '{name}' in DATASETS"

    def test_download_types_valid(self):
        valid_types = {"ply", "zip"}
        for key, config in DATASETS.items():
            assert config["download_type"] in valid_types, (
                f"Dataset '{key}' has invalid download_type: {config['download_type']}"
            )

    def test_lego_has_extract_subdir(self):
        lego = DATASETS["lego"]
        assert "extract_subdir" in lego
        assert lego["extract_subdir"] == "nerf_synthetic/lego"

    def test_hf_scenes_are_ply_type(self):
        for name in ["bonsai", "bicycle", "stump"]:
            assert DATASETS[name]["download_type"] == "ply"


# ---------------------------------------------------------------------------
# PLY validation
# ---------------------------------------------------------------------------

class TestValidatePly:
    def test_valid_ply(self, tmp_path: Path):
        ply = tmp_path / "test.ply"
        ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
        assert _validate_ply(ply) is True

    def test_invalid_ply(self, tmp_path: Path):
        ply = tmp_path / "bad.ply"
        ply.write_bytes(b"NOT A PLY FILE AT ALL")
        assert _validate_ply(ply) is False

    def test_nonexistent_file(self, tmp_path: Path):
        assert _validate_ply(tmp_path / "missing.ply") is False


# ---------------------------------------------------------------------------
# Find PLY files
# ---------------------------------------------------------------------------

class TestFindPlyFiles:
    def test_finds_ply_recursively(self, tmp_path: Path):
        (tmp_path / "a.ply").write_bytes(b"ply")
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "b.ply").write_bytes(b"ply")
        (tmp_path / "c.txt").write_text("not ply")

        result = _find_ply_files(tmp_path)
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"a.ply", "b.ply"}

    def test_empty_dir(self, tmp_path: Path):
        assert _find_ply_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Download (mocked)
# ---------------------------------------------------------------------------

class TestDownload:
    def test_http_download(self, tmp_path: Path):
        output = tmp_path / "test.ply"
        content = b"ply\nfake data here"

        with patch("scripts.download_sample_data.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.headers = {"content-length": str(len(content))}
            mock_resp.iter_content = Mock(return_value=[content])
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            _download_with_progress("http://example.com/test.ply", output)

        assert output.exists()
        assert output.read_bytes() == content

    def test_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "nested" / "deep" / "file.ply"
        content = b"ply data"

        with patch("scripts.download_sample_data.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.headers = {"content-length": str(len(content))}
            mock_resp.iter_content = Mock(return_value=[content])
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            _download_with_progress("http://example.com/f.ply", output)

        assert output.parent.exists()
        assert output.exists()

    def test_http_error_propagates(self, tmp_path: Path):
        output = tmp_path / "fail.ply"

        with patch("scripts.download_sample_data.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
            mock_get.return_value = mock_resp

            with pytest.raises(requests.HTTPError):
                _download_with_progress("http://example.com/404", output)

    def test_google_drive_url(self, tmp_path: Path):
        output = tmp_path / "drive.ply"
        url = "https://drive.google.com/file/d/ABC123/view"

        # gdown is conditionally imported inside _download_with_progress,
        # so we patch builtins.__import__ to inject a mock.
        mock_gdown = Mock()
        mock_gdown.download.side_effect = lambda u, p, quiet, fuzzy: (
            Path(p).write_bytes(b"ply gd data")
        )

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "gdown":
                return mock_gdown
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            _download_with_progress(url, output)

        mock_gdown.download.assert_called_once_with(
            url, str(output), quiet=False, fuzzy=True,
        )
        assert output.exists()


# ---------------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_zip(tmp_path: Path) -> Path:
    """Create a mock zip with nested structure."""
    src = tmp_path / "_src"
    nested = src / "nerf_synthetic" / "lego"
    nested.mkdir(parents=True)
    (nested / "train").mkdir()
    (nested / "train" / "r_0.png").write_bytes(b"img0")
    (nested / "transforms_train.json").write_text('{"frames":[]}')

    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src))
    return zip_path


class TestExtractZip:
    def test_basic_extraction(self, tmp_path: Path, mock_zip: Path):
        out = tmp_path / "out"
        out.mkdir()
        _extract_zip(mock_zip, out, subdir=None)
        assert (out / "nerf_synthetic" / "lego" / "transforms_train.json").exists()

    def test_subdir_flattening(self, tmp_path: Path, mock_zip: Path):
        out = tmp_path / "flat"
        out.mkdir()
        _extract_zip(mock_zip, out, subdir="nerf_synthetic/lego")
        assert (out / "transforms_train.json").exists()
        assert (out / "train" / "r_0.png").exists()
        assert not (out / "nerf_synthetic").exists()

    def test_invalid_zip(self, tmp_path: Path):
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(zipfile.BadZipFile):
            _extract_zip(bad, out, subdir=None)

    def test_nonexistent_subdir(self, tmp_path: Path, mock_zip: Path):
        out = tmp_path / "out2"
        out.mkdir()
        _extract_zip(mock_zip, out, subdir="nonexistent/path")
        # Original structure preserved when subdir not found
        assert (out / "nerf_synthetic" / "lego").exists()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_download_and_extract_workflow(self, tmp_path: Path, mock_zip: Path):
        """Simulate full download → extract → find PLY workflow."""
        target = tmp_path / "dataset"
        target.mkdir()

        # Simulate zip download
        zip_content = mock_zip.read_bytes()
        dl_path = tmp_path / "dl.zip"

        with patch("scripts.download_sample_data.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.headers = {"content-length": str(len(zip_content))}
            mock_resp.iter_content = Mock(return_value=[zip_content])
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            _download_with_progress("http://example.com/data.zip", dl_path)

        assert dl_path.exists()

        _extract_zip(dl_path, target, subdir="nerf_synthetic/lego")
        assert (target / "transforms_train.json").exists()

    def test_ply_download_workflow(self, tmp_path: Path):
        """Simulate PLY download → validate workflow."""
        ply_content = b"ply\nformat binary_little_endian 1.0\nend_header\nDATA"
        ply_path = tmp_path / "scene" / "point_cloud.ply"

        with patch("scripts.download_sample_data.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.headers = {"content-length": str(len(ply_content))}
            mock_resp.iter_content = Mock(return_value=[ply_content])
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            _download_with_progress("http://example.com/pc.ply", ply_path)

        assert _validate_ply(ply_path) is True
        found = _find_ply_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "point_cloud.ply"
