"""Unit tests for download_sample_data script."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from scripts.download_sample_data import (
    DATASETS,
    download_file,
    extract_zip,
    validate_dataset,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_lego_dataset(tmp_path: Path) -> Path:
    """Create a mock lego dataset structure for testing."""
    dataset_dir = tmp_path / "lego"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create expected directories
    for subdir in ["train", "val", "test"]:
        img_dir = dataset_dir / subdir
        img_dir.mkdir(parents=True, exist_ok=True)

        # Create mock images
        for i in range(3):
            img_file = img_dir / f"r_{i}.png"
            img_file.write_bytes(b"fake_png_data")

    # Create transforms files
    for split in ["train", "val", "test"]:
        transforms_file = dataset_dir / f"transforms_{split}.json"
        transforms_data = {
            "camera_angle_x": 0.6911112070083618,
            "frames": [
                {
                    "file_path": f"./{split}/r_0",
                    "rotation": 0.012566370614359171,
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 4.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
                for i in range(3)
            ],
        }
        transforms_file.write_text(json.dumps(transforms_data, indent=2))

    return dataset_dir


@pytest.fixture
def mock_zip_file(tmp_path: Path, mock_lego_dataset: Path) -> Path:
    """Create a mock zip file with lego dataset structure."""
    zip_path = tmp_path / "test_dataset.zip"

    # Create a nested structure like the real dataset
    temp_extract = tmp_path / "temp_extract"
    temp_extract.mkdir(parents=True, exist_ok=True)
    nested_dir = temp_extract / "nerf_synthetic" / "lego"
    nested_dir.mkdir(parents=True, exist_ok=True)

    # Copy mock dataset to nested directory
    import shutil

    for item in mock_lego_dataset.iterdir():
        if item.is_dir():
            shutil.copytree(item, nested_dir / item.name)
        else:
            shutil.copy(item, nested_dir / item.name)

    # Create zip file
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in temp_extract.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_extract)
                zf.write(file_path, arcname)

    return zip_path


@pytest.mark.parametrize(
    "url,is_google_drive",
    [
        ("http://example.com/test.zip", False),
        ("https://example.org/data.zip", False),
        ("https://drive.google.com/file/d/123456/view", True),
        ("https://drive.usercontent.google.com/download?id=123456", True),
    ],
)
def test_download_file(tmp_path: Path, url: str, is_google_drive: bool):
    """Test download_file with various URL types (regular HTTP and Google Drive)."""
    output_path = tmp_path / "test_download.zip"
    mock_content = b"fake_zip_content"

    if is_google_drive:
        # Mock gdown.download for Google Drive URLs
        with patch("scripts.download_sample_data.gdown.download") as mock_gdown:
            # Simulate gdown creating the file
            mock_gdown.side_effect = lambda u, p, quiet, fuzzy: output_path.write_bytes(mock_content)

            download_file(url, output_path)

            # Verify gdown was called correctly
            mock_gdown.assert_called_once_with(url, str(output_path), quiet=False, fuzzy=True)
            assert output_path.exists(), "Downloaded file should exist"
            logger.info(f"✓ download_file Google Drive test passed: {url}")
    else:
        # Mock requests.get for regular HTTP URLs
        with patch("scripts.download_sample_data.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.headers = {"content-length": str(len(mock_content))}
            mock_response.iter_content = Mock(return_value=[mock_content])
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            download_file(url, output_path)

            # Verify file was created
            assert output_path.exists(), "Downloaded file should exist"
            assert output_path.read_bytes() == mock_content, "Downloaded content should match"
            mock_get.assert_called_once_with(url, stream=True)
            mock_response.raise_for_status.assert_called_once()
            logger.info(f"✓ download_file HTTP test passed: {url}")


def test_download_file_creates_parent_directory(tmp_path: Path):
    """Test that download_file creates parent directories if they don't exist."""
    nested_path = tmp_path / "nested" / "deep" / "dir" / "test.zip"
    mock_url = "http://example.com/test.zip"
    mock_content = b"fake_zip_content"

    # Mock requests.get
    with patch("scripts.download_sample_data.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(mock_content))}
        mock_response.iter_content = Mock(return_value=[mock_content])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        download_file(mock_url, nested_path)

        # Verify parent directory was created
        assert nested_path.parent.exists(), "Parent directory should be created"
        assert nested_path.exists(), "Downloaded file should exist"
        logger.info("✓ download_file parent directory creation test passed")


def test_download_file_handles_http_error(tmp_path: Path):
    """Test that download_file properly handles HTTP errors."""
    output_path = tmp_path / "test_download.zip"
    mock_url = "http://example.com/notfound.zip"

    # Mock requests.get to raise an error
    with patch("scripts.download_sample_data.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Verify that the error is raised
        with pytest.raises(requests.HTTPError):
            download_file(mock_url, output_path)

        logger.info("✓ download_file HTTP error handling test passed")


def test_download_file_chunked_download(tmp_path: Path):
    """Test that download_file properly handles chunked downloads."""
    output_path = tmp_path / "test_download.zip"
    mock_url = "http://example.com/largefile.zip"

    # Create multiple chunks
    chunk1 = b"chunk1_data"
    chunk2 = b"chunk2_data"
    chunk3 = b"chunk3_data"
    expected_content = chunk1 + chunk2 + chunk3

    # Mock requests.get
    with patch("scripts.download_sample_data.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(expected_content))}
        mock_response.iter_content = Mock(return_value=[chunk1, chunk2, chunk3])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        download_file(mock_url, output_path)

        # Verify all chunks were written
        assert output_path.exists(), "Downloaded file should exist"
        assert output_path.read_bytes() == expected_content, "All chunks should be written"
        logger.info("✓ download_file chunked download test passed")


def test_extract_zip_basic(tmp_path: Path, mock_zip_file: Path):
    """Test basic zip extraction without subdirectory handling."""
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(mock_zip_file, extract_dir, subdir=None)

    # Verify extraction
    assert extract_dir.exists(), "Extract directory should exist"
    assert (extract_dir / "nerf_synthetic" / "lego").exists(), "Nested structure should be preserved"
    logger.info(f"✓ extract_zip basic test passed: {extract_dir}")


def test_extract_zip_with_subdir(tmp_path: Path, mock_zip_file: Path):
    """Test zip extraction with subdirectory flattening."""
    extract_dir = tmp_path / "extracted_flat"
    extract_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(mock_zip_file, extract_dir, subdir="nerf_synthetic/lego")

    # Verify subdirectory was flattened
    assert extract_dir.exists(), "Extract directory should exist"
    assert (extract_dir / "train").exists(), "Train directory should be at root"
    assert (extract_dir / "transforms_train.json").exists(), "Transform files should be at root"
    assert not (extract_dir / "nerf_synthetic").exists(), "Parent directory should be removed"
    logger.info(f"✓ extract_zip with subdir test passed: {extract_dir}")


def test_validate_dataset_valid(mock_lego_dataset: Path):
    """Test validate_dataset with a valid lego dataset."""
    config = DATASETS["lego"]

    result = validate_dataset(mock_lego_dataset, config)

    assert result is True, "Validation should pass for valid dataset"
    logger.info("✓ validate_dataset valid test passed")


def test_validate_dataset_missing_file(mock_lego_dataset: Path):
    """Test validate_dataset with missing expected file."""
    config = DATASETS["lego"]

    # Remove one of the expected files
    (mock_lego_dataset / "transforms_train.json").unlink()

    result = validate_dataset(mock_lego_dataset, config)

    assert result is False, "Validation should fail when expected file is missing"
    logger.info("✓ validate_dataset missing file test passed")


def test_validate_dataset_missing_directory(mock_lego_dataset: Path):
    """Test validate_dataset with missing expected directory."""
    config = DATASETS["lego"]

    # Remove one of the expected directories
    import shutil

    shutil.rmtree(mock_lego_dataset / "train")

    result = validate_dataset(mock_lego_dataset, config)

    assert result is False, "Validation should fail when expected directory is missing"
    logger.info("✓ validate_dataset missing directory test passed")


def test_validate_dataset_invalid_json(mock_lego_dataset: Path):
    """Test validate_dataset with invalid JSON structure."""
    config = DATASETS["lego"]

    # Create invalid JSON (missing 'frames' key)
    invalid_json = {"camera_angle_x": 0.69}
    (mock_lego_dataset / "transforms_train.json").write_text(json.dumps(invalid_json))

    result = validate_dataset(mock_lego_dataset, config)

    assert result is False, "Validation should fail for invalid JSON structure"
    logger.info("✓ validate_dataset invalid JSON test passed")


def test_validate_dataset_nonexistent_dir(tmp_path: Path):
    """Test validate_dataset with nonexistent directory."""
    config = DATASETS["lego"]
    nonexistent_dir = tmp_path / "nonexistent"

    result = validate_dataset(nonexistent_dir, config)

    assert result is False, "Validation should fail for nonexistent directory"
    logger.info("✓ validate_dataset nonexistent dir test passed")


def test_datasets_configuration():
    """Test that DATASETS configuration is properly structured."""
    assert "lego" in DATASETS, "DATASETS should contain 'lego' dataset"
    assert "room_sample" in DATASETS, "DATASETS should contain 'room_sample' dataset"

    lego_config = DATASETS["lego"]
    assert "name" in lego_config, "Dataset config should have 'name'"
    assert "url" in lego_config, "Dataset config should have 'url'"
    assert "target_dir" in lego_config, "Dataset config should have 'target_dir'"
    assert "expected_files" in lego_config, "Dataset config should have 'expected_files'"
    assert "expected_dirs" in lego_config, "Dataset config should have 'expected_dirs'"

    # Verify lego dataset has proper expected files
    assert "transforms_train.json" in lego_config["expected_files"]
    assert "train" in lego_config["expected_dirs"]
    logger.info("✓ DATASETS configuration test passed")


def test_download_integration_mock(tmp_path: Path, mock_zip_file: Path):
    """Integration test simulating full download workflow with mocked HTTP."""
    target_dir = tmp_path / "lego_download"
    mock_url = "http://example.com/nerf_data.zip"

    # Mock the download
    with patch("scripts.download_sample_data.requests.get") as mock_get:
        mock_response = Mock()
        mock_content = mock_zip_file.read_bytes()
        mock_response.headers = {"content-length": str(len(mock_content))}
        mock_response.iter_content = Mock(return_value=[mock_content])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Download the zip
        zip_path = tmp_path / "downloaded.zip"
        download_file(mock_url, zip_path)

        # Extract it
        extract_zip(zip_path, target_dir, subdir="nerf_synthetic/lego")

        # Validate it
        config = DATASETS["lego"]
        result = validate_dataset(target_dir, config)

    assert result is True, "Full download workflow should succeed"
    assert (target_dir / "train").exists(), "Train directory should exist"
    assert (target_dir / "transforms_train.json").exists(), "Transform file should exist"
    logger.info("✓ Download integration test passed")
