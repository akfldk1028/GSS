"""Tests for gRPC security — checkpoint path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.grpc.security import validate_checkpoint_path


@pytest.fixture
def tmp_checkpoints(tmp_path: Path):
    """Create test checkpoint files in allowed/disallowed dirs."""
    # Allowed directory
    allowed = tmp_path / "data" / "jobs"
    allowed.mkdir(parents=True)
    valid_ckpt = allowed / "model.pt"
    valid_ckpt.write_bytes(b"fake checkpoint")

    # Disallowed directory
    disallowed = tmp_path / "etc"
    disallowed.mkdir()
    evil_ckpt = disallowed / "evil.pt"
    evil_ckpt.write_bytes(b"malicious payload")

    # Wrong extension
    wrong_ext = allowed / "model.txt"
    wrong_ext.write_text("not a checkpoint")

    return {
        "allowed_dir": str(tmp_path / "data"),
        "valid_ckpt": str(valid_ckpt),
        "evil_ckpt": str(evil_ckpt),
        "wrong_ext": str(wrong_ext),
    }


class TestCheckpointValidation:
    def test_valid_path(self, tmp_checkpoints):
        result = validate_checkpoint_path(
            tmp_checkpoints["valid_ckpt"],
            allowed_dirs=[tmp_checkpoints["allowed_dir"]],
        )
        assert result.exists()
        assert result.suffix == ".pt"

    def test_rejects_outside_whitelist(self, tmp_checkpoints):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            validate_checkpoint_path(
                tmp_checkpoints["evil_ckpt"],
                allowed_dirs=[tmp_checkpoints["allowed_dir"]],
            )

    def test_rejects_nonexistent(self, tmp_checkpoints):
        with pytest.raises(FileNotFoundError):
            validate_checkpoint_path(
                "/nonexistent/model.pt",
                allowed_dirs=[tmp_checkpoints["allowed_dir"]],
            )

    def test_rejects_wrong_extension(self, tmp_checkpoints):
        with pytest.raises(PermissionError, match="Invalid checkpoint extension"):
            validate_checkpoint_path(
                tmp_checkpoints["wrong_ext"],
                allowed_dirs=[tmp_checkpoints["allowed_dir"]],
            )

    def test_rejects_directory(self, tmp_checkpoints):
        with pytest.raises(PermissionError, match="not a file"):
            validate_checkpoint_path(
                tmp_checkpoints["allowed_dir"],
                allowed_dirs=[tmp_checkpoints["allowed_dir"]],
            )

    def test_multiple_allowed_dirs(self, tmp_checkpoints, tmp_path):
        # Create another allowed dir
        alt_dir = tmp_path / "alt_models"
        alt_dir.mkdir()
        alt_ckpt = alt_dir / "alt.ckpt"
        alt_ckpt.write_bytes(b"alt model")

        result = validate_checkpoint_path(
            str(alt_ckpt),
            allowed_dirs=[tmp_checkpoints["allowed_dir"], str(alt_dir)],
        )
        assert result == alt_ckpt.resolve()

    def test_path_traversal_blocked(self, tmp_checkpoints, tmp_path):
        """Ensure ../../ traversal is caught."""
        traversal_path = str(
            Path(tmp_checkpoints["allowed_dir"]) / ".." / "etc" / "evil.pt"
        )
        with pytest.raises((PermissionError, FileNotFoundError)):
            validate_checkpoint_path(
                traversal_path,
                allowed_dirs=[tmp_checkpoints["allowed_dir"]],
            )

    def test_pth_extension_allowed(self, tmp_path):
        allowed = tmp_path / "models"
        allowed.mkdir()
        ckpt = allowed / "weights.pth"
        ckpt.write_bytes(b"data")

        result = validate_checkpoint_path(str(ckpt), allowed_dirs=[str(allowed)])
        assert result.suffix == ".pth"

    def test_ckpt_extension_allowed(self, tmp_path):
        allowed = tmp_path / "models"
        allowed.mkdir()
        ckpt = allowed / "epoch_10.ckpt"
        ckpt.write_bytes(b"data")

        result = validate_checkpoint_path(str(ckpt), allowed_dirs=[str(allowed)])
        assert result.suffix == ".ckpt"
