"""Integration test: gRPC checkpoint whitelist enforcement.

Verifies that the gRPC server actually rejects checkpoint paths
outside the whitelist — not just unit-testing the validator.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

try:
    import grpc
    import torch

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="grpcio or torch not installed")


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def secured_server(tmp_path):
    """gRPC server with strict checkpoint whitelist."""
    from server.grpc.render_server import create_server
    from server.grpc.render_client import RenderClient

    # Allowed directory
    allowed_dir = tmp_path / "safe_models"
    allowed_dir.mkdir()

    # Disallowed directory
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()

    # Create checkpoints in both
    safe_ckpt = allowed_dir / "good.pt"
    torch.save({"means": torch.randn(10, 3)}, safe_ckpt)

    unsafe_ckpt = unsafe_dir / "evil.pt"
    torch.save({"means": torch.randn(10, 3)}, unsafe_ckpt)

    port = _find_free_port()
    server, _ = create_server(
        port=port,
        max_models=4,
        allowed_checkpoint_dirs=[str(allowed_dir)],
    )
    server.start()
    time.sleep(0.1)

    client = RenderClient(f"localhost:{port}")

    yield client, safe_ckpt, unsafe_ckpt

    client.close()
    server.stop(grace=1)


class TestCheckpointWhitelistIntegration:
    def test_load_from_allowed_dir_succeeds(self, secured_server):
        client, safe_ckpt, _ = secured_server
        result = client.load_model("good_model", str(safe_ckpt))
        assert result["success"] is True

    def test_load_from_disallowed_dir_rejected(self, secured_server):
        client, _, unsafe_ckpt = secured_server
        result = client.load_model("evil_model", str(unsafe_ckpt))
        assert result["success"] is False
        assert "outside allowed directories" in result["error"]

    def test_load_nonexistent_path_rejected(self, secured_server):
        client, _, _ = secured_server
        result = client.load_model("ghost", "/tmp/nonexistent/model.pt")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_load_wrong_extension_rejected(self, secured_server, tmp_path):
        """Even if in allowed dir, wrong extension is blocked."""
        client, safe_ckpt, _ = secured_server
        txt_file = safe_ckpt.parent / "readme.txt"
        txt_file.write_text("not a model")

        result = client.load_model("txt_model", str(txt_file))
        assert result["success"] is False
        assert "extension" in result["error"].lower()

    def test_path_traversal_rejected(self, secured_server):
        """../../../ traversal is caught even over gRPC."""
        client, safe_ckpt, unsafe_ckpt = secured_server
        traversal = str(safe_ckpt.parent / ".." / "unsafe" / "evil.pt")

        result = client.load_model("traversal", traversal)
        assert result["success"] is False
        assert "outside allowed" in result["error"].lower() or "not found" in result["error"].lower()
