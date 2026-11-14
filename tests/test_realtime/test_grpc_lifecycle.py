"""Integration test: gRPC server lifecycle — start, load, render, unload, stop.

Spins up a real gRPC server on a random port, loads a fake checkpoint,
renders frames, and verifies the full lifecycle.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

try:
    import grpc

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.skipif(not HAS_GRPC, reason="grpcio not installed"),
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
]


def _find_free_port() -> int:
    """Find a free port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def grpc_server_and_client(fake_checkpoint):
    """Start a real gRPC server and return (server, client, checkpoint_path)."""
    from server.grpc.render_server import create_server
    from server.grpc.render_client import RenderClient

    port = _find_free_port()
    allowed_dirs = [str(fake_checkpoint.parent)]

    server, servicer = create_server(
        port=port,
        max_models=2,
        allowed_checkpoint_dirs=allowed_dirs,
    )
    server.start()
    time.sleep(0.1)  # Let server bind

    client = RenderClient(f"localhost:{port}")

    yield server, client, fake_checkpoint

    client.close()
    server.stop(grace=1)


class TestGRPCLifecycle:
    """Full lifecycle: start → list (empty) → load → info → render → unload → stop."""

    def test_server_starts_and_lists_empty(self, grpc_server_and_client):
        _, client, _ = grpc_server_and_client
        models = client.get_available_models()
        assert models == []

    def test_load_model(self, grpc_server_and_client):
        _, client, ckpt_path = grpc_server_and_client

        result = client.load_model("test_scene", str(ckpt_path))
        assert result["success"] is True
        assert result["num_gaussians"] == 100

    def test_load_model_idempotent(self, grpc_server_and_client):
        """Loading the same model_id twice returns success without error."""
        _, client, ckpt_path = grpc_server_and_client

        r1 = client.load_model("test_scene", str(ckpt_path))
        r2 = client.load_model("test_scene", str(ckpt_path))
        assert r1["success"] is True
        assert r2["success"] is True

    def test_model_info_after_load(self, grpc_server_and_client):
        _, client, ckpt_path = grpc_server_and_client

        client.load_model("test_scene", str(ckpt_path))
        info = client.get_model_info("test_scene")
        assert info is not None
        assert info["model_id"] == "test_scene"
        assert info["num_gaussians"] == 100

    def test_model_info_not_found(self, grpc_server_and_client):
        _, client, _ = grpc_server_and_client

        info = client.get_model_info("nonexistent")
        assert info is None

    def test_render_frame(self, grpc_server_and_client, identity_c2w):
        """Load model and render a single JPEG frame."""
        _, client, ckpt_path = grpc_server_and_client

        client.load_model("test_scene", str(ckpt_path))

        result = client.render(
            model_id="test_scene",
            width=320,
            height=240,
            camera_to_world=np.array(identity_c2w, dtype=np.float32).reshape(4, 4),
            fx=256.0, fy=256.0,
            cx=160.0, cy=120.0,
            fmt="jpeg",
            jpeg_quality=50,
        )

        assert result["success"] is True
        assert result["render_time_ms"] > 0
        assert len(result["image_data"]) > 0
        # JPEG starts with FFD8
        assert result["image_data"][:2] == b"\xff\xd8"

    def test_render_png(self, grpc_server_and_client, identity_c2w):
        """Render a PNG frame."""
        _, client, ckpt_path = grpc_server_and_client

        client.load_model("test_scene", str(ckpt_path))

        result = client.render(
            model_id="test_scene",
            width=64, height=64,
            camera_to_world=np.array(identity_c2w, dtype=np.float32).reshape(4, 4),
            fx=50.0, fy=50.0, cx=32.0, cy=32.0,
            fmt="png",
        )

        assert result["success"] is True
        # PNG starts with 89 50 4E 47
        assert result["image_data"][:4] == b"\x89PNG"

    def test_render_raw(self, grpc_server_and_client, identity_c2w):
        """Render raw RGB bytes."""
        _, client, ckpt_path = grpc_server_and_client

        client.load_model("test_scene", str(ckpt_path))

        w, h = 32, 24
        result = client.render(
            model_id="test_scene",
            width=w, height=h,
            camera_to_world=np.array(identity_c2w, dtype=np.float32).reshape(4, 4),
            fx=25.0, fy=25.0, cx=16.0, cy=12.0,
            fmt="raw",
        )

        assert result["success"] is True
        assert len(result["image_data"]) == w * h * 3

    def test_render_unloaded_model_fails(self, grpc_server_and_client, identity_c2w):
        _, client, _ = grpc_server_and_client

        result = client.render(
            model_id="not_loaded",
            width=64, height=64,
            camera_to_world=np.array(identity_c2w, dtype=np.float32).reshape(4, 4),
            fx=50.0, fy=50.0, cx=32.0, cy=32.0,
        )

        assert result["success"] is False
        assert "not loaded" in result["error"]

    def test_unload_model(self, grpc_server_and_client):
        _, client, ckpt_path = grpc_server_and_client

        client.load_model("test_scene", str(ckpt_path))
        result = client.unload_model("test_scene")
        assert result["success"] is True

        # Verify removed
        models = client.get_available_models()
        assert len(models) == 0

    def test_unload_nonexistent_fails(self, grpc_server_and_client):
        _, client, _ = grpc_server_and_client

        result = client.unload_model("nonexistent")
        assert result["success"] is False

    def test_max_models_limit(self, grpc_server_and_client):
        """Server rejects loading beyond max_models (2)."""
        _, client, ckpt_path = grpc_server_and_client

        r1 = client.load_model("model_1", str(ckpt_path))
        r2 = client.load_model("model_2", str(ckpt_path))
        r3 = client.load_model("model_3", str(ckpt_path))

        assert r1["success"] is True
        assert r2["success"] is True
        assert r3["success"] is False
        assert "Max models" in r3["error"]

    def test_available_models_after_load(self, grpc_server_and_client):
        _, client, ckpt_path = grpc_server_and_client

        client.load_model("scene_a", str(ckpt_path))
        client.load_model("scene_b", str(ckpt_path))

        models = client.get_available_models()
        ids = {m["model_id"] for m in models}
        assert ids == {"scene_a", "scene_b"}
