"""Integration test: WebSocket render streaming (FastAPI ↔ gRPC ↔ frame).

Tests the full path:
  Browser (WebSocket) → FastAPI route → gRPC client → gRPC server → render → JPEG → WebSocket
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
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
def render_app(fake_checkpoint, monkeypatch):
    """FastAPI app with a real gRPC render server behind it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.grpc.render_server import create_server
    from server.grpc.render_client import RenderClient
    from server.routes.render import router

    port = _find_free_port()
    allowed_dirs = [str(fake_checkpoint.parent)]

    # Override server settings so REST checkpoint validation uses test dirs
    monkeypatch.setattr(
        "server.config.settings.grpc_allowed_checkpoint_dirs",
        str(fake_checkpoint.parent),
    )

    server, _ = create_server(
        port=port,
        max_models=2,
        allowed_checkpoint_dirs=allowed_dirs,
    )
    server.start()
    time.sleep(0.1)

    client = RenderClient(f"localhost:{port}")

    # Pre-load the test model
    result = client.load_model("test_scene", str(fake_checkpoint))
    assert result["success"], f"Failed to load model: {result['error']}"

    app = FastAPI()
    app.include_router(router)
    app.state.render_client = client

    test_client = TestClient(app)

    yield test_client, client

    client.close()
    server.stop(grace=1)


class TestWebSocketRenderStream:
    def test_connect_sends_model_info(self, render_app):
        """WebSocket connect → receive 'connected' message with model info."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "connected"
            assert msg["model_id"] == "test_scene"
            assert msg["num_gaussians"] == 100

    def test_camera_pose_returns_jpeg(self, render_app, identity_c2w):
        """Send camera pose → receive JPEG binary frame."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            ws.receive_text()  # connected

            ws.send_text(json.dumps({
                "type": "camera",
                "camera_to_world": identity_c2w,
                "fx": 200.0,
                "fy": 200.0,
                "cx": 160.0,
                "cy": 120.0,
            }))

            frame = ws.receive_bytes()
            assert len(frame) > 0
            # JPEG magic bytes
            assert frame[:2] == b"\xff\xd8"

    def test_multiple_frames(self, render_app, orbit_cameras):
        """Send multiple camera poses → receive multiple JPEG frames."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            ws.receive_text()  # connected

            frames_received = 0
            for c2w in orbit_cameras[:4]:  # Send 4 of 8 orbit poses
                ws.send_text(json.dumps({
                    "type": "camera",
                    "camera_to_world": c2w,
                    "fx": 200.0, "fy": 200.0,
                    "cx": 160.0, "cy": 120.0,
                }))

                frame = ws.receive_bytes()
                assert len(frame) > 0
                frames_received += 1

            assert frames_received == 4

    def test_resize_then_render(self, render_app, identity_c2w):
        """Resize viewport → render → frame at new resolution."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            ws.receive_text()  # connected

            # Resize
            ws.send_text(json.dumps({
                "type": "resize",
                "width": 128,
                "height": 96,
            }))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "resized"
            assert msg["width"] == 128

            # Render at new size
            ws.send_text(json.dumps({
                "type": "camera",
                "camera_to_world": identity_c2w,
                "fx": 100.0, "fy": 100.0,
                "cx": 64.0, "cy": 48.0,
            }))

            frame = ws.receive_bytes()
            assert len(frame) > 0

    def test_invalid_camera_returns_error(self, render_app):
        """Bad camera_to_world → error message, not crash."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            ws.receive_text()  # connected

            # Too few values
            ws.send_text(json.dumps({
                "type": "camera",
                "camera_to_world": [1, 0, 0],
            }))

            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert "16 floats" in msg["error"]

    def test_missing_camera_key_returns_error(self, render_app):
        """Missing camera_to_world key → error message."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            ws.receive_text()  # connected

            ws.send_text(json.dumps({
                "type": "camera",
                # no camera_to_world
            }))

            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"

    def test_ping_pong(self, render_app):
        """Keepalive works."""
        test_client, _ = render_app

        with test_client.websocket_connect("/ws/render/test_scene") as ws:
            ws.receive_text()  # connected

            ws.send_text(json.dumps({"type": "ping"}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "pong"

    def test_connect_unloaded_model_rejected(self, render_app):
        """Connecting to a model that isn't loaded fails gracefully."""
        test_client, _ = render_app

        with pytest.raises(Exception):
            with test_client.websocket_connect("/ws/render/nonexistent"):
                pass


class TestRESTRenderEndpoints:
    def test_list_models(self, render_app):
        test_client, _ = render_app

        response = test_client.get("/render/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) >= 1
        ids = {m["model_id"] for m in data["models"]}
        assert "test_scene" in ids

    def test_load_model_via_rest(self, render_app, fake_checkpoint):
        test_client, _ = render_app

        response = test_client.post("/render/load", json={
            "model_id": "rest_loaded",
            "checkpoint_path": str(fake_checkpoint),
        })
        data = response.json()
        assert data["success"] is True

    def test_load_invalid_path_rejected(self, render_app):
        test_client, _ = render_app

        response = test_client.post("/render/load", json={
            "model_id": "bad",
            "checkpoint_path": "/etc/passwd",
        })
        data = response.json()
        assert data["success"] is False
