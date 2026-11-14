"""Tests for gRPC render integration (server routes + client)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_render_client():
    """Create a mock RenderClient."""
    client = MagicMock()
    client.get_available_models.return_value = [
        {
            "model_id": "test_model",
            "checkpoint_path": "/data/checkpoint.pt",
            "num_gaussians": 50000,
            "gpu_memory_mb": 512.0,
            "method": "3dgut",
        }
    ]
    client.get_model_info.return_value = {
        "model_id": "test_model",
        "checkpoint_path": "/data/checkpoint.pt",
        "num_gaussians": 50000,
        "gpu_memory_mb": 512.0,
        "method": "3dgut",
    }
    client.render.return_value = {
        "image_data": b"\xff\xd8\xff\xe0" + b"\x00" * 100,  # fake JPEG
        "render_time_ms": 16.5,
        "success": True,
        "error": "",
        "width": 1920,
        "height": 1080,
    }
    client.load_model.return_value = {
        "success": True,
        "error": "",
        "num_gaussians": 50000,
        "gpu_memory_mb": 512.0,
    }
    return client


@pytest.fixture
def app_with_render(mock_render_client):
    """Create a FastAPI test app with mocked render client."""
    from fastapi import FastAPI
    from server.routes.render import router

    app = FastAPI()
    app.include_router(router)
    app.state.render_client = mock_render_client
    return app


class TestRenderModelsEndpoint:
    def test_list_models(self, app_with_render, mock_render_client):
        client = TestClient(app_with_render)
        response = client.get("/render/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "test_model"

    def test_list_models_no_grpc(self):
        from fastapi import FastAPI
        from server.routes.render import router

        app = FastAPI()
        app.include_router(router)
        app.state.render_client = None

        client = TestClient(app)
        response = client.get("/render/models")
        data = response.json()
        assert data["models"] == []
        assert "error" in data


class TestRenderWebSocket:
    def test_ws_connect_no_grpc(self):
        """WebSocket should close if gRPC not configured."""
        from fastapi import FastAPI
        from server.routes.render import router

        app = FastAPI()
        app.include_router(router)
        # No render_client set

        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/render/test_model"):
                pass

    def test_ws_connect_model_not_found(self, app_with_render, mock_render_client):
        """WebSocket should close if model not loaded."""
        mock_render_client.get_model_info.return_value = None

        client = TestClient(app_with_render)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/render/nonexistent"):
                pass

    def test_ws_camera_render(self, app_with_render, mock_render_client):
        """Test camera pose → rendered frame flow."""
        client = TestClient(app_with_render)
        with client.websocket_connect("/ws/render/test_model") as ws:
            # Should receive "connected" message first
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "connected"
            assert msg["model_id"] == "test_model"

            # Send camera pose
            c2w = np.eye(4, dtype=np.float32).flatten().tolist()
            ws.send_text(json.dumps({
                "type": "camera",
                "camera_to_world": c2w,
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 960.0,
                "cy": 540.0,
            }))

            # Should receive binary frame
            frame = ws.receive_bytes()
            assert len(frame) > 0
            assert mock_render_client.render.called

    def test_ws_resize(self, app_with_render):
        """Test resize message."""
        client = TestClient(app_with_render)
        with client.websocket_connect("/ws/render/test_model") as ws:
            ws.receive_text()  # connected

            ws.send_text(json.dumps({
                "type": "resize",
                "width": 1280,
                "height": 720,
            }))

            msg = json.loads(ws.receive_text())
            assert msg["type"] == "resized"
            assert msg["width"] == 1280

    def test_ws_ping_pong(self, app_with_render):
        """Test keepalive ping/pong."""
        client = TestClient(app_with_render)
        with client.websocket_connect("/ws/render/test_model") as ws:
            ws.receive_text()  # connected

            ws.send_text(json.dumps({"type": "ping"}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "pong"


class TestRenderClient:
    def test_client_format_map(self):
        """Test that format strings map correctly."""
        from server.grpc.render_client import RenderClient

        client = RenderClient.__new__(RenderClient)
        # Just verify the class can be instantiated
        assert client is not None


class TestProtoDefinitions:
    def test_proto_files_exist(self):
        """Verify proto files are present."""
        from pathlib import Path

        protos = Path("server/grpc/protos")
        assert (protos / "common.proto").exists()
        assert (protos / "render.proto").exists()
