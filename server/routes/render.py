"""Render routes — WebSocket bridge to gRPC render server.

Clients send camera poses via WebSocket, receive rendered JPEG frames back.
The FastAPI server acts as a bridge to the gRPC render server.

Protocol:
  Client → Server (JSON text):
    {"type": "camera", "camera_to_world": [16 floats], "fx": ..., "fy": ..., "cx": ..., "cy": ...}
    {"type": "resize", "width": 1920, "height": 1080}
    {"type": "load", "model_id": "...", "checkpoint_path": "..."}

  Server → Client:
    Binary: JPEG frame data
    JSON text: {"type": "loaded", ...} / {"type": "error", ...} / {"type": "models", ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_render_client(websocket: WebSocket):
    """Get or create the shared render client from app state."""
    return getattr(websocket.app.state, "render_client", None)


@router.get("/render/models")
async def list_render_models(request: Request):
    """List all models loaded on the gRPC render server."""
    from ..grpc.render_client import RenderClient

    client: RenderClient | None = getattr(request.app.state, "render_client", None)
    if client is None:
        return {"models": [], "error": "gRPC render server not configured"}

    try:
        models = await asyncio.to_thread(client.get_available_models)
        return {"models": models}
    except Exception as e:
        logger.error("Failed to list render models: %s", e)
        return {"models": [], "error": str(e)}


@router.post("/render/load")
async def load_render_model(request: Request):
    """Load a model on the gRPC render server."""
    from ..grpc.render_client import RenderClient

    client: RenderClient | None = getattr(request.app.state, "render_client", None)
    if client is None:
        return {"success": False, "error": "gRPC render server not configured"}

    body = await request.json()
    model_id = body.get("model_id")
    checkpoint_path = body.get("checkpoint_path")

    if not model_id or not checkpoint_path:
        return {"success": False, "error": "model_id and checkpoint_path required"}

    # Validate checkpoint path before sending to gRPC server
    try:
        from ..config import settings
        from ..grpc.security import validate_checkpoint_path

        allowed_dirs = [
            d.strip() for d in settings.grpc_allowed_checkpoint_dirs.split(",")
            if d.strip()
        ]
        validate_checkpoint_path(checkpoint_path, allowed_dirs=allowed_dirs)
    except (FileNotFoundError, PermissionError) as e:
        return {"success": False, "error": str(e)}

    try:
        result = await asyncio.to_thread(client.load_model, model_id, checkpoint_path)
        return result
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return {"success": False, "error": str(e)}


@router.websocket("/ws/render/{model_id}")
async def render_stream(websocket: WebSocket, model_id: str) -> None:
    """Real-time rendering via WebSocket.

    Client sends camera poses, receives rendered JPEG frames.
    """
    from ..grpc.render_client import RenderClient

    client: RenderClient | None = _get_render_client(websocket)
    if client is None:
        await websocket.close(code=4011, reason="gRPC render server not configured")
        return

    # Check model is loaded
    try:
        info = client.get_model_info(model_id)
    except Exception as e:
        await websocket.close(code=4012, reason=f"gRPC server unreachable: {e}")
        return

    if info is None:
        await websocket.close(
            code=4013,
            reason=f"Model '{model_id}' not loaded on render server",
        )
        return

    await websocket.accept()
    logger.info("Render stream connected for model '%s'", model_id)

    # Default resolution
    width = 1920
    height = 1080

    try:
        # Send model info
        await websocket.send_text(json.dumps({
            "type": "connected",
            "model_id": model_id,
            "num_gaussians": info["num_gaussians"],
            "method": info["method"],
        }))

        _pending_camera = None  # latest camera msg (frame dropping)

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "camera":
                # Render frame from camera pose
                c2w_raw = msg.get("camera_to_world")
                if c2w_raw is None or len(c2w_raw) != 16:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "camera_to_world must be 16 floats (4x4 row-major)",
                    }))
                    continue

                # Frame dropping: store latest camera and drain any queued msgs
                _pending_camera = msg
                try:
                    while True:
                        # Non-blocking check for more queued messages
                        raw2 = await asyncio.wait_for(
                            websocket.receive_text(), timeout=0.001,
                        )
                        msg2 = json.loads(raw2)
                        if msg2.get("type") == "camera":
                            c2w2 = msg2.get("camera_to_world")
                            if c2w2 is not None and len(c2w2) == 16:
                                _pending_camera = msg2  # replace with newer
                except asyncio.TimeoutError:
                    pass  # no more queued messages

                cam = _pending_camera
                c2w = np.array(cam["camera_to_world"], dtype=np.float32).reshape(4, 4)
                fx = cam.get("fx", width * 0.8)
                fy = cam.get("fy", fx)
                cx = cam.get("cx", width / 2.0)
                cy = cam.get("cy", height / 2.0)
                fmt = cam.get("format", "jpeg")
                quality = cam.get("quality", 75)

                # Run sync gRPC call in thread to avoid blocking event loop
                result = await asyncio.to_thread(
                    client.render,
                    model_id=model_id,
                    width=width,
                    height=height,
                    camera_to_world=c2w,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    fmt=fmt,
                    jpeg_quality=quality,
                )

                if result["success"]:
                    await websocket.send_bytes(result["image_data"])
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": result["error"],
                    }))

            elif msg_type == "resize":
                new_w = msg.get("width", width)
                new_h = msg.get("height", height)
                new_w = max(1, min(int(new_w), 3840))
                new_h = max(1, min(int(new_h), 2160))
                width = new_w
                height = new_h
                await websocket.send_text(json.dumps({
                    "type": "resized",
                    "width": width,
                    "height": height,
                }))

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info("Render stream disconnected for model '%s'", model_id)
    except Exception as e:
        logger.error("Render stream error for model '%s': %s", model_id, e)
        await websocket.close(code=1011, reason=str(e)[:120])
