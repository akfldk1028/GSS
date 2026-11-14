"""gRPC client — used by FastAPI routes to communicate with the render server.

Usage:
    client = RenderClient("localhost:46435")
    response = client.render(model_id, width, height, camera_matrix, intrinsics)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ADDRESS = "localhost:46435"


class RenderClient:
    """gRPC client for the GSS render server."""

    # Default timeouts (seconds)
    RENDER_TIMEOUT = 30.0
    LOAD_TIMEOUT = 120.0
    DEFAULT_TIMEOUT = 10.0

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        tls_ca: str | None = None,
        tls_client_cert: str | None = None,
        tls_client_key: str | None = None,
    ):
        self.address = address
        self._tls_ca = tls_ca
        self._tls_client_cert = tls_client_cert
        self._tls_client_key = tls_client_key
        self._channel = None
        self._stub = None

    def _ensure_connected(self):
        """Lazy connection to gRPC server."""
        if self._stub is not None:
            return

        import grpc
        from .generated import render_pb2_grpc

        options = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]

        if self._tls_ca:
            from .security import load_tls_channel_credentials

            credentials = load_tls_channel_credentials(
                self._tls_ca, self._tls_client_cert, self._tls_client_key,
            )
            self._channel = grpc.secure_channel(self.address, credentials, options=options)
            logger.info("Connected to gRPC render server at %s (TLS)", self.address)
        else:
            self._channel = grpc.insecure_channel(self.address, options=options)
            logger.info("Connected to gRPC render server at %s", self.address)

        self._stub = render_pb2_grpc.GSSRenderServiceStub(self._channel)

    def load_model(
        self, model_id: str, checkpoint_path: str
    ) -> dict:
        """Load a model on the render server.

        Returns: {"success": bool, "error": str, "num_gaussians": int, "gpu_memory_mb": float}
        """
        self._ensure_connected()
        from .generated import render_pb2

        response = self._stub.LoadModel(
            render_pb2.LoadModelRequest(
                model_id=model_id,
                checkpoint_path=checkpoint_path,
            ),
            timeout=self.LOAD_TIMEOUT,
        )
        return {
            "success": response.success,
            "error": response.error,
            "num_gaussians": response.num_gaussians,
            "gpu_memory_mb": response.gpu_memory_mb,
        }

    def unload_model(self, model_id: str) -> dict:
        """Unload a model from GPU memory."""
        self._ensure_connected()
        from .generated import render_pb2

        response = self._stub.UnloadModel(
            render_pb2.UnloadModelRequest(model_id=model_id),
            timeout=self.DEFAULT_TIMEOUT,
        )
        return {"success": response.success, "error": response.error}

    def render(
        self,
        model_id: str,
        width: int,
        height: int,
        camera_to_world: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        fmt: str = "jpeg",
        jpeg_quality: int = 85,
    ) -> dict:
        """Render a single frame.

        Args:
            model_id: ID of loaded model.
            width, height: Output resolution.
            camera_to_world: 4x4 numpy array (row-major).
            fx, fy, cx, cy: Camera intrinsics.
            fmt: "jpeg", "png", or "raw".
            jpeg_quality: JPEG quality (1-100).

        Returns: {"image_data": bytes, "render_time_ms": float, "success": bool, "error": str}
        """
        self._ensure_connected()
        from .generated import render_pb2

        format_map = {"jpeg": 0, "png": 1, "raw": 2}
        image_format = format_map.get(fmt.lower(), 0)

        c2w_flat = camera_to_world.astype(np.float32).flatten().tolist()

        response = self._stub.RenderRGB(
            render_pb2.RenderRequest(
                model_id=model_id,
                width=width,
                height=height,
                camera=render_pb2.CameraParams(
                    camera_to_world=c2w_flat,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                ),
                format=image_format,
                jpeg_quality=jpeg_quality,
            ),
            timeout=self.RENDER_TIMEOUT,
        )
        return {
            "image_data": response.image_data,
            "render_time_ms": response.render_time_ms,
            "success": response.success,
            "error": response.error,
            "width": response.width,
            "height": response.height,
        }

    def get_available_models(self) -> list[dict]:
        """List all loaded models."""
        self._ensure_connected()
        from .generated import common_pb2

        response = self._stub.GetAvailableModels(
            common_pb2.Empty(), timeout=self.DEFAULT_TIMEOUT,
        )
        return [
            {
                "model_id": m.model_id,
                "checkpoint_path": m.checkpoint_path,
                "num_gaussians": m.num_gaussians,
                "gpu_memory_mb": m.gpu_memory_mb,
                "method": m.method,
            }
            for m in response.models
        ]

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get info about a specific model. Returns None if not found."""
        self._ensure_connected()
        from .generated import render_pb2

        response = self._stub.GetModelInfo(
            render_pb2.ModelInfoRequest(model_id=model_id),
            timeout=self.DEFAULT_TIMEOUT,
        )
        if not response.found:
            return None
        info = response.info
        return {
            "model_id": info.model_id,
            "checkpoint_path": info.checkpoint_path,
            "num_gaussians": info.num_gaussians,
            "gpu_memory_mb": info.gpu_memory_mb,
            "method": info.method,
        }

    def close(self):
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass
