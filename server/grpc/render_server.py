"""gRPC rendering server — loads 3DGRUT models and renders on demand.

Usage:
    python -m server.grpc.render_server [--port 46435]
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent import futures
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default gRPC port (NuRec convention: 46435)
DEFAULT_PORT = 46435
MAX_WORKERS = 4
MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64 MB


class GSSRenderServicer:
    """gRPC servicer that manages 3DGRUT models and renders frames."""

    def __init__(
        self,
        max_models: int = 4,
        allowed_checkpoint_dirs: list[str] | None = None,
    ):
        self.max_models = max_models
        self.models: dict[str, dict] = {}  # model_id → {trainer, info}
        self._lock = threading.Lock()  # Protects self.models across gRPC threads
        self._allowed_checkpoint_dirs = allowed_checkpoint_dirs
        logger.info("GSSRenderServicer initialized (max_models=%d)", max_models)

    def LoadModel(self, request, context):
        """Load a 3DGRUT checkpoint into GPU memory."""
        from .generated import render_pb2

        model_id = request.model_id
        checkpoint_path = request.checkpoint_path

        # Early check (under lock) for already-loaded or capacity
        with self._lock:
            if model_id in self.models:
                return render_pb2.LoadModelResponse(
                    success=True,
                    num_gaussians=self.models[model_id]["num_gaussians"],
                    gpu_memory_mb=self.models[model_id].get("gpu_memory_mb", 0),
                )

            if len(self.models) >= self.max_models:
                return render_pb2.LoadModelResponse(
                    success=False,
                    error=f"Max models ({self.max_models}) reached. Unload one first.",
                )
            # Reserve the slot to prevent TOCTOU race
            self.models[model_id] = None  # placeholder

        # Validate checkpoint path against whitelist
        try:
            from .security import validate_checkpoint_path

            validated_path = validate_checkpoint_path(
                checkpoint_path,
                allowed_dirs=self._allowed_checkpoint_dirs,
            )
            checkpoint_path = str(validated_path)
        except (FileNotFoundError, PermissionError) as e:
            with self._lock:
                self.models.pop(model_id, None)  # release slot
            if isinstance(e, PermissionError):
                logger.warning("Checkpoint path rejected: %s", e)
            return render_pb2.LoadModelResponse(success=False, error=str(e))

        try:
            trainer, num_gaussians, gpu_mb = _load_3dgrut_model(checkpoint_path)
            with self._lock:
                self.models[model_id] = {
                    "trainer": trainer,
                    "checkpoint_path": checkpoint_path,
                    "num_gaussians": num_gaussians,
                    "gpu_memory_mb": gpu_mb,
                    "method": "3dgut",
                }
            logger.info(
                "Loaded model '%s': %d gaussians, %.1f MB GPU",
                model_id, num_gaussians, gpu_mb,
            )
            return render_pb2.LoadModelResponse(
                success=True,
                num_gaussians=num_gaussians,
                gpu_memory_mb=gpu_mb,
            )
        except Exception as e:
            with self._lock:
                self.models.pop(model_id, None)  # release slot on failure
            logger.error("Failed to load model '%s': %s", model_id, e)
            return render_pb2.LoadModelResponse(success=False, error=str(e))

    def UnloadModel(self, request, context):
        """Remove a model from GPU memory."""
        from .generated import render_pb2

        model_id = request.model_id
        with self._lock:
            if model_id not in self.models:
                return render_pb2.UnloadModelResponse(
                    success=False, error=f"Model '{model_id}' not loaded",
                )
            del self.models[model_id]

        # Trigger GPU garbage collection
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Unloaded model '%s'", model_id)
        return render_pb2.UnloadModelResponse(success=True)

    def RenderRGB(self, request, context):
        """Render a single frame from the given camera parameters."""
        from .generated import render_pb2

        model_id = request.model_id
        with self._lock:
            model_entry = self.models.get(model_id)
        if model_entry is None:
            return render_pb2.RenderResponse(
                success=False, error=f"Model '{model_id}' not loaded",
            )

        t0 = time.perf_counter()

        try:
            trainer = model_entry["trainer"]
            width = request.width or 1920
            height = request.height or 1080
            camera = request.camera

            # Build camera-to-world 4x4 from proto
            c2w = np.array(camera.camera_to_world, dtype=np.float32).reshape(4, 4)
            intrinsics = {
                "fx": camera.fx, "fy": camera.fy,
                "cx": camera.cx, "cy": camera.cy,
            }

            # Render
            rgb_array = _render_frame(trainer, c2w, intrinsics, width, height)

            # Encode
            fmt = request.format  # 0=JPEG, 1=PNG, 2=RAW
            quality = request.jpeg_quality or 85
            image_bytes = _encode_image(rgb_array, fmt, quality)

            render_ms = (time.perf_counter() - t0) * 1000
            return render_pb2.RenderResponse(
                image_data=image_bytes,
                render_time_ms=render_ms,
                success=True,
                width=width,
                height=height,
            )
        except Exception as e:
            logger.error("Render failed for '%s': %s", model_id, e)
            return render_pb2.RenderResponse(
                success=False,
                error=str(e),
                render_time_ms=(time.perf_counter() - t0) * 1000,
            )

    def GetAvailableModels(self, request, context):
        """Return list of all loaded models."""
        from .generated import render_pb2

        models = []
        with self._lock:
            snapshot = {
                mid: info for mid, info in self.models.items()
                if info is not None  # skip placeholders from in-progress loads
            }
        for mid, info in snapshot.items():
            models.append(render_pb2.ModelInfo(
                model_id=mid,
                checkpoint_path=info["checkpoint_path"],
                num_gaussians=info["num_gaussians"],
                gpu_memory_mb=info.get("gpu_memory_mb", 0),
                method=info.get("method", "3dgut"),
            ))
        return render_pb2.ModelsResponse(models=models)

    def GetModelInfo(self, request, context):
        """Return info about a specific model."""
        from .generated import render_pb2

        model_id = request.model_id
        with self._lock:
            info = self.models.get(model_id)
        if info is None:
            return render_pb2.ModelInfoResponse(found=False)

        return render_pb2.ModelInfoResponse(
            found=True,
            info=render_pb2.ModelInfo(
                model_id=model_id,
                checkpoint_path=info["checkpoint_path"],
                num_gaussians=info["num_gaussians"],
                gpu_memory_mb=info.get("gpu_memory_mb", 0),
                method=info.get("method", "3dgut"),
            ),
        )


# ── 3DGRUT integration helpers ─────────────────────────────────────


def _load_3dgrut_model(checkpoint_path: str) -> tuple:
    """Load a 3DGRUT checkpoint and return (trainer, num_gaussians, gpu_mb).

    Falls back to a torch checkpoint if 3DGRUT's Trainer is not available.
    """
    import torch

    try:
        # Try 3DGRUT's native loader
        from threedgrut.scripts.train import Trainer3DGRUT

        trainer = Trainer3DGRUT.create_from_ckpt(checkpoint_path)
        trainer.eval()
        num_gaussians = trainer.model.num_gaussians
    except ImportError:
        logger.warning(
            "threedgrut package not installed — loading as raw checkpoint. "
            "Rendering will use fallback rasterization."
        )
        # Fallback: load as raw torch checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        trainer = ckpt  # Store the dict; render will handle it
        num_gaussians = 0
        if isinstance(ckpt, dict):
            for key in ("means", "positions", "_xyz"):
                if key in ckpt:
                    num_gaussians = len(ckpt[key])
                    break

    # Estimate GPU memory
    gpu_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    return trainer, num_gaussians, gpu_mb


def _render_frame(
    trainer, c2w: np.ndarray, intrinsics: dict, width: int, height: int,
) -> np.ndarray:
    """Render an RGB frame using the loaded model.

    Returns: uint8 numpy array (H, W, 3).
    """
    import torch

    try:
        # 3DGRUT native rendering path
        from threedgrut.utils.camera import Camera

        camera = Camera(
            c2w=torch.tensor(c2w, device="cuda"),
            fx=intrinsics["fx"], fy=intrinsics["fy"],
            cx=intrinsics["cx"], cy=intrinsics["cy"],
            width=width, height=height,
        )
        with torch.no_grad():
            output = trainer.model(camera)
        rgb = output["rgb"].clamp(0, 1).cpu().numpy()
        return (rgb * 255).astype(np.uint8)
    except ImportError:
        pass

    # Fallback: if model is a raw checkpoint dict, render with gsplat
    if isinstance(trainer, dict):
        return _fallback_render(trainer, c2w, intrinsics, width, height)

    raise RuntimeError("Cannot render: no compatible renderer found")


def _fallback_render(
    ckpt: dict, c2w: np.ndarray, intrinsics: dict, width: int, height: int,
) -> np.ndarray:
    """Fallback rendering using gsplat for raw checkpoint dicts."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract Gaussian parameters
    means = ckpt.get("means", ckpt.get("positions", ckpt.get("_xyz")))
    if means is None:
        raise RuntimeError("No position data in checkpoint")

    if not isinstance(means, torch.Tensor):
        means = torch.tensor(means, device=device, dtype=torch.float32)
    else:
        means = means.to(device=device)

    # Build w2c from c2w (use torch.linalg.inv to avoid numpy LAPACK conflicts)
    # gsplat expects OpenCV convention (Y-down, Z-forward).
    # If c2w is OpenGL (Y-up, Z-backward), convert: flip Y and Z columns.
    c2w_t = torch.tensor(c2w, device=device, dtype=torch.float32)
    c2w_cv = c2w_t.clone()
    c2w_cv[:, 1] *= -1  # flip Y column (OpenGL Y-up → OpenCV Y-down)
    c2w_cv[:, 2] *= -1  # flip Z column (OpenGL -Z forward → OpenCV +Z forward)
    w2c_t = torch.linalg.inv(c2w_cv)
    viewmat = w2c_t.unsqueeze(0)

    K = torch.tensor([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], device=device, dtype=torch.float32).unsqueeze(0)

    # Get other params with defaults
    n = len(means)
    quats = ckpt.get("quats")
    if quats is not None:
        quats = torch.as_tensor(quats, device=device, dtype=torch.float32)
    else:
        quats = torch.zeros(n, 4, device=device, dtype=torch.float32)
        quats[:, 0] = 1.0

    scales = ckpt.get("log_scales")
    if scales is not None:
        scales = torch.exp(torch.as_tensor(scales, device=device, dtype=torch.float32))
    else:
        scales = torch.full((n, 3), 0.01, device=device, dtype=torch.float32)

    opacities = ckpt.get("raw_opacities")
    if opacities is not None:
        opacities = torch.sigmoid(torch.as_tensor(opacities, device=device, dtype=torch.float32))
    else:
        opacities = torch.ones(n, device=device, dtype=torch.float32)

    # Full SH coefficients (N, K, 3) or DC-only colors (N, 3)
    sh_coeffs = ckpt.get("sh_coeffs")
    sh_degree = ckpt.get("sh_degree", 0)
    if sh_coeffs is not None:
        colors = torch.as_tensor(sh_coeffs, device=device, dtype=torch.float32)
    else:
        colors = ckpt.get("colors")
        if colors is not None:
            colors = torch.as_tensor(colors, device=device, dtype=torch.float32)
            # SH DC coefficients → RGB: color = SH_C0 * sh_dc + 0.5
            SH_C0 = 0.28209479177387814
            colors = (SH_C0 * colors + 0.5).clamp(0, 1)
        else:
            colors = torch.ones(n, 3, device=device, dtype=torch.float32) * 0.5

    try:
        from gsplat import rasterization

        raster_kwargs = dict(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
        )
        # If colors are SH coefficients (N, K, 3), pass sh_degree
        if colors.dim() == 3:
            raster_kwargs["sh_degree"] = sh_degree

        with torch.no_grad():
            renders, _, _ = rasterization(**raster_kwargs)
        # gsplat returns (B, C, H, W); permute to (H, W, C) for cv2
        img = renders[0].clamp(0, 1)
        if img.shape[0] <= 4 and img.dim() == 3:
            img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        rgb = img.cpu().numpy()
        return (rgb * 255).astype(np.uint8)
    except ImportError:
        # No renderer available — return placeholder
        logger.warning("No renderer available (gsplat/3dgrut) — returning blank frame")
        return np.zeros((height, width, 3), dtype=np.uint8)
    except Exception as e:
        # gsplat JIT compile failure (e.g., no MSVC on Windows) — return placeholder
        logger.warning("gsplat rasterization failed (%s) — returning blank frame", e)
        return np.zeros((height, width, 3), dtype=np.uint8)


def _encode_image(rgb: np.ndarray, fmt: int, quality: int = 85) -> bytes:
    """Encode numpy RGB array to JPEG/PNG/RAW bytes."""
    if fmt == 2:  # RAW_RGB
        return rgb.tobytes()

    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == 1:  # PNG
        _, buf = cv2.imencode(".png", bgr)
    else:  # JPEG (default)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ── Server lifecycle ───────────────────────────────────────────────


def serve(
    port: int = DEFAULT_PORT,
    max_models: int = 4,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    allowed_checkpoint_dirs: list[str] | None = None,
) -> None:
    """Start the gRPC rendering server (blocking).

    Args:
        port: Port to listen on.
        max_models: Max concurrent models in GPU memory.
        tls_cert: Path to server certificate (PEM). None = insecure.
        tls_key: Path to server private key (PEM).
        tls_ca: Path to CA cert for mutual TLS (optional).
        allowed_checkpoint_dirs: Whitelist for checkpoint paths.
    """
    import grpc
    from .generated import render_pb2_grpc

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ],
    )
    servicer = GSSRenderServicer(
        max_models=max_models,
        allowed_checkpoint_dirs=allowed_checkpoint_dirs,
    )
    render_pb2_grpc.add_GSSRenderServiceServicer_to_server(servicer, server)

    if tls_cert and tls_key:
        from .security import load_tls_server_credentials

        credentials = load_tls_server_credentials(tls_cert, tls_key, tls_ca)
        server.add_secure_port(f"[::]:{port}", credentials)
        logger.info("gRPC render server listening on port %d (TLS)", port)
    else:
        server.add_insecure_port(f"[::]:{port}")
        logger.info("gRPC render server listening on port %d (insecure)", port)

    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(grace=5)


def create_server(
    port: int = DEFAULT_PORT,
    max_models: int = 4,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    allowed_checkpoint_dirs: list[str] | None = None,
):
    """Create (but don't start) a gRPC server. Returns (server, servicer)."""
    import grpc
    from .generated import render_pb2_grpc

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ],
    )
    servicer = GSSRenderServicer(
        max_models=max_models,
        allowed_checkpoint_dirs=allowed_checkpoint_dirs,
    )
    render_pb2_grpc.add_GSSRenderServiceServicer_to_server(servicer, server)

    if tls_cert and tls_key:
        from .security import load_tls_server_credentials

        credentials = load_tls_server_credentials(tls_cert, tls_key, tls_ca)
        server.add_secure_port(f"[::]:{port}", credentials)
        logger.info("gRPC server configured with TLS on port %d", port)
    else:
        server.add_insecure_port(f"[::]:{port}")

    return server, servicer


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GSS gRPC Render Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-models", type=int, default=4)
    parser.add_argument("--tls-cert", type=str, default=None, help="TLS cert path")
    parser.add_argument("--tls-key", type=str, default=None, help="TLS key path")
    parser.add_argument("--tls-ca", type=str, default=None, help="CA cert for mTLS")
    parser.add_argument(
        "--allowed-dirs", type=str, default=None,
        help="Comma-separated checkpoint whitelist dirs",
    )
    args = parser.parse_args()

    allowed = args.allowed_dirs.split(",") if args.allowed_dirs else None

    serve(
        port=args.port,
        max_models=args.max_models,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        tls_ca=args.tls_ca,
        allowed_checkpoint_dirs=allowed,
    )
