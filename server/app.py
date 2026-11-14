"""FastAPI application assembly + lifespan."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import settings
from .events import EventBus
from .jobs import JobManager
from .mqtt import TwinState
from .nucleus import NucleusManager
from .routes import health, nucleus_sync, render, scan, sensors, stream, ws
from .worker import worker_loop

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure jobs directory exists at import time (needed for StaticFiles mount)
settings.jobs_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start GPU worker(s) on startup, cancel on shutdown."""
    # Core services
    manager = JobManager()
    event_bus = EventBus()
    app.state.job_manager = manager
    app.state.event_bus = event_bus

    # Phase 3: Nucleus manager (lazy — connects on demand via API)
    app.state.nucleus_manager = NucleusManager()

    # Phase 4: Digital twin state (sensors)
    app.state.twin_state = TwinState()

    # gRPC render server (3DGRUT real-time rendering)
    grpc_server = None
    app.state.render_client = None
    if settings.grpc_enabled:
        try:
            from .grpc.render_server import create_server
            from .grpc.render_client import RenderClient

            # Parse checkpoint whitelist
            allowed_dirs = [
                d.strip() for d in settings.grpc_allowed_checkpoint_dirs.split(",")
                if d.strip()
            ]

            # TLS config
            tls_cert = settings.grpc_tls_cert if settings.grpc_tls_enabled else None
            tls_key = settings.grpc_tls_key if settings.grpc_tls_enabled else None
            tls_ca = settings.grpc_tls_ca if settings.grpc_tls_enabled and settings.grpc_tls_ca else None

            grpc_server, _servicer = create_server(
                port=settings.grpc_port,
                max_models=settings.grpc_max_models,
                tls_cert=tls_cert,
                tls_key=tls_key,
                tls_ca=tls_ca,
                allowed_checkpoint_dirs=allowed_dirs,
            )
            grpc_server.start()
            app.state.render_client = RenderClient(
                f"localhost:{settings.grpc_port}",
                tls_ca=tls_cert if settings.grpc_tls_enabled else None,
            )
            logger.info("gRPC render server started on port %d", settings.grpc_port)
        except Exception as e:
            logger.warning("gRPC render server failed to start: %s", e)

    # Start GPU worker tasks
    tasks = [
        asyncio.create_task(worker_loop(manager, event_bus))
        for _ in range(settings.worker_count)
    ]
    logger.info("GSS API server started — %d worker(s)", settings.worker_count)

    yield

    # Shutdown
    mqtt_bridge = getattr(app.state, "mqtt_bridge", None)
    if mqtt_bridge:
        mqtt_bridge.stop()
    if grpc_server is not None:
        grpc_server.stop(grace=5)
        logger.info("gRPC render server stopped")
    render_client = getattr(app.state, "render_client", None)
    if render_client is not None:
        render_client.close()
    for t in tasks:
        t.cancel()
    logger.info("GSS API server shutting down")


app = FastAPI(
    title="GSS API",
    description="3D Gaussian Splatting → BIM pipeline API — real-time digital twin",
    version="0.2.0",
    lifespan=lifespan,
)

# REST + WebSocket routes
app.include_router(health.router)
app.include_router(scan.router)
app.include_router(ws.router)
app.include_router(stream.router)
app.include_router(nucleus_sync.router)
app.include_router(sensors.router)
app.include_router(render.router)

# Static file serving for job outputs
app.mount(
    "/downloads",
    StaticFiles(directory=str(settings.jobs_dir)),
    name="downloads",
)
