"""Health / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..models import HealthResponse

router = APIRouter()


def _gpu_info() -> tuple[bool, str | None]:
    """Best-effort GPU detection via torch."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, None


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    manager = request.app.state.job_manager
    gpu_ok, gpu_name = _gpu_info()
    return HealthResponse(
        status="ok",
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        queue_size=manager.queue_size,
        active_jobs=manager.active_count,
    )
