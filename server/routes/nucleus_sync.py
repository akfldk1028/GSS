"""Nucleus sync configuration endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jobs import JobManager, JobStatus
from ..nucleus import NucleusManager

router = APIRouter(prefix="/nucleus", tags=["nucleus"])


class NucleusConfigRequest(BaseModel):
    server_url: str  # e.g. "omniverse://192.168.1.100"


class NucleusConfigResponse(BaseModel):
    connected: bool
    server_url: str | None = None
    message: str = ""


class NucleusPushResponse(BaseModel):
    job_id: str
    pushed: dict[str, bool]


class NucleusStatusResponse(BaseModel):
    available: bool
    server_url: str | None = None


@router.get("/status", response_model=NucleusStatusResponse)
async def nucleus_status(request: Request) -> NucleusStatusResponse:
    """Check Nucleus connection status."""
    nm: NucleusManager = request.app.state.nucleus_manager
    return NucleusStatusResponse(
        available=nm.is_available,
        server_url=nm.connection.server_url if nm.connection else None,
    )


@router.post("/connect", response_model=NucleusConfigResponse)
async def nucleus_connect(request: Request, body: NucleusConfigRequest) -> NucleusConfigResponse:
    """Connect to an Omniverse Nucleus server."""
    nm: NucleusManager = request.app.state.nucleus_manager
    success = nm.configure(body.server_url)
    return NucleusConfigResponse(
        connected=success,
        server_url=body.server_url,
        message="Connected" if success else "Connection failed (omni.client may not be installed)",
    )


@router.post("/push/{job_id}", response_model=NucleusPushResponse)
async def nucleus_push(request: Request, job_id: str) -> NucleusPushResponse:
    """Push a completed job's USD outputs to Nucleus."""
    nm: NucleusManager = request.app.state.nucleus_manager
    manager: JobManager = request.app.state.job_manager

    if not nm.is_available:
        raise HTTPException(503, "Nucleus not connected. POST /nucleus/connect first.")

    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, "Job not yet completed")

    results = nm.push_job_outputs(job_id, Path(job.job_dir))
    return NucleusPushResponse(job_id=job_id, pushed=results)
