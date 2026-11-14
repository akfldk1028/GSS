"""API request / response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ScanRequest(BaseModel):
    """Optional metadata sent alongside the file upload."""

    pipeline_config: str | None = Field(
        None, description="Pipeline config name (e.g. 'pipeline_import.yaml')"
    )


class ScanResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    message: str = "Job created"


class JobOutput(BaseModel):
    ifc: str | None = None
    glb: str | None = None
    usdc: str | None = None
    usdz: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    outputs: JobOutput | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu_available: bool = False
    gpu_name: str | None = None
    queue_size: int = 0
    active_jobs: int = 0
