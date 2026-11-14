"""Scan endpoints — upload files and query job status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from ..config import settings
from ..jobs import JobManager
from ..models import JobOutput, JobResponse, JobStatus, ScanResponse

router = APIRouter()


def _resolve_pipeline_config(name: str | None) -> Path:
    """Return the absolute path to the requested (or default) pipeline config."""
    if name:
        candidate = Path("configs") / name
        if not candidate.exists():
            raise HTTPException(400, f"Unknown pipeline config: {name}")
        return candidate.resolve()
    return settings.default_pipeline_config.resolve()


@router.post("/scan", response_model=ScanResponse)
async def create_scan(
    request: Request,
    file: UploadFile,
    pipeline_config: str | None = None,
) -> ScanResponse:
    """Accept PLY / MP4 upload and enqueue a pipeline job."""
    manager: JobManager = request.app.state.job_manager

    # Validate file extension
    if file.filename is None:
        raise HTTPException(400, "Filename is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".ply", ".mp4", ".mov", ".avi"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Resolve config
    config_path = _resolve_pipeline_config(pipeline_config)

    # Create job directory
    # Temporary job_id for directory — will be replaced by Job's uuid
    import uuid

    tmp_id = uuid.uuid4().hex[:12]
    job_dir = settings.jobs_dir / tmp_id
    upload_dir = job_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Stream upload to disk
    dest = upload_dir / file.filename
    total = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):  # 8 MB chunks
            total += len(chunk)
            if total > settings.upload_max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File too large")
            f.write(chunk)

    # Enqueue
    job = await manager.create_job(
        upload_path=str(dest),
        pipeline_config=str(config_path),
        job_dir=str(job_dir),
    )

    return ScanResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(request: Request, job_id: str) -> JobResponse:
    """Query job status and results."""
    manager: JobManager = request.app.state.job_manager
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    outputs: JobOutput | None = None
    if job.status == JobStatus.completed and job.outputs:
        # Convert relative paths to download URLs
        base = f"/downloads/{Path(job.job_dir).name}"
        outputs = JobOutput(
            ifc=f"{base}/{job.outputs.ifc}" if job.outputs.ifc else None,
            glb=f"{base}/{job.outputs.glb}" if job.outputs.glb else None,
            usdc=f"{base}/{job.outputs.usdc}" if job.outputs.usdc else None,
            usdz=f"{base}/{job.outputs.usdz}" if job.outputs.usdz else None,
        )

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        outputs=outputs,
    )
