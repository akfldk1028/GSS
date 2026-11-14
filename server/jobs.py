"""Job lifecycle management — in-memory store + asyncio queue."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from .models import JobOutput, JobStatus


class Job:
    """Mutable job record stored in memory."""

    __slots__ = (
        "job_id", "status", "created_at", "started_at", "finished_at",
        "error", "outputs", "upload_path", "pipeline_config", "job_dir",
    )

    def __init__(self, *, upload_path: str, pipeline_config: str, job_dir: str):
        self.job_id: str = uuid.uuid4().hex[:12]
        self.status: JobStatus = JobStatus.queued
        self.created_at: datetime = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.outputs: JobOutput | None = None
        self.upload_path: str = upload_path
        self.pipeline_config: str = pipeline_config
        self.job_dir: str = job_dir


class JobManager:
    """Thread-safe (via GIL + asyncio) job bookkeeping."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def create_job(
        self, *, upload_path: str, pipeline_config: str, job_dir: str
    ) -> Job:
        job = Job(
            upload_path=upload_path,
            pipeline_config=pipeline_config,
            job_dir=job_dir,
        )
        self._jobs[job.job_id] = job
        await self.queue.put(job.job_id)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.running)
