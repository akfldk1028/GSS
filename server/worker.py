"""GPU worker — pulls jobs from the queue and runs the GSS pipeline."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import settings
from .events import EventBus
from .jobs import Job, JobManager, JobStatus
from .models import JobOutput

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=settings.worker_count)


# ---------------------------------------------------------------------------
# Synchronous pipeline execution (runs in thread pool)
# ---------------------------------------------------------------------------

def _run_pipeline_sync(job: Job, event_bus: EventBus) -> JobOutput:
    """Execute the GSS pipeline for a single job.

    This runs in a worker thread so it doesn't block the event loop.
    Publishes progress events via EventBus for WebSocket clients.
    """
    from gss.core.pipeline_runner import (
        import_step_class,
        load_pipeline_config,
        load_step_config,
    )

    job_dir = Path(job.job_dir)

    # 1. Prepare per-job directory layout matching data_root convention
    raw_dir = job_dir / "raw"
    interim_dir = job_dir / "interim"
    processed_dir = job_dir / "processed"
    for d in (raw_dir, interim_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Copy uploaded file into raw/
    upload = Path(job.upload_path)
    dest = raw_dir / upload.name
    shutil.copy2(upload, dest)

    # 3. Create per-job pipeline config with data_root override
    src_config = Path(job.pipeline_config)
    with open(src_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data_root"] = str(job_dir.resolve())
    job_config_path = job_dir / "pipeline.yaml"
    with open(job_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 4. Run pipeline with per-step event publishing
    pipeline_cfg = load_pipeline_config(job_config_path)
    data_root = pipeline_cfg.data_root
    results = {}

    enabled_steps = [s for s in pipeline_cfg.steps if s.enabled]
    total_steps = len(enabled_steps)

    for i, entry in enumerate(enabled_steps):
        step_name = entry.name
        progress = i / total_steps

        # Publish step start
        event_bus.publish_sync(job.job_id, {
            "type": "progress",
            "step": step_name,
            "step_index": i,
            "total_steps": total_steps,
            "status": "running",
            "progress": progress,
        })

        t0 = time.time()
        step_cls = import_step_class(entry.module)
        step_config = load_step_config(Path(entry.config_file), step_cls.config_type)
        step_instance = step_cls(config=step_config, data_root=data_root)

        # Build input from previous step outputs
        input_data = {}
        if entry.depends_on:
            for dep in entry.depends_on:
                if dep in results:
                    input_data.update(results[dep].model_dump())

        step_input = step_cls.input_type(**input_data) if input_data else step_cls.input_type()
        output = step_instance.execute(step_input)
        results[entry.name] = output
        elapsed = time.time() - t0

        # Publish step complete
        event_bus.publish_sync(job.job_id, {
            "type": "step_complete",
            "step": step_name,
            "step_index": i,
            "total_steps": total_steps,
            "elapsed": round(elapsed, 2),
            "progress": (i + 1) / total_steps,
        })

    # 5. Collect outputs
    outputs = JobOutput()
    for p in processed_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(job_dir).as_posix()
        match p.suffix.lower():
            case ".ifc":
                outputs.ifc = rel
            case ".glb":
                outputs.glb = rel
            case ".usdc":
                outputs.usdc = rel
            case ".usdz":
                outputs.usdz = rel

    return outputs


# ---------------------------------------------------------------------------
# Async worker loop
# ---------------------------------------------------------------------------

async def worker_loop(manager: JobManager, event_bus: EventBus) -> None:
    """Continuously pull jobs from the queue and execute them."""
    loop = asyncio.get_event_loop()
    logger.info("Worker started — waiting for jobs")

    while True:
        job_id = await manager.queue.get()
        job = manager.get_job(job_id)
        if job is None:
            logger.warning("Job %s not found, skipping", job_id)
            continue

        job.status = JobStatus.running
        job.started_at = datetime.now(timezone.utc)
        logger.info("Job %s started", job_id)

        try:
            outputs = await loop.run_in_executor(
                _executor, _run_pipeline_sync, job, event_bus
            )
            job.outputs = outputs
            job.status = JobStatus.completed
            logger.info("Job %s completed", job_id)

            # Publish completion event
            await event_bus.publish(job_id, {
                "type": "completed",
                "outputs": outputs.model_dump(exclude_none=True),
            })
        except Exception:
            job.status = JobStatus.failed
            job.error = traceback.format_exc()
            logger.error("Job %s failed:\n%s", job_id, job.error)

            # Publish error event
            await event_bus.publish(job_id, {
                "type": "error",
                "message": job.error,
            })
        finally:
            job.finished_at = datetime.now(timezone.utc)
