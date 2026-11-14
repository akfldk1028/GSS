"""WebSocket endpoint — real-time job progress streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..events import EventBus
from ..jobs import JobManager, JobStatus

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    """Stream real-time events for a specific job.

    Client connects → receives progress/step_complete/completed/error events
    until the job finishes or the client disconnects.
    """
    manager: JobManager = websocket.app.state.job_manager
    event_bus: EventBus = websocket.app.state.event_bus

    job = manager.get_job(job_id)
    if job is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    logger.info("WebSocket connected for job %s", job_id)

    # If already finished, send final status and close
    if job.status in (JobStatus.completed, JobStatus.failed):
        await _send_final(websocket, job)
        await websocket.close()
        return

    # Subscribe to events
    queue = event_bus.subscribe(job_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(json.dumps(event))

                # Close after terminal events
                if event.get("type") in ("completed", "error"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_text(json.dumps({"type": "heartbeat"}))

                # Check if job finished while we were waiting
                job = manager.get_job(job_id)
                if job and job.status in (JobStatus.completed, JobStatus.failed):
                    await _send_final(websocket, job)
                    break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    finally:
        event_bus.unsubscribe(job_id, queue)


async def _send_final(ws: WebSocket, job) -> None:
    """Send the terminal event for a finished job."""
    if job.status == JobStatus.completed:
        outputs = {}
        if job.outputs:
            outputs = job.outputs.model_dump(exclude_none=True)
        await ws.send_text(json.dumps({
            "type": "completed",
            "outputs": outputs,
        }))
    elif job.status == JobStatus.failed:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": job.error or "Unknown error",
        }))
