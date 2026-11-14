"""Gaussian splat streaming endpoint — binary WebSocket.

Client connects with a job_id, receives the full Gaussian scene,
then sends camera pose updates to receive view-adaptive deltas.

Protocol:
  Client → Server (JSON text):
    {"type": "camera", "position": [x,y,z], "forward": [fx,fy,fz]}

  Server → Client (binary):
    Header (12B) + N * 48B gaussian data
    See splat_encoder.py for format details.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..jobs import JobManager, JobStatus
from ..splat_encoder import MsgType, encode_frame, load_ply_gaussians
from ..spatial_index import CuboidIndex

router = APIRouter()
logger = logging.getLogger(__name__)

# Maximum gaussians per frame to avoid overwhelming the client
MAX_GAUSSIANS_PER_FRAME = 100_000


@router.websocket("/ws/stream/{job_id}")
async def stream_gaussians(websocket: WebSocket, job_id: str) -> None:
    """Stream Gaussian splat data for a completed job.

    1. Client connects → server sends full scene (chunked)
    2. Client sends camera poses → server sends view-adaptive subsets
    """
    manager: JobManager = websocket.app.state.job_manager
    job = manager.get_job(job_id)

    if job is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    if job.status != JobStatus.completed:
        await websocket.close(code=4009, reason="Job not yet completed")
        return

    # Find the PLY file in job directory
    job_dir = Path(job.job_dir)
    ply_files = list(job_dir.rglob("*.ply"))
    if not ply_files:
        await websocket.close(code=4010, reason="No PLY file found")
        return

    await websocket.accept()
    logger.info("Stream connected for job %s", job_id)

    try:
        # Load gaussians
        gaussians = load_ply_gaussians(ply_files[0])
        n_total = gaussians["positions"].shape[0]
        logger.info("Loaded %d gaussians from %s", n_total, ply_files[0].name)

        # Build spatial index
        index = CuboidIndex(gaussians["positions"], cuboid_size=2.0)
        logger.info("Spatial index: %d cuboids", index.cuboid_count)

        # Send initial full scene (chunked)
        seq = 0
        for start in range(0, n_total, MAX_GAUSSIANS_PER_FRAME):
            end = min(start + MAX_GAUSSIANS_PER_FRAME, n_total)
            frame = encode_frame(
                MsgType.FULL,
                seq,
                gaussians["positions"][start:end],
                gaussians["rotations"][start:end],
                gaussians["scales"][start:end],
                gaussians["opacities"][start:end],
                gaussians["sh_dc"][start:end],
                gaussians["sh_rest"][start:end] if gaussians["sh_rest"] is not None else None,
            )
            await websocket.send_bytes(frame)
            seq += 1

        # Send metadata
        await websocket.send_text(json.dumps({
            "type": "scene_loaded",
            "total_gaussians": n_total,
            "cuboids": index.cuboid_count,
            "chunks_sent": seq,
        }))

        # Listen for camera pose updates
        prev_visible: set[int] = set(range(n_total))

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "camera":
                cam_pos = np.array(msg["position"], dtype=np.float32)
                cam_fwd = np.array(msg["forward"], dtype=np.float32)
                max_dist = msg.get("max_distance", 20.0)

                # Query visible gaussians
                visible_idx = index.query_frustum(cam_pos, cam_fwd, max_distance=max_dist)
                new_visible = set(visible_idx.tolist())

                # Delta: newly visible gaussians
                to_add = new_visible - prev_visible
                if to_add:
                    add_idx = np.array(sorted(to_add), dtype=np.int32)
                    # Limit per frame
                    if len(add_idx) > MAX_GAUSSIANS_PER_FRAME:
                        add_idx = add_idx[:MAX_GAUSSIANS_PER_FRAME]

                    frame = encode_frame(
                        MsgType.DELTA_ADD,
                        seq,
                        gaussians["positions"][add_idx],
                        gaussians["rotations"][add_idx],
                        gaussians["scales"][add_idx],
                        gaussians["opacities"][add_idx],
                        gaussians["sh_dc"][add_idx],
                        gaussians["sh_rest"][add_idx] if gaussians["sh_rest"] is not None else None,
                    )
                    await websocket.send_bytes(frame)
                    seq += 1

                prev_visible = new_visible

    except WebSocketDisconnect:
        logger.info("Stream disconnected for job %s", job_id)
    except Exception as e:
        logger.error("Stream error for job %s: %s", job_id, e)
        await websocket.close(code=1011, reason=str(e)[:120])
