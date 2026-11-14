"""Integration test: render performance & throughput measurement.

Measures actual render latency and frame throughput through the full
gRPC pipeline. Not a pass/fail test — logs metrics for comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

try:
    import grpc
    import torch

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="grpcio or torch not installed")


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def perf_server(fake_checkpoint):
    """gRPC server + client for performance testing."""
    from server.grpc.render_server import create_server
    from server.grpc.render_client import RenderClient

    port = _find_free_port()
    server, _ = create_server(
        port=port,
        max_models=2,
        allowed_checkpoint_dirs=[str(fake_checkpoint.parent)],
    )
    server.start()
    time.sleep(0.1)

    client = RenderClient(f"localhost:{port}")
    client.load_model("perf_model", str(fake_checkpoint))

    yield client

    client.close()
    server.stop(grace=1)


class TestRenderLatency:
    """Measure single-frame render latency at different resolutions."""

    @pytest.mark.parametrize("width,height", [
        (64, 48),
        (320, 240),
        (640, 480),
        (1280, 720),
    ])
    def test_single_frame_latency(self, perf_server, identity_c2w, width, height):
        c2w = np.array(identity_c2w, dtype=np.float32).reshape(4, 4)

        # Warm-up
        perf_server.render(
            model_id="perf_model",
            width=width, height=height,
            camera_to_world=c2w,
            fx=width * 0.8, fy=width * 0.8,
            cx=width / 2.0, cy=height / 2.0,
        )

        # Measure
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            result = perf_server.render(
                model_id="perf_model",
                width=width, height=height,
                camera_to_world=c2w,
                fx=width * 0.8, fy=width * 0.8,
                cx=width / 2.0, cy=height / 2.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            assert result["success"] is True

        avg_ms = sum(latencies) / len(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)

        print(f"\n  {width}x{height}: avg={avg_ms:.1f}ms, min={min_ms:.1f}ms, max={max_ms:.1f}ms")

        # Sanity: should complete within 10 seconds (very generous)
        assert avg_ms < 10_000


class TestRenderThroughput:
    """Measure sustained frame throughput."""

    def test_burst_30_frames(self, perf_server, orbit_cameras):
        """Render 30 frames as fast as possible, measure FPS."""
        c2w_list = orbit_cameras * 4  # 8 * 4 = 32, take 30

        t0 = time.perf_counter()
        frames_ok = 0
        total_bytes = 0

        for c2w in c2w_list[:30]:
            c2w_arr = np.array(c2w, dtype=np.float32).reshape(4, 4)
            result = perf_server.render(
                model_id="perf_model",
                width=320, height=240,
                camera_to_world=c2w_arr,
                fx=256.0, fy=256.0,
                cx=160.0, cy=120.0,
                fmt="jpeg",
                jpeg_quality=50,
            )
            if result["success"]:
                frames_ok += 1
                total_bytes += len(result["image_data"])

        elapsed = time.perf_counter() - t0
        fps = frames_ok / elapsed if elapsed > 0 else 0
        bandwidth_mbps = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

        print(
            f"\n  30-frame burst: {frames_ok}/30 ok, "
            f"{fps:.1f} FPS, {bandwidth_mbps:.2f} MB/s, "
            f"{elapsed:.2f}s total"
        )

        assert frames_ok == 30

    def test_jpeg_vs_raw_size(self, perf_server, identity_c2w):
        """Compare JPEG vs RAW output size."""
        c2w = np.array(identity_c2w, dtype=np.float32).reshape(4, 4)
        w, h = 320, 240

        jpeg_result = perf_server.render(
            model_id="perf_model",
            width=w, height=h,
            camera_to_world=c2w,
            fx=256.0, fy=256.0, cx=160.0, cy=120.0,
            fmt="jpeg", jpeg_quality=75,
        )

        raw_result = perf_server.render(
            model_id="perf_model",
            width=w, height=h,
            camera_to_world=c2w,
            fx=256.0, fy=256.0, cx=160.0, cy=120.0,
            fmt="raw",
        )

        jpeg_kb = len(jpeg_result["image_data"]) / 1024
        raw_kb = len(raw_result["image_data"]) / 1024
        ratio = raw_kb / jpeg_kb if jpeg_kb > 0 else 0

        print(
            f"\n  {w}x{h}: JPEG={jpeg_kb:.1f}KB, RAW={raw_kb:.1f}KB, "
            f"ratio={ratio:.1f}x"
        )

        assert jpeg_result["success"] is True
        assert raw_result["success"] is True
        assert raw_kb == w * h * 3 / 1024  # Exact: H*W*3 bytes
        assert jpeg_kb < raw_kb  # JPEG should be smaller
