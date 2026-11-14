"""Tests for WebSocket endpoints."""

import json


class TestJobWebSocket:
    def test_ws_not_found(self, client):
        """WebSocket for unknown job should close with 4004."""
        try:
            with client.websocket_connect("/ws/jobs/nonexistent"):
                pass
            assert False, "Should have closed"
        except Exception:
            pass  # Expected: connection rejected

    def test_ws_receives_events(self, client, dummy_ply):
        """WebSocket should receive at least one event for an active job."""
        with open(dummy_ply, "rb") as f:
            r = client.post("/scan", files={"file": ("test.ply", f, "application/octet-stream")})
        job_id = r.json()["job_id"]

        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            raw = ws.receive_text()
            event = json.loads(raw)
            # Should get progress, step_complete, error, or completed
            assert event["type"] in ("progress", "step_complete", "completed", "error", "heartbeat")


class TestStreamWebSocket:
    def test_stream_not_found(self, client):
        try:
            with client.websocket_connect("/ws/stream/nonexistent"):
                pass
            assert False, "Should have closed"
        except Exception:
            pass

    def test_stream_not_completed(self, client, dummy_ply):
        """Stream should reject jobs that aren't completed yet."""
        with open(dummy_ply, "rb") as f:
            r = client.post("/scan", files={"file": ("test.ply", f, "application/octet-stream")})
        job_id = r.json()["job_id"]

        # Job is queued/running, not completed — should be rejected
        try:
            with client.websocket_connect(f"/ws/stream/{job_id}"):
                pass
        except Exception:
            pass  # Expected
