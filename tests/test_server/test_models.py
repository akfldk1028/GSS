"""Tests for server models / schemas."""

from server.models import (
    HealthResponse,
    JobOutput,
    JobResponse,
    JobStatus,
    ScanResponse,
)


class TestModels:
    def test_job_status_values(self):
        assert JobStatus.queued == "queued"
        assert JobStatus.running == "running"
        assert JobStatus.completed == "completed"
        assert JobStatus.failed == "failed"

    def test_scan_response_defaults(self):
        r = ScanResponse(job_id="abc123")
        assert r.status == JobStatus.queued
        assert r.message == "Job created"

    def test_job_output_all_none(self):
        o = JobOutput()
        assert o.ifc is None
        assert o.glb is None
        assert o.usdc is None
        assert o.usdz is None

    def test_health_response_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.gpu_available is False
        assert h.queue_size == 0
