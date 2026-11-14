"""Tests for POST /scan and GET /jobs/{id} endpoints."""

import pytest


class TestScanUpload:
    def test_upload_ply(self, client, dummy_ply):
        with open(dummy_ply, "rb") as f:
            r = client.post("/scan", files={"file": ("test.ply", f, "application/octet-stream")})
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_upload_rejects_invalid_extension(self, client):
        r = client.post("/scan", files={"file": ("bad.txt", b"hello", "text/plain")})
        assert r.status_code == 400
        assert "Unsupported" in r.json()["detail"]

    def test_upload_rejects_no_filename(self, client):
        r = client.post("/scan", files={"file": ("", b"data", "application/octet-stream")})
        assert r.status_code in (400, 422)

    @pytest.mark.parametrize("ext", [".ply", ".mp4", ".mov", ".avi"])
    def test_upload_accepts_valid_extensions(self, client, ext):
        r = client.post(
            "/scan",
            files={"file": (f"test{ext}", b"dummy", "application/octet-stream")},
        )
        assert r.status_code == 200


class TestJobQuery:
    def test_get_job_not_found(self, client):
        r = client.get("/jobs/nonexistent_id")
        assert r.status_code == 404

    def test_get_job_after_upload(self, client, dummy_ply):
        with open(dummy_ply, "rb") as f:
            r = client.post("/scan", files={"file": ("test.ply", f, "application/octet-stream")})
        job_id = r.json()["job_id"]

        r2 = client.get(f"/jobs/{job_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "running", "completed", "failed")
        assert "created_at" in data
