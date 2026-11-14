"""Tests for GET /health endpoint."""


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "gpu_available" in data
        assert "queue_size" in data
        assert "active_jobs" in data
        assert isinstance(data["queue_size"], int)
