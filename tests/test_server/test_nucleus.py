"""Tests for Nucleus sync endpoints (Phase 3)."""


class TestNucleusStatus:
    def test_status_disconnected(self, client):
        r = client.get("/nucleus/status")
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is False
        assert data["server_url"] is None

    def test_connect_without_omni_client(self, client):
        """Connection should fail gracefully without omni.client."""
        r = client.post("/nucleus/connect", json={"server_url": "omniverse://localhost"})
        assert r.status_code == 200
        assert r.json()["connected"] is False

    def test_push_without_connection(self, client):
        r = client.post("/nucleus/push/some_job_id")
        assert r.status_code == 503
