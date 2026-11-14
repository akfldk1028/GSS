"""Tests for sensor/MQTT endpoints (Phase 4)."""


class TestSensors:
    def test_get_sensors_empty(self, client):
        r = client.get("/sensors/")
        assert r.status_code == 200
        assert r.json()["sensors"] == {}

    def test_mqtt_connect_without_aiomqtt(self, client):
        """MQTT connect should fail gracefully without aiomqtt."""
        r = client.post("/sensors/mqtt/connect", json={
            "broker_host": "localhost",
            "broker_port": 1883,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "unavailable"
