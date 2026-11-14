"""MQTT integration — IoT sensor data ingestion for digital twin state sync.

Subscribes to sensor topics, updates the twin state, and pushes
changes to WebSocket clients and optionally to Nucleus USD live layers.

Requires: aiomqtt (pip install aiomqtt).
Falls back gracefully when not installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _has_aiomqtt() -> bool:
    try:
        import aiomqtt  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class SensorReading:
    """Single sensor data point."""
    sensor_id: str
    topic: str
    value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TwinState:
    """In-memory digital twin state — sensors + derived properties."""

    def __init__(self) -> None:
        self._sensors: dict[str, SensorReading] = {}
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def update_sensor(self, reading: SensorReading) -> None:
        """Update a sensor value and notify subscribers."""
        self._sensors[reading.sensor_id] = reading
        event = {
            "type": "sensor_update",
            "sensor_id": reading.sensor_id,
            "topic": reading.topic,
            "value": reading.value,
            "timestamp": reading.timestamp.isoformat(),
        }
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # drop if subscriber is slow

    def get_sensor(self, sensor_id: str) -> SensorReading | None:
        return self._sensors.get(sensor_id)

    def get_all_sensors(self) -> dict[str, Any]:
        return {
            sid: {
                "topic": r.topic,
                "value": r.value,
                "timestamp": r.timestamp.isoformat(),
            }
            for sid, r in self._sensors.items()
        }

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)


class MqttBridge:
    """Bridge between MQTT broker and digital twin state."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topics: list[str] | None = None,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics = topics or ["gss/sensors/#"]
        self._running = False

    async def run(self, twin_state: TwinState) -> None:
        """Connect to MQTT broker and forward messages to TwinState."""
        if not _has_aiomqtt():
            logger.warning(
                "aiomqtt not installed — MQTT bridge disabled. "
                "Install with: pip install aiomqtt"
            )
            return

        import aiomqtt

        self._running = True
        logger.info("MQTT bridge connecting to %s:%d", self.broker_host, self.broker_port)

        while self._running:
            try:
                async with aiomqtt.Client(
                    hostname=self.broker_host,
                    port=self.broker_port,
                ) as client:
                    for topic in self.topics:
                        await client.subscribe(topic)
                    logger.info("MQTT subscribed to: %s", self.topics)

                    async for message in client.messages:
                        try:
                            payload = json.loads(message.payload.decode())
                            sensor_id = payload.get("sensor_id", str(message.topic))
                            reading = SensorReading(
                                sensor_id=sensor_id,
                                topic=str(message.topic),
                                value=payload.get("value", payload),
                            )
                            twin_state.update_sensor(reading)
                        except (json.JSONDecodeError, Exception) as e:
                            logger.debug("MQTT message parse error: %s", e)

            except Exception as e:
                if self._running:
                    logger.warning("MQTT connection lost (%s), reconnecting in 5s...", e)
                    await asyncio.sleep(5)

    def stop(self) -> None:
        self._running = False
