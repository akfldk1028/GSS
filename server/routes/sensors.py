"""Sensor data endpoints — REST + WebSocket for digital twin state."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..mqtt import TwinState

router = APIRouter(prefix="/sensors", tags=["sensors"])


class SensorStateResponse(BaseModel):
    sensors: dict


class MqttConfigRequest(BaseModel):
    broker_host: str = "localhost"
    broker_port: int = 1883
    topics: list[str] = ["gss/sensors/#"]


class MqttConfigResponse(BaseModel):
    status: str
    message: str


@router.get("/", response_model=SensorStateResponse)
async def get_sensors(request: Request) -> SensorStateResponse:
    """Get all current sensor readings."""
    twin_state: TwinState = request.app.state.twin_state
    return SensorStateResponse(sensors=twin_state.get_all_sensors())


@router.post("/mqtt/connect", response_model=MqttConfigResponse)
async def connect_mqtt(request: Request, body: MqttConfigRequest) -> MqttConfigResponse:
    """Start MQTT bridge to receive sensor data."""
    from ..mqtt import MqttBridge, _has_aiomqtt

    if not _has_aiomqtt():
        return MqttConfigResponse(
            status="unavailable",
            message="aiomqtt not installed. pip install aiomqtt",
        )

    twin_state: TwinState = request.app.state.twin_state

    # Stop existing bridge if any
    existing = getattr(request.app.state, "mqtt_bridge", None)
    if existing:
        existing.stop()

    bridge = MqttBridge(
        broker_host=body.broker_host,
        broker_port=body.broker_port,
        topics=body.topics,
    )
    request.app.state.mqtt_bridge = bridge
    asyncio.create_task(bridge.run(twin_state))

    return MqttConfigResponse(
        status="started",
        message=f"MQTT bridge started: {body.broker_host}:{body.broker_port}",
    )


@router.websocket("/ws")
async def sensor_websocket(websocket: WebSocket) -> None:
    """Stream real-time sensor updates via WebSocket."""
    twin_state: TwinState = websocket.app.state.twin_state

    await websocket.accept()
    queue = twin_state.subscribe()

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(json.dumps(event))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    finally:
        twin_state.unsubscribe(queue)
