"""Tests for EventBus (Phase 1)."""

import asyncio

import pytest

from server.events import EventBus


@pytest.mark.asyncio
class TestEventBus:
    async def test_subscribe_and_publish(self):
        bus = EventBus()
        q = bus.subscribe("job1")
        await bus.publish("job1", {"type": "test", "data": 42})
        event = await asyncio.wait_for(q.get(), timeout=1.0)
        assert event["type"] == "test"
        assert event["data"] == 42

    async def test_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe("job1")
        bus.unsubscribe("job1", q)
        await bus.publish("job1", {"type": "test"})
        assert q.empty()

    async def test_multiple_subscribers(self):
        bus = EventBus()
        q1 = bus.subscribe("job1")
        q2 = bus.subscribe("job1")
        await bus.publish("job1", {"type": "broadcast"})

        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert e1["type"] == "broadcast"
        assert e2["type"] == "broadcast"

    async def test_publish_to_nonexistent_job(self):
        bus = EventBus()
        # Should not raise
        await bus.publish("nonexistent", {"type": "test"})
