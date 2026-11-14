"""Event bus — asyncio-based pub/sub for job-level events."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Per-job event channels.  Workers publish, WebSocket clients subscribe."""

    def __init__(self) -> None:
        # job_id → set of asyncio.Queue (one per subscriber)
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(q)
        logger.debug("Subscriber added for job %s (total: %d)", job_id, len(self._subscribers[job_id]))
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(job_id)
        if subs:
            subs.discard(q)
            if not subs:
                del self._subscribers[job_id]

    async def publish(self, job_id: str, event: dict[str, Any]) -> None:
        subs = self._subscribers.get(job_id)
        if not subs:
            return
        for q in subs:
            await q.put(event)

    def publish_sync(self, job_id: str, event: dict[str, Any]) -> None:
        """Fire-and-forget from sync code (worker thread).

        Schedules the publish on the running event loop.
        """
        subs = self._subscribers.get(job_id)
        if not subs:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.publish(job_id, event))
                )
        except RuntimeError:
            pass  # no event loop — skip
