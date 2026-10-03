"""Thread-safe fan-out of live updates to Server-Sent-Event clients."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading


class Hub:
    def __init__(self) -> None:
        self._subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._lock = threading.Lock()

    def publish(self, message: dict) -> None:
        data = json.dumps(message, default=str)
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(_put, q, data)

    async def subscribe(self, keepalive: float = 15.0):
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        sub = (asyncio.get_running_loop(), q)
        with self._lock:
            self._subs.add(sub)
        try:
            while True:
                try:
                    yield await asyncio.wait_for(q.get(), timeout=keepalive)
                except TimeoutError:
                    yield None  # lets the caller send a keep-alive
        finally:
            with self._lock:
                self._subs.discard(sub)


def _put(q: asyncio.Queue, data: str) -> None:
    if q.full():  # slow client: drop the oldest message
        q.get_nowait()
    q.put_nowait(data)
