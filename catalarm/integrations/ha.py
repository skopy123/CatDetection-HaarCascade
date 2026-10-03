"""Home Assistant publisher (REST states API, ported from modHaHttpSensorPush.py).

One sensor per camera. A state is sent whenever it changes and re-sent every
``heartbeat_s``, because entities created through ``/api/states`` vanish on HA restart.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import httpx

from catalarm.config import HAConfig, expand

log = logging.getLogger(__name__)


class HAPublisher:
    def __init__(self, cfg: HAConfig):
        self.cfg = cfg
        self._latest: dict[str, tuple[str, dict]] = {}  # entity -> (state, attributes)
        self._sent: dict[str, float] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ha-publisher", daemon=True)
        self.last_error: str | None = None
        self.last_ok: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.url)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")
        self._thread.join(timeout=5)

    def update_config(self, cfg: HAConfig) -> None:
        self.cfg = cfg
        for entity in self._latest:
            self._queue.put(entity)

    def publish(self, entity_id: str, state: str, attributes: dict) -> None:
        previous = self._latest.get(entity_id)
        self._latest[entity_id] = (state, attributes)
        if previous is None or previous[0] != state:
            self._queue.put(entity_id)

    def forget(self, entity_id: str) -> None:
        self._latest.pop(entity_id, None)

    def _run(self) -> None:
        with httpx.Client(timeout=5) as client:
            while not self._stop.is_set():
                try:
                    entity = self._queue.get(timeout=1)
                except queue.Empty:
                    entity = None
                if self._stop.is_set():
                    break
                now = time.time()
                due = {entity} if entity else set()
                due |= {e for e in self._latest if now - self._sent.get(e, 0) >= self.cfg.heartbeat_s}
                for e in due:
                    if e in self._latest:
                        self._send(client, e)

    def _send(self, client: httpx.Client, entity_id: str) -> None:
        self._sent[entity_id] = time.time()
        if not self.enabled:
            return
        state, attributes = self._latest[entity_id]
        url = f"{expand(self.cfg.url).rstrip('/')}/api/states/{entity_id}"
        headers = {"Authorization": f"Bearer {expand(self.cfg.token)}"}
        try:
            r = client.post(url, headers=headers, json={"state": state, "attributes": attributes})
            r.raise_for_status()
            self.last_ok, self.last_error = time.time(), None
        except httpx.HTTPError as e:
            if self.last_error != str(e):
                log.warning("HA update of %s failed: %s", entity_id, e)
            self.last_error = str(e)
