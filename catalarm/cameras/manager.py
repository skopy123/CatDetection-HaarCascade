from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from catalarm.cameras.pipeline import CameraPipeline
from catalarm.config import AppConfig

if TYPE_CHECKING:
    from catalarm.services import Services

log = logging.getLogger(__name__)


class CameraManager:
    """Keeps one running pipeline per enabled camera in the config."""

    def __init__(self, services: Services):
        self.s = services
        self.pipelines: dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()

    def get(self, camera_id: str) -> CameraPipeline | None:
        return self.pipelines.get(camera_id)

    def sync(self, cfg: AppConfig) -> None:
        with self._lock:
            wanted = {c.id: c for c in cfg.cameras if c.enabled}
            for cam_id in list(self.pipelines):
                p = self.pipelines[cam_id]
                new = wanted.get(cam_id)
                if new is None or new.url != p.cfg.url:
                    log.info("stopping camera %s", cam_id)
                    p.stop()
                    self.s.ha.forget(p.cfg.entity_id)
                    del self.pipelines[cam_id]
                else:
                    p.update_config(new)
            for cam_id, cam in wanted.items():
                if cam_id not in self.pipelines:
                    log.info("starting camera %s", cam_id)
                    p = CameraPipeline(cam, self.s)
                    self.pipelines[cam_id] = p
                    p.start()

    def stop_all(self) -> None:
        with self._lock:
            for p in self.pipelines.values():
                p.stop()
            self.pipelines.clear()
