"""Wires the long-lived components together and reacts to config changes."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from catalarm.cameras.manager import CameraManager
from catalarm.config import AppConfig, ConfigStore
from catalarm.gallery.store import Gallery
from catalarm.hub import Hub
from catalarm.integrations.ha import HAPublisher
from catalarm.stats.db import StatsDB
from catalarm.vision.detector import load_detector
from catalarm.vision.embedder import load_embedder
from catalarm.vision.worker import InferenceWorker

log = logging.getLogger(__name__)


class Services:
    def __init__(self, data_dir: Path, models_dir: Path, seed_templates: Path | None = None):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.snapshots_dir = data_dir / "snapshots"
        self.captures_dir = data_dir / "captures"
        self.config = ConfigStore(data_dir / "config.yaml")
        cfg = self.config.config

        self.gallery = Gallery(data_dir / "gallery", load_embedder(cfg.embedder, models_dir))
        if seed_templates and not self.gallery.cats() and not (data_dir / "gallery" / "cats.yaml").exists():
            n = self.gallery.import_legacy_templates(seed_templates)
            log.info("imported %d legacy reference images from %s", n, seed_templates)

        self.db = StatsDB(data_dir / "catalarm.db")
        self.hub = Hub()
        self.ha = HAPublisher(cfg.ha)
        self.worker = InferenceWorker(
            load_detector(cfg.detector.model, models_dir, cfg.detector.confidence)
            if cfg.detector.enabled
            else load_detector("", models_dir, 0),
            self.gallery,
            cfg,
        )
        self.cameras = CameraManager(self)
        self._applied = cfg
        self._stop = threading.Event()
        self._housekeeping = threading.Thread(target=self._housekeep, name="housekeeping", daemon=True)
        self.config.subscribe(self._on_config)

    def start(self) -> None:
        self.ha.start()
        self.cameras.sync(self.config.config)
        self._housekeeping.start()

    def stop(self) -> None:
        self._stop.set()
        self.cameras.stop_all()
        self.worker.shutdown()
        self.ha.stop()
        self.db.close()

    def _on_config(self, cfg: AppConfig) -> None:
        old, self._applied = self._applied, cfg
        self.worker.cfg = cfg
        self.ha.update_config(cfg.ha)
        if cfg.detector != old.detector:
            name = cfg.detector.model if cfg.detector.enabled else ""
            self.worker.set_detector(load_detector(name, self.models_dir, cfg.detector.confidence))
        if cfg.embedder != old.embedder:
            # re-embedding the gallery takes a moment, do not block the request
            threading.Thread(
                target=lambda: self.gallery.set_embedder(load_embedder(cfg.embedder, self.models_dir)),
                daemon=True,
            ).start()
        self.cameras.sync(cfg)

    def _housekeep(self) -> None:
        while not self._stop.wait(3600):
            try:
                for rel in self.db.prune(self.config.config.retention_days):
                    for suffix in (".jpg", "_crop.jpg"):
                        (self.snapshots_dir / f"{rel}{suffix}").unlink(missing_ok=True)
            except Exception:
                log.exception("housekeeping failed")


def from_env() -> Services:
    from catalarm.config import data_dir, models_dir

    seed = os.environ.get("CATALARM_SEED_TEMPLATES")
    return Services(data_dir(), models_dir(), Path(seed) if seed else None)
