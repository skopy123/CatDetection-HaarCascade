import time

import cv2
import numpy as np
import yaml

from catalarm.config import CameraConfig, ConfigStore, expand
from catalarm.gallery.store import Gallery
from catalarm.stats.db import StatsDB
from catalarm.vision.embedder import HistogramEmbedder

from .conftest import TEMPLATES, TEST_IMG


def test_env_references_are_not_written_back(tmp_path, monkeypatch):
    monkeypatch.setenv("RTSP_STREAM", "rtsp://user:secret@cam/stream")
    monkeypatch.delenv("RTSP_URL", raising=False)
    store = ConfigStore(tmp_path / "config.yaml")
    cam = store.config.cameras[0]
    assert cam.url == "${RTSP_STREAM}"
    assert expand(cam.url) == "rtsp://user:secret@cam/stream"
    assert "secret" not in (tmp_path / "config.yaml").read_text()
    assert cam.motion.mask and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in cam.motion.mask[0])


def test_config_update_notifies_and_persists(tmp_path, monkeypatch):
    monkeypatch.delenv("RTSP_STREAM", raising=False)
    monkeypatch.delenv("RTSP_URL", raising=False)
    store = ConfigStore(tmp_path / "config.yaml")
    seen = []
    store.subscribe(lambda c: seen.append(len(c.cameras)))
    store.update(lambda c: c.cameras.append(CameraConfig(id="garden", url="rtsp://x")))
    assert seen == [1]
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert raw["cameras"][0]["id"] == "garden"
    assert ConfigStore(tmp_path / "config.yaml").config.camera("garden").entity_id == "sensor.cat_ai_garden"


def test_gallery_legacy_import_and_cache(tmp_path):
    g = Gallery(tmp_path / "gallery", HistogramEmbedder())
    n = g.import_legacy_templates(TEMPLATES)
    day = len(list((TEMPLATES / "DayTemplates").glob("*.png")))
    night = len(list((TEMPLATES / "NightTemplates").glob("*.png")))
    assert n == day + night
    assert {c.id for c in g.cats()} == {"amalka", "gita", "kocour", "maugli"}
    assert len(g.images(mode="day")) == day and len(g.images(mode="night")) == night
    embs, labels, _ = g.references("night")
    assert len(labels) == night and embs.shape[0] == night

    class CountingEmbedder(HistogramEmbedder):
        calls = 0

        def embed(self, images):
            CountingEmbedder.calls += len(images)
            return super().embed(images)

    g2 = Gallery(tmp_path / "gallery", CountingEmbedder())  # same model name -> cache hit
    assert CountingEmbedder.calls == 0
    img = cv2.imread(str(TEST_IMG / "gita1.jpg"))
    ref = g2.add_image("gita", "day", img)
    assert CountingEmbedder.calls == 1
    assert max(cv2.imread(str(ref.path)).shape[:2]) <= 320
    g2.move_image(ref.id, "maugli", "night")
    assert g2.image(ref.id).cat_id == "maugli"
    g2.delete_image(ref.id)
    assert g2.image(ref.id) is None
    matrix = g2.similarity_matrix("day")
    assert np.allclose(np.diag(matrix["distance"]), 0, atol=1e-3)


def test_stats_db(tmp_path):
    db = StatsDB(tmp_path / "s.db")
    now = time.time()
    db.add_event("cam", "maugli", "maugli", 0.9, 0.2, False, "cam/x/1", now)
    db.add_event("cam", "no_cat", None, None, None, True, None, now + 1)
    v = db.start_visit("cam", "maugli", now)
    db.end_visit(v, now + 30)
    db.add_perf("cam", 5.0, 1.0, 12.0, 30.0, 4, now)
    assert db.events(cat_id="maugli")[0]["confidence"] == 0.9
    assert db.daily_visits(1)[0]["seconds"] == 30
    assert sum(r["visits"] for r in db.hourly_visits(1)) == 1
    assert {r["state"] for r in db.state_counts(1)} == {"maugli", "no_cat"}
    assert db.perf(1)[0]["detect_ms"] == 12.0
    assert db.prune(0) == ["cam/x/1"]
    assert [e["state"] for e in db.events()] == ["no_cat"]  # newer than the cutoff
