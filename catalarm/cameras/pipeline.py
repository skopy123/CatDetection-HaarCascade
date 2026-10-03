"""Per-camera processing loop: night vision -> motion -> (shared) inference -> vote -> publish."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from catalarm.cameras.reader import CameraReader
from catalarm.config import CameraConfig, expand
from catalarm.vision.motion import MotionDetector, MotionResult
from catalarm.vision.nightvision import is_night_vision
from catalarm.vision.tracker import NO_MOTION, UNCONFIRMED, Tracker, TrackState
from catalarm.vision.worker import Analysis

if TYPE_CHECKING:
    from catalarm.services import Services

log = logging.getLogger(__name__)

OVERLAYS = {"mask", "motion", "boxes", "label"}


@dataclass
class PerfWindow:
    motion_ms: list[float] = field(default_factory=list)
    detect_ms: list[float] = field(default_factory=list)
    embed_ms: list[float] = field(default_factory=list)
    frames: int = 0
    started: float = field(default_factory=time.time)


class CameraPipeline:
    def __init__(self, cfg: CameraConfig, services: Services):
        self.cfg = cfg
        self.s = services
        self.reader = CameraReader(cfg.id, expand(cfg.url) or "")
        self.motion = MotionDetector(cfg.motion)
        self.tracker = Tracker(services.config.config.thresholds)
        self.track = TrackState(NO_MOTION, None, 0.0, False)
        self.night = False
        self.last_motion: MotionResult | None = None
        self.last_analysis: Analysis | None = None
        self.last_analysis_frame: np.ndarray | None = None
        self.paused = False
        self._step = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"pipeline-{cfg.id}", daemon=True)
        self._visit_id: int | None = None
        self._last_cat_seen = 0.0
        self._perf = PerfWindow()
        self._render_cache: tuple[tuple, bytes] | None = None

    # ---- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        self.reader.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._step.set()
        self._thread.join(timeout=10)
        self.reader.stop()
        if self._visit_id is not None:
            self.s.db.end_visit(self._visit_id, self._last_cat_seen or time.time())

    def update_config(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.motion.update_config(cfg.motion)

    def pause(self, paused: bool) -> None:
        self.paused = paused

    def step(self) -> None:
        self._step.set()

    # ---- loop ----------------------------------------------------------------------------
    def _run(self) -> None:
        last_no, last_infer = -1, 0.0
        while not self._stop.is_set():
            app = self.s.config.config
            if self.paused and not self._step.is_set():
                self._stop.wait(0.2)
                continue
            forced = self._step.is_set()
            self._step.clear()
            start = time.time()
            frame, no = self.reader.latest()
            try:
                if frame is None or (no == last_no and not forced):
                    self._handle(self.tracker.update(start, motion=False), None, None)
                else:
                    last_no = no
                    last_infer = self._process(frame, start, last_infer, app.process_fps, forced)
            except Exception:
                log.exception("camera %s: processing failed", self.cfg.id)
                self._stop.wait(5)
            self._flush_perf()
            self._stop.wait(max(0.0, 1.0 / app.motion_fps - (time.time() - start)))

    def _process(
        self, frame: np.ndarray, now: float, last_infer: float, process_fps: float, forced: bool
    ) -> float:
        self.tracker.update_thresholds(self.s.config.config.thresholds)
        t0 = time.perf_counter()
        self.night = is_night_vision(frame)
        self.last_motion = self.motion.apply(frame, self.night)
        self._perf.motion_ms.append((time.perf_counter() - t0) * 1000)
        self._perf.frames += 1

        analysis = None
        if forced or (self.last_motion.detected and now - last_infer >= 1.0 / process_fps):
            analysis = self.s.worker.submit(frame, self.night).result(timeout=60)
            last_infer = now
            self.last_analysis, self.last_analysis_frame = analysis, frame
            self._perf.detect_ms.append(analysis.detect_ms)
            self._perf.embed_ms.append(analysis.embed_ms)
        track = self.tracker.update(now, self.last_motion.detected, analysis.best if analysis else None)
        self._handle(track, analysis, frame)
        return last_infer

    # ---- outputs -------------------------------------------------------------------------
    def _handle(self, track: TrackState, analysis: Analysis | None, frame: np.ndarray | None) -> None:
        now = time.time()
        if track.is_cat:
            self._last_cat_seen = now
        if track.changed:
            self._on_change(track, analysis, frame, now)
        self.track = track
        self._publish_ha()
        if track.changed or analysis is not None:
            self.s.hub.publish({"type": "camera", "camera": self.status()})

    def _on_change(
        self, track: TrackState, analysis: Analysis | None, frame: np.ndarray | None, now: float
    ) -> None:
        log.info("camera %s: %s -> %s", self.cfg.id, self.track.state, track.state)
        if self._visit_id is not None and (not track.is_cat or track.cat_id != self.track.cat_id):
            self.s.db.end_visit(self._visit_id, self._last_cat_seen)
            self._visit_id = None
        if track.is_cat and self._visit_id is None:
            self._visit_id = self.s.db.start_visit(self.cfg.id, track.cat_id, now)

        snapshot = None
        use_frame = self.last_analysis_frame if analysis is None else frame
        if (track.is_cat or track.state == UNCONFIRMED) and use_frame is not None:
            snapshot = self._save_snapshot(use_frame, self.last_analysis, now)
        best = self.last_analysis.best if self.last_analysis else None
        event_id = self.s.db.add_event(
            self.cfg.id,
            track.state,
            track.cat_id or (best.cat_id if best and track.state == UNCONFIRMED else None),
            track.confidence or (best.confidence if best else None),
            best.margin if best else None,
            self.night,
            snapshot,
            now,
        )
        self.s.hub.publish({"type": "event", "event": self.s.db.event(event_id)})

    def _save_snapshot(self, frame: np.ndarray, analysis: Analysis | None, now: float) -> str:
        rel = Path(self.cfg.id) / time.strftime("%Y%m%d", time.localtime(now)) / f"{int(now * 1000)}"
        base = self.s.snapshots_dir / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{base}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if analysis and analysis.best_index is not None:
            det = analysis.detections[analysis.best_index]
            crop = det.crop(frame, self.s.config.config.detector.padding)
            cv2.imwrite(f"{base}_crop.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return rel.as_posix()

    def _publish_ha(self) -> None:
        best = self.last_analysis.best if self.last_analysis else None
        cat = self.s.gallery.cat(self.track.cat_id) if self.track.cat_id else None
        public = self.s.config.config.public_url
        attrs = {
            "friendly_name": f"Cat AI {self.cfg.name or self.cfg.id}",
            "icon": "mdi:cat",
            "camera": self.cfg.id,
            "cat": cat.name if cat else None,
            "cat_id": self.track.cat_id,
            "confidence": round(self.track.confidence, 3),
            "margin": round(best.margin, 3) if best else None,
            "distances": {k: round(v, 3) for k, v in best.distances.items()} if best else {},
            "night_mode": self.night,
            "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self._last_cat_seen))
            if self._last_cat_seen
            else None,
            "snapshot_url": f"{public.rstrip('/')}/api/v1/cameras/{self.cfg.id}/snapshot.jpg"
            if public
            else None,
        }
        self.s.ha.publish(self.cfg.entity_id, self.track.state, attrs)

    def _flush_perf(self) -> None:
        p = self._perf
        elapsed = time.time() - p.started
        if elapsed < 60:
            return

        def avg(v: list[float]) -> float | None:
            return sum(v) / len(v) if v else None

        self.s.db.add_perf(
            self.cfg.id,
            p.frames / elapsed,
            avg(p.motion_ms),
            avg(p.detect_ms),
            avg(p.embed_ms),
            len(p.detect_ms),
        )
        self._perf = PerfWindow()

    # ---- views for the API ---------------------------------------------------------------
    def status(self) -> dict:
        r = self.reader
        return {
            "id": self.cfg.id,
            "name": self.cfg.name or self.cfg.id,
            "enabled": self.cfg.enabled,
            "connected": r.connected,
            "error": r.error,
            "fps": round(r.fps, 1),
            "frame_age": round(time.time() - r.frame_ts, 1) if r.frame_ts else None,
            "state": self.track.state,
            "cat_id": self.track.cat_id,
            "confidence": round(self.track.confidence, 3),
            "night": self.night,
            "motion": round(self.last_motion.changed_ratio, 5) if self.last_motion else None,
            "motion_detected": self.last_motion.detected if self.last_motion else False,
            "paused": self.paused,
            "entity_id": self.cfg.entity_id,
            "last_analysis": self.last_analysis.to_dict() if self.last_analysis else None,
        }

    def frame_jpeg(self, overlays: set[str] | None = None, quality: int = 80) -> bytes | None:
        frame, no = self.reader.latest()
        if frame is None:
            return None
        key = (no, frozenset(overlays or ()), quality, self.last_analysis.ts if self.last_analysis else 0)
        if self._render_cache and self._render_cache[0] == key:
            return self._render_cache[1]
        img = self.render(frame, overlays or set()) if overlays else frame
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        self._render_cache = (key, data)
        return data

    def motion_jpeg(self) -> bytes | None:
        if self.last_motion is None:
            return None
        return cv2.imencode(".png", self.last_motion.mask)[1].tobytes()

    def render(self, frame: np.ndarray, overlays: set[str]) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        if "motion" in overlays and self.last_motion is not None:
            m = cv2.resize(self.last_motion.mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            img[m] = (0.5 * img[m] + (0, 0, 127)).astype(np.uint8)
        if "mask" in overlays:
            layer = img.copy()
            for poly in self.cfg.motion.mask:
                if len(poly) >= 3:
                    pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
                    cv2.fillPoly(layer, [pts], (80, 80, 80))
            img = cv2.addWeighted(layer, 0.6, img, 0.4, 0)
        a = self.last_analysis
        fresh = a is not None and time.time() - a.ts < 3
        if "boxes" in overlays and fresh:
            for i, d in enumerate(a.detections):
                r = a.results[i] if i < len(a.results) else None
                color = (60, 200, 60) if r and r.state == "match" else (0, 180, 255)
                x1, y1, x2, y2 = d.box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"{r.cat_id} {r.distance:.2f}" if r else f"cat {d.score:.2f}"
                cv2.putText(img, text, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if "label" in overlays:
            lines = [f"{self.track.state}  {'NIGHT' if self.night else 'DAY'}"]
            if fresh and a.results:
                lines += [f"{k}: {v:.2f}" for k, v in list(a.best.distances.items())[:5]]
            for i, line in enumerate(lines):
                y = h - 12 - 20 * (len(lines) - 1 - i)
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return img
