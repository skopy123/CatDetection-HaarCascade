"""Shared inference: one detector and one embedder session serve all cameras."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from catalarm.config import AppConfig
from catalarm.gallery.store import Gallery
from catalarm.vision.detector import Detection, Detector
from catalarm.vision.matcher import NO_CAT, MatchResult, match

MAX_CROPS = 3


@dataclass
class Analysis:
    detections: list[Detection]
    results: list[MatchResult]  # one per analysed detection
    best: MatchResult
    best_index: int | None
    detect_ms: float
    embed_ms: float
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "detections": [{"box": d.box, "score": round(d.score, 3)} for d in self.detections],
            "results": [r.to_dict() for r in self.results],
            "best": self.best.to_dict(),
            "best_index": self.best_index,
            "detect_ms": round(self.detect_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
        }


class InferenceWorker:
    def __init__(self, detector: Detector, gallery: Gallery, cfg: AppConfig):
        self.detector = detector
        self.gallery = gallery
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._lock = threading.Lock()

    def set_detector(self, detector: Detector) -> None:
        with self._lock:
            self.detector = detector

    def submit(self, frame: np.ndarray, night: bool) -> Future[Analysis]:
        return self._pool.submit(self.analyse, frame, night)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def analyse(self, frame: np.ndarray, night: bool) -> Analysis:
        with self._lock:
            cfg = self.cfg
            t0 = time.perf_counter()
            detections = self.detector.detect(frame)[:MAX_CROPS]
            t1 = time.perf_counter()
            results: list[MatchResult] = []
            if detections:
                crops = [d.crop(frame, cfg.detector.padding) for d in detections]
                # hold the gallery lock so embedder and references always belong together
                with self.gallery.lock:
                    embs = self.gallery.embedder.embed(crops)
                    refs, labels, _ = self.gallery.references("night" if night else "day")
                results = [match(e, refs, labels, cfg.thresholds) for e in embs]
            t2 = time.perf_counter()
        best_index = min(range(len(results)), key=lambda i: results[i].distance) if results else None
        best = results[best_index] if best_index is not None else MatchResult(NO_CAT, None, 2.0, 0.0)
        return Analysis(detections, results, best, best_index, (t1 - t0) * 1000, (t2 - t1) * 1000)
