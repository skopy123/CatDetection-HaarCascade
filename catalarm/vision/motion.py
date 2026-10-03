"""Cheap motion gate (ported from modMotionDetection.py), one instance per camera."""

from dataclasses import dataclass

import cv2
import numpy as np

from catalarm.config import MotionConfig

WORK_WIDTH = 320


@dataclass
class MotionResult:
    detected: bool
    changed_ratio: float
    mask: np.ndarray  # binary motion mask at work resolution


class MotionDetector:
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._bg = cv2.createBackgroundSubtractorKNN()
        self._ignore: np.ndarray | None = None
        self._size: tuple[int, int] | None = None
        self._night: bool | None = None

    def reset(self) -> None:
        self._bg = cv2.createBackgroundSubtractorKNN()

    def update_config(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self._ignore = None

    def _ignore_mask(self, w: int, h: int) -> np.ndarray:
        if self._ignore is None or self._size != (w, h):
            mask = np.zeros((h, w), np.uint8)
            for poly in self.cfg.mask:
                if len(poly) >= 3:
                    pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            self._ignore, self._size = mask, (w, h)
        return self._ignore

    def apply(self, frame_bgr: np.ndarray, night: bool = False) -> MotionResult:
        # the IR switch changes the whole image, start learning the background again
        if self._night is not None and night != self._night:
            self.reset()
        self._night = night

        h, w = frame_bgr.shape[:2]
        size = (WORK_WIDTH, max(1, round(h * WORK_WIDTH / w)))
        gray = cv2.cvtColor(cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur[self._ignore_mask(*size) > 0] = 0
        fg = self._bg.apply(blur, learningRate=-1)
        mask = cv2.threshold(fg, self.cfg.threshold, 255, cv2.THRESH_BINARY)[1]
        ratio = float(np.count_nonzero(mask) / mask.size)
        detected = (not self.cfg.enabled) or ratio > self.cfg.min_area_ratio
        return MotionResult(detected, ratio, mask)
