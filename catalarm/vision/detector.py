"""Cat detector. YOLO (COCO) exported to ONNX; only the ``cat`` class is used."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

COCO_CAT = 15


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    score: float

    def crop(self, frame: np.ndarray, padding: float = 0.0) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        pw, ph = (x2 - x1) * padding / 2, (y2 - y1) * padding / 2
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        x2, y2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
        return frame[y1:y2, x1:x2]


class Detector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class FullFrameDetector:
    """No detection: treat the whole frame as the cat (behaviour of the old system)."""

    name = "full-frame"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        return [Detection((0, 0, w, h), 1.0)]


def letterbox(img: np.ndarray, size: int, fill: int = 114) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), fill, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy : dy + nh, dx : dx + nw] = resized
    return out, scale, dx, dy


class YoloOnnxDetector:
    def __init__(self, model_path: Path, confidence: float = 0.35, iou: float = 0.45):
        import onnxruntime as ort

        self.name = model_path.stem
        self.confidence = confidence
        self.iou = iou
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self._size = inp.shape[2] if isinstance(inp.shape[2], int) else 640

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        img, scale, dx, dy = letterbox(frame_bgr, self._size)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self._session.run(None, {self._input: blob})[0][0]  # (4 + classes, anchors)
        scores = out[4 + COCO_CAT]
        keep = scores >= self.confidence
        if not np.any(keep):
            return []
        cx, cy, bw, bh = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
        scores = scores[keep]
        x1 = (cx - bw / 2 - dx) / scale
        y1 = (cy - bh / 2 - dy) / scale
        boxes = np.stack([x1, y1, bw / scale, bh / scale], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence, self.iou)
        h, w = frame_bgr.shape[:2]
        result = []
        for i in np.array(idx).flatten():
            bx, by, bw_, bh_ = boxes[i]
            box = (
                int(max(0, bx)),
                int(max(0, by)),
                int(min(w, bx + bw_)),
                int(min(h, by + bh_)),
            )
            result.append(Detection(box, float(scores[i])))
        result.sort(key=lambda d: d.score, reverse=True)
        return result


def load_detector(name: str, models_dir: Path, confidence: float) -> Detector:
    path = models_dir / f"{name}.onnx"
    if not path.exists():
        log.warning("detector model %s not found, using full frame (no cropping)", path)
        return FullFrameDetector()
    return YoloOnnxDetector(path, confidence)
