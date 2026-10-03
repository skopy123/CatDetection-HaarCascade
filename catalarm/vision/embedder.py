"""Image embedders. All return L2-normalized vectors so cosine distance = 1 - dot."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_size: int
    description: str


# models that tools/export_models.py knows how to produce
MODEL_SPECS = {
    s.name: s
    for s in [
        ModelSpec("dinov2_vits14_reg", 224, "DINOv2 ViT-S/14 with registers (Meta, 2023)"),
        ModelSpec("dinov2_vitb14_reg", 224, "DINOv2 ViT-B/14 with registers - model of the old system"),
        ModelSpec("dinov3_vits16", 224, "DINOv3 ViT-S/16 (Meta, 2025, gated on HF)"),
        ModelSpec("megadescriptor_t_224", 224, "MegaDescriptor-T - trained for animal re-identification"),
    ]
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
        ...


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def square_pad(img: np.ndarray) -> np.ndarray:
    """Pad to a square with the ImageNet mean color, so crops keep their aspect ratio."""
    h, w = img.shape[:2]
    side = max(h, w)
    fill = (IMAGENET_MEAN[::-1] * 255).astype(np.uint8).tolist()  # BGR
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=fill
    )


def preprocess(img_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(square_pad(img_bgr), (size, size), interpolation=cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


class OnnxEmbedder:
    def __init__(self, spec: ModelSpec, model_path: Path, batch: int = 8):
        import onnxruntime as ort

        self.name = spec.name
        self.spec = spec
        self.batch = batch
        opts = ort.SessionOptions()
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.dim = int(self._session.get_outputs()[0].shape[-1])

    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(images_bgr), self.batch):
            chunk = np.stack([preprocess(im, self.spec.input_size) for im in images_bgr[i : i + self.batch]])
            out.append(self._session.run(None, {self._input: chunk})[0])
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return l2norm(np.concatenate(out).astype(np.float32))


class HistogramEmbedder:
    """Model-free color histogram embedding. Used in tests and when no model is available."""

    name = "histogram"
    dim = 8 * 8 * 4

    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images_bgr), self.dim), np.float32)
        for i, img in enumerate(images_bgr):
            hsv = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
            out[i] = np.sqrt(hist.flatten())
        return l2norm(out)


def available_embedders(models_dir: Path) -> list[str]:
    names = [n for n in MODEL_SPECS if (models_dir / f"{n}.onnx").exists()]
    return names + ["histogram"]


def load_embedder(name: str, models_dir: Path) -> Embedder:
    if name != "histogram":
        spec = MODEL_SPECS.get(name)
        path = models_dir / f"{name}.onnx"
        if spec and path.exists():
            return OnnxEmbedder(spec, path)
        log.warning("embedder %s not available in %s, falling back to histogram", name, models_dir)
    return HistogramEmbedder()
