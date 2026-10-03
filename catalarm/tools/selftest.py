"""End-to-end check with the real models: identify the test photos against the seed gallery.

    python -m catalarm.tools.selftest [--images testImg] [--templates templates]

File names encode the expected cat (``maugli1.jpg`` -> maugli). Exit code 1 on mismatch.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2

from catalarm.config import AppConfig, models_dir
from catalarm.gallery.store import Gallery
from catalarm.vision.detector import FullFrameDetector, load_detector
from catalarm.vision.embedder import HistogramEmbedder, load_embedder
from catalarm.vision.nightvision import is_night_vision
from catalarm.vision.worker import InferenceWorker


def run(images: Path, templates: Path, models: Path) -> list[str]:
    cfg = AppConfig()
    embedder = load_embedder(cfg.embedder, models)
    detector = load_detector(cfg.detector.model, models, cfg.detector.confidence)
    if isinstance(embedder, HistogramEmbedder) or isinstance(detector, FullFrameDetector):
        return [f"models missing in {models}: embedder={embedder.name} detector={detector.name}"]
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        gallery = Gallery(Path(tmp), embedder)
        gallery.import_legacy_templates(templates)
        worker = InferenceWorker(detector, gallery, cfg)
        for path in sorted(images.glob("*.jpg")):
            expected = path.stem.rstrip("0123456789").lower()
            if not gallery.cat(expected):
                continue
            img = cv2.imread(str(path))
            a = worker.analyse(img, is_night_vision(img))
            got = a.best.cat_id
            print(f"{path.name}: detections={len(a.detections)} best={got} {a.best.to_dict()['distances']}")
            if got != expected:
                errors.append(f"{path.name}: expected {expected}, got {got}")
        worker.shutdown()
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("testImg"))
    ap.add_argument("--templates", type=Path, default=Path("templates"))
    ap.add_argument("--models", type=Path, default=models_dir())
    args = ap.parse_args()
    errors = run(args.images, args.templates, args.models)
    for e in errors:
        print("FAIL", e, file=sys.stderr)
    print("selftest", "failed" if errors else "passed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
