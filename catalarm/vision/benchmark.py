"""Leave-one-out evaluation of the gallery: how well does an embedder separate the cats?"""

from __future__ import annotations

import time

import cv2
import numpy as np

from catalarm.config import Thresholds
from catalarm.gallery.store import MODES, Gallery
from catalarm.vision.embedder import Embedder
from catalarm.vision.matcher import MATCH, match


def evaluate(embs: np.ndarray, labels: list[str], th: Thresholds) -> dict:
    cats = sorted(set(labels))
    confusion = {a: dict.fromkeys(cats, 0) for a in cats}
    correct = accepted = accepted_correct = 0
    same, other = [], []
    n = len(labels)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        if not keep:
            continue
        r = match(embs[i], embs[keep], [labels[j] for j in keep], th)
        if r.cat_id is None:
            continue
        confusion[labels[i]][r.cat_id] += 1
        correct += r.cat_id == labels[i]
        if r.state == MATCH:
            accepted += 1
            accepted_correct += r.cat_id == labels[i]
        for j in keep:
            d = float(1 - embs[i] @ embs[j])
            (same if labels[j] == labels[i] else other).append(d)

    def stats(v: list[float]) -> dict | None:
        if not v:
            return None
        a = np.asarray(v)
        return {k: round(float(f(a)), 4) for k, f in (("mean", np.mean), ("min", np.min), ("max", np.max))}

    return {
        "images": n,
        "cats": cats,
        "top1_accuracy": round(correct / n, 4) if n else None,
        "accepted_rate": round(accepted / n, 4) if n else None,  # passed the match threshold
        "accepted_precision": round(accepted_correct / accepted, 4) if accepted else None,
        "same_cat_distance": stats(same),
        "other_cat_distance": stats(other),
        "confusion": confusion,
    }


def benchmark(gallery: Gallery, embedder: Embedder, th: Thresholds) -> dict:
    t0 = time.perf_counter()
    refs = gallery.images()
    imgs = [cv2.imread(str(r.path)) for r in refs]
    embs = embedder.embed(imgs) if refs else np.zeros((0, embedder.dim), np.float32)
    elapsed = time.perf_counter() - t0
    result = {
        "embedder": embedder.name,
        "ms_per_image": round(elapsed * 1000 / max(1, len(refs)), 1),
        "modes": {},
    }
    for mode in MODES:
        idx = [i for i, r in enumerate(refs) if r.mode == mode]
        if idx:
            result["modes"][mode] = evaluate(embs[idx], [refs[i].cat_id for i in idx], th)
    return result
