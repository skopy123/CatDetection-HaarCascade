"""Reference gallery shared by all cameras.

Layout (all plain files, no database)::

    gallery/cats.yaml                      # [{id, name, color}]
    gallery/<cat_id>/<day|night>/<id>.jpg  # low-res reference crops
    gallery/embeddings-<model>.npz         # cache, rebuilt when files or the model change
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from catalarm.config import check_id
from catalarm.vision.embedder import Embedder

log = logging.getLogger(__name__)

MODES = ("day", "night")
MAX_SIDE = 320
# validated categorical order (colorblind-safe for adjacent series); a cat keeps its color
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# filename prefixes used by the original templates folder (m1.png -> Maugli)
LEGACY_NAMES = {"a": "Amalka", "g": "Gita", "k": "Kocour", "m": "Maugli"}


@dataclass
class Cat:
    id: str
    name: str
    color: str


@dataclass
class RefImage:
    id: str
    cat_id: str
    mode: str
    path: Path

    def to_dict(self) -> dict:
        return {"id": self.id, "cat_id": self.cat_id, "mode": self.mode}


def downscale(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


class Gallery:
    def __init__(self, root: Path, embedder: Embedder):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embedder = embedder
        self._cats: list[Cat] = []
        self._images: dict[str, RefImage] = {}
        self._emb: dict[str, np.ndarray] = {}  # image id -> embedding
        self.version = 0
        self._listeners: list[Callable[[], None]] = []
        self.reload()

    # ---- loading -------------------------------------------------------------------------
    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def set_embedder(self, embedder: Embedder) -> None:
        with self._lock:
            self._embedder = embedder
            self._emb = {}
            self.reload()

    def subscribe(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _cache_path(self) -> Path:
        return self.root / f"embeddings-{self._embedder.name}.npz"

    def reload(self) -> None:
        with self._lock:
            cats_file = self.root / "cats.yaml"
            raw = yaml.safe_load(cats_file.read_text(encoding="utf-8")) if cats_file.exists() else []
            self._cats = [Cat(**c) for c in raw or []]
            self._images = {}
            for cat in self._cats:
                for mode in MODES:
                    for p in sorted((self.root / cat.id / mode).glob("*.jpg")):
                        self._images[p.stem] = RefImage(p.stem, cat.id, mode, p)
            self._refresh_embeddings()

    def _refresh_embeddings(self) -> None:
        cache = self._cache_path()
        stamps = {i: f"{r.path.stat().st_mtime_ns}" for i, r in self._images.items()}
        cached: dict[str, np.ndarray] = {}
        if cache.exists():
            try:
                with np.load(cache) as npz:
                    for key in npz.files:
                        img_id, stamp = key.rsplit("@", 1)
                        if stamps.get(img_id) == stamp:
                            cached[img_id] = npz[key]
            except Exception:
                log.warning("embedding cache %s unreadable, rebuilding", cache)
        missing = [i for i in self._images if i not in cached]
        if missing:
            log.info("embedding %d reference images with %s", len(missing), self._embedder.name)
            imgs = [cv2.imread(str(self._images[i].path)) for i in missing]
            for img_id, vec in zip(missing, self._embedder.embed(imgs), strict=True):
                cached[img_id] = vec
        self._emb = cached
        if missing or len(cached) != len(stamps) or not cache.exists():
            np.savez(cache, **{f"{i}@{stamps[i]}": v for i, v in cached.items()})
        self.version += 1
        for fn in self._listeners:
            fn()

    def _save_cats(self) -> None:
        data = [asdict(c) for c in self._cats]
        (self.root / "cats.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    # ---- queries -------------------------------------------------------------------------
    def cats(self) -> list[Cat]:
        with self._lock:
            return list(self._cats)

    def cat(self, cat_id: str) -> Cat | None:
        return next((c for c in self.cats() if c.id == cat_id), None)

    def images(self, cat_id: str | None = None, mode: str | None = None) -> list[RefImage]:
        with self._lock:
            return [
                r
                for r in self._images.values()
                if (cat_id is None or r.cat_id == cat_id) and (mode is None or r.mode == mode)
            ]

    def image(self, image_id: str) -> RefImage | None:
        with self._lock:
            return self._images.get(image_id)

    def references(self, mode: str) -> tuple[np.ndarray, list[str], list[str]]:
        """Embeddings, cat labels and image ids for one mode (falls back to all images
        when the mode has no references, e.g. no night pictures yet)."""
        with self._lock:
            refs = [r for r in self._images.values() if r.mode == mode and r.id in self._emb]
            if not refs:
                refs = [r for r in self._images.values() if r.id in self._emb]
            if not refs:
                return np.zeros((0, self._embedder.dim), np.float32), [], []
            return (
                np.stack([self._emb[r.id] for r in refs]),
                [r.cat_id for r in refs],
                [r.id for r in refs],
            )

    def similarity_matrix(self, mode: str) -> dict:
        embs, labels, ids = self.references(mode)
        sim = embs @ embs.T if len(ids) else np.zeros((0, 0))
        return {"ids": ids, "labels": labels, "distance": np.round(1 - sim, 4).tolist()}

    # ---- mutations -----------------------------------------------------------------------
    def add_cat(self, cat_id: str, name: str, color: str | None = None) -> Cat:
        check_id(cat_id)
        with self._lock:
            if self.cat(cat_id):
                raise ValueError(f"cat {cat_id} already exists")
            cat = Cat(cat_id, name, color or COLORS[len(self._cats) % len(COLORS)])
            self._cats.append(cat)
            self._save_cats()
            return cat

    def update_cat(self, cat_id: str, name: str | None = None, color: str | None = None) -> Cat:
        with self._lock:
            cat = self.cat(cat_id)
            if not cat:
                raise KeyError(cat_id)
            if name is not None:
                cat.name = name
            if color is not None:
                cat.color = color
            self._save_cats()
            return cat

    def delete_cat(self, cat_id: str) -> None:
        with self._lock:
            if not self.cat(cat_id):
                raise KeyError(cat_id)
            for ref in self.images(cat_id):
                ref.path.unlink(missing_ok=True)
            self._cats = [c for c in self._cats if c.id != cat_id]
            self._save_cats()
            self.reload()

    def add_image(self, cat_id: str, mode: str, img: np.ndarray) -> RefImage:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            if not self.cat(cat_id):
                raise KeyError(cat_id)
            folder = self.root / cat_id / mode
            folder.mkdir(parents=True, exist_ok=True)
            img_id = uuid.uuid4().hex[:12]
            path = folder / f"{img_id}.jpg"
            cv2.imwrite(str(path), downscale(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.reload()
            return self._images[img_id]

    def move_image(self, image_id: str, cat_id: str | None = None, mode: str | None = None) -> RefImage:
        with self._lock:
            ref = self._images.get(image_id)
            if not ref:
                raise KeyError(image_id)
            cat_id, mode = cat_id or ref.cat_id, mode or ref.mode
            if not self.cat(cat_id) or mode not in MODES:
                raise ValueError("unknown cat or mode")
            target = self.root / cat_id / mode / ref.path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            ref.path.replace(target)
            self.reload()
            return self._images[image_id]

    def delete_image(self, image_id: str) -> None:
        with self._lock:
            ref = self._images.get(image_id)
            if not ref:
                raise KeyError(image_id)
            ref.path.unlink(missing_ok=True)
            self.reload()

    # ---- migration -----------------------------------------------------------------------
    def import_legacy_templates(self, templates_dir: Path) -> int:
        """Import ``DayTemplates``/``NightTemplates`` from the old system (``m1.png`` style)."""
        count = 0
        with self._lock:
            for folder, mode in (("DayTemplates", "day"), ("NightTemplates", "night")):
                for p in sorted((templates_dir / folder).glob("*.png")):
                    prefix = p.stem.rstrip("0123456789")
                    name = LEGACY_NAMES.get(prefix)
                    img = cv2.imread(str(p))
                    if name is None or img is None:
                        continue
                    cat_id = name.lower()
                    if not self.cat(cat_id):
                        self.add_cat(cat_id, name)
                    dest = self.root / cat_id / mode
                    dest.mkdir(parents=True, exist_ok=True)
                    out = dest / f"legacy-{mode}-{p.stem}.jpg"  # ids must be unique across folders
                    cv2.imwrite(str(out), downscale(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    count += 1
            self.reload()
        return count
