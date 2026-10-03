"""Application configuration.

Settings live in ``<data_dir>/config.yaml``. String values may reference environment
variables as ``${NAME}``; they are stored unexpanded (so secrets never get written back
to the file) and expanded only where they are used, via :func:`expand`.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import AfterValidator, BaseModel, Field

log = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def expand(value: str | None) -> str | None:
    """Replace ``${VAR}`` references with environment values (missing -> empty)."""
    if value is None:
        return None
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def data_dir() -> Path:
    return Path(os.environ.get("CATALARM_DATA_DIR", "/data"))


def models_dir() -> Path:
    return Path(os.environ.get("CATALARM_MODELS_DIR", "/models"))


def check_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError("id must be 1-32 chars of a-z, 0-9, '_' or '-'")
    return value


Polygon = list[tuple[float, float]]
EntityId = Annotated[str, AfterValidator(check_id)]


class MotionConfig(BaseModel):
    # polygons (normalized 0..1 coordinates) that are ignored by motion detection,
    # e.g. the camera's timestamp overlay
    mask: list[Polygon] = Field(default_factory=list)
    # fraction of (downscaled) frame pixels that must change to count as motion
    min_area_ratio: float = Field(0.0026, ge=0, le=1)
    threshold: int = Field(30, ge=1, le=254)
    enabled: bool = True


class CameraConfig(BaseModel):
    id: EntityId
    name: str = ""
    url: str
    enabled: bool = True
    motion: MotionConfig = Field(default_factory=MotionConfig)
    ha_entity: str | None = None

    @property
    def entity_id(self) -> str:
        return self.ha_entity or f"sensor.cat_ai_{self.id.replace('-', '_')}"


class Thresholds(BaseModel):
    match: float = Field(0.30, ge=0, le=2)  # cosine distance for a confirmed match
    unconfirmed: float = Field(0.50, ge=0, le=2)  # cosine distance for "unconfirmed"
    margin: float = Field(0.05, ge=0, le=2)  # required gap between best and second cat
    knn_k: int = Field(1, ge=1, le=10)  # score a cat by mean of its k closest references
    vote_k: int = Field(3, ge=1, le=20)  # cat confirmed when it wins vote_k ...
    vote_n: int = Field(5, ge=1, le=50)  # ... of the last vote_n observations
    hold_s: float = Field(20, ge=0, le=600)  # keep reporting a cat this long after last seen


class DetectorConfig(BaseModel):
    enabled: bool = True
    model: str = "yolo11s"
    confidence: float = Field(0.35, ge=0.01, le=1)
    padding: float = Field(0.10, ge=0, le=1)  # enlarge crop box by this fraction


class HAConfig(BaseModel):
    url: str | None = None
    token: str | None = "${HA_TOKEN}"
    heartbeat_s: float = Field(60, ge=5)
    enabled: bool = True


class AppConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    embedder: str = "dinov2_vits14_reg"
    ha: HAConfig = Field(default_factory=HAConfig)
    motion_fps: float = Field(5.0, gt=0, le=30)  # rate frames are fed to motion detection
    process_fps: float = Field(2.0, gt=0, le=15)  # max inference rate per camera
    retention_days: int = Field(30, ge=1)
    public_url: str | None = None  # base URL used for snapshot links sent to HA

    def camera(self, camera_id: str) -> CameraConfig | None:
        return next((c for c in self.cameras if c.id == camera_id), None)


# mask of the original single-camera setup, in 640x360 pixel coordinates
LEGACY_MASK_PX = [(0, 0), (0, 360), (80, 360), (150, 18), (227, 18), (227, 0)]


def default_config() -> AppConfig:
    """Config created on first start; migrates the old single-camera env setup."""
    cfg = AppConfig(ha=HAConfig(url=os.environ.get("HA_URL") or None))
    url = os.environ.get("RTSP_URL") or os.environ.get("RTSP_STREAM")
    if url:
        mask = [[(x / 640, y / 360) for x, y in LEGACY_MASK_PX]]
        env_name = "RTSP_URL" if os.environ.get("RTSP_URL") else "RTSP_STREAM"
        cfg.cameras.append(
            CameraConfig(
                id="cathouse",
                name="Cat house",
                url=f"${{{env_name}}}",
                motion=MotionConfig(mask=mask),
            )
        )
    return cfg


class ConfigStore:
    """Thread-safe holder of the current config with change notification."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._listeners: list[Callable[[AppConfig], None]] = []
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self._cfg = AppConfig.model_validate(raw)
        else:
            self._cfg = default_config()
            self._write(self._cfg)
            log.info("created default config at %s", path)

    @property
    def config(self) -> AppConfig:
        with self._lock:
            return self._cfg.model_copy(deep=True)

    def subscribe(self, fn: Callable[[AppConfig], None]) -> None:
        self._listeners.append(fn)

    def update(self, fn: Callable[[AppConfig], None]) -> AppConfig:
        """Apply ``fn`` to a copy of the config, validate, persist and notify."""
        with self._lock:
            cfg = self._cfg.model_copy(deep=True)
            fn(cfg)
            cfg = AppConfig.model_validate(cfg.model_dump())
            self._write(cfg)
            self._cfg = cfg
        for listener in self._listeners:
            try:
                listener(cfg.model_copy(deep=True))
            except Exception:
                log.exception("config listener failed")
        return cfg

    def _write(self, cfg: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = cfg.model_dump(mode="json")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.path)
