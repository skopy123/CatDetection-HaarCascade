"""Frame sources. A reader thread keeps only the most recent frame (no growing queue).

URL forms:
* ``rtsp://...``, ``http://...`` or a video file path - read with OpenCV/FFmpeg
  (video files loop, handy for replaying recorded clips)
* ``fake:<dir-or-glob>[?fps=N]`` - cycles through still images (tests, demos)
"""

from __future__ import annotations

import glob
import logging
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs

import cv2
import numpy as np

log = logging.getLogger(__name__)


class FrameSource:
    def open(self) -> bool: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class OpenCVSource(FrameSource):
    def __init__(self, url: str):
        self.url = url
        self.is_file = Path(url).is_file()
        self._cap: cv2.VideoCapture | None = None
        self._frame_interval = 0.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if self.is_file:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 15
            self._frame_interval = 1.0 / fps
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok and self.is_file:  # loop recorded clips
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if self._frame_interval:
            time.sleep(self._frame_interval)
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FakeSource(FrameSource):
    def __init__(self, spec: str):
        pattern, _, query = spec.partition("?")
        self.fps = float(parse_qs(query).get("fps", ["5"])[0])
        p = Path(pattern)
        files = sorted(str(f) for f in p.iterdir()) if p.is_dir() else sorted(glob.glob(pattern))
        self.files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self._frames: list[np.ndarray] = []
        self._i = 0

    def open(self) -> bool:
        self._frames = [img for f in self.files if (img := cv2.imread(f)) is not None]
        return bool(self._frames)

    def read(self) -> np.ndarray | None:
        time.sleep(1.0 / self.fps)
        if not self._frames:
            return None
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return frame.copy()

    def close(self) -> None:
        self._frames = []


def make_source(url: str) -> FrameSource:
    if url.startswith("fake:"):
        return FakeSource(url[5:])
    return OpenCVSource(url)


class CameraReader:
    """Background thread that reads frames and reconnects with exponential backoff."""

    def __init__(self, camera_id: str, url: str):
        self.camera_id = camera_id
        self.url = url
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self.frame_ts = 0.0
        self.frame_no = 0
        self.fps = 0.0
        self.connected = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"reader-{camera_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return self._frame, self.frame_no

    def _run(self) -> None:
        delay = 2.0
        while not self._stop.is_set():
            source = make_source(self.url)
            if not source.open():
                self.connected, self.error = False, "cannot open stream"
                log.warning("camera %s: cannot open stream, retry in %.0fs", self.camera_id, delay)
                source.close()
                self._stop.wait(delay)
                delay = min(delay * 2, 60)
                continue
            log.info("camera %s: connected", self.camera_id)
            self.connected, self.error, delay = True, None, 2.0
            failures, count, t0 = 0, 0, time.time()
            while not self._stop.is_set():
                frame = source.read()
                if frame is None:
                    failures += 1
                    if failures > 10:
                        self.error = "stream stopped delivering frames"
                        log.warning("camera %s: %s, reconnecting", self.camera_id, self.error)
                        break
                    self._stop.wait(0.5)
                    continue
                failures = 0
                with self._lock:
                    self._frame = frame
                    self.frame_no += 1
                    self.frame_ts = time.time()
                count += 1
                if count >= 50:
                    now = time.time()
                    self.fps, count, t0 = count / (now - t0), 0, now
            source.close()
            self.connected = False
