import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from catalarm import __version__
from catalarm.api.deps import services
from catalarm.services import Services

router = APIRouter(tags=["system"])

STALE_S = 30


@router.get("/health")
def health(strict: bool = False, s: Services = Depends(services)) -> JSONResponse:
    """Always 200 unless ``strict`` is set, then 503 when a camera delivers no frames."""
    cams = []
    for p in list(s.cameras.pipelines.values()):
        age = time.time() - p.reader.frame_ts if p.reader.frame_ts else None
        cams.append(
            {
                "id": p.cfg.id,
                "connected": p.reader.connected,
                "frame_age": round(age, 1) if age is not None else None,
            }
        )
    healthy = all(c["connected"] and c["frame_age"] is not None and c["frame_age"] < STALE_S for c in cams)
    body = {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "embedder": s.gallery.embedder.name,
        "detector": s.worker.detector.name,
        "cameras": cams,
        "ha": {"enabled": s.ha.enabled, "last_ok": s.ha.last_ok, "last_error": s.ha.last_error},
    }
    return JSONResponse(body, status_code=503 if strict and not healthy else 200)
