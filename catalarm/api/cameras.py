import asyncio
import re
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from catalarm.api.deps import pipeline, services
from catalarm.cameras.pipeline import OVERLAYS
from catalarm.config import CameraConfig, MotionConfig
from catalarm.services import Services

router = APIRouter(prefix="/cameras", tags=["cameras"])

_CRED_RE = re.compile(r"(://[^:/@]+:)([^@]+)(@)")
MASKED = "***"


def mask_url(url: str) -> str:
    return _CRED_RE.sub(rf"\g<1>{MASKED}\g<3>", url)


def camera_view(s: Services, cam: CameraConfig) -> dict:
    data = cam.model_dump()
    data["url"] = mask_url(cam.url)
    data["entity_id"] = cam.entity_id
    p = s.cameras.get(cam.id)
    data["status"] = p.status() if p else None
    return data


def parse_overlays(overlay: str | None) -> set[str]:
    if not overlay:
        return set()
    if overlay == "all":
        return set(OVERLAYS)
    return {o for o in overlay.split(",") if o in OVERLAYS}


@router.get("")
def list_cameras(s: Services = Depends(services)) -> list[dict]:
    return [camera_view(s, c) for c in s.config.config.cameras]


@router.post("", status_code=201)
def create_camera(cam: CameraConfig, s: Services = Depends(services)) -> dict:
    if s.config.config.camera(cam.id):
        raise HTTPException(409, f"camera {cam.id} already exists")
    s.config.update(lambda c: c.cameras.append(cam))
    return camera_view(s, cam)


@router.get("/{camera_id}")
def get_camera(camera_id: str, s: Services = Depends(services)) -> dict:
    cam = s.config.config.camera(camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    return camera_view(s, cam)


@router.put("/{camera_id}")
def update_camera(camera_id: str, cam: CameraConfig, s: Services = Depends(services)) -> dict:
    old = s.config.config.camera(camera_id)
    if not old:
        raise HTTPException(404, "camera not found")
    if cam.id != camera_id:
        raise HTTPException(400, "camera id cannot be changed")
    if MASKED in cam.url:  # the client sent back the masked URL, keep the stored password
        cam.url = old.url

    def apply(c):
        c.cameras = [cam if x.id == camera_id else x for x in c.cameras]

    s.config.update(apply)
    return camera_view(s, cam)


@router.put("/{camera_id}/motion")
def update_motion(camera_id: str, motion: MotionConfig, s: Services = Depends(services)) -> dict:
    if not s.config.config.camera(camera_id):
        raise HTTPException(404, "camera not found")

    def apply(c):
        c.camera(camera_id).motion = motion

    s.config.update(apply)
    return motion.model_dump()


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: str, s: Services = Depends(services)) -> None:
    if not s.config.config.camera(camera_id):
        raise HTTPException(404, "camera not found")
    s.config.update(lambda c: setattr(c, "cameras", [x for x in c.cameras if x.id != camera_id]))


@router.get("/{camera_id}/status")
def camera_status(camera_id: str, s: Services = Depends(services)) -> dict:
    return pipeline(s, camera_id).status()


@router.get("/{camera_id}/snapshot.jpg")
async def snapshot(camera_id: str, overlay: str | None = None, s: Services = Depends(services)) -> Response:
    p = pipeline(s, camera_id)
    data = await run_in_threadpool(p.frame_jpeg, parse_overlays(overlay))
    if data is None:
        raise HTTPException(503, "no frame yet")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{camera_id}/motion.png")
def motion_mask(camera_id: str, s: Services = Depends(services)) -> Response:
    data = pipeline(s, camera_id).motion_jpeg()
    if data is None:
        raise HTTPException(503, "no motion data yet")
    return Response(data, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/{camera_id}/stream.mjpg")
async def stream(
    camera_id: str,
    request: Request,
    overlay: str | None = None,
    fps: float = Query(5, gt=0, le=15),
    s: Services = Depends(services),
) -> StreamingResponse:
    p = pipeline(s, camera_id)
    overlays = parse_overlays(overlay)

    async def frames():
        while not await request.is_disconnected():
            t = time.monotonic()
            data = await run_in_threadpool(p.frame_jpeg, overlays)
            if data:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(data)).encode()
                    + b"\r\n\r\n"
                    + data
                    + b"\r\n"
                )
            await asyncio.sleep(max(0.0, 1 / fps - (time.monotonic() - t)))

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


class PauseRequest(BaseModel):
    paused: bool


@router.post("/{camera_id}/pause")
def pause(camera_id: str, body: PauseRequest, s: Services = Depends(services)) -> dict:
    p = pipeline(s, camera_id)
    p.pause(body.paused)
    return {"paused": p.paused}


@router.post("/{camera_id}/step")
def step(camera_id: str, s: Services = Depends(services)) -> dict:
    p = pipeline(s, camera_id)
    p.step()
    return {"ok": True}


@router.post("/{camera_id}/capture", status_code=201)
def capture(camera_id: str, s: Services = Depends(services)) -> dict:
    """Store the current raw frame so it can be cropped into the gallery later."""
    data = pipeline(s, camera_id).frame_jpeg(None, quality=95)
    if data is None:
        raise HTTPException(503, "no frame yet")
    s.captures_dir.mkdir(parents=True, exist_ok=True)
    capture_id = f"{camera_id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    (s.captures_dir / f"{capture_id}.jpg").write_bytes(data)
    return {"id": capture_id, "url": f"/api/v1/captures/{capture_id}.jpg"}


captures_router = APIRouter(prefix="/captures", tags=["cameras"])


@captures_router.get("")
def list_captures(s: Services = Depends(services)) -> list[dict]:
    files = sorted(s.captures_dir.glob("*.jpg"), reverse=True) if s.captures_dir.exists() else []
    return [{"id": f.stem, "url": f"/api/v1/captures/{f.name}"} for f in files[:100]]


@captures_router.get("/{capture_id}.jpg")
def get_capture(capture_id: str, s: Services = Depends(services)) -> FileResponse:
    path = s.captures_dir / f"{capture_id}.jpg"
    if "/" in capture_id or "\\" in capture_id or not path.is_file():
        raise HTTPException(404, "capture not found")
    return FileResponse(path)


@captures_router.delete("/{capture_id}", status_code=204)
def delete_capture(capture_id: str, s: Services = Depends(services)) -> None:
    path = s.captures_dir / f"{capture_id}.jpg"
    if "/" in capture_id or "\\" in capture_id or not path.is_file():
        raise HTTPException(404, "capture not found")
    path.unlink()
