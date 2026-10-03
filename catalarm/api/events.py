import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from catalarm.api.deps import services
from catalarm.services import Services

router = APIRouter(tags=["events"])


class Relabel(BaseModel):
    cat_id: str | None


def event_view(e: dict) -> dict:
    snap = e.get("snapshot")
    e = dict(e)
    e["snapshot_url"] = f"/api/v1/events/{e['id']}/snapshot.jpg" if snap else None
    e["crop_url"] = f"/api/v1/events/{e['id']}/snapshot.jpg?crop=true" if snap else None
    return e


@router.get("/events")
def list_events(
    camera_id: str | None = None,
    cat_id: str | None = None,
    state: str | None = None,
    before: float | None = None,
    limit: int = 50,
    include_idle: bool = False,
    s: Services = Depends(services),
) -> list[dict]:
    rows = s.db.events(camera_id, cat_id, state, before, min(limit * (1 if include_idle else 4), 1000))
    if not include_idle:
        rows = [r for r in rows if r["state"] not in ("no_motion", "no_cat")]
    return [event_view(r) for r in rows[:limit]]


@router.get("/events/{event_id}/snapshot.jpg")
def event_snapshot(event_id: int, crop: bool = False, s: Services = Depends(services)) -> FileResponse:
    e = s.db.event(event_id)
    if not e or not e["snapshot"]:
        raise HTTPException(404, "no snapshot")
    path = s.snapshots_dir / f"{e['snapshot']}{'_crop' if crop else ''}.jpg"
    if crop and not path.exists():
        path = s.snapshots_dir / f"{e['snapshot']}.jpg"
    if not path.is_file():
        raise HTTPException(404, "snapshot file missing")
    return FileResponse(path, media_type="image/jpeg")


@router.patch("/events/{event_id}")
def relabel_event(event_id: int, body: Relabel, s: Services = Depends(services)) -> dict:
    if not s.db.event(event_id):
        raise HTTPException(404, "event not found")
    if body.cat_id and not s.gallery.cat(body.cat_id):
        raise HTTPException(400, "unknown cat")
    s.db.relabel_event(event_id, body.cat_id)
    return event_view(s.db.event(event_id))


@router.get("/live")
async def live(request: Request, s: Services = Depends(services)) -> StreamingResponse:
    """Server-Sent Events: camera status changes and new events."""

    async def gen():
        cams = [p.status() for p in list(s.cameras.pipelines.values())]
        yield f"data: {json.dumps({'type': 'hello', 'cameras': cams}, default=str)}\n\n"
        async for msg in s.hub.subscribe():
            if await request.is_disconnected():
                break
            yield ": keep-alive\n\n" if msg is None else f"data: {msg}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})
