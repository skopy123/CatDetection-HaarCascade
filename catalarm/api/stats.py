import time

from fastapi import APIRouter, Depends, Query

from catalarm.api.deps import services
from catalarm.services import Services

router = APIRouter(prefix="/stats", tags=["stats"])


def tz_offset() -> int:
    return time.localtime().tm_gmtoff


@router.get("/summary")
def summary(days: int = Query(7, ge=1, le=365), s: Services = Depends(services)) -> dict:
    return {
        "days": days,
        "cats": [{"id": c.id, "name": c.name, "color": c.color} for c in s.gallery.cats()],
        "daily": s.db.daily_visits(days, tz_offset()),
        "hourly": s.db.hourly_visits(days, tz_offset()),
        "states": s.db.state_counts(days),
    }


@router.get("/perf")
def perf(
    hours: int = Query(24, ge=1, le=24 * 30), camera_id: str | None = None, s: Services = Depends(services)
) -> list[dict]:
    return s.db.perf(hours, camera_id)
