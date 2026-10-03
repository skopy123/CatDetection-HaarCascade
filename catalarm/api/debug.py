from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from catalarm.api.deps import read_image, services
from catalarm.services import Services
from catalarm.vision.benchmark import benchmark
from catalarm.vision.embedder import available_embedders, load_embedder
from catalarm.vision.nightvision import is_night_vision

router = APIRouter(tags=["debug"])


@router.post("/debug/identify")
async def identify(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    s: Services = Depends(services),
) -> dict:
    """Run detection + identification on an uploaded image (no tracker, no HA, no stats)."""
    img = await read_image(file)
    night = is_night_vision(img) if mode == "auto" else mode == "night"
    analysis = await run_in_threadpool(s.worker.analyse, img, night)
    h, w = img.shape[:2]
    return analysis.to_dict() | {"night": night, "width": w, "height": h}


class BenchmarkRequest(BaseModel):
    embedders: list[str] | None = None  # default: all available models


@router.post("/benchmark")
def run_benchmark(body: BenchmarkRequest, s: Services = Depends(services)) -> dict:
    available = available_embedders(s.models_dir)
    names = body.embedders or [n for n in available if n != "histogram"] or ["histogram"]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise HTTPException(400, f"not available: {', '.join(unknown)}")
    th = s.config.config.thresholds
    results = []
    for name in names:
        emb = s.gallery.embedder if name == s.gallery.embedder.name else load_embedder(name, s.models_dir)
        results.append(benchmark(s.gallery, emb, th))
    return {"current": s.gallery.embedder.name, "thresholds": th.model_dump(), "results": results}
