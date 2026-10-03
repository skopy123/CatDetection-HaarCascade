from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from catalarm.api.deps import services
from catalarm.config import AppConfig, DetectorConfig, HAConfig, Thresholds
from catalarm.services import Services
from catalarm.vision.embedder import MODEL_SPECS, available_embedders

router = APIRouter(prefix="/settings", tags=["settings"])

MASKED = "***"


class SettingsUpdate(BaseModel):
    thresholds: Thresholds | None = None
    detector: DetectorConfig | None = None
    embedder: str | None = None
    ha: HAConfig | None = None
    motion_fps: float | None = None
    process_fps: float | None = None
    retention_days: int | None = None
    public_url: str | None = None


def settings_view(cfg: AppConfig) -> dict:
    data = cfg.model_dump(exclude={"cameras"})
    token = cfg.ha.token or ""
    # a ${VAR} reference is not secret, a literal token is
    data["ha"]["token"] = token if token.startswith("${") or not token else MASKED
    return data


@router.get("")
def get_settings(s: Services = Depends(services)) -> dict:
    return settings_view(s.config.config)


@router.put("")
def update_settings(body: SettingsUpdate, s: Services = Depends(services)) -> dict:
    if body.embedder and body.embedder not in available_embedders(s.models_dir):
        raise HTTPException(400, f"embedder {body.embedder} is not available")

    def apply(cfg: AppConfig) -> None:
        for field in body.model_fields_set:
            value = getattr(body, field)
            if field == "ha" and value is not None and value.token == MASKED:
                value = value.model_copy(update={"token": cfg.ha.token})
            if value is not None:
                setattr(cfg, field, value)

    try:
        cfg = s.config.update(apply)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return settings_view(cfg)


@router.get("/embedders")
def embedders(s: Services = Depends(services)) -> list[dict]:
    available = set(available_embedders(s.models_dir))
    out = [
        {"name": spec.name, "description": spec.description, "available": spec.name in available}
        for spec in MODEL_SPECS.values()
    ]
    out.append(
        {"name": "histogram", "description": "Color histogram (no model, testing only)", "available": True}
    )
    return out
