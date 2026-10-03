from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from catalarm.api.deps import read_image, services
from catalarm.gallery.store import MODES
from catalarm.services import Services

router = APIRouter(prefix="/gallery", tags=["gallery"])


class CatCreate(BaseModel):
    id: str
    name: str
    color: str | None = None


class CatUpdate(BaseModel):
    name: str | None = None
    color: str | None = None


class ImageUpdate(BaseModel):
    cat_id: str | None = None
    mode: str | None = None


def cat_view(s: Services, cat) -> dict:
    counts = {m: len(s.gallery.images(cat.id, m)) for m in MODES}
    return {"id": cat.id, "name": cat.name, "color": cat.color, "images": counts}


def image_view(ref) -> dict:
    return ref.to_dict() | {"url": f"/api/v1/gallery/images/{ref.id}.jpg"}


@router.get("/cats")
def list_cats(s: Services = Depends(services)) -> list[dict]:
    return [cat_view(s, c) for c in s.gallery.cats()]


@router.post("/cats", status_code=201)
def create_cat(body: CatCreate, s: Services = Depends(services)) -> dict:
    try:
        return cat_view(s, s.gallery.add_cat(body.id, body.name, body.color))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.patch("/cats/{cat_id}")
def update_cat(cat_id: str, body: CatUpdate, s: Services = Depends(services)) -> dict:
    try:
        return cat_view(s, s.gallery.update_cat(cat_id, body.name, body.color))
    except KeyError as e:
        raise HTTPException(404, "cat not found") from e


@router.delete("/cats/{cat_id}", status_code=204)
def delete_cat(cat_id: str, s: Services = Depends(services)) -> None:
    try:
        s.gallery.delete_cat(cat_id)
    except KeyError as e:
        raise HTTPException(404, "cat not found") from e


@router.get("/images")
def list_images(cat_id: str | None = None, mode: str | None = None, s: Services = Depends(services)) -> list:
    return [image_view(r) for r in s.gallery.images(cat_id, mode)]


@router.post("/images", status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    cat_id: str = Form(...),
    mode: str = Form(...),
    s: Services = Depends(services),
) -> dict:
    img = await read_image(file)
    try:
        return image_view(s.gallery.add_image(cat_id, mode, img))
    except KeyError as e:
        raise HTTPException(404, "cat not found") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/images/{image_id}.jpg")
def get_image(image_id: str, s: Services = Depends(services)) -> FileResponse:
    ref = s.gallery.image(image_id)
    if not ref:
        raise HTTPException(404, "image not found")
    return FileResponse(ref.path, media_type="image/jpeg")


@router.patch("/images/{image_id}")
def update_image(image_id: str, body: ImageUpdate, s: Services = Depends(services)) -> dict:
    try:
        return image_view(s.gallery.move_image(image_id, body.cat_id, body.mode))
    except KeyError as e:
        raise HTTPException(404, "image not found") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: str, s: Services = Depends(services)) -> None:
    try:
        s.gallery.delete_image(image_id)
    except KeyError as e:
        raise HTTPException(404, "image not found") from e


@router.get("/matrix")
def similarity_matrix(mode: str = "day", s: Services = Depends(services)) -> dict:
    if mode not in MODES:
        raise HTTPException(400, "mode must be day or night")
    return s.gallery.similarity_matrix(mode) | {"embedder": s.gallery.embedder.name}
