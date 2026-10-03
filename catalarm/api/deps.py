import cv2
import numpy as np
from fastapi import HTTPException, Request, UploadFile

from catalarm.cameras.pipeline import CameraPipeline
from catalarm.services import Services


def services(request: Request) -> Services:
    return request.app.state.services


def pipeline(s: Services, camera_id: str) -> CameraPipeline:
    p = s.cameras.get(camera_id)
    if p is None:
        if s.config.config.camera(camera_id):
            raise HTTPException(409, f"camera {camera_id} is disabled")
        raise HTTPException(404, f"camera {camera_id} not found")
    return p


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a valid image")
    return img
