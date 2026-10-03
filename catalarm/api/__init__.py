from fastapi import APIRouter

from catalarm.api import cameras, debug, events, gallery, settings, stats, system

router = APIRouter(prefix="/api/v1")
for module in (system, cameras, gallery, events, stats, settings, debug):
    router.include_router(module.router)
router.include_router(cameras.captures_router)
