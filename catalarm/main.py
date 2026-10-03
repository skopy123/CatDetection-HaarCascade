"""FastAPI application. Serves the JSON API under /api/v1 and the static frontend at /."""

from __future__ import annotations

import base64
import logging
import os
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from catalarm import __version__
from catalarm.api import router
from catalarm.services import Services, from_env

log = logging.getLogger("catalarm")

DEFAULT_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def create_app(make_services: Callable[[], Services] = from_env, start: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = make_services()
        app.state.services = s
        if start:
            s.start()
        log.info("CatAlarm %s started, cameras: %s", __version__, [c.id for c in s.config.config.cameras])
        yield
        s.stop()

    app = FastAPI(title="CatAlarm", version=__version__, lifespan=lifespan)
    app.include_router(router)

    user, password = os.environ.get("AUTH_USER"), os.environ.get("AUTH_PASSWORD")
    if user and password:
        expected = base64.b64encode(f"{user}:{password}".encode()).decode()

        @app.middleware("http")
        async def basic_auth(request: Request, call_next):
            if request.url.path != "/api/v1/health":
                given = request.headers.get("authorization", "")
                if not secrets.compare_digest(given, f"Basic {expected}"):
                    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="CatAlarm"'})
            return await call_next(request)

    frontend = Path(os.environ.get("CATALARM_FRONTEND_DIR", DEFAULT_FRONTEND))
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


def run() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_config=None)


if __name__ == "__main__":
    run()
