from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catalarm.main import create_app
from catalarm.services import Services

REPO = Path(__file__).resolve().parent.parent
TEST_IMG = REPO / "testImg"
TEMPLATES = REPO / "templates"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"  # empty: histogram embedder + full-frame detector
    d.mkdir()
    return d


@pytest.fixture
def services(data_dir: Path, models_dir: Path, monkeypatch) -> Services:
    monkeypatch.delenv("RTSP_URL", raising=False)
    monkeypatch.delenv("RTSP_STREAM", raising=False)
    s = Services(data_dir, models_dir, TEMPLATES)
    yield s
    s.stop()


@pytest.fixture
def client(services: Services) -> TestClient:
    app = create_app(lambda: services)
    with TestClient(app) as c:
        yield c


def fake_camera(camera_id: str, fps: int = 20) -> dict:
    return {"id": camera_id, "name": camera_id.title(), "url": f"fake:{TEST_IMG}?fps={fps}"}
