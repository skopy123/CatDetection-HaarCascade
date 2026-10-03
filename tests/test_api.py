import time

from .conftest import TEST_IMG, fake_camera


def wait_for(fn, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if value := fn():
            return value
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def test_health_and_frontend(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["embedder"] == "histogram"
    assert r.json()["detector"] == "full-frame"
    assert client.get("/").status_code == 200


def test_camera_crud_and_stream(client):
    r = client.post("/api/v1/cameras", json=fake_camera("cathouse"))
    assert r.status_code == 201
    assert client.post("/api/v1/cameras", json=fake_camera("cathouse")).status_code == 409
    wait_for(lambda: client.get("/api/v1/cameras/cathouse/snapshot.jpg").status_code == 200)
    r = client.get("/api/v1/cameras/cathouse/snapshot.jpg?overlay=all")
    assert r.headers["content-type"] == "image/jpeg" and len(r.content) > 1000

    cam = client.get("/api/v1/cameras/cathouse").json()
    cam["name"] = "Renamed"
    assert client.put("/api/v1/cameras/cathouse", json=cam).json()["name"] == "Renamed"
    motion = {"mask": [[[0, 0], [1, 0], [1, 0.1]]], "min_area_ratio": 0.01, "threshold": 40, "enabled": True}
    assert client.put("/api/v1/cameras/cathouse/motion", json=motion).status_code == 200

    assert client.post("/api/v1/cameras/cathouse/pause", json={"paused": True}).json() == {"paused": True}
    assert client.post("/api/v1/cameras/cathouse/step").status_code == 200
    cap = client.post("/api/v1/cameras/cathouse/capture").json()
    assert client.get(cap["url"]).status_code == 200
    assert client.delete("/api/v1/cameras/cathouse").status_code == 204
    assert client.get("/api/v1/cameras/cathouse").status_code == 404


def test_camera_url_password_is_masked(client):
    cam = {"id": "garden", "url": "rtsp://admin:hunter2@10.0.0.9/stream", "enabled": False}
    assert client.post("/api/v1/cameras", json=cam).json()["url"] == "rtsp://admin:***@10.0.0.9/stream"
    view = client.get("/api/v1/cameras/garden").json()
    view["name"] = "Garden"
    client.put("/api/v1/cameras/garden", json=view)
    services = client.app.state.services
    assert services.config.config.camera("garden").url == "rtsp://admin:hunter2@10.0.0.9/stream"


def test_two_cameras_share_gallery_and_have_separate_state(client):
    client.post("/api/v1/cameras", json=fake_camera("house"))
    client.post("/api/v1/cameras", json=fake_camera("garden"))
    s = client.app.state.services
    pipes = s.cameras.pipelines
    wait_for(lambda: len(pipes) == 2 and all(p.last_analysis for p in pipes.values()))
    cams = client.get("/api/v1/cameras").json()
    assert {c["entity_id"] for c in cams} == {"sensor.cat_ai_house", "sensor.cat_ai_garden"}
    assert s.cameras.get("house").tracker is not s.cameras.get("garden").tracker
    health = client.get("/api/v1/health").json()
    assert {c["id"] for c in health["cameras"]} == {"house", "garden"}


def test_gallery_endpoints(client):
    cats = client.get("/api/v1/gallery/cats").json()
    assert {c["id"] for c in cats} == {"amalka", "gita", "kocour", "maugli"}
    assert client.post("/api/v1/gallery/cats", json={"id": "micka", "name": "Micka"}).status_code == 201
    assert client.post("/api/v1/gallery/cats", json={"id": "Bad Id", "name": "x"}).status_code == 400
    with open(TEST_IMG / "gita1.jpg", "rb") as f:
        r = client.post(
            "/api/v1/gallery/images",
            files={"file": ("g.jpg", f, "image/jpeg")},
            data={"cat_id": "micka", "mode": "day"},
        )
    assert r.status_code == 201
    img = r.json()
    assert client.get(img["url"]).status_code == 200
    moved = client.patch(f"/api/v1/gallery/images/{img['id']}", json={"mode": "night"}).json()
    assert moved["mode"] == "night"
    m = client.get("/api/v1/gallery/matrix?mode=day").json()
    assert len(m["distance"]) == len(m["ids"]) > 0
    assert client.delete("/api/v1/gallery/cats/micka").status_code == 204
    assert client.get(img["url"]).status_code == 404


def test_identify_and_benchmark(client):
    with open(TEST_IMG / "maugli1.jpg", "rb") as f:
        r = client.post("/api/v1/debug/identify", files={"file": ("m.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["best"]["cat_id"] in {"amalka", "gita", "kocour", "maugli"}
    assert body["night"] is False and len(body["detections"]) == 1

    r = client.post("/api/v1/benchmark", json={})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["embedder"] == "histogram"
    assert set(res["modes"]) == {"day", "night"}
    assert 0 <= res["modes"]["day"]["top1_accuracy"] <= 1


def test_settings_mask_token(client):
    r = client.put("/api/v1/settings", json={"ha": {"url": "http://ha:8123", "token": "abc"}})
    assert r.json()["ha"]["token"] == "***"
    r = client.put("/api/v1/settings", json={"ha": {"url": "http://ha2:8123", "token": "***"}})
    s = client.app.state.services
    assert s.config.config.ha.token == "abc" and s.config.config.ha.url == "http://ha2:8123"
    assert client.put("/api/v1/settings", json={"embedder": "nope"}).status_code == 400
    th = client.put("/api/v1/settings", json={"thresholds": {"match": 0.25}}).json()["thresholds"]
    assert th["match"] == 0.25 and th["vote_k"] == 3
    names = {e["name"]: e["available"] for e in client.get("/api/v1/settings/embedders").json()}
    assert names["histogram"] and not names["dinov2_vits14_reg"]


def test_events_and_stats(client):
    s = client.app.state.services
    now = time.time()
    s.db.add_event("house", "maugli", "maugli", 0.9, 0.2, False, None, now)
    s.db.add_event("house", "no_motion", None, None, None, False, None, now + 1)
    events = client.get("/api/v1/events").json()
    assert [e["state"] for e in events] == ["maugli"]
    assert len(client.get("/api/v1/events?include_idle=true").json()) == 2
    r = client.patch(f"/api/v1/events/{events[0]['id']}", json={"cat_id": "gita"})
    assert r.json()["cat_id"] == "gita"
    summary = client.get("/api/v1/stats/summary?days=7").json()
    assert {c["id"] for c in summary["cats"]} >= {"maugli"}
    assert client.get("/api/v1/stats/perf").status_code == 200
