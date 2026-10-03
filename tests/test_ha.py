import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from catalarm.config import HAConfig
from catalarm.integrations.ha import HAPublisher


class FakeHA:
    def __init__(self):
        self.requests: list[tuple[str, str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, self.headers["Authorization"], body))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.05)
    return cond()


def test_publishes_on_change_and_heartbeat(monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "secret-token")
    ha = FakeHA()
    pub = HAPublisher(HAConfig(url=ha.url, token="${HA_TOKEN}", heartbeat_s=5))
    pub.start()
    try:
        pub.publish("sensor.cat_ai_cathouse", "maugli", {"confidence": 0.9})
        assert wait(lambda: len(ha.requests) == 1)
        path, auth, body = ha.requests[0]
        assert path == "/api/states/sensor.cat_ai_cathouse"
        assert auth == "Bearer secret-token"
        assert body == {"state": "maugli", "attributes": {"confidence": 0.9}}

        # same state again: no immediate request (only the heartbeat re-sends it)
        pub.publish("sensor.cat_ai_cathouse", "maugli", {"confidence": 0.8})
        time.sleep(0.5)
        assert len(ha.requests) == 1
        pub.publish("sensor.cat_ai_cathouse", "no_cat", {})
        assert wait(lambda: len(ha.requests) == 2)
        assert ha.requests[1][2]["state"] == "no_cat"

        pub.cfg = HAConfig(url=ha.url, token="${HA_TOKEN}", heartbeat_s=5)
        pub._sent["sensor.cat_ai_cathouse"] -= 10  # pretend the heartbeat is due
        assert wait(lambda: len(ha.requests) == 3)
        assert pub.last_error is None and pub.last_ok
    finally:
        pub.stop()
        ha.server.shutdown()


def test_errors_are_reported_not_raised():
    pub = HAPublisher(HAConfig(url="http://127.0.0.1:9", token="x"))
    pub.start()
    try:
        pub.publish("sensor.cat_ai_x", "maugli", {})
        assert wait(lambda: pub.last_error is not None)
    finally:
        pub.stop()
