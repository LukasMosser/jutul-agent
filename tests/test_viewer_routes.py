"""The viewer routes work on any app, given only the two session lookups.

The web server's own tests cover these routes inside ``create_app``; these pin
the contract another front end (the MCP server) relies on: install them on a
bare app with ``live_session``/``output_dir`` and artifacts, popouts, and the
live-plot proxy all resolve through those lookups alone.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jutul_agent.interfaces.server.viewer import add_viewer_routes


@pytest.fixture
def upstream_port():
    """A stand-in for a session's Bonito server: echoes the path it was asked for."""

    class _Echo(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = f"bonito saw {self.path}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _client(tmp_path: Path, sessions: dict[str, SimpleNamespace]) -> TestClient:
    app = FastAPI()
    add_viewer_routes(
        app,
        live_session=sessions.get,
        output_dir=lambda sid: tmp_path / sid if sid in sessions else None,
    )
    return TestClient(app)


def test_artifacts_resolve_through_output_dir(tmp_path: Path) -> None:
    artifacts = tmp_path / "s1" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "plot.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "s1" / "secret.txt").write_text("outside artifacts")
    client = _client(tmp_path, {"s1": SimpleNamespace(web_plot_port=None)})

    assert client.get("/sessions/s1/artifacts/plot.png").content == b"\x89PNG fake"
    assert client.get("/sessions/s1/artifacts/missing.png").status_code == 404
    # Nothing outside the artifacts dir is reachable.
    assert client.get("/sessions/s1/artifacts/..%2Fsecret.txt").status_code == 404
    assert client.get("/sessions/nope/artifacts/plot.png").status_code == 404


def test_live_proxy_forwards_to_the_sessions_port(tmp_path: Path, upstream_port: int) -> None:
    sessions = {
        "live": SimpleNamespace(web_plot_port=upstream_port),
        "idle": SimpleNamespace(web_plot_port=None),
    }
    client = _client(tmp_path, sessions)

    response = client.get("/live/live/viz/plot-1?x=1")
    assert response.status_code == 200
    assert response.text == "bonito saw /viz/plot-1?x=1"
    # No live server yet, or no such session: a plain 404, never a dial.
    assert client.get("/live/idle/viz/plot-1").status_code == 404
    assert client.get("/live/nope/viz/plot-1").status_code == 404


def test_popout_page_stages_the_live_route(tmp_path: Path) -> None:
    client = _client(tmp_path, {"s1": SimpleNamespace(web_plot_port=None)})

    page = client.get("/popout/s1", params={"route": "plot-1", "w": 800, "h": 600})
    assert page.status_code == 200
    assert "/live/s1/viz/plot-1" in page.text
    assert client.get("/popout/s1", params={"route": "../x", "w": 800, "h": 600}).status_code == 404
    # A re-fit for a session that isn't live is a no-op, not an error.
    refit = client.post("/popout/nope/refit", params={"route": "plot-1", "w": 800, "h": 600})
    assert refit.status_code == 204
