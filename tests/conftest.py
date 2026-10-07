from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from mbt_server.services.buggy_app import create_app as create_buggy
from mbt_server.services.reference_app import create_app as create_reference

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(url: str) -> None:
    deadline = time.time() + 5
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=0.2, trust_env=False)
            if response.status_code < 500:
                return
        except httpx.TransportError as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"server did not become ready: {last_error}")


@pytest.fixture(scope="session")
def server_factory():
    servers = []

    def start(app_name: str, app) -> str:
        port = _free_port()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        _wait_for(base_url + "/docs")
        servers.append((server, thread))
        return base_url

    yield start

    for server, thread in reversed(servers):
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def reference_url(server_factory) -> str:
    return server_factory("reference", create_reference())


@pytest.fixture(scope="session")
def buggy_url(server_factory) -> str:
    return server_factory("buggy", create_buggy())


@pytest.fixture
def order_model():
    with (ROOT / "fixtures" / "order_model.json").open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)
