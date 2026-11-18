"""Owned synthetic data and a loopback model; never call the configured real provider."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Generator

import pytest

import memory_core
from memory_core import MemoryStore
from memory_extract import Extracted
from memory_models import IngestRequest, MemoryConfig, MemoryRecord


@pytest.fixture
def model_server(tmp_path: Path):
    requests = []
    control = {"status": 200, "body": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "key": self.headers.get("api-key"), "body": body})
            if "input" in body:
                response = control.get("embedding_body")
                if response is None:
                    vectors = control.get("vectors", {})
                    response = {"data": [
                        {"index": index, "embedding": vectors.get(text, [0.0, 1.0])}
                        for index, text in enumerate(body["input"])
                    ]}
                status = control.get("embedding_status", control["status"])
            else:
                text = body["messages"][-1]["content"]
                extraction = (
                    {"store": False, "reason": "Greeting has no durable information."}
                    if text == "Hello!"
                    else {"store": True, "kind": "event" if text.startswith("Decision:") else "state",
                          "content": text, "evidence": text}
                )
                response = control["body"]
                if response is None:
                    response = {"choices": [{"message": {"content": json.dumps(extraction)}}]}
                status = control["status"]
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "endpoint": f"http://127.0.0.1:{server.server_port}",
        "deployment": "fixture-model", "api_version": "2024-06-01",
        "subscription_key": "fixture-no-secret",
    }), encoding="utf-8")
    try:
        yield path, requests, control
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    """Storage tests use a deterministic extraction fixture, not semantic acceptance."""
    monkeypatch.setattr(memory_core, "extract", lambda text, path, kind=None: Extracted(
        store=True, kind=kind or ("event" if text.startswith("Decision:") else "state"),
        content=text, evidence=text,
    ))
    path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(path))
    monkeypatch.setenv("MEMORY_PROJECT_ID", "project-a")
    monkeypatch.setenv("MEMORY_USE_VECTOR", "false")
    monkeypatch.setenv("MEMORY_USE_LLM", "true")
    instance = MemoryStore(MemoryConfig(
        db_path=path, project_id="project-a", model_config_path=tmp_path / "unused-model.json",
    ))
    instance.initialize()
    return instance


@pytest.fixture
def proposal() -> IngestRequest:
    return IngestRequest(target="project", text="Database MySQL", source="fixture://single-table")


@pytest.fixture
def saved(store: MemoryStore, proposal: IngestRequest) -> MemoryRecord:
    result = store.ingest(proposal)
    assert result.memory is not None
    return result.memory
