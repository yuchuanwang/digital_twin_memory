"""Actual stdio to a loopback model: extract, store, recall; no approval fixture."""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager, closing
from datetime import timedelta
from pathlib import Path
from typing import AsyncIterator

import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

import server
from memory_core import MemoryStore
from memory_models import MemoryConfig


ROOT = Path(server.__file__).parent
INGEST = {"target": "project", "text": "Protocol PostgreSQL 并发", "source": "fixture://mcp"}


@pytest.fixture(autouse=True)
def model_environment(model_server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_MODEL_CONFIG", str(model_server[0]))


def environment(path: Path, project: str = "project-a") -> dict[str, str]:
    env = {
        "MEMORY_DB_PATH": str(path), "MEMORY_PROJECT_ID": project,
        "MEMORY_USE_VECTOR": "false", "MEMORY_USE_LLM": "true",
        "MEMORY_MODEL_CONFIG": os.environ["MEMORY_MODEL_CONFIG"],
    }
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return env


@asynccontextmanager
async def connected(
    path: Path, project: str = "project-a", mode: str = "direct",
) -> AsyncIterator[ClientSession]:
    args = [str(ROOT / "server.py")]
    if mode != "direct":
        setup = "import asyncio\nimport server\n"
        if mode == "fault":
            setup += (
                "from memory_core import MemoryStore\n"
                "def fail(self, command):\n"
                "    raise RuntimeError('private-runtime-marker')\n"
                "MemoryStore.recall_page = fail\n"
            )
        elif mode == "busy":
            setup += "import memory_core\nmemory_core._BUSY_TIMEOUT_MS = 25\n"
        elif mode != "import":
            raise ValueError("Unknown owned test bootstrap.")
        args = ["-c", setup + "raise SystemExit(asyncio.run(server.run()))"]
    params = StdioServerParameters(
        command=sys.executable, args=args, cwd=ROOT, env=environment(path, project),
    )
    with path.with_suffix(f".{project or 'unbound'}.{mode}.stderr.log").open("w") as errlog:
        async with stdio_client(params, errlog=errlog) as streams:
            async with ClientSession(
                *streams, read_timeout_seconds=timedelta(seconds=5)
            ) as session:
                await session.initialize()
                yield session


async def call(session: ClientSession, name: str, arguments: dict[str, object]):
    result = await session.call_tool(name, arguments)
    assert result.structuredContent is not None
    assert len(result.content) == 1 and isinstance(result.content[0], types.TextContent)
    body = result.structuredContent
    assert json.loads(result.content[0].text) == body
    assert body["ok"] is (not result.isError)
    return body


@pytest.mark.parametrize("mode", ["direct", "import"])
def test_actual_model_ingestion_is_immediately_visible_with_history(
    tmp_path: Path, mode: str,
) -> None:
    async def scenario() -> None:
        async with connected(tmp_path / "mcp.db", mode=mode) as session:
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert set(tools) == {"memory_context", "memory_ingest", "memory_recall"}
            assert tools["memory_ingest"].annotations.idempotentHint is False
            assert tools["memory_ingest"].annotations.openWorldHint is True
            assert all(tools[name].annotations.openWorldHint is False for name in (
                "memory_context", "memory_recall",
            ))
            context = (await call(session, "memory_context", {}))["data"]
            assert context["schema_version"] == 1 and context["review_required"] is False
            assert context["ingestion"] == "server_model"
            saved = (await call(session, "memory_ingest", INGEST))["data"]
            assert saved["stored"] is True
            record = saved["memory"]
            assert record["status"] == "extracted" and record["reviewed_at"] is None
            memory_id = record["memory_id"]
            current = (await call(session, "memory_recall", {"query": "并发"}))["data"]
            assert current == [record]
            updated = (await call(session, "memory_ingest", {
                **INGEST, "memory_id": memory_id, "expected_version": 1,
                "text": "Protocol SQLite",
            }))["data"]["memory"]
            assert updated["version"] == 2 and updated["base_version"] == 1
            assert (await call(session, "memory_recall", {"query": "PostgreSQL"}))["data"] == []
            history = (await call(session, "memory_recall", {
                "memory_id": memory_id, "view": "history",
            }))["data"]
            assert {row["version"] for row in history} == {1, 2}
            blocked = await call(session, "memory_forget", {"memory_id": memory_id})
            assert blocked["error"]["code"] == "REVIEW_REQUIRED"
            assert (await call(session, "memory_recall", {
                "memory_id": memory_id, "view": "history",
            }))["data"] == history
    asyncio.run(scenario())


def test_stdio_recall_exposes_pagination_without_changing_data_array(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with connected(tmp_path / "pages.db") as session:
            saved = []
            for suffix in ("one", "two", "three"):
                result = await call(session, "memory_ingest", {**INGEST, "text": f"Protocol {suffix}"})
                saved.append(result["data"]["memory"])
            first = await call(session, "memory_recall", {"query": "Protocol", "limit": 2})
            assert first["data"] == list(reversed(saved[1:]))
            assert first["pagination"] == {"offset": 0, "limit": 2, "has_more": True, "next_offset": 2}
            last = await call(session, "memory_recall", {
                "query": "Protocol", "limit": 2, "offset": first["pagination"]["next_offset"],
            })
            assert last["data"] == saved[:1]
            assert last["pagination"] == {"offset": 2, "limit": 2, "has_more": False, "next_offset": None}
    asyncio.run(scenario())


def test_stdio_automatically_uses_embeddings_and_preserves_record_shape(
    tmp_path: Path, model_server,
) -> None:
    path, requests, control = model_server
    config = json.loads(path.read_text(encoding="utf-8"))
    config["embedding_deployment"] = "fixture-embedding"
    path.write_text(json.dumps(config), encoding="utf-8")
    control["vectors"] = {"coding language": [1.0, 0.0], "Examples use Python.": [1.0, 0.0]}

    async def scenario() -> None:
        async with connected(tmp_path / "hybrid.db") as session:
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert tools["memory_recall"].annotations.openWorldHint is True
            assert tools["memory_context"].annotations.openWorldHint is False
            context = (await call(session, "memory_context", {}))["data"]
            assert context["retrieval"] == "hybrid" and context["schema_version"] == 1
            record = (await call(session, "memory_ingest", {
                **INGEST, "text": "Examples use Python.",
            }))["data"]["memory"]
            recalled = await call(session, "memory_recall", {"query": "coding language"})
            assert recalled["data"] == [record]
            assert recalled["pagination"]["has_more"] is False
            control["embedding_status"] = 401
            failed = await call(session, "memory_recall", {"query": "another query"})
            assert failed["error"]["code"] == "MODEL_UNAVAILABLE"
            direct = await call(session, "memory_recall", {"memory_id": record["memory_id"]})
            assert direct["data"] == [record]
            assert len([item for item in requests if "input" in item["body"]]) == 2
    asyncio.run(scenario())


def test_smalltalk_skip_and_model_failure_have_distinct_results(tmp_path: Path, model_server) -> None:
    async def scenario() -> None:
        async with connected(tmp_path / "mcp.db") as session:
            skipped = (await call(session, "memory_ingest", {**INGEST, "text": "Hello!"}))["data"]
            assert skipped["stored"] is False and skipped["reason"] and skipped["memory"] is None
            model_server[2]["status"] = 401
            failed = await call(session, "memory_ingest", INGEST)
            assert failed["error"]["code"] == "MODEL_UNAVAILABLE" and failed["ok"] is False
            assert (await call(session, "memory_recall", {"query": "Protocol"}))["data"] == []
    asyncio.run(scenario())


def test_scope_and_conflict_errors_are_not_sent_to_the_model(tmp_path: Path, model_server) -> None:
    async def scenario() -> None:
        path = tmp_path / "shared.db"
        async with connected(path) as primary:
            record = (await call(primary, "memory_ingest", INGEST))["data"]["memory"]
            async with connected(path, project="") as unbound, connected(path, project="other") as other:
                for session in (unbound, other):
                    assert (await call(session, "memory_recall", {"query": "Protocol"}))["data"] == []
                    result = await call(session, "memory_ingest", {
                        **INGEST, "target": "user" if session is unbound else "project",
                        "memory_id": record["memory_id"], "expected_version": 1,
                    })
                    assert result["error"]["code"] == "SCOPE_NOT_ALLOWED"
                result = await call(unbound, "memory_ingest", INGEST)
                assert result["error"]["code"] == "SCOPE_REQUIRED"
            stale = await call(primary, "memory_ingest", {
                **INGEST, "memory_id": record["memory_id"], "expected_version": 2,
            })
            assert stale["error"]["code"] == "VERSION_CONFLICT"
            assert len(model_server[1]) == 1
    asyncio.run(scenario())


def test_removed_approval_tools_and_input_injections_are_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with connected(tmp_path / "mcp.db") as session:
            for name, arguments in [
                ("memory_review", {"memory_id": "test", "version": 1}),
                ("memory_submit", {}), ("memory_list_candidates", {}),
                ("memory_ingest", {**INGEST, "confirmed": True}),
                ("memory_ingest", {**INGEST, "kind": "state"}),
                ("memory_ingest", {**INGEST, "expected_version": True}),
                ("memory_recall", {"query": "test", "limit": True}),
            ]:
                assert (await call(session, name, arguments))["error"]["code"] == "VALIDATION_ERROR"
    asyncio.run(scenario())


def test_busy_and_runtime_faults_are_explicit_and_redacted(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "mcp.db"
        async with connected(path, mode="busy") as session:
            store = MemoryStore(MemoryConfig(db_path=path))
            store.initialize()
            with store.connection(write=True):
                result = await call(session, "memory_ingest", INGEST)
                assert result["error"]["code"] == "DATABASE_BUSY"
            assert (await call(session, "memory_recall", {"query": "Protocol"}))["data"] == []
        async with connected(path, mode="fault") as session:
            result = await call(session, "memory_recall", {"query": "test"})
            assert result["error"]["code"] == "INTERNAL_ERROR"
            assert "private-runtime-marker" not in json.dumps(result)
    asyncio.run(scenario())


def test_imports_do_not_load_model_config_or_open_db_or_network(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c",
         "from unittest.mock import patch\n"
         "with patch('sqlite3.connect', side_effect=AssertionError('database I/O')):\n"
         "    with patch('httpx.post', side_effect=AssertionError('network I/O')):\n"
         "        import memory_extract, memory_core, memory_admin, server\n"],
        cwd=ROOT, env={**os.environ, **environment(tmp_path / "untouched.db")},
        text=True, capture_output=True, timeout=10, check=True,
    )
    assert result.stdout == result.stderr == "" and not (tmp_path / "untouched.db").exists()


def test_stdout_jsonrpc_redaction_shutdown_and_bad_schema_startup(tmp_path: Path) -> None:
    async def scenario(path: Path, legacy: bool) -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / "server.py"), cwd=ROOT,
            env={**os.environ, **environment(path)}, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert process.stdin and process.stdout and process.stderr
            if legacy:
                assert await asyncio.wait_for(process.wait(), 5) == 1
                assert await process.stdout.read() == b""
                assert b"SCHEMA_MIGRATION_REQUIRED" in await process.stderr.read()
                return
            message = {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": types.LATEST_PROTOCOL_VERSION, "capabilities": {},
                           "clientInfo": {"name": "owned-test", "version": "1"}},
            }
            process.stdin.write((json.dumps(message) + "\n").encode())
            await process.stdin.drain()
            assert json.loads(await asyncio.wait_for(process.stdout.readline(), 5))["id"] == 1
            process.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            process.stdin.write(b'{"invalid":"private-wire-marker"}\n')
            await process.stdin.drain()
            diagnostic = await asyncio.wait_for(process.stdout.readline(), 5)
            assert b"private-wire-marker" not in diagnostic
            process.stdin.close()
            assert await asyncio.wait_for(process.wait(), 5) == 0
            assert b"private-wire-marker" not in await process.stderr.read()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    asyncio.run(scenario(tmp_path / "wire.db", False))
    legacy = tmp_path / "legacy.db"
    with closing(sqlite3.connect(legacy)) as db, db:
        db.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY)")
    before = legacy.read_bytes()
    asyncio.run(scenario(legacy, True))
    assert legacy.read_bytes() == before
