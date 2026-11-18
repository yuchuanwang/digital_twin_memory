"""Exercise the entry script only in disposable copied workspaces."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from contextlib import closing
from pathlib import Path

import pytest

import manage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "memory project [test]"
    root.mkdir()
    for name in (
        "manage.py", "memory_schema.py", "memory_models.py", "memory_core.py",
        "memory_admin.py", "memory_extract.py", "memory_search.py", "server.py", "requirements.txt",
    ):
        shutil.copy2(ROOT / name, root / name)
    config = {
        "mcpServers": {
            "digital-twin-memory": {
                "type": "local", "command": "replaced-on-deploy",
                "args": [], "cwd": "replaced-on-deploy",
                "env": {
                    "MEMORY_DB_PATH": str(Path(".test-data") / "memory.db"),
                    "MEMORY_PROJECT_ID": "script-fixture",
                },
                "tools": ["*"],
            },
            "keep-existing-server": {"type": "local", "command": "unchanged"},
        }
    }
    (root / ".mcp.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({
        "endpoint": "https://example.invalid", "deployment": "fixture-model",
        "api_version": "2024-06-01", "subscription_key": "fixture-no-secret",
    }), encoding="utf-8")
    return root


def run_script(
    root: Path, *arguments: str, answer: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "manage.py"), *arguments],
        cwd=root.parent, input=answer, text=True, capture_output=True, timeout=30,
    )


def test_deploy_initializes_schema_one_and_preserves_existing_configuration(workspace: Path) -> None:
    for _ in range(2):
        result = run_script(workspace, "deploy")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "schema 1" in result.stdout
    config = json.loads((workspace / ".mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["keep-existing-server"]["command"] == "unchanged"
    server = config["mcpServers"]["digital-twin-memory"]
    assert server["command"] == Path(sys.executable).as_posix()
    assert server["args"] == [(workspace / "server.py").as_posix()]
    assert server["cwd"] == workspace.as_posix()
    assert server["env"]["MEMORY_PROJECT_ID"] == "script-fixture"
    assert server["env"]["MEMORY_DB_PATH"] == (workspace / ".test-data" / "memory.db").as_posix()
    path = Path(server["env"]["MEMORY_DB_PATH"])
    assert path == workspace / ".test-data" / "memory.db"
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        ).fetchall() == [("memories",)]
    result = run_script(workspace, "admin", "list")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == []
    blocked = run_script(workspace, "admin", "forget", "--memory-id", "owned-test")
    assert blocked.returncode == 1 and "REVIEW_REQUIRED" in blocked.stderr


@pytest.mark.parametrize("keep_other_server", [False, True])
def test_deploy_creates_missing_host_configuration(
    workspace: Path, keep_other_server: bool,
) -> None:
    config_path = workspace / ".mcp.json"
    if keep_other_server:
        config_path.write_text(json.dumps({
            "mcpServers": {"keep-existing-server": {"type": "local", "command": "unchanged"}},
        }), encoding="utf-8")
    else:
        config_path.unlink()
    result = run_script(workspace, "deploy")
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server = config["mcpServers"]["digital-twin-memory"]
    assert server["command"] == Path(sys.executable).as_posix()
    assert server["args"] == [(workspace / "server.py").as_posix()]
    assert server["cwd"] == workspace.as_posix()
    assert server["env"]["MEMORY_DB_PATH"] == (workspace / ".test-data" / "memory.db").as_posix()
    assert server["env"]["MEMORY_PROJECT_ID"] == "digital-twin-memory-test"
    if keep_other_server:
        assert config["mcpServers"]["keep-existing-server"]["command"] == "unchanged"


@pytest.mark.parametrize("action", ["deploy", "test"])
def test_commands_require_system_python_before_any_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], action: str,
) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "non-system-python"))
    monkeypatch.setattr(sys, "argv", ["manage.py", action])
    monkeypatch.setattr(
        manage, "_configured", lambda *args: pytest.fail("Must not deploy or write configuration")
    )
    monkeypatch.setattr(
        manage, "_dependencies", lambda *args: pytest.fail("Must not check or install dependencies")
    )
    assert manage.main() == 1
    assert "Use system Python" in capsys.readouterr().err


def test_dependency_installation_targets_the_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt: "INSTALL")
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0, "", "")

    monkeypatch.setattr(manage.subprocess, "run", run)
    manage._dependencies("mcp")
    assert calls[1] == [
        sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt",
    ]


def test_externally_managed_python_requires_system_package_management(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "EXTERNALLY-MANAGED").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt: "INSTALL")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["-c", "import mcp"], "Must not install into externally managed Python"
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(manage.subprocess, "run", run)
    with pytest.raises(ValueError, match="system package manager"):
        manage._dependencies("mcp")


def test_deploy_rejects_foreign_database_paths_until_explicitly_reconfigured(workspace: Path) -> None:
    native = workspace / "foreign-memory.db"
    foreign = native.as_posix()[len(native.drive):] if os.name == "nt" else r"C:\foreign-memory.db"
    config_path = workspace / ".mcp.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mcpServers"]["digital-twin-memory"]["env"]["MEMORY_DB_PATH"] = foreign
    config_path.write_text(json.dumps(config), encoding="utf-8")
    before = config_path.read_bytes()
    rejected = run_script(workspace, "deploy")
    assert rejected.returncode == 1 and "CONFIG_ERROR" in rejected.stderr
    assert "Ready:" not in rejected.stdout and config_path.read_bytes() == before
    assert not list(workspace.glob("*.db"))
    local_database = workspace / "native-memory.db"
    deployed = run_script(workspace, "deploy", "--db", str(local_database))
    assert deployed.returncode == 0, deployed.stdout + deployed.stderr
    server = json.loads(config_path.read_text(encoding="utf-8"))["mcpServers"]["digital-twin-memory"]
    assert Path(server["env"]["MEMORY_DB_PATH"]) == local_database


def test_deploy_rejects_an_incompatible_file_without_relabeling_it(workspace: Path) -> None:
    path = workspace / "existing.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA user_version = 3")
        db.execute("CREATE TABLE memories(memory_id TEXT, version INTEGER)")
    before, config_before = path.read_bytes(), (workspace / ".mcp.json").read_bytes()
    result = run_script(workspace, "deploy", "--db", str(path))
    assert result.returncode != 0
    assert path.read_bytes() == before
    assert (workspace / ".mcp.json").read_bytes() == config_before


def test_missing_dependencies_require_explicit_installation_consent(workspace: Path) -> None:
    (workspace / "mcp.py").write_text("raise ImportError('owned missing-dependency fixture')\n")
    before = (workspace / ".mcp.json").read_bytes()
    result = run_script(workspace, "deploy", answer="cancel\n")
    assert result.returncode == 1 and "Cancelled; dependencies are missing." in result.stderr
    assert (workspace / ".mcp.json").read_bytes() == before
    assert not (workspace / ".test-data").exists()


def test_invalid_model_config_does_not_expose_credentials_or_modify_config(workspace: Path) -> None:
    (workspace / "config.json").write_text('{"subscription_key":"private-config-fixture"}')
    before = (workspace / ".mcp.json").read_bytes()
    result = run_script(workspace, "deploy")
    assert result.returncode == 1
    assert "private-config-fixture" not in result.stdout + result.stderr
    assert "CONFIG_ERROR" in result.stderr
    assert (workspace / ".mcp.json").read_bytes() == before
    assert not (workspace / ".test-data").exists()


@pytest.mark.parametrize("setting,value", [
    ("MEMORY_USE_LLM", "false"),
    ("MEMORY_MODEL_CONFIG", "missing-model.json"),
])
@pytest.mark.parametrize("inherited", [False, True])
def test_deploy_validates_effective_startup_environment_before_reporting_ready(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
    setting: str, value: str, inherited: bool,
) -> None:
    if inherited:
        monkeypatch.setenv(setting, value)
    else:
        path = workspace / ".mcp.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["mcpServers"]["digital-twin-memory"]["env"][setting] = value
        path.write_text(json.dumps(config), encoding="utf-8")
    before = (workspace / ".mcp.json").read_bytes()
    result = run_script(workspace, "deploy")
    assert result.returncode == 1
    assert "Ready:" not in result.stdout and "CONFIG_ERROR" in result.stderr
    assert (workspace / ".mcp.json").read_bytes() == before
    assert not (workspace / ".test-data").exists()


def test_deploy_configured_environment_overrides_inherited_settings(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_USE_LLM", "false")
    path = workspace / ".mcp.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["mcpServers"]["digital-twin-memory"]["env"]["MEMORY_USE_LLM"] = "true"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_script(workspace, "deploy")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ready:" in result.stdout


@pytest.mark.parametrize("succeeds", [True, False])
def test_test_command_uses_isolated_tests_and_preserves_runner_exit_status(
    workspace: Path, succeeds: bool,
) -> None:
    tests = workspace / "tests"
    tests.mkdir()
    (tests / "test_owned.py").write_text(
        f"def test_owned():\n    assert {succeeds}\n", encoding="utf-8"
    )
    untouched = workspace / "existing-data.db"
    untouched.write_bytes(b"not a test database")
    result = run_script(workspace, "test")
    assert result.returncode == (0 if succeeds else 1), result.stdout + result.stderr
    assert ("1 passed" if succeeds else "1 failed") in result.stdout
    assert untouched.read_bytes() == b"not a test database"


def test_clean_requires_confirmation_and_never_touches_a_configured_production_db(
    workspace: Path,
) -> None:
    production = workspace / "production.db"
    result = run_script(workspace, "deploy", "--db", str(production), "--project", "production")
    assert result.returncode == 0, result.stdout + result.stderr
    production_before = production.read_bytes()
    directory = workspace / ".test-data"
    directory.mkdir()
    paths = [directory / ("memory.db" + suffix) for suffix in ("", "-wal", "-shm", "-journal")]
    for path in paths:
        path.write_bytes(b"owned synthetic cleanup fixture")
    backup = directory / "keep-backup.db"
    backup.write_bytes(b"keep this backup")
    cancelled = run_script(workspace, "clean", answer="cancel\n")
    assert cancelled.returncode == 0 and "Cancelled" in cancelled.stdout
    assert all(path.exists() for path in paths)
    cleaned = run_script(workspace, "clean", answer="DELETE TEST\n")
    assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
    assert all(not path.exists() for path in paths)
    assert backup.read_bytes() == b"keep this backup"
    assert production.read_bytes() == production_before
    assert directory.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows refuses to unlink an open SQLite file")
def test_clean_refuses_an_open_file_before_removing_sidecars(workspace: Path) -> None:
    directory = workspace / ".test-data"
    directory.mkdir()
    path, sidecar = directory / "memory.db", directory / "memory.db-wal"
    path.write_bytes(b"owned locked fixture")
    sidecar.write_bytes(b"owned WAL fixture")
    with path.open("rb"):
        result = run_script(workspace, "clean", answer="DELETE TEST\n")
    assert result.returncode != 0
    assert path.read_bytes() == b"owned locked fixture"
    assert sidecar.read_bytes() == b"owned WAL fixture"


def trace_input(
    response: str, *, role: str = "tool", body: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "span", "spanId": "owned-test-span",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": json.dumps([{
                "role": role,
                "parts": [{
                    "type": "tool_call_response", "mcp_server_name": "digital-twin-memory",
                    "mcp_tool_name": "memory_recall",
                    "response": json.dumps(
                        body if body is not None else {"ok": True, "data": [{"content": response}]}
                    ),
                }],
            }]),
        },
    }


def test_trace_matching_ignores_user_messages_tool_spans_and_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "trace.jsonl"
    ignored = [
        trace_input("DB-ONLY-MARKER", role="user"),
        {"type": "span", "attributes": {
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.call.result": "DB-ONLY-MARKER",
        }},
        {"type": "span", "attributes": {
            "gen_ai.operation.name": "chat", "gen_ai.output.messages": "DB-ONLY-MARKER",
        }},
    ]
    for response in (
        {"ok": True, "data": []},
        {"ok": False, "error": {"message": "DB-ONLY-MARKER"}},
    ):
        ignored.append(trace_input("ignored", body=response))
    path.write_text("\n".join(json.dumps(row) for row in ignored), encoding="utf-8")
    assert manage._trace_matches(path, "DB-ONLY-MARKER") == 0
    with path.open("a", encoding="utf-8") as target:
        target.write("\n" + json.dumps(trace_input("DB-ONLY-MARKER")))
    assert manage._trace_matches(path, "DB-ONLY-MARKER") == 1
    assert f"trace line {len(ignored) + 1}" in capsys.readouterr().out
    assert manage._trace_matches(path, "NOT-IN-INPUT") == 0


@pytest.mark.parametrize("matched,cli_status,expected", [(True, 0, 0), (False, 0, 2), (True, 9, 9)])
def test_trace_uses_child_only_local_exporter_and_checks_actual_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    matched: bool, cli_status: int, expected: int,
) -> None:
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["manage.py", "trace", "--find", "DB-ONLY-MARKER", "--prompt", "Query the fictional project."])
    monkeypatch.setattr(manage.shutil, "which", lambda name: "owned-copilot")
    monkeypatch.setattr("builtins.input", lambda prompt: "TRACE")
    monkeypatch.setenv("COPILOT_OTEL_EXPORTER_TYPE", "otlp-http")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.invalid")
    monkeypatch.delenv("COPILOT_OTEL_FILE_EXPORTER_PATH", raising=False)

    def run(command, *, cwd, env):
        assert command[0] == "owned-copilot" and command[-1] == "Query the fictional project."
        assert "DB-ONLY-MARKER" not in command
        assert env["COPILOT_OTEL_EXPORTER_TYPE"] == "file"
        assert env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT"] == "SPAN_ONLY"
        assert cwd == tmp_path
        path = Path(env["COPILOT_OTEL_FILE_EXPORTER_PATH"])
        assert path.parent == tmp_path / ".test-data" / "prompt-traces"
        path.write_text(json.dumps(trace_input("DB-ONLY-MARKER" if matched else "other")), encoding="utf-8")
        return subprocess.CompletedProcess(command, cli_status)

    monkeypatch.setattr(manage.subprocess, "run", run)
    assert manage.main() == expected
    assert os.environ["COPILOT_OTEL_EXPORTER_TYPE"] == "otlp-http"
    assert "COPILOT_OTEL_FILE_EXPORTER_PATH" not in os.environ
    if not matched:
        assert "UNVERIFIED" in capsys.readouterr().err


def test_trace_cancel_does_not_launch_or_create_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["manage.py", "trace"])
    monkeypatch.setattr(manage.shutil, "which", lambda name: "owned-copilot")
    monkeypatch.setattr("builtins.input", lambda prompt: "cancel")
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not launch"))
    assert manage.main() == 0 and not (tmp_path / ".test-data").exists()


def test_trace_missing_file_is_unverified_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["manage.py", "trace"])
    monkeypatch.setattr(manage.shutil, "which", lambda name: "owned-copilot")
    monkeypatch.setattr("builtins.input", lambda prompt: "TRACE")
    monkeypatch.setattr(manage.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    assert manage.main() == 2
    assert "UNVERIFIED: no local trace file" in capsys.readouterr().err
