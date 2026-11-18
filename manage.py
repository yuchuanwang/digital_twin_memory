"""Cross-platform system-Python commands; no background service or global setup."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_DATABASE = ROOT / ".test-data" / "memory.db"


def _confirm(message: str, expected: str) -> bool:
    try:
        return input(message).strip() == expected
    except (EOFError, KeyboardInterrupt):
        return False


def _dependencies(*modules: str) -> None:
    check = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(modules)], cwd=ROOT
    )
    if check.returncode:
        if not _confirm("Missing dependencies. Type INSTALL to install for your user account: ", "INSTALL"):
            raise ValueError("Cancelled; dependencies are missing.")
        if (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").is_file():
            raise ValueError(
                "This Python is externally managed. Install dependencies with your system package manager "
                "or use an independently managed, non-virtual Python installation."
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"],
            cwd=ROOT, check=True,
        )


def _clean() -> int:
    """Only reset the fixed test DB; never follow a redirected directory or file."""
    paths = [TEST_DATABASE.with_name(TEST_DATABASE.name + suffix)
             for suffix in ("", "-wal", "-shm", "-journal")]
    if TEST_DATABASE.parent.resolve() != TEST_DATABASE.parent:
        raise ValueError("Refusing to clean a redirected test directory.")
    if any(path.is_symlink() or path.resolve() != path for path in paths):
        raise ValueError("Refusing to clean a redirected test file.")
    files = [path for path in paths if path.exists()]
    if not files:
        print("No test database to clean.")
        return 0
    print("Stop MCP and close database viewers first. Only these test files will be removed:")
    for path in files:
        print(f"  {path}")
    if not _confirm("Type DELETE TEST to confirm, anything else cancels: ", "DELETE TEST"):
        print("Cancelled; no files removed.")
        return 0
    # Main file first: Windows refuses deletion of an open SQLite DB before its WAL is touched.
    for path in files:
        path.unlink()
    print("Test database removed. Deploy or restart MCP to create an empty one.")
    return 0


def _trace_matches(path: Path, needle: str | None) -> int:
    """Only count memory results inside model input, not tool spans or model output."""
    matches = 0
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            attrs = row.get("attributes", {})
            if row.get("type") != "span" or attrs.get("gen_ai.operation.name") != "chat":
                continue
            raw = attrs.get("gen_ai.input.messages")
            if raw is None:
                continue
            messages = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(messages, list):
                raise ValueError(f"Unsupported model-input format at trace line {number}.")
            for message in messages:
                if message.get("role") != "tool":
                    continue
                for part in message.get("parts", []):
                    if (
                        part.get("type") == "tool_call_response"
                        and part.get("mcp_server_name") == "digital-twin-memory"
                        and part.get("mcp_tool_name") == "memory_recall"
                    ):
                        response = part.get("response")
                        try:
                            body = json.loads(response) if isinstance(response, str) else response
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(body, dict) or body.get("ok") is not True:
                            continue
                        memories = body.get("data")
                        if not isinstance(memories, list) or not memories:
                            continue
                        text = json.dumps(memories, ensure_ascii=False)
                        if needle is None or needle in text:
                            matches += 1
                            print(f"MATCH: model input contains a memory_recall result; "
                                  f"trace line {number}, span {row.get('spanId')}.")
    return matches


def _trace(args: argparse.Namespace) -> int:
    if args.find is not None and not args.find.strip():
        raise ValueError("Trace search text must not be blank.")
    copilot = shutil.which("copilot")
    if copilot is None:
        raise ValueError("Copilot CLI was not found on PATH.")
    print("WARNING: local traces contain prompts, code and tool results. Use synthetic data only.")
    print("This starts a NEW CLI session using this project's MCP config, not an existing VS Code conversation.")
    if not _confirm("Type TRACE to enable local content capture, anything else cancels: ", "TRACE"):
        print("Cancelled; no CLI session or trace was created.")
        return 0
    directory = ROOT / ".test-data" / "prompt-traces"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"prompt-{uuid.uuid4().hex}.jsonl"
    env = {
        **os.environ,
        "COPILOT_OTEL_ENABLED": "true",
        "COPILOT_OTEL_EXPORTER_TYPE": "file",
        "COPILOT_OTEL_FILE_EXPORTER_PATH": str(path),
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT": "SPAN_ONLY",
    }
    command = [copilot, "-C", str(ROOT), "--sync-session=off"]
    if args.prompt is not None:
        command.extend([
            "--trust-folder",
            "--allow-tool", "digital-twin-memory(memory_context)",
            "--allow-tool", "digital-twin-memory(memory_recall)", "-p", args.prompt,
        ])
    print(f"Local trace: {path}")
    result = subprocess.run(command, cwd=ROOT, env=env)
    print(f"Local trace: {path}")
    if not path.is_file():
        print("UNVERIFIED: no local trace file was produced.", file=sys.stderr)
        return result.returncode or 2
    matches = _trace_matches(path, args.find)
    if not matches:
        print("UNVERIFIED: no matching memory result was observed in captured model input. "
              "Recall may not have run, or content capture/format may be unavailable.", file=sys.stderr)
    return result.returncode or (0 if matches else 2)


def _configured(args: argparse.Namespace) -> int:
    config_path = ROOT / ".mcp.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    elif args.action == "deploy":
        config = {"mcpServers": {}}
    else:
        raise ValueError("Project MCP configuration is missing. Run deploy first.")
    if not isinstance(config, dict):
        raise ValueError("MCP configuration must be a JSON object.")
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcpServers must be a JSON object.")
    if args.action == "deploy" and "digital-twin-memory" not in servers:
        servers["digital-twin-memory"] = {
            "type": "local", "tools": ["*"],
            "env": {
                "MEMORY_DB_PATH": TEST_DATABASE.as_posix(),
                "MEMORY_PROJECT_ID": "digital-twin-memory-test",
            },
        }
    server = servers["digital-twin-memory"]
    if not isinstance(server, dict) or not isinstance(server.get("env"), dict):
        raise ValueError("The memory MCP entry and its env must be JSON objects.")
    database = args.db if args.db is not None else server["env"]["MEMORY_DB_PATH"]
    if not isinstance(database, str) or not database.strip():
        raise ValueError("Database path must not be blank.")
    project = args.project if args.project is not None else server["env"]["MEMORY_PROJECT_ID"]
    if not isinstance(project, str):
        raise ValueError("Project ID must be a string.")
    if args.action == "admin":
        if not args.arguments:
            raise ValueError("Specify an admin command, for example: python manage.py admin list")
        return subprocess.run([
            sys.executable, str(ROOT / "memory_admin.py"),
            "--db", database, "--project", project, *args.arguments,
        ], cwd=ROOT).returncode

    _dependencies("anyio", "mcp", "pydantic", "httpx")
    env = {
        **os.environ, **server["env"],
        "MEMORY_DB_PATH": database, "MEMORY_PROJECT_ID": project,
    }
    result = subprocess.run([
        sys.executable, "-c",
        "import json, os; from memory_core import MemoryStore; from memory_models import MemoryConfig; "
        "from memory_extract import ModelConfig; "
        "config = MemoryConfig.from_env(os.environ); "
        "ModelConfig.load(config.model_config_path); "
        "config.db_path.parent.mkdir(parents=True, exist_ok=True); "
        "MemoryStore(config).initialize(); "
        "print(json.dumps(str(config.db_path)))",
    ], cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, text=True)
    database = json.loads(result.stdout)
    if not isinstance(database, str) or not database:
        raise ValueError("Deployment did not return a database path.")
    server.update(
        command=Path(sys.executable).as_posix(),
        args=[(ROOT / "server.py").as_posix()], cwd=ROOT.as_posix(),
    )
    server["env"].update(MEMORY_DB_PATH=Path(database).as_posix(), MEMORY_PROJECT_ID=project)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: schema 1; Python {sys.executable}\nDatabase: {database}\nProject: {project}")
    print("Model config checked locally; no model request made. Reload MCP or start copilot.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy, test and clean the local memory service with system Python."
    )
    commands = parser.add_subparsers(dest="action")
    deploy = commands.add_parser("deploy", help="Prepare the database and project MCP configuration.")
    commands.add_parser("test", help="Run all isolated tests.")
    commands.add_parser("clean", help="Confirm and reset only the fixed test database.")
    trace = commands.add_parser("trace", help="Inspect memory in a new CLI session's local model-input trace.")
    trace.add_argument("--find", help="Only match this text inside memory_recall results in model input.")
    trace.add_argument("--prompt", help="Optional one-shot read-only memory query; otherwise opens interactive CLI.")
    admin = commands.add_parser("admin", help="Forward an operator command with configured DB/project.")
    for command in (deploy, admin):
        command.add_argument("--db", help="Override the configured database path.")
        command.add_argument("--project", help="Override the configured project ID.")
    admin.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action is None:
        parser.print_help()
        return 0
    try:
        if sys.prefix != sys.base_prefix:
            raise ValueError("Use system Python, not an activated virtual environment.")
        if args.action == "clean":
            return _clean()
        if args.action == "trace":
            return _trace(args)
        if args.action == "test":
            _dependencies("pytest")
            return subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT
            ).returncode
        return _configured(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
