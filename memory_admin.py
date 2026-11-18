"""Inspect memories, confirm deletion and create manual database snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Sequence

from memory_core import MemoryStore
from memory_models import (
    ListRequest, MemoryConfig, MemoryKey, MemoryServiceError, validate_request,
)
from memory_schema import schema_exists


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local memory.")
    parser.add_argument("--db", help="Database path; must be a new destination for restore.")
    parser.add_argument(
        "--project", help="Bound project for memory operations; snapshots always copy the entire DB."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List records, including model-extracted memories.")
    listing.add_argument("--scope", choices=("default", "user", "project"), default="default")
    listing.add_argument("--limit", type=int, default=10)
    listing.add_argument("--offset", type=int, default=0)
    show = commands.add_parser("show", help="Read all versions of one memory, including legacy statuses.")
    show.add_argument("--memory-id", required=True)
    forget = commands.add_parser("forget", help="Preview and delete all versions of a memory.")
    forget.add_argument("--memory-id", required=True)
    backup = commands.add_parser("backup", help="Snapshot the entire database to a new file.")
    backup.add_argument("--output", required=True)
    restore = commands.add_parser("restore", help="Restore a snapshot to a new --db path.")
    restore.add_argument("--source", required=True)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _require_interactive() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise MemoryServiceError("REVIEW_REQUIRED", "Use an independent interactive terminal.")


def _prompt(message: str) -> str | None:
    try:
        answer = input(message).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer in ("", "cancel"):
        print("Cancelled; no changes made.", file=sys.stderr)
        return None
    return answer


def _forget(store: MemoryStore, key: MemoryKey) -> int:
    preview = store.inspect(key.memory_id)
    print("Deletion preview: all displayed versions, source and evidence will be removed.")
    _print_json([row.model_dump(mode="json") for row in preview])
    answer = _prompt(f"Type delete {key.memory_id} to confirm, or cancel: ")
    if answer is None:
        return 130
    if answer != f"delete {key.memory_id}":
        raise MemoryServiceError("VALIDATION_ERROR", "Deletion confirmation did not match.")
    deleted = store.forget(preview)
    _print_json({"memory_id": key.memory_id, "deleted_versions": deleted})
    return 0


def _copy_database(source: Path, destination: Path) -> None:
    """Copy one consistent snapshot; never replace files or migrate the source."""
    if not source.is_file() or not destination.parent.is_dir():
        raise MemoryServiceError("CONFIG_ERROR", "Source file and destination parent must exist.")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        # Pin the schema check and online backup to the same committed snapshot.
        original.execute("BEGIN")
        if not schema_exists(original):
            raise MemoryServiceError("CONFIG_ERROR", "Source must be an initialized memory database.")
        try:
            fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError as error:
            raise MemoryServiceError("CONFIG_ERROR", "Destination exists; choose a new file.") from error
        complete = False
        try:
            os.close(fd)
            with closing(sqlite3.connect(destination)) as copied:
                original.backup(copied)
                if not schema_exists(copied) or copied.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise MemoryServiceError("INTERNAL_ERROR", "Copied database failed validation.")
                if copied.execute("PRAGMA journal_mode = DELETE").fetchone()[0] != "delete":
                    raise MemoryServiceError("CONFIG_ERROR", "Cannot finalize a standalone snapshot.")
            complete = True
        finally:
            if not complete:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    destination.with_name(destination.name + suffix).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in ("backup", "restore"):
            if args.db is None:
                raise MemoryServiceError("CONFIG_ERROR", "Backup and restore require an explicit --db path.")
            database = MemoryConfig.from_env({"MEMORY_DB_PATH": args.db}).db_path
            other = MemoryConfig.from_env({
                "MEMORY_DB_PATH": args.output if args.command == "backup" else args.source
            }).db_path
            source, destination = (database, other) if args.command == "backup" else (other, database)
            _copy_database(source, destination)
            _print_json({
                "status": "backed_up" if args.command == "backup" else "restored",
                "source": str(source), "output": str(destination), "scope": "entire_database",
            })
            return 0
        if args.command == "forget":
            _require_interactive()
        env = dict(os.environ)
        if args.db is not None:
            env["MEMORY_DB_PATH"] = args.db
        if args.project is not None:
            env["MEMORY_PROJECT_ID"] = args.project
        config = MemoryConfig.from_env(env)
        if not config.db_path.is_file():
            raise MemoryServiceError("CONFIG_ERROR", "Admin commands require an existing database.")
        store = MemoryStore(config)
        store.initialize()
        if args.command == "list":
            page = store.list_page(validate_request(
                ListRequest, {"scope": args.scope, "limit": args.limit, "offset": args.offset}
            ))
            _print_json([row.model_dump(mode="json") for row in page.data])
            print(json.dumps({"pagination": page.pagination.model_dump()}), file=sys.stderr)
            return 0
        if args.command == "show":
            key = validate_request(MemoryKey, {"memory_id": args.memory_id})
            _print_json([row.model_dump(mode="json") for row in store.inspect(key.memory_id)])
            return 0
        return _forget(store, validate_request(MemoryKey, {"memory_id": args.memory_id}))
    except MemoryServiceError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        if error.details:
            print(json.dumps(error.details, sort_keys=True), file=sys.stderr)
        return 1
    except (sqlite3.Error, OSError, ValueError) as error:
        print(
            f"INTERNAL_ERROR: Operation failed ({type(error).__name__}); "
            "inspect stored state before retrying.", file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
