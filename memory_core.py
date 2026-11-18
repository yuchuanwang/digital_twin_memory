"""Single-table model-extracted memory; only the operator CLI can delete records."""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import closing, contextmanager
from typing import Generator, Sequence

from memory_extract import Skipped, extract
from memory_models import (
    IngestRequest, IngestResult, ListRequest, MemoryConfig, MemoryPage, MemoryRecord,
    MemoryServiceError, Pagination, RecallRequest, Scope,
)
from memory_schema import CREATE_TABLE, SCHEMA_VERSION, schema_exists
from memory_search import MemorySearch


_BUSY_TIMEOUT_MS = 5000
_RECORD_ORDER = " ORDER BY m.created_at DESC, m.memory_id, m.version DESC"


def _database_error(error: sqlite3.OperationalError) -> None:
    if (getattr(error, "sqlite_errorcode", 0) or 0) & 255 in (5, 6) or str(error).startswith(
        ("database is locked", "database table is locked", "database schema is locked")
    ):
        raise MemoryServiceError(
            "DATABASE_BUSY", "Memory database is busy.", retryable=True
        ) from error
    raise error


def _current_version(records: Sequence[MemoryRecord]) -> int:
    return max(
        (row.version for row in records if row.status in ("extracted", "approved")), default=0
    )


class MemoryStore:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._initialized = False
        self._search = MemorySearch(config.model_config_path)

    def _open(self, *, readonly: bool = False) -> sqlite3.Connection:
        mode = "ro" if readonly else "rw"
        db = sqlite3.connect(
            f"{self.config.db_path.as_uri()}?mode={mode}", uri=True,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        db.row_factory = sqlite3.Row
        return db

    def initialize(self) -> None:
        self._initialized = False
        path = self.config.db_path
        if not path.parent.is_dir():
            raise MemoryServiceError("CONFIG_ERROR", "Database parent directory must exist.")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            if not path.is_file():
                raise MemoryServiceError("CONFIG_ERROR", "Database path must be a file.") from None
        except OSError as error:
            raise MemoryServiceError("CONFIG_ERROR", "Cannot create database file.") from error
        else:
            os.close(fd)
        try:
            # Reject old files before journal_mode can make persistent changes.
            with closing(self._open(readonly=True)) as db:
                schema_exists(db)
            with closing(self._open()) as db, db:
                if db.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                    raise MemoryServiceError("CONFIG_ERROR", "Database must support WAL.")
                db.execute("BEGIN IMMEDIATE")
                if not schema_exists(db):
                    db.execute(CREATE_TABLE)
                    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except sqlite3.OperationalError as error:
            _database_error(error)
        self._initialized = True

    @contextmanager
    def connection(self, *, write: bool = False) -> Generator[sqlite3.Connection, None, None]:
        if not self._initialized:
            raise MemoryServiceError("CONFIG_ERROR", "Initialize the store first.")
        try:
            with closing(self._open()) as db, db:
                db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield db
        except sqlite3.OperationalError as error:
            _database_error(error)

    def context(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION, "project_id": self.config.project_id,
            "default_read_scopes": (
                ["user", "project"] if self.config.project_id is not None else ["user"]
            ),
            "write_targets": (
                ["user", "project"] if self.config.project_id is not None else ["user"]
            ),
            "retrieval": "hybrid" if self._search.configuration() else "literal_substring",
            "views": ["current", "history"],
            "pagination": "offset",
            "kinds": ["state", "event"], "ingestion": "server_model",
            "verification": "model_extracted_not_user_reviewed", "review_required": False,
            "automatic_trigger": "agent_instructions_best_effort", "mcp_deletion": False,
            "idempotency": False, "batch_write": False, "automatic_expiry": False,
        }

    def _scope(self, scope: Scope) -> tuple[str, list[str | int]]:
        project = self.config.project_id
        if scope == "project":
            if project is None:
                raise MemoryServiceError("SCOPE_REQUIRED", "Bind a project first.")
            return "m.project_id = ?", [project]
        if scope == "user" or project is None:
            return "m.project_id IS NULL", []
        return "(m.project_id IS NULL OR m.project_id = ?)", [project]

    def _records(self, db: sqlite3.Connection, memory_id: str) -> list[MemoryRecord]:
        rows = db.execute(
            "SELECT * FROM memories WHERE memory_id = ? ORDER BY version", (memory_id,)
        ).fetchall()
        if not rows:
            raise MemoryServiceError("MEMORY_NOT_FOUND", "Memory does not exist.")
        if any(row["project_id"] not in (None, self.config.project_id) for row in rows):
            raise MemoryServiceError("SCOPE_NOT_ALLOWED", "Memory belongs to another project.")
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    def inspect(self, memory_id: str) -> list[MemoryRecord]:
        with self.connection() as db:
            return self._records(db, memory_id)

    def _insert(self, db: sqlite3.Connection, record: MemoryRecord) -> MemoryRecord:
        db.execute(
            """INSERT INTO memories (
                memory_id, version, base_version, project_id, kind,
                content, source, evidence, status, created_at, reviewed_at
            ) VALUES (
                :memory_id, :version, :base_version, :project_id, :kind,
                :content, :source, :evidence, :status, :created_at, :reviewed_at
            )""",
            record.model_dump(exclude={"reference"}),
        )
        return record

    def _baseline(
        self, db: sqlite3.Connection, command: IngestRequest, project: str | None,
    ) -> list[MemoryRecord]:
        if command.memory_id is None:
            return []
        records = self._records(db, command.memory_id)
        if records[0].project_id != project:
            raise MemoryServiceError("SCOPE_NOT_ALLOWED", "Revision target cannot change scope.")
        current = _current_version(records)
        if command.expected_version != current:
            raise MemoryServiceError(
                "VERSION_CONFLICT", "Current stored version changed.",
                details={"expected_version": command.expected_version or 0, "current_version": current},
            )
        return records

    def ingest(self, command: IngestRequest) -> IngestResult:
        project = self.config.project_id if command.target == "project" else None
        if command.target == "project" and project is None:
            raise MemoryServiceError("SCOPE_REQUIRED", "Bind a project first.")
        with self.connection() as db:
            records = self._baseline(db, command, project)
        # Model calls must not hold a database lock; scope is checked before any outbound request.
        extracted = extract(
            command.text, self.config.model_config_path, records[0].kind if records else None
        )
        if isinstance(extracted, Skipped):
            return IngestResult(stored=False, reason=extracted.reason)
        with self.connection(write=True) as db:
            records = self._baseline(db, command, project)
            if records and extracted.kind != records[0].kind:
                raise MemoryServiceError("MODEL_OUTPUT_INVALID", "A revision cannot change memory kind.")
            record = self._insert(db, MemoryRecord(
                memory_id=command.memory_id or uuid.uuid4().hex,
                version=records[-1].version + 1 if records else 1,
                base_version=_current_version(records), project_id=project,
                kind=extracted.kind, content=extracted.content,
                source=command.source, evidence=extracted.evidence,
            ))
        return IngestResult(stored=True, memory=record)

    def list_records(self, command: ListRequest) -> list[MemoryRecord]:
        return self.list_page(command).data

    def list_page(self, command: ListRequest) -> MemoryPage:
        where, params = self._scope(command.scope)
        return self._page(where, params, command)

    def recall(self, command: RecallRequest) -> list[MemoryRecord]:
        return self.recall_page(command).data

    def recall_page(self, command: RecallRequest) -> MemoryPage:
        where, params = self._scope(command.scope)
        clauses = [where, "m.status IN ('extracted', 'approved')"]
        if command.view == "current":
            clauses.extend([
                "m.kind = 'state'",
                """m.version = (SELECT max(version) FROM memories
                   WHERE memory_id = m.memory_id AND status IN ('extracted', 'approved'))""",
            ])
        if command.memory_id is not None:
            clauses.append("m.memory_id = ?")
            params.append(command.memory_id)
        if command.query is not None and command.search_mode != "keyword":
            model_config = self._search.configuration()
            if model_config is None and command.search_mode == "hybrid":
                raise MemoryServiceError(
                    "CONFIG_ERROR", "Hybrid recall requires an embedding section in config.json."
                )
            if model_config is not None:
                with self.connection() as db:
                    rows = db.execute(
                        "SELECT m.* FROM memories m WHERE " + " AND ".join(clauses) + _RECORD_ORDER,
                        params,
                    ).fetchall()
                records = [MemoryRecord.model_validate(dict(row)) for row in rows]
                ranked = self._search.rank(command.query, records, model_config)
                return MemoryPage(
                    data=ranked[command.offset:command.offset + command.limit],
                    pagination=Pagination(
                        offset=command.offset, limit=command.limit,
                        has_more=len(ranked) > command.offset + command.limit,
                    ),
                )
        if command.query is not None:
            clauses.append("instr(lower(m.content), lower(?)) > 0")
            params.append(command.query)
        return self._page(" AND ".join(clauses), params, command)

    def _page(
        self, where: str, params: list[str | int], command: ListRequest,
    ) -> MemoryPage:
        with self.connection() as db:
            rows = db.execute(
                "SELECT m.* FROM memories m WHERE " + where
                + _RECORD_ORDER + " LIMIT ? OFFSET ?",
                [*params, command.limit + 1, command.offset],
            ).fetchall()
        return MemoryPage(
            data=[MemoryRecord.model_validate(dict(row)) for row in rows[:command.limit]],
            pagination=Pagination(
                offset=command.offset, limit=command.limit, has_more=len(rows) > command.limit,
            ),
        )

    def forget(self, preview: Sequence[MemoryRecord]) -> int:
        """Bind deletion to the entire displayed chain without a persistent receipt."""
        if not preview:
            raise MemoryServiceError("VALIDATION_ERROR", "A deletion preview is required.")
        with self.connection(write=True) as db:
            records = self._records(db, preview[0].memory_id)
            if records != list(preview):
                raise MemoryServiceError("VERSION_CONFLICT", "Memory changed after the deletion preview.")
            return db.execute(
                "DELETE FROM memories WHERE memory_id = ?", (preview[0].memory_id,)
            ).rowcount
