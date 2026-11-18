"""Single-table storage and conflict handling; no fixture calls an approval API."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

import memory_core
from memory_core import MemoryStore
from memory_extract import Extracted, Skipped
from memory_models import IngestRequest, ListRequest, MemoryConfig, MemoryRecord, MemoryServiceError
from memory_schema import CREATE_TABLE


def test_schema_is_one_table_with_eleven_fields_and_explicit_initialization(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    store = MemoryStore(MemoryConfig(db_path=path))
    assert not path.exists()
    with pytest.raises(MemoryServiceError, match="Initialize"):
        with store.connection():
            pytest.fail("Must initialize explicitly.")
    store.initialize()
    with store.connection() as db:
        assert [(row["type"], row["name"]) for row in db.execute(
            "SELECT type, name FROM sqlite_schema WHERE type != 'index'"
        )] == [("table", "memories")]
        assert len(db.execute("PRAGMA table_info(memories)").fetchall()) == 11
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")
    store.initialize()


@pytest.mark.parametrize("version", [0, 1, 2, 3])
def test_incompatible_database_is_not_modified(tmp_path: Path, version: int) -> None:
    path = tmp_path / "old.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT)")
        db.execute("INSERT INTO memories VALUES (1, 'keep old evidence')")
        db.execute(f"PRAGMA user_version = {version}")
    before = path.read_bytes()
    with pytest.raises(MemoryServiceError) as caught:
        MemoryStore(MemoryConfig(db_path=path)).initialize()
    assert caught.value.code == "SCHEMA_MIGRATION_REQUIRED" and path.read_bytes() == before


@pytest.mark.parametrize("ddl", [
    "CREATE TABLE memories(memory_id TEXT, version INTEGER, PRIMARY KEY(memory_id, version))",
    CREATE_TABLE.replace("    evidence TEXT NOT NULL,\n", ""),
    CREATE_TABLE.replace("version INTEGER NOT NULL,", "version TEXT NOT NULL,", 1),
    CREATE_TABLE.replace("content TEXT NOT NULL", "content TEXT"),
    CREATE_TABLE.replace("PRIMARY KEY (memory_id, version)", "PRIMARY KEY (version, memory_id)"),
])
def test_schema_one_rejects_incomplete_or_incompatible_columns_without_writes(
    tmp_path: Path, ddl: str,
) -> None:
    path = tmp_path / "incompatible.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(ddl)
        db.execute("PRAGMA user_version = 1")
    before = path.read_bytes()
    with pytest.raises(MemoryServiceError) as caught:
        MemoryStore(MemoryConfig(db_path=path)).initialize()
    assert caught.value.code == "SCHEMA_MIGRATION_REQUIRED"
    assert path.read_bytes() == before
    assert not path.with_name(path.name + "-wal").exists()
    assert not path.with_name(path.name + "-shm").exists()


def test_legacy_status_default_does_not_block_compatible_databases(tmp_path: Path) -> None:
    path = tmp_path / "legacy-compatible.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(CREATE_TABLE.replace("DEFAULT 'extracted'", "DEFAULT 'candidate'"))
        db.execute("PRAGMA user_version = 1")
    store = MemoryStore(MemoryConfig(db_path=path))
    store.initialize()
    assert store.list_records(ListRequest()) == []


def test_stored_extraction_is_visible_without_claiming_user_review(
    store: MemoryStore, saved: MemoryRecord,
) -> None:
    assert saved.status == "extracted" and saved.reviewed_at is None
    assert saved.evidence == saved.content
    assert store.list_records(ListRequest()) == [saved]
    reopened = MemoryStore(store.config)
    reopened.initialize()
    assert reopened.inspect(saved.memory_id) == [saved]


def test_skip_and_model_failure_do_not_create_rows(
    store: MemoryStore, proposal: IngestRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_core, "extract", lambda *args: Skipped(store=False, reason="Only a greeting."))
    result = store.ingest(proposal)
    assert not result.stored and result.reason and result.memory is None
    def fail(*args):
        raise MemoryServiceError("MODEL_UNAVAILABLE", "Owned model fixture failure.")
    monkeypatch.setattr(memory_core, "extract", fail)
    with pytest.raises(MemoryServiceError, match="Owned model"):
        store.ingest(proposal)
    assert store.list_records(ListRequest()) == []


@pytest.mark.parametrize("extra", [
    {"request_id": "obsolete"}, {"status": "approved"}, {"kind": "state"},
    {"evidence": "model must extract this"}, {"text": " "}, {"expected_version": 1},
    {"memory_id": "without-version"}, {"memory_id": "test", "expected_version": True},
])
def test_invalid_or_agent_controlled_extraction_fields_are_rejected(
    proposal: IngestRequest, extra: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({**proposal.model_dump(), **extra})


def test_competing_model_calls_hold_no_db_lock_and_recheck_versions(
    store: MemoryStore, saved: MemoryRecord, monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    def extract(text, path, kind=None):
        barrier.wait(timeout=5)
        return Extracted(store=True, kind="state", content=text, evidence=text)
    monkeypatch.setattr(memory_core, "extract", extract)
    request = IngestRequest(
        target="project", memory_id=saved.memory_id, expected_version=1,
        text="Database PostgreSQL", source="fixture://race",
    )
    def ingest():
        try:
            return store.ingest(request).stored
        except MemoryServiceError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest(), range(2)))
    assert True in results and "VERSION_CONFLICT" in results
    assert [row.version for row in store.inspect(saved.memory_id)] == [1, 2]


def test_scope_and_stale_baseline_are_checked_before_outbound_model(
    store: MemoryStore, saved: MemoryRecord, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args):
        pytest.fail("Invalid scope/baseline must not be sent to a model.")
    monkeypatch.setattr(memory_core, "extract", forbidden)
    for project in ("other", None):
        service = MemoryStore(MemoryConfig(db_path=store.config.db_path, project_id=project))
        service.initialize()
        with pytest.raises(MemoryServiceError) as caught:
            service.ingest(IngestRequest(
                target="project" if project else "user", memory_id=saved.memory_id,
                expected_version=1, text="Must not send", source="fixture://scope",
            ))
        assert caught.value.code == "SCOPE_NOT_ALLOWED"
    with pytest.raises(MemoryServiceError) as caught:
        store.ingest(IngestRequest(
            target="project", memory_id=saved.memory_id, expected_version=2,
            text="Must not send", source="fixture://stale",
        ))
    assert caught.value.code == "VERSION_CONFLICT"


def test_old_pending_row_stays_pending_and_unrecalled(
    store: MemoryStore, saved: MemoryRecord,
) -> None:
    from memory_models import RecallRequest
    with store.connection(write=True) as db:
        db.execute(
            "UPDATE memories SET status = 'candidate' WHERE memory_id = ?", (saved.memory_id,)
        )
    assert store.recall(RecallRequest(memory_id=saved.memory_id)) == []
    assert store.inspect(saved.memory_id)[0].status == "candidate"


def test_deletion_binds_the_complete_snapshot_and_removes_all_versions(
    store: MemoryStore, saved: MemoryRecord,
) -> None:
    preview = store.inspect(saved.memory_id)
    store.ingest(IngestRequest(
        target="project", memory_id=saved.memory_id, expected_version=1,
        text="Database PostgreSQL", source="fixture://replacement",
    ))
    with pytest.raises(MemoryServiceError) as caught:
        store.forget(preview)
    assert caught.value.code == "VERSION_CONFLICT"
    assert store.forget(store.inspect(saved.memory_id)) == 2
