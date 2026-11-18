"""Inspection, operator-only deletion and unchanged snapshot safety."""

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Callable

import pytest

import memory_admin
from memory_core import MemoryStore
from memory_models import IngestRequest, MemoryConfig, MemoryRecord, RecallRequest


def test_list_shows_extracted_records_without_review(
    saved: MemoryRecord, capsys: pytest.CaptureFixture[str],
) -> None:
    assert memory_admin.main(["list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["reference"] == saved.reference
    assert rows[0]["status"] == "extracted" and rows[0]["reviewed_at"] is None


def test_list_paginates_and_show_reads_full_history_without_confirmation(
    store: MemoryStore, saved: MemoryRecord, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for version in range(2, 26):
        store.ingest(IngestRequest(
            target="project", memory_id=saved.memory_id, expected_version=version - 1,
            text=f"Database version {version}", source="fixture://pagination",
        ))
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Read-only commands must not prompt"))
    assert memory_admin.main(["list", "--limit", "20"]) == 0
    output = capsys.readouterr()
    assert len(json.loads(output.out)) == 20
    assert json.loads(output.err)["pagination"] == {
        "offset": 0, "limit": 20, "has_more": True, "next_offset": 20,
    }
    assert memory_admin.main(["list", "--limit", "20", "--offset", "20"]) == 0
    output = capsys.readouterr()
    assert len(json.loads(output.out)) == 5
    assert json.loads(output.err)["pagination"]["has_more"] is False
    assert memory_admin.main(["show", "--memory-id", saved.memory_id]) == 0
    assert [row["version"] for row in json.loads(capsys.readouterr().out)] == list(range(1, 26))
    assert memory_admin.main(["--project", "other", "show", "--memory-id", saved.memory_id]) == 1
    output = capsys.readouterr()
    assert "SCOPE_NOT_ALLOWED" in output.err and saved.content not in output.out
    assert len(store.inspect(saved.memory_id)) == 25


def test_review_command_is_no_longer_an_approval_path(saved: MemoryRecord) -> None:
    with pytest.raises(SystemExit) as caught:
        memory_admin.main(["review", "--memory-id", saved.memory_id, "--version", "1"])
    assert caught.value.code == 2


def test_real_process_rejects_piped_deletion(
    saved: MemoryRecord, store: MemoryStore,
) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(memory_admin.__file__)), "forget", "--memory-id", saved.memory_id],
        input=f"delete {saved.memory_id}\n", text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 1 and "REVIEW_REQUIRED" in result.stderr
    assert store.inspect(saved.memory_id) == [saved]


@pytest.mark.parametrize("answer", ["cancel", "", EOFError(), KeyboardInterrupt()])
def test_cancelled_deletion_preserves_data(
    saved: MemoryRecord, store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
    answer: str | BaseException,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    def respond(prompt: str) -> str:
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr("builtins.input", respond)
    assert memory_admin.main(["forget", "--memory-id", saved.memory_id]) == 130
    assert store.inspect(saved.memory_id) == [saved]


def test_deletion_detects_a_new_extraction_during_confirmation(
    saved: MemoryRecord, store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    def change_then_confirm(prompt: str) -> str:
        store.ingest(IngestRequest(
            target="project", memory_id=saved.memory_id, expected_version=1,
            text="Changed while waiting", source="fixture://concurrent",
        ))
        return f"delete {saved.memory_id}"
    monkeypatch.setattr("builtins.input", change_then_confirm)
    assert memory_admin.main(["forget", "--memory-id", saved.memory_id]) == 1
    assert "VERSION_CONFLICT" in capsys.readouterr().err
    monkeypatch.setattr("builtins.input", lambda prompt: f"delete {saved.memory_id}")
    assert memory_admin.main(["forget", "--memory-id", saved.memory_id]) == 0


def test_snapshot_preserves_extractions_history_and_legacy_pending_rows(
    store: MemoryStore, saved: MemoryRecord, tmp_path: Path,
) -> None:
    snapshot, restored = tmp_path / "snapshot.db", tmp_path / "restored.db"
    with store.connection() as keeper:
        keeper.execute("SELECT count(*) FROM memories").fetchone()
        store.ingest(IngestRequest(
            target="project", memory_id=saved.memory_id, expected_version=1,
            text="Database PostgreSQL", source="fixture://snapshot",
        ))
        legacy = store.ingest(IngestRequest(
            target="user", text="Legacy pending fixture", source="fixture://legacy",
        )).memory
        assert legacy is not None
        with store.connection(write=True) as writer:
            writer.execute("UPDATE memories SET status = 'candidate' WHERE memory_id = ?", (legacy.memory_id,))
        with store.connection() as reader:
            expected = [tuple(row) for row in reader.execute("SELECT * FROM memories ORDER BY memory_id, version")]
        assert store.config.db_path.with_name("memory.db-wal").stat().st_size > 0
        with store.connection(write=True) as writer:
            writer.execute("UPDATE memories SET content = 'uncommitted value'")
            assert memory_admin.main([
                "--db", str(store.config.db_path), "backup", "--output", str(snapshot),
            ]) == 0
            writer.rollback()
    assert memory_admin.main(["--db", str(restored), "restore", "--source", str(snapshot)]) == 0
    with closing(sqlite3.connect(restored)) as db:
        assert db.execute("SELECT * FROM memories ORDER BY memory_id, version").fetchall() == expected
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    recovered = MemoryStore(MemoryConfig(db_path=restored, project_id="project-a"))
    recovered.initialize()
    assert recovered.recall(RecallRequest(memory_id=saved.memory_id)) == store.recall(
        RecallRequest(memory_id=saved.memory_id)
    )


@pytest.mark.parametrize("command", ["backup", "restore"])
def test_snapshots_never_overwrite_an_existing_file_or_source(
    store: MemoryStore, saved: MemoryRecord, tmp_path: Path, command: str,
) -> None:
    existing = tmp_path / "existing.db"
    existing.write_bytes(b"keep this file")
    for destination in (existing, store.config.db_path):
        before = destination.read_bytes()
        args = (
            ["--db", str(store.config.db_path), "backup", "--output", str(destination)]
            if command == "backup"
            else ["--db", str(destination), "restore", "--source", str(store.config.db_path)]
        )
        assert memory_admin.main(args) == 1 and destination.read_bytes() == before


def test_invalid_snapshot_and_failed_copy_leave_no_partial_output(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "partial.db"
    assert memory_admin.main([
        "--db", str(destination), "restore", "--source", str(tmp_path / "missing.db"),
    ]) == 1
    connect = sqlite3.connect
    class FailedCopy(sqlite3.Connection):
        def backup(self, target: sqlite3.Connection, **kwargs) -> None:
            target.execute("CREATE TABLE partial(content TEXT)")
            raise sqlite3.OperationalError("Owned copy failure")
    monkeypatch.setattr(memory_admin.sqlite3, "connect", lambda *args, **kwargs: connect(
        *args, **kwargs, factory=FailedCopy
    ))
    assert memory_admin.main([
        "--db", str(store.config.db_path), "backup", "--output", str(destination),
    ]) == 1
    assert not destination.exists()


def test_snapshot_subprocesses_work_without_operator_input(
    store: MemoryStore, saved: MemoryRecord, tmp_path: Path,
) -> None:
    snapshot, restored = tmp_path / "process-backup.db", tmp_path / "process-restored.db"
    for args, status in (
        (["--db", str(store.config.db_path), "backup", "--output", str(snapshot)], "backed_up"),
        (["--db", str(restored), "restore", "--source", str(snapshot)], "restored"),
    ):
        result = subprocess.run(
            [sys.executable, str(Path(memory_admin.__file__)), *args],
            text=True, capture_output=True, timeout=10, check=True,
        )
        assert json.loads(result.stdout)["status"] == status and result.stderr == ""
