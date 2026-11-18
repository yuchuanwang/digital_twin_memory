"""First-release schema: one table, no migration engine or derived indexes."""

import sqlite3

from memory_models import MemoryServiceError


SCHEMA_VERSION = 1
CREATE_TABLE = """
CREATE TABLE memories (
    memory_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    base_version INTEGER NOT NULL,
    project_id TEXT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    evidence TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'extracted',
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    PRIMARY KEY (memory_id, version)
)
"""

_COLUMNS = {
    "memory_id": ("TEXT", 1, 1),
    "version": ("INTEGER", 1, 2),
    "base_version": ("INTEGER", 1, 0),
    "project_id": ("TEXT", 0, 0),
    "kind": ("TEXT", 1, 0),
    "content": ("TEXT", 1, 0),
    "source": ("TEXT", 1, 0),
    "evidence": ("TEXT", 1, 0),
    "status": ("TEXT", 1, 0),
    "created_at": ("TEXT", 1, 0),
    "reviewed_at": ("TEXT", 0, 0),
}


def schema_exists(db: sqlite3.Connection) -> bool:
    """Recognize this schema or an empty file; never rewrite an incompatible DB."""
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        columns = db.execute("PRAGMA table_xinfo(memories)").fetchall()
        # Defaults may differ in legacy databases; all inserts supply status explicitly.
        if (
            {row[1]: (row[2].upper(), row[3], row[5]) for row in columns} == _COLUMNS
            and all(row[6] == 0 for row in columns)
        ):
            return True
    if version == 0 and not db.execute(
        "SELECT 1 FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*' LIMIT 1"
    ).fetchone():
        return False
    raise MemoryServiceError(
        "SCHEMA_MIGRATION_REQUIRED",
        "Database schema is incompatible. Use a new path; no data was migrated.",
    )
