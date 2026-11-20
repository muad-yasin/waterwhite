"""The schema and its stepwise migration (plan 3.4, Decision 2).

The store's schema version lives in ``meta.schema_version``. Version n is reached
from version n - 1 by step n. Step 1 creates the schema of ``schema.sql``; step 2
runs ``schema_2.sql`` (a harness log names its harness; a map that did not verify
no longer holds its namespace and derivation), step 3 runs ``schema_3.sql`` (the
hand-off of a harness log keeps its order; one array of a blend has one digest), step 4
runs ``schema_4.sql`` (a harness log says whether its handed texts were compared), step 5 runs
``schema_5.sql`` (a trace row names its segment, and a shard's files carry the result of a
check). All
ship as package data. A later schema adds a step and leaves the earlier ones as they
are, so a store of any age is brought forward one step at a time and never rewritten
in one jump, and a new store runs every step, so it holds the same schema as a
migrated one.

A step may refuse a store: step 3 does when a segment holds two rows for one array
of its blend, which no write of this tool makes. The step raises, its transaction is
rolled back and the store is left at the version it had.
"""

from __future__ import annotations

import sqlite3
from importlib import resources

from ..errors import CacheDigestConflict, WriteRefused

SCHEMA_VERSION = 5


def schema_sql(name: str = "schema.sql") -> str:
    """The text of one schema script from the installed package."""
    return resources.files("waterwhite.store").joinpath(name).read_text(encoding="utf-8")


def split_statements(script: str) -> list[str]:
    """Split an SQL script into statements.

    The statements are run one by one inside the caller's transaction;
    ``sqlite3``'s own script runner would commit first. A buffer that holds only
    comments is never complete, so a comment stays with the statement after it."""
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    tail = [ln for ln in buffer.splitlines() if ln.strip() and not ln.strip().startswith("--")]
    if tail:
        raise ValueError("the schema script ends inside a statement")
    return statements


def _step_1(conn: sqlite3.Connection) -> None:
    for statement in split_statements(schema_sql()):
        conn.execute(statement)


def _step_2(conn: sqlite3.Connection) -> None:
    for statement in split_statements(schema_sql("schema_2.sql")):
        conn.execute(statement)


def _step_3(conn: sqlite3.Connection) -> None:
    """Refuse a store that holds two rows for one array of a blend (the same digest
    twice or two digests), then run ``schema_3.sql``. No row is changed or removed
    to make the index fit: the store says which segment and array, and stays as it
    was."""
    clash = conn.execute(
        "SELECT COALESCE(s.name, 'segment row ' || d.segment_id), d.kind, COUNT(*) "
        "FROM segment_cache_digest d LEFT JOIN run_segment s ON s.segment_id = d.segment_id "
        "WHERE d.shard_position IS NULL "
        "GROUP BY d.segment_id, d.kind HAVING COUNT(*) > 1 ORDER BY d.segment_id, d.kind LIMIT 1"
    ).fetchone()
    if clash is not None:
        raise CacheDigestConflict(
            f"the store cannot be brought to schema version 3: segment {clash[0]} holds "
            f"{clash[2]} digests for its {clash[1]} array, which has one; step 3 changes "
            "nothing (no row is changed to make the index fit)",
            segment=clash[0], kind=clash[1], held=clash[2], version=3,
        )
    for statement in split_statements(schema_sql("schema_3.sql")):
        conn.execute(statement)


def _step_4(conn: sqlite3.Connection) -> None:
    """The mark that says whether a log's hand-off was compared (``schema_4.sql``).
    Every log already in the store gets 0."""
    for statement in split_statements(schema_sql("schema_4.sql")):
        conn.execute(statement)


def _step_5(conn: sqlite3.Connection) -> None:
    """The segment of each trace row, with the index that finds one segment's rows of a
    document, and the table of what a check of a shard's files found (``schema_5.sql``).
    Every shard already in the store has no row there: nothing says it was checked."""
    for statement in split_statements(schema_sql("schema_5.sql")):
        conn.execute(statement)


# version reached -> the step that reaches it from the version before
STEPS = {1: _step_1, 2: _step_2, 3: _step_3, 4: _step_4, 5: _step_5}


def apply_step(conn: sqlite3.Connection, version: int) -> None:
    """Run one step and record the version it reaches. The caller holds the
    transaction, so a step and its version row are written together or not at all."""
    STEPS[version](conn)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (str(version),),
    )


PENDING_KEY = "migration_unsaid"  # meta: the steps committed by a migration whose note was not given yet


def pending_steps(conn: sqlite3.Connection) -> list[int]:
    """The versions of steps that a migration committed and whose note has not been given:
    a migration that stopped in a later step (or was killed in it) leaves them."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (PENDING_KEY,)).fetchone()
    return [int(n) for n in row[0].split(",")] if row and row[0] else []


def migrate(conn: sqlite3.Connection, current: int, on_step=None) -> int:
    """Bring a store from ``current`` to ``SCHEMA_VERSION``, one transaction per
    step. Returns the number of steps run. ``on_step(version)`` is called for each step that
    has committed and whose note is not given yet: first the steps an earlier migration
    committed and stopped after, which it recorded in ``meta`` in the step's own transaction
    (so a process killed in a later step leaves them), then each step this one commits. The
    record is cleared once the last step has committed; a process killed between that and the
    printing of the note loses the note, and nothing else. A step that stops with a database
    error is rolled back and refused as a write that did not happen (``WriteRefused``),
    naming the step and the version the store is left at; a step that refuses the store
    (``CacheDigestConflict``) raises its own error."""
    ran = 0
    carried = pending_steps(conn)
    if on_step is not None:
        for version in carried:
            on_step(version)
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.execute("BEGIN IMMEDIATE")
        try:
            apply_step(conn, version)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = value || ',' || excluded.value",
                (PENDING_KEY, str(version)))
        except BaseException as error:
            conn.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):
                raise WriteRefused(
                    f"the migration step {version} stopped: {error}; the store is left at "
                    f"version {version - 1} and the command can be run again",
                    step=version, left_at=version - 1) from error
            raise
        conn.execute("COMMIT")
        ran += 1
        if on_step is not None:
            on_step(version)
    if pending_steps(conn):
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM meta WHERE key = ?", (PENDING_KEY,))
        conn.execute("COMMIT")
    return ran
