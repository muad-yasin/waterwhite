"""The schema and its stepwise migration (plan 3.4, Decision 2).

The store's schema version lives in ``meta.schema_version``. Version n is reached
from version n - 1 by step n. Step 1 creates the schema of ``schema.sql``; step 2
runs ``schema_2.sql`` (a harness log names its harness; a map that did not verify
no longer holds its namespace and derivation), step 3 runs ``schema_3.sql`` (the
hand-off of a harness log keeps its order; one array of a blend has one digest), step 4
runs ``schema_4.sql`` (a harness log says whether its handed texts were compared). All
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

from ..errors import CacheDigestConflict

SCHEMA_VERSION = 4


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


# version reached -> the step that reaches it from the version before
STEPS = {1: _step_1, 2: _step_2, 3: _step_3, 4: _step_4}


def apply_step(conn: sqlite3.Connection, version: int) -> None:
    """Run one step and record the version it reaches. The caller holds the
    transaction, so a step and its version row are written together or not at all."""
    STEPS[version](conn)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (str(version),),
    )


def migrate(conn: sqlite3.Connection, current: int) -> int:
    """Bring a store from ``current`` to ``SCHEMA_VERSION``, one transaction per
    step. Returns the number of steps run."""
    ran = 0
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.execute("BEGIN IMMEDIATE")
        try:
            apply_step(conn, version)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        ran += 1
    return ran
