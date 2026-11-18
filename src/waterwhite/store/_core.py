"""What the store's write, read and check code share: the connection, the one
transaction rule, stored paths, and small lookups.

Plan sections: 3.4 (one SQLite file, write-ahead logging, full synchronous
writes, one transaction per write), Decision 2.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .. import ids
from .. import vocabulary as v
from ..errors import InvalidRecord, UnsupportedFormat, WriteRefused
from . import artifact_files
from .views import FILE_ABSENT, FILE_NONE, FILE_PARTIAL, FILE_PRESENT

# --- An artifact's file as found on disk --------------------------------------------


@dataclass(frozen=True)
class DiskState:
    """What a look at an artifact's file gives.

    ``state`` is one of the FILE_* words. ``digest`` is set when the file is
    present. ``count`` is set by ``look_at`` when the file is present; it is None,
    with ``unreadable`` saying why, when a shard's index cannot be read as an
    index."""

    state: str
    path: str
    count: int | None = None
    digest: str | None = None
    unreadable: str | None = None


def look_at(kind: str, path: Path | None) -> DiskState:
    """Read the artifact at ``path``. A missing file is a state, never an error."""
    if path is None or kind not in (v.KIND_JSONL, v.KIND_SHARD):
        return DiskState(FILE_NONE, "")
    if kind == v.KIND_JSONL:
        if not path.is_file():
            return DiskState(FILE_ABSENT, str(path))
        found = artifact_files.read_jsonl(path)
        return DiskState(FILE_PRESENT, str(path), found.line_count, found.digest)
    bin_path, idx_path = artifact_files.shard_paths(path)
    have = (bin_path.is_file(), idx_path.is_file())
    if not any(have):
        return DiskState(FILE_ABSENT, str(path))
    if not all(have):
        return DiskState(FILE_PARTIAL, str(path))
    digest = artifact_files.shard_digest_only(path)
    try:
        count = artifact_files.read_shard_index(idx_path).document_count
    except UnsupportedFormat as exc:
        return DiskState(FILE_PRESENT, str(path), None, digest, exc.detail)
    return DiskState(FILE_PRESENT, str(path), count, digest)


# --- The shared base ----------------------------------------------------------------

RowHook = Callable[[str, int], None]


class StoreCore:
    """Connection, transaction and path handling. ``Store`` adds the methods."""

    def __init__(self, conn: sqlite3.Connection, path: Path, readonly: bool) -> None:
        self._conn = conn
        self._path = path
        self._dir = path.parent
        self._readonly = readonly
        self._closed = False
        self._row_hook: RowHook | None = None

    # -- transactions --

    def _require_writable(self) -> None:
        if self._closed:
            raise WriteRefused("the store is closed")
        if self._readonly:
            raise WriteRefused("the store was opened read-only")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One write: BEGIN IMMEDIATE, then everything or nothing.

        Any error rolls the whole write back. A constraint of the schema that a
        record breaks is reported as ``InvalidRecord``."""
        self._require_writable()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise WriteRefused(f"the store cannot be written now: {exc}") from exc
        try:
            yield self._conn
        except sqlite3.IntegrityError as exc:
            self._conn.execute("ROLLBACK")
            raise InvalidRecord(f"the schema refused the record: {exc}") from exc
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _feed(self, table: str, rows: Iterable[tuple]) -> Iterator[tuple]:
        """Hand rows to SQLite one by one, telling the row hook how many rows of
        ``table`` this write has stored so far (see ``Store.set_row_hook``)."""
        hook = self._row_hook
        if hook is None:
            yield from rows
            return
        written = 0
        for row in rows:
            hook(table, written)
            yield row
            written += 1
        hook(table, written)

    # -- paths --

    def _stored_path(self, path) -> str:
        """The form a path takes in the store: relative to the store's folder when
        the artifact lies under it, absolute otherwise."""
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self._dir).as_posix()
        except ValueError:
            return str(resolved)

    def _disk_path(self, stored: str | None) -> Path | None:
        if stored is None:
            return None
        p = Path(stored)
        return p if p.is_absolute() else self._dir / p

    # -- lookups --

    def _artifact_row(self, instance_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM artifact WHERE instance_id = ?", (instance_id,)
        ).fetchone()

    def _artifact_id(self, instance_id: str) -> int | None:
        row = self._conn.execute(
            "SELECT artifact_id FROM artifact WHERE instance_id = ?", (instance_id,)
        ).fetchone()
        return None if row is None else row[0]

    def _segment_row(self, name: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT s.*, r.name AS run_name FROM run_segment s JOIN run r ON r.run_id = s.run_id "
            "WHERE s.name = ?",
            (name,),
        ).fetchone()


class OccurrenceResolver:
    """Turns (instance id, locator) into the store's row id, for one write.

    Lookups go to the store one by one; once many go to the same artifact, its
    whole locator table is loaded instead."""

    _BULK_AFTER = 64

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._artifact_ids: dict[str, int | None] = {}
        self._single: dict[tuple[str, str], int | None] = {}
        self._bulk: dict[str, dict[str, int]] = {}
        self._asked: dict[str, int] = {}

    def artifact_id(self, instance_id: str) -> int | None:
        if instance_id not in self._artifact_ids:
            row = self._conn.execute(
                "SELECT artifact_id FROM artifact WHERE instance_id = ?", (instance_id,)
            ).fetchone()
            self._artifact_ids[instance_id] = None if row is None else row[0]
        return self._artifact_ids[instance_id]

    def find(self, instance_id: str, locator: str) -> int | None:
        table = self._bulk.get(instance_id)
        if table is not None:
            return table.get(locator)
        key = (instance_id, locator)
        if key in self._single:
            return self._single[key]
        artifact_id = self.artifact_id(instance_id)
        if artifact_id is None:
            self._single[key] = None
            return None
        asked = self._asked.get(instance_id, 0) + 1
        self._asked[instance_id] = asked
        if asked > self._BULK_AFTER:
            table = {
                row[0]: row[1]
                for row in self._conn.execute(
                    "SELECT locator, occurrence_id FROM occurrence WHERE artifact_id = ?",
                    (artifact_id,),
                )
            }
            self._bulk[instance_id] = table
            return table.get(locator)
        row = self._conn.execute(
            "SELECT occurrence_id FROM occurrence WHERE artifact_id = ? AND locator = ?",
            (artifact_id, locator),
        ).fetchone()
        found = None if row is None else row[0]
        self._single[key] = found
        return found

    def forget(self, instance_id: str) -> None:
        """Drop what is cached for an artifact whose rows this write has changed."""
        self._artifact_ids.pop(instance_id, None)
        self._bulk.pop(instance_id, None)
        self._asked.pop(instance_id, None)
        for key in [k for k in self._single if k[0] == instance_id]:
            del self._single[key]
