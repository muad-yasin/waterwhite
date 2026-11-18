"""The scratch store of Check 4 (fixture/table/README.md sections 4 and 7): the writes
its rows make to the map of one shard of 500 documents, and the edit one fault makes
to the complete map.

The replica builds the store with the shard registered and no map
(``replica.fixture.build_table_store(folder, "scratch-500", tokenize=True)``). Here:

- ``kill_writer``: the map of the shard is written by a process of its own
  (``jobs/write_scratch_map.py``) that kills itself with SIGKILL as its 101st row of the
  map table goes in (set-up ``w.kill-after-100``);
- ``write_complete_map``: the same writer, not killed (the retry of ``w.retry``, and the
  complete map of ``l.count-499``);
- ``declare_wrong_count``: ``Store.write_map`` called in this process with a declared
  count that is not the shard's (set-up ``w.declare-499``);
- ``count_edit``: the stored count of the complete map set to 499 outside the write
  API (set-up ``l.count-499``).

What the store holds afterwards is read by the rows through ``show``, never from the
tables, except by the edit that a fault is.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "fixture" / "spec" / "fixture_spec.toml"
WRITER = Path(__file__).resolve().parent / "jobs" / "write_scratch_map.py"

SHARD = "c04/shard-500"
INPUT = "c04/shard-500.input"
DOCUMENTS = 500
WRITER_TIMEOUT = 120.0  # seconds


class ScratchError(Exception):
    """A set-up of the scratch store did not do what the table says it does."""


def _settings() -> str:
    """The tokenizer settings of the fixture spec, as the writer takes them."""
    with open(SPEC, "rb") as f:
        tokenizer = tomllib.load(f)["tokenizer"]
    return json.dumps({"append_eod": tokenizer["append_eod"],
                       "eod_in_span": tokenizer["eod_in_span"],
                       "eod_token_id": tokenizer["eod_token_id"]})


def _run_writer(store_file: Path, kill_after: int) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(
        [sys.executable, str(WRITER), str(store_file), SHARD, INPUT, _settings(),
         str(kill_after)],
        capture_output=True, text=True, timeout=WRITER_TIMEOUT, env=env,
        cwd=Path(store_file).parent)


def kill_writer(store_file: Path, after: int = 100) -> None:
    """Write the shard's map and kill the writer as its row number ``after`` (counted
    from 0) of the map table is handed to the store. The process must have died by the
    kill: a writer that finished, or failed another way, is an error of the set-up."""
    done = _run_writer(store_file, after)
    if done.returncode != -signal.SIGKILL or "finished" in done.stdout:
        raise ScratchError(f"the writer was to be killed after {after} rows and ended with "
                           f"exit code {done.returncode}: {done.stdout[-200:]} "
                           f"{done.stderr[-400:]}")


def write_complete_map(store_file: Path) -> subprocess.CompletedProcess:
    """Write the shard's map again, not killed. The finished process is returned; a
    refusal is the caller's to read (``refusal``)."""
    return _run_writer(store_file, -1)


def refusal(done: subprocess.CompletedProcess) -> str | None:
    """The fixed error name a writer ended with, from its last line of standard error
    (``name: detail``), or None when it finished."""
    if done.returncode == 0:
        return None
    lines = done.stderr.strip().splitlines()
    last = lines[-1] if lines else ""
    parts = last.split(": ")
    return parts[1] if len(parts) > 1 else last


def instance_and_digest(store_file: Path) -> tuple[str, str]:
    """The instance id and the registered content digest of the shard, from the
    store's names (a set-up looks its target up, as the faults do)."""
    with closing(sqlite3.connect(store_file)) as db:
        rows = db.execute(
            "SELECT a.instance_id, a.content_digest FROM artifact a "
            "JOIN artifact_name n USING (artifact_id) WHERE n.name = ?", (SHARD,)).fetchall()
    if len(rows) != 1:
        raise ScratchError(f"the store holds {len(rows)} artifacts named {SHARD}, one was "
                           "expected")
    return rows[0][0], rows[0][1]


def declare_wrong_count(store_file: Path, declared: int = 499) -> str:
    """Call ``Store.write_map`` for the shard with ``declared`` documents, which is not
    the shard's count. Returns the fixed name of the refusal. A write that is accepted
    is an error of the set-up: nothing would then be refused."""
    from waterwhite import vocabulary as v
    from waterwhite.errors import WaterwhiteError
    from waterwhite.records import MapRecord, MapRow
    from waterwhite.store import Store

    instance, digest = instance_and_digest(store_file)
    rows = tuple(MapRow(n, n, n + 1, v.UNKNOWN) for n in range(declared))
    store = Store.open(store_file)
    try:
        try:
            store.write_map(
                instance,
                MapRecord(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_NOT_RUN,
                          v.UNKNOWN, True, True, rows),
                declared_count=declared, declared_digest=digest)
        except WaterwhiteError as refused:
            return refused.name
    finally:
        store.close()
    raise ScratchError(f"write_map accepted {declared} declared documents for a shard of "
                       f"{DOCUMENTS}")


def count_edit(store_file: Path) -> None:
    """l.count-499: the stored count of the complete map is set to 499, outside the write
    API. The map is written first (it is the table's "complete 500-row map") and the edit
    must change exactly the one binding."""
    done = write_complete_map(store_file)
    if done.returncode != 0:
        raise ScratchError(f"the complete map could not be written: {done.stderr[-400:]}")
    with closing(sqlite3.connect(store_file)) as db:
        with db:
            changed = db.execute(
                "UPDATE binding SET declared_count = 499 WHERE artifact_id = "
                "(SELECT artifact_id FROM artifact_name WHERE name = ?) AND declared_count = ?",
                (SHARD, DOCUMENTS)).rowcount
    if changed != 1:
        raise ScratchError(f"the edit changed {changed} binding(s), the catalogue names one")


# --- What the store holds, in the words of the table ---------------------------------------

NO_MAP = "no map for the shard"
COMPLETE = "the complete map: 500 rows"
PARTIAL = "a map with fewer than 500 rows"
PARTIAL_MAP = "a partial map"
ROWS_WITHOUT_BINDING = "rows without a binding"
MATCHES = "a map that matches its shard: 500 rows, its count and digest bound to the shard"
MISMATCHES = "a map whose count or digest does not match the shard"
REJECTED = ("a rejection with a named mismatch: declared count mismatch or declared digest "
            "mismatch")
SENTENCES = frozenset({NO_MAP, COMPLETE, PARTIAL, PARTIAL_MAP, ROWS_WITHOUT_BINDING, MATCHES,
                       MISMATCHES, REJECTED})
NAMED_MISMATCHES = ("declared count mismatch", "declared digest mismatch")


def held(shown: dict, refused: str | None = None) -> set[str]:
    """The sentences of the table's rows that describe what ``show`` of the shard prints
    (its bindings, its stored rows, its registered digest) and how a write ended
    (``refused``: the fixed name of the refusal, or None)."""
    bindings = shown.get("bindings") or []
    rows = shown.get("stored_rows") or 0
    digest = shown.get("content_digest")
    out: set[str] = set()
    if not bindings and rows == 0:
        out.add(NO_MAP)
    if not bindings and rows > 0:
        out.add(ROWS_WITHOUT_BINDING)
    if bindings and rows < DOCUMENTS:
        out |= {PARTIAL, PARTIAL_MAP}
    if bindings and any(b["declared_count"] != DOCUMENTS or b["declared_digest"] != digest
                        for b in bindings):
        out.add(MISMATCHES)
    if (len(bindings) == 1 and rows == DOCUMENTS and bindings[0]["declared_count"] == DOCUMENTS
            and bindings[0]["declared_digest"] == digest):
        out |= {COMPLETE, MATCHES}
    if refused in NAMED_MISMATCHES:
        out.add(REJECTED)
    return out
