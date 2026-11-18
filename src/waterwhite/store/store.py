"""The evidence store: one SQLite file and the only writer of every record.

``Store`` opens the file with write-ahead logging, full synchronous writes and
foreign keys on. Every write method is exactly one transaction: it checks first,
writes afterwards, and a refusal, an error or a killed process leaves nothing of
it behind. The read methods live in ``_reads``, the loud checks in ``_integrity``;
``docs/store.md`` describes all of them.

Plan sections: 3.4 (the store, its refusals and its four loud checks), 3.1 and
Decision 29 (identity), 3.2 (labels), 3.3 (how each link is obtained), 3.5 and
Decision 19 (step traces, written once), Decision 2 (one transactional file).
DECISIONS.md D-14 (a second registration that disagrees is refused), D-15 (a
stated parent lies in a declared input), D-18 (what each loud check compares),
D-20 (one verified map per namespace and derivation), D-21 (a sole handed record), D-24
(an edge keeps the name of its job; the reader's order of maps).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .. import extensions, ids, normalise
from .. import vocabulary as v
from ..errors import (
    ArtifactDigestMismatch,
    DeclaredCountMismatch,
    DeclaredDigestMismatch,
    InvalidRecord,
    LabelStrongerThanDerivation,
    NewerSchemaVersion,
    NotAStore,
    Refused,
    StoreFileExists,
    UnsupportedFormat,
    WriteRefused,
)
from ..ids import checkpoint_instance_id, mixture_instance_id
from ..records import (
    OWN_ARTIFACT,
    DropRecord,
    EdgeInput,
    EdgeRecord,
    HarnessLog,
    HarnessRead,
    MapRecord,
    MapRow,
    OccurrenceRecord,
    OccurrenceRef,
    SnapshotShard,
    StepRef,
    StepTrace,
    TraceRow,
    TraceSample,
)
from . import artifact_files
from ._core import OccurrenceResolver, RowHook, look_at
from ._integrity import IntegrityMixin
from ._migrate import SCHEMA_VERSION, apply_step, migrate
from .views import (
    FILE_PRESENT,
    CacheDigest,
    Launch,
    Segment,
    SegmentStep,
    ShuffleDigest,
    StoredEdge,
)

# ``OWN_ARTIFACT`` (``records``) and the two id functions (``ids``) are named here
# as well, for code that imports them from the store.
__all__ = ["Store", "checkpoint_instance_id", "mixture_instance_id", "OWN_ARTIFACT"]

RUN_RECORD_FORMAT = v.RUN_RECORD_FORMAT

# The derivations an edge between JSONL records can carry. Position and token
# matching derive a tokenization map, never an edge.
_EDGE_DERIVATIONS = (
    v.DERIVED_STATED,
    v.DERIVED_POINTER,
    v.DERIVED_FUNCTION,
    v.DERIVED_FINGERPRINT,
    v.DERIVED_FILE,
    v.DERIVED_NONE,
)
_MAP_DERIVATIONS = (v.DERIVED_POSITION, v.DERIVED_TOKEN_MATCHING)
# A source link has no parent record behind it, so it is never stated by a step
# and never inferred by a rule that names a parent: it rests on a publisher's
# pointer, on what is known of the record's file, or on nothing.
_SOURCE_LINK_DERIVATIONS = (v.DERIVED_POINTER, v.DERIVED_FILE, v.DERIVED_NONE)
_TRACE_CAUSES = frozenset(v.TRACE_CAUSES)
_TRACE_MISSING_REASONS = frozenset(v.TRACE_MISSING_STORED_REASONS)
_CACHE_KINDS = v.CACHE_KINDS_PER_SHARD + v.CACHE_KINDS_PER_BLEND


# --- Small checks (no store, no disk) ------------------------------------------------


def _need(condition: object, detail: str, **context: object) -> None:
    if not condition:
        raise InvalidRecord(detail, **context)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _line_number(locator: object, what: str) -> int:
    _need(isinstance(locator, str), f"{what}: a locator is text, got {locator!r}")
    try:
        kind, number = ids.parse_locator(locator)
    except ValueError:
        raise InvalidRecord(f"{what}: not a locator: {locator!r}") from None
    _need(kind == "L" and number >= 1, f"{what}: a JSONL record is named by a line locator, got {locator!r}")
    return number


def _check_ref(ref: object, what: str) -> None:
    _need(isinstance(ref, OccurrenceRef), f"{what}: expected an occurrence reference, got {ref!r}")
    _need(isinstance(ref.instance_id, str), f"{what}: the instance id is text")
    _line_number(ref.locator, what)


def _check_fingerprint(fingerprint: object, what: str) -> None:
    _need(
        fingerprint is None
        or (isinstance(fingerprint, bytes) and len(fingerprint) == ids.FINGERPRINT_BYTES),
        f"{what}: a fingerprint is {ids.FINGERPRINT_BYTES} bytes or None",
    )


def _check_step(step: object, what: str) -> None:
    _need(isinstance(step, StepRef), f"{what}: expected a step reference, got {step!r}")
    _need(
        _is_text(step.name) and isinstance(step.version, str) and isinstance(step.param_digest, str),
        f"{what}: a step has a name, a version and a parameter digest",
    )


def _check_inputs(inputs: Iterable[str]) -> tuple[str, ...]:
    listed = tuple(inputs)
    _need(all(_is_text(i) for i in listed), "an input is named by its instance id")
    return listed


def _check_label(label: str, derivation: str, verification: str | None, what: str) -> None:
    """The label rule (plan 3.2, 3.4, 2.2(f)), stated once for edges, maps and rows."""
    _need(label in v.LABELS, f"{what}: unknown label {label!r}")
    allowed = v.labels_allowed(derivation, verification)
    if label not in allowed:
        raise LabelStrongerThanDerivation(
            f"{what}: label {label!r} with derivation {derivation!r} and verification "
            f"{verification!r}; allowed: {', '.join(allowed)}",
            label=label, derivation=derivation, verification=verification,
        )


def _check_log(log: object, what: str) -> None:
    _need(isinstance(log, HarnessLog), f"{what}: expected a harness log")
    _need(_is_text(log.job_name), f"{what}: a harness log names its job")
    _need(_is_text(log.log_digest), f"{what}: a harness log carries the digest of its file")
    _need(log.model is None or isinstance(log.model, str), f"{what}: the model entry is text or None")
    _need(log.harness is None or log.harness in v.HARNESS_KINDS,
          f"{what}: a harness log names its harness as one of "
          f"{', '.join(repr(k) for k in v.HARNESS_KINDS)}, or does not name it")
    for ref in log.handed:
        _check_ref(ref, f"{what}, hand-off")
    for read in log.reads:
        _need(isinstance(read, HarnessRead), f"{what}: expected a harness read")
        if read.occurrence is not None:
            _check_ref(read.occurrence, f"{what}, read")
        _need(
            read.occurrence is not None or _is_text(read.outside_ref),
            f"{what}: a read names a handed record or what was asked for",
        )


def _check_edge(edge: object) -> None:
    _need(isinstance(edge, EdgeRecord), f"expected an edge record, got {edge!r}")
    what = f"edge of {edge.child_locator!r}"
    _line_number(edge.child_locator, what)
    _check_step(edge.transform, what)
    _need(edge.parent_mode in v.PARENT_MODES, f"{what}: unknown parent mode {edge.parent_mode!r}")
    _need(
        edge.derivation in _EDGE_DERIVATIONS,
        f"{what}: an edge cannot carry the derivation {edge.derivation!r}",
    )
    _need(
        edge.verification is None or edge.verification in v.VERIFICATIONS,
        f"{what}: unknown verification result {edge.verification!r}",
    )
    _check_label(edge.label, edge.derivation, edge.verification, what)
    seen: set[OccurrenceRef] = set()
    selected = 0
    for item in edge.inputs:
        _need(isinstance(item, EdgeInput), f"{what}: expected an edge input")
        _check_ref(item.parent, what)
        _need(item.parent not in seen, f"{what}: {item.parent.key} is named twice")
        seen.add(item.parent)
        if item.selected:
            selected += 1
            _need(item.used, f"{what}: the selected parent is a parent, not in scope only")
        _need(
            item.used or edge.is_generation,
            f"{what}: only a generation edge holds a record that is in scope only",
        )
    if edge.parent_mode == v.MODE_ALL:
        _need(selected == 0, f'{what}: an "all" edge selects no parent')
    elif edge.derivation == v.DERIVED_STATED:
        _need(selected == 1, f'{what}: a stated "any" edge names the one parent its step kept')
    else:
        _need(selected <= 1, f'{what}: an "any" edge selects at most one parent')
    # An edge keeps the name of its job (DECISIONS.md D-24): every generation edge
    # names one and no other edge does.
    if edge.is_generation:
        _need(
            _is_text(edge.job_name),
            f"{what}: a generation edge names its job (job_name); for a job run in the "
            "harness that is the job name of its harness log",
        )
    else:
        _need(edge.job_name is None, f"{what}: only a generation edge names a job")
    _need(
        not edge.handed_texts or edge.is_generation,
        f"{what}: only a generation edge carries the texts a job was handed",
    )
    _need(isinstance(edge.handed_texts, tuple), f"{what}: the handed texts are a tuple")
    for pair in edge.handed_texts:
        _need(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], str),
            f"{what}: a handed text is a (reference, text) pair",
        )
        _check_ref(pair[0], f"{what}, handed text")
    if edge.harness_log is not None:
        _need(edge.is_generation, f"{what}: only a generation edge carries a harness log")
        _check_log(edge.harness_log, what)
        _need(
            edge.harness_log.job_name == edge.job_name,
            f"{what}: the edge names the job {edge.job_name!r} and carries the harness log "
            f"of the job {edge.harness_log.job_name!r}",
            job=edge.job_name, log_job=edge.harness_log.job_name,
        )
        log_texts = edge.harness_log.handed_texts
        _need(isinstance(log_texts, tuple), f"{what}: the handed texts of its harness log are a tuple")
        for pair in log_texts:
            _need(
                isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], str),
                f"{what}: a handed text of its harness log is a (reference, text) pair",
            )
            _check_ref(pair[0], f"{what}, handed text of its harness log")
        handed = set(edge.harness_log.handed)
        for ref, _text in (*edge.handed_texts, *log_texts):
            _need(
                ref in handed,
                f"{what}: a text was handed under {ref.key}, which lies outside the hand-off "
                f"of job {edge.harness_log.job_name!r}",
                job=edge.harness_log.job_name, ref=ref.key,
            )
        for item in edge.inputs:
            _need(
                item.parent in handed,
                f"{what}: {item.parent.key} lies outside the hand-off of job "
                f"{edge.harness_log.job_name!r}",
                job=edge.harness_log.job_name, parent=item.parent.key,
            )
        if len(handed) == 1:
            # A job handed exactly one record: that record is a parent of each
            # output, whatever the output states (DECISIONS.md D-21).
            (sole,) = handed
            _need(
                any(item.parent == sole and item.used for item in edge.inputs),
                f"{what}: job {edge.harness_log.job_name!r} was handed one record, "
                f"{sole.key}, and the edge does not name it as a parent; a sole handed "
                "record is a parent of each output",
                job=edge.harness_log.job_name, parent=sole.key,
            )


def _check_drop(drop: object) -> None:
    _need(isinstance(drop, DropRecord), f"expected a drop record, got {drop!r}")
    _check_ref(drop.input, "drop")
    _need(drop.reason in v.DROP_REASONS, f"drop of {drop.input.key}: unknown reason {drop.reason!r}")
    _need(_is_text(drop.step_name), f"drop of {drop.input.key}: a drop names its step")
    _need(drop.group_key is None or isinstance(drop.group_key, str), "a group key is text or None")


def _check_stated_parents(edge: EdgeRecord, inputs: tuple[str, ...]) -> None:
    """A stated edge's parent lies in an artifact the step declared as an input, or
    in the artifact being written (DECISIONS.md D-15). The inputs are part of the
    instance id; a parent outside them would be history the id does not cover. A
    record that is in scope only is not a parent and is not held to this."""
    for item in edge.inputs:
        if not item.used or item.parent.instance_id == OWN_ARTIFACT:
            continue
        _need(
            item.parent.instance_id in inputs,
            f"edge of {edge.child_locator!r}: the parent {item.parent.key} lies in no "
            "declared input of the step and not in the artifact itself",
            parent=item.parent.key,
        )


def _edge_identity(edge: EdgeRecord) -> list:
    """An edge in the form the edge digest takes (plan 3.1, Decision 29): the child's
    locator, its parents, and for an "any" edge the parent the step kept. A record
    that is in scope only is not a parent and is not part of it."""
    parents = [(i.parent.instance_id or None, i.parent.locator) for i in edge.inputs if i.used]
    selected = None
    for item in edge.inputs:
        if item.selected:
            selected = (item.parent.instance_id or None, item.parent.locator)
    return ids.edge_record(edge.child_locator, parents, selected)


def _bind_ref(ref: OccurrenceRef, instance: str) -> OccurrenceRef:
    return OccurrenceRef(instance, ref.locator) if ref.instance_id == OWN_ARTIFACT else ref


def _bind_edge(edge: EdgeRecord, instance: str) -> EdgeRecord:
    """Replace ``OWN_ARTIFACT`` by the artifact's instance id, once it is known."""
    log = edge.harness_log
    named = [i.parent for i in edge.inputs] + [ref for ref, _ in edge.handed_texts]
    if log is not None:
        named += list(log.handed) + [r.occurrence for r in log.reads if r.occurrence is not None]
        named += [ref for ref, _ in log.handed_texts]
    if all(ref.instance_id != OWN_ARTIFACT for ref in named):
        return edge
    inputs = tuple(dataclasses.replace(i, parent=_bind_ref(i.parent, instance)) for i in edge.inputs)
    if log is not None:
        log = dataclasses.replace(
            log,
            handed=tuple(_bind_ref(r, instance) for r in log.handed),
            reads=tuple(
                r if r.occurrence is None
                else dataclasses.replace(r, occurrence=_bind_ref(r.occurrence, instance))
                for r in log.reads
            ),
            handed_texts=tuple((_bind_ref(r, instance), t) for r, t in log.handed_texts),
        )
    return dataclasses.replace(
        edge, inputs=inputs, harness_log=log,
        handed_texts=tuple((_bind_ref(r, instance), t) for r, t in edge.handed_texts),
    )


def _row_form(row: MapRow) -> tuple:
    return (
        row.ordinal, row.token_start, row.token_end, row.label, row.parent,
        frozenset(row.candidates), row.fingerprint,
    )


def _opt(holder: dict, key: str, kind: type, what: str):
    """A field of the run record: its value, or None when absent or null."""
    value = holder.get(key)
    if value is None:
        return None
    ok = _is_int(value) if kind is int else isinstance(value, kind)
    _need(ok, f"run record: {what}.{key} should be {kind.__name__}, got {value!r}")
    return value


def _stated_version(path) -> int | None:
    """The schema version a file says it has, read without opening it as a store;
    None when it says none or cannot be read. Every refusal is ``Store.open``'s."""
    target = Path(path).resolve()
    if not target.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        return None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
    return int(row[0]) if row is not None and str(row[0]).isdigit() else None

class Store(IntegrityMixin):
    """The evidence store. Use ``Store.create`` or ``Store.open``."""

    # Called as (version found, version reached) whenever ``open`` migrates a store of an
    # older schema version, so that a command that opens a store for writing can say so
    # in its output instead of migrating silently (the command line sets it for a command).
    migration_hook: Callable[[int, int], None] | None = None

    # ------------------------------------------------------------------------------
    # Opening and closing
    # ------------------------------------------------------------------------------

    @staticmethod
    def _connect(path: Path, readonly: bool) -> sqlite3.Connection:
        if readonly:
            conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, isolation_level=None)
        else:
            conn = sqlite3.connect(str(path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        if not readonly:
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                conn.close()
                raise WriteRefused(f"the store file cannot use write-ahead logging (got {mode!r})")
            conn.execute("PRAGMA synchronous = FULL")
        return conn

    @classmethod
    def create(
        cls,
        path,
        *,
        normaliser: str,
        verified_joins: Iterable[tuple[str, str, str]] = (),
    ) -> "Store":
        """Create a store file. Writes ``meta`` (schema version, normaliser) and the
        given ``verified_join`` rows, all in one transaction. ``verified_joins``
        are the (release tag, sampler type, sampler class) combinations the
        exposure join was verified for (plan F6); they are written here and at no
        other time. Refuses a path that already holds a file (``StoreFileExists``)."""
        target = Path(path).resolve()
        if target.exists():
            raise StoreFileExists(f"{target} exists; a store is never created over a file",
                                  path=str(target))
        if not target.parent.is_dir():
            raise WriteRefused(f"the folder {target.parent} does not exist")
        name, _, version = normaliser.partition("/") if isinstance(normaliser, str) else ("", "", "")
        _need(name and version, f'a normaliser is named "<name>/<version>", got {normaliser!r}')
        listed = [tuple(j) for j in verified_joins]
        for join in listed:
            _need(
                len(join) == 3 and all(_is_text(x) for x in join),
                f"a verified join is (release tag, sampler type, sampler class), got {join!r}",
            )
        joins = sorted(set(listed))
        conn = cls._connect(target, readonly=False)
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for step in range(1, SCHEMA_VERSION + 1):
                    apply_step(conn, step)
                conn.execute("INSERT INTO meta (key, value) VALUES ('normaliser', ?)", (normaliser,))
                conn.executemany(
                    "INSERT INTO verified_join (release_tag, sampler_type, sampler_class) "
                    "VALUES (?, ?, ?)",
                    joins,
                )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        except BaseException:
            conn.close()
            # Nothing was there before this call: leave nothing half-made behind.
            for leftover in (target, Path(f"{target}-wal"), Path(f"{target}-shm")):
                if leftover.exists():
                    os.unlink(leftover)
            raise
        return cls(conn, target, readonly=False)

    @classmethod
    def migrate(cls, path) -> tuple[int, int]:
        """Bring the store at ``path`` to this build's schema version, step by step
        (each step one transaction), and close it. Returns the version the file had
        and the version it has now; a store already at this version is opened
        read-only and left as it is. The refusals are those of ``open``."""
        found = _stated_version(path)
        current = found is not None and found == SCHEMA_VERSION
        with cls.open(path, readonly=current) as store:
            reached = store.schema_version
        return (found if found is not None else reached), reached

    @classmethod
    def open(cls, path, *, readonly: bool = False) -> "Store":
        """Open an existing store. A store of an older schema version is migrated
        step by step first; a read-only open of such a store is refused. A path
        that holds no store is ``NotAStore``; a store written by a newer version
        of the tool is ``NewerSchemaVersion``."""
        target = Path(path).resolve()
        if not target.is_file():
            raise NotAStore(f"no file at {target}", path=str(target))
        try:
            conn = cls._connect(target, readonly)
        except sqlite3.DatabaseError as exc:
            raise NotAStore(f"{target} cannot be opened as a store: {exc}", path=str(target)) from exc
        try:
            try:
                row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            except sqlite3.DatabaseError as exc:
                raise NotAStore(f"{target} holds no store: {exc}", path=str(target)) from exc
            if row is None:
                raise NotAStore(f"{target} holds no schema version", path=str(target))
            try:
                found = int(row[0])
            except ValueError:
                raise NotAStore(
                    f"{target} holds the schema version {row[0]!r}, which is not a number",
                    path=str(target),
                ) from None
            if found > SCHEMA_VERSION:
                raise NewerSchemaVersion(
                    f"the store has schema version {found}; this tool reads up to {SCHEMA_VERSION}",
                    found=found, reads=SCHEMA_VERSION, path=str(target),
                )
            # Version 1 was revised in place once, before any store left the test
            # folders: the edge gained the name of its job (DECISIONS.md D-24). A
            # file made before that says "1" and lacks the column. It is refused by
            # name here, before any migration step, and not by an SQL error in the
            # middle of a step or a query.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(edge)")}
            if "job_name" not in columns:
                raise NotAStore(
                    f"{target} holds a store of schema version {found} as it was before an "
                    "edge kept the name of its job; this build does not read it, and there "
                    "is no migration step for it: make the store again",
                    path=str(target),
                )
            if found < SCHEMA_VERSION:
                if readonly:
                    raise Refused(
                        f"the store has schema version {found} and must be migrated to "
                        f"{SCHEMA_VERSION} first (the command line: waterwhite migrate --store FILE)"
                    )
                migrate(conn, found)
                if cls.migration_hook is not None:
                    cls.migration_hook(found, SCHEMA_VERSION)
        except BaseException:
            conn.close()
            raise
        return cls(conn, target, readonly)

    def close(self) -> None:
        """Close the store. A writable store first folds its write-ahead log into
        the file. That needs no other connection to be reading; when one is, the
        log stays beside the file (``<file>-wal``) and nothing is lost."""
        if self._closed:
            return
        if not self._readonly:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass  # a reader holds the log; every committed write is still in it
        self._conn.close()
        self._closed = True

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def set_row_hook(self, hook: RowHook | None) -> None:
        """Install a function called inside a write as its rows go in: ``hook(table,
        written)`` runs before each row of ``table`` is handed to SQLite, with the
        number of rows of that table this write has stored so far, and once more
        after the last. It exists so that a test can stop a writer in the middle of
        a transaction (plan Check 4). It must not write to the store."""
        self._row_hook = hook

    # ------------------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------------------

    def settings(self) -> dict[str, str]:
        """The ``setting`` rows. A missing key means off (see ``extensions``)."""
        return {r[0]: r[1] for r in self._conn.execute("SELECT key, value FROM setting")}

    def set_setting(self, key: str, value: str) -> None:
        """Write one setting. The switch of an extension (a key that starts with
        ``extensions.SETTING_PREFIX``) names an extension of this build and holds
        "on" or "off"; anything else under that prefix is refused, because a
        switch the tool cannot read would be taken for off."""
        _need(_is_text(key) and isinstance(value, str), "a setting is a text key and a text value")
        if key.startswith(extensions.SETTING_PREFIX):
            name = key[len(extensions.SETTING_PREFIX):]
            _need(name in extensions.EXTENSIONS, f"no extension is named {name!r}", key=key)
            _need(value in extensions.SWITCH_VALUES,
                  f"the switch of the extension {name!r} is \"on\" or \"off\", not {value!r}",
                  key=key)
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO setting (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def verify_hand_off(self, job_name: str, texts: Sequence[tuple[OccurrenceRef, str]]) -> None:
        """Show the store the texts a stored job was handed, and mark its log compared.

        ``texts`` holds, for every reference of the stored log's hand-off and for no
        other, the text the job was handed under it. Each is compared with the record
        its reference names exactly as at registration (the fingerprint after the
        store's normaliser; a record that holds no text takes only a text that is empty
        after the normaliser). If every one matches, the log is marked compared and its
        edges can establish a bound again; a text that is not its record, a missing or
        an extra reference, a job with no stored log and a log with an empty hand-off
        are refused and change nothing. A log that is already marked is compared again
        and stays marked. A refusal names the job and the reference and never a text."""
        _need(_is_text(job_name), "a job has a text name")
        listed = tuple(texts)
        for pair in listed:
            _need(isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], OccurrenceRef)
                  and isinstance(pair[1], str),
                  "a text of a hand-off is a pair of a record reference and a text")
            _check_ref(pair[0], f"the hand-off of job {job_name!r}")
        with self._transaction() as conn:
            held = conn.execute(
                "SELECT log_id FROM harness_log WHERE job_name = ?", (job_name,)
            ).fetchone()
            _need(held is not None, f"job {job_name!r} has no stored harness log", job=job_name)
            log = self._harness_log(held[0])
            handed = set(log.handed)
            _need(bool(handed), f"the hand-off of job {job_name!r} is empty: there is no text to "
                  "compare, so nothing is changed", job=job_name)
            given = {ref for ref, _ in listed}
            outside = sorted(ref.key for ref in given - handed)
            _need(not outside, f"job {job_name!r} was not handed {outside[0] if outside else ''}; "
                  "nothing is changed", job=job_name)
            missing = sorted(ref.key for ref in handed - given)
            _need(not missing, f"no text was given for {missing[0] if missing else ''} of the "
                  f"{len(handed)} references handed to job {job_name!r}; nothing is changed",
                  job=job_name)
            resolver = OccurrenceResolver(conn)
            for ref, text in listed:
                self._compare_text(conn, resolver, f"the hand-off of job {job_name!r}", job_name,
                                   ref, text, None, 0, (), unchanged="nothing is changed")
            conn.execute("UPDATE harness_log SET hand_off_checked = 1 WHERE log_id = ?", (held[0],))

    def remove_setting(self, key: str) -> None:
        """Remove one setting. Removing a switch row means off, and it is how a switch
        the tool cannot read (a key under ``extensions.SETTING_PREFIX`` that names no
        extension of this build, or a value other than "on" or "off", written from
        outside the tool) is cleared: ``set_setting`` refuses to write one. Removing
        a key the store does not hold changes nothing."""
        _need(_is_text(key), "a setting has a text key")
        with self._transaction() as conn:
            conn.execute("DELETE FROM setting WHERE key = ?", (key,))

    # ------------------------------------------------------------------------------
    # Sources and members (plan 2.3)
    # ------------------------------------------------------------------------------

    def add_source(self, name: str) -> None:
        """Register a source name. Doing it again changes nothing."""
        _need(_is_text(name), "a source has a name")
        with self._transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO source (name) VALUES (?)", (name,))

    def add_member(
        self,
        source: str,
        name: str,
        revision: str,
        part_kind: str,
        parts: Sequence[tuple[str, str]],
    ) -> None:
        """Register a member of a source and the parts it covers: (instance id,
        locator), with "" as the locator for a whole file. An identical call again
        changes nothing. A member that exists with the same source, revision and
        part kind gains the parts it does not hold yet; any other difference is
        refused."""
        _need(_is_text(name) and isinstance(revision, str), "a member has a name and a revision")
        _need(part_kind in v.PART_KINDS, f"unknown part kind {part_kind!r}")
        listed = [tuple(p) for p in parts]
        for part in listed:
            _need(len(part) == 2 and _is_text(part[0]) and isinstance(part[1], str),
                  f"a part is (instance id, locator or \"\"), got {part!r}")
            if part[1]:
                _line_number(part[1], f"member {name!r}")
        with self._transaction() as conn:
            row = conn.execute("SELECT source_id FROM source WHERE name = ?", (source,)).fetchone()
            _need(row is not None, f"source {source!r} is not registered", source=source)
            source_id = row[0]
            resolver = OccurrenceResolver(conn)
            rows = []
            for instance_id, locator in listed:
                artifact_id = resolver.artifact_id(instance_id)
                _need(artifact_id is not None,
                      f"member {name!r}: artifact {instance_id} is not registered")
                if locator:
                    _need(resolver.find(instance_id, locator) is not None,
                          f"member {name!r}: {instance_id}:{locator} is not a record of the store")
                rows.append((artifact_id, locator))
            held = conn.execute(
                "SELECT member_id, source_id, revision, part_kind FROM member WHERE name = ?", (name,)
            ).fetchone()
            if held is None:
                member_id = conn.execute(
                    "INSERT INTO member (source_id, name, revision, part_kind) VALUES (?, ?, ?, ?)",
                    (source_id, name, revision, part_kind),
                ).lastrowid
            else:
                _need(
                    (held["source_id"], held["revision"], held["part_kind"])
                    == (source_id, revision, part_kind),
                    f"member {name!r} is registered with another source, revision or part kind",
                    member=name,
                )
                member_id = held["member_id"]
            conn.executemany(
                "INSERT OR IGNORE INTO member_part (member_id, artifact_id, locator) VALUES (?, ?, ?)",
                [(member_id, artifact_id, locator) for artifact_id, locator in rows],
            )

    # ------------------------------------------------------------------------------
    # Shared pieces of the artifact writes
    # ------------------------------------------------------------------------------

    def _add_name(self, conn: sqlite3.Connection, name: str, artifact_id: int) -> None:
        """Give an instance a name. A name names one instance for good."""
        held = conn.execute(
            "SELECT a.artifact_id, a.instance_id FROM artifact_name n "
            "JOIN artifact a ON a.artifact_id = n.artifact_id WHERE n.name = ?",
            (name,),
        ).fetchone()
        if held is None:
            conn.execute("INSERT INTO artifact_name (name, artifact_id) VALUES (?, ?)", (name, artifact_id))
            return
        _need(
            held["artifact_id"] == artifact_id,
            f"the name {name!r} already names another instance ({held['instance_id']}); "
            "an artifact with another history needs another name",
            name=name, instance=held["instance_id"],
        )

    def _existing(self, conn: sqlite3.Connection, instance: str, kind: str, name: str) -> bool:
        """True when the instance is already registered; its name is then added."""
        row = self._artifact_row(instance)
        if row is None:
            return False
        _need(row["kind"] == kind, f"instance {instance} is registered as a {row['kind']}")
        self._add_name(conn, name, row["artifact_id"])
        return True

    def _insert_artifact(
        self,
        conn: sqlite3.Connection,
        *,
        instance: str,
        kind: str,
        role: str | None,
        content_digest: str,
        path: str | None,
        step: tuple[str, str, str] | None,
        member_name: str | None,
        edge_digest: str,
        name: str,
        inputs: Sequence[str],
    ) -> int:
        held = conn.execute(
            "SELECT a.instance_id FROM artifact_name n JOIN artifact a "
            "ON a.artifact_id = n.artifact_id WHERE n.name = ?",
            (name,),
        ).fetchone()
        _need(
            held is None,
            f"the name {name!r} already names another instance"
            + ("" if held is None else f" ({held[0]})")
            + "; an artifact with another history needs another name",
            name=name,
        )
        step = step or (None, None, None)
        artifact_id = conn.execute(
            "INSERT INTO artifact (instance_id, kind, role, content_digest, path, step_name, "
            "step_version, param_digest, member_name, edge_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (instance, kind, role, content_digest, path, *step, member_name, edge_digest),
        ).lastrowid
        conn.execute("INSERT INTO artifact_name (name, artifact_id) VALUES (?, ?)", (name, artifact_id))
        conn.executemany(
            "INSERT INTO artifact_input (artifact_id, position, input_instance_id) VALUES (?, ?, ?)",
            [(artifact_id, position, value) for position, value in enumerate(inputs)],
        )
        return artifact_id

    @staticmethod
    def _ref_known(
        resolver: OccurrenceResolver, ref: OccurrenceRef, own: str | None, own_lines: int
    ) -> bool:
        """Does this occurrence exist: in the store, or among the lines of the
        artifact this write is about to store."""
        if own is not None and ref.instance_id == own:
            return 1 <= ids.parse_locator(ref.locator)[1] <= own_lines
        return resolver.find(ref.instance_id, ref.locator) is not None

    def _log_form(
        self, resolver: OccurrenceResolver, log: HarnessLog, own: str | None, own_lines: int
    ) -> tuple:
        """A harness log as it reads back from the store. A read of a record the
        store has no row for is kept as text in ``outside_ref``."""
        reads = []
        for read in log.reads:
            ref = read.occurrence
            if ref is not None and self._ref_known(resolver, ref, own, own_lines):
                reads.append((ref, read.outside_ref, bool(read.refused)))
            else:
                outside = read.outside_ref if read.outside_ref is not None else ref.key
                reads.append((None, outside, bool(read.refused)))
        # The hand-off is a list: the order it was handed in, each record once.
        return (log.log_digest, log.model, log.harness, tuple(dict.fromkeys(log.handed)),
                tuple(reads))

    def _log_matches(self, form: tuple, stored: HarnessLog | None) -> bool:
        """Does a log in the form of ``_log_form`` equal the stored one. The hand-off
        is compared in order. A log written before the store kept the order (schema
        version 2 and earlier) has none to give back, and its hand-off is compared
        as a set."""
        other = self._stored_log_form(stored)
        if other is None:
            return False
        if self._conn.execute(
            "SELECT 1 FROM harness_handed h JOIN harness_log l ON l.log_id = h.log_id "
            "WHERE l.job_name = ? AND h.position IS NULL LIMIT 1",
            (stored.job_name,),
        ).fetchone() is not None:
            form, other = (
                (*f[:3], frozenset(f[3]), *f[4:]) for f in (form, other)
            )
        return form == other

    def _check_handed_texts(
        self,
        conn: sqlite3.Connection,
        resolver: OccurrenceResolver,
        edge: EdgeRecord,
        own: str | None,
        own_lines: int,
        own_fingerprints: Sequence[bytes | None],
    ) -> None:
        """Each text the edge's job was handed is the record its reference names.

        The fingerprint of the text after the store's normaliser (the computation of
        an ingest, ``normalise.text_fingerprint``) is compared with the fingerprint
        stored for the record. A mismatch refuses the registration, naming the job
        and the reference and never the text; this runs before any row of the write
        goes in. A record that has no fingerprint (a line with no text, or one that
        is empty after the normaliser) holds no text, so a handed text that is empty
        after the normaliser is that record and any other text is not: it is refused.
        The texts are those of the edge and those of the log it carries (each once);
        every one is compared, so an edge and a log that differ are refused unless
        each of them is the record. An edge and a log that carry no texts are not
        compared."""
        log = edge.harness_log
        texts = tuple(dict.fromkeys(edge.handed_texts + (log.handed_texts if log is not None else ())))
        if not texts:
            return
        what = f"edge of {edge.child_locator!r}"
        job = edge.job_name
        for ref, text in texts:
            self._compare_text(conn, resolver, what, job, ref, text, own, own_lines, own_fingerprints)

    def _compare_text(
        self,
        conn: sqlite3.Connection,
        resolver: OccurrenceResolver,
        what: str,
        job: str | None,
        ref: OccurrenceRef,
        text: str,
        own: str | None,
        own_lines: int,
        own_fingerprints: Sequence[bytes | None],
        unchanged: str = "nothing is registered",
    ) -> None:
        """One handed text is the record its reference names (see ``_check_handed_texts``).
        A refusal names the job and the reference and never the text, and says what
        was left unchanged (``unchanged``)."""
        normaliser = self.normaliser
        if own is not None and ref.instance_id == own:
            known = self._ref_known(resolver, ref, own, own_lines)
            recorded = own_fingerprints[ids.parse_locator(ref.locator)[1] - 1] if known else None
        else:
            occurrence_id = resolver.find(ref.instance_id, ref.locator)
            known = occurrence_id is not None
            recorded = None
            if known:
                recorded = conn.execute(
                    "SELECT fingerprint FROM occurrence WHERE occurrence_id = ?", (occurrence_id,)
                ).fetchone()[0]
        _need(
            known,
            f"{what}: job {job!r} was handed {ref.key}, which is not a record of the store",
            job=job, ref=ref.key,
        )
        if recorded is None:
            _need(
                not normalise.apply(normaliser, text),
                f"{what}: job {job!r} was handed {ref.key} with a text, and that record "
                f"holds no text (no text, or empty after the normaliser {normaliser!r}); "
                f"{unchanged}",
                job=job, ref=ref.key,
            )
            return
        try:
            given = normalise.text_fingerprint(normaliser, text)
        except UnicodeEncodeError:
            given = None  # a lone surrogate has no UTF-8 form, so no fingerprint
        _need(
            given == bytes(recorded),
            f"{what}: job {job!r} was handed {ref.key} with a text that is not that "
            f"record: its fingerprint after the normaliser {normaliser!r} is not the "
            f"fingerprint the store holds for the record; {unchanged}",
            job=job, ref=ref.key,
        )

    def _check_edges_against_store(
        self,
        conn: sqlite3.Connection,
        resolver: OccurrenceResolver,
        edges: Sequence[EdgeRecord],
        own: str | None,
        own_lines: int,
        own_fingerprints: Sequence[bytes | None] = (),
    ) -> None:
        """The checks of an edge that need the store: every record it names exists,
        the texts its job was handed are the records their references name, and its
        harness log does not contradict a stored log of the same job.
        ``own_fingerprints`` are the fingerprints of the lines of the artifact being
        written (``own``), which are not stored yet."""
        logs: dict[str, tuple] = {}
        for edge in edges:
            what = f"edge of {edge.child_locator!r}"
            for item in edge.inputs:
                _need(
                    self._ref_known(resolver, item.parent, own, own_lines),
                    f"{what}: {item.parent.key} is not a record of the store",
                    parent=item.parent.key,
                )
            log = edge.harness_log
            if log is not None:
                for ref in log.handed:
                    _need(
                        self._ref_known(resolver, ref, own, own_lines),
                        f"{what}: job {log.job_name!r} was handed {ref.key}, which is not a "
                        "record of the store",
                        job=log.job_name,
                    )
            self._check_handed_texts(conn, resolver, edge, own, own_lines, own_fingerprints)
            if log is None:
                continue
            form = self._log_form(resolver, log, own, own_lines)
            if log.job_name in logs:
                _need(logs[log.job_name] == form,
                      f"two different logs are given for job {log.job_name!r}", job=log.job_name)
                continue
            logs[log.job_name] = form
            held = conn.execute(
                "SELECT log_id FROM harness_log WHERE job_name = ?", (log.job_name,)
            ).fetchone()
            if held is not None:
                _need(
                    self._log_matches(form, self._harness_log(held[0])),
                    f"job {log.job_name!r} already has a stored log with other contents; "
                    "a second run of a job needs its own job name",
                    job=log.job_name,
                )

    @staticmethod
    def _stored_log_form(stored: HarnessLog | None) -> tuple | None:
        """A log read back from the store, in the form of ``_log_form``."""
        if stored is None:
            return None
        return (
            stored.log_digest, stored.model, stored.harness, tuple(stored.handed),
            tuple((r.occurrence, r.outside_ref, r.refused) for r in stored.reads),
        )

    def _write_log(
        self, conn: sqlite3.Connection, resolver: OccurrenceResolver, log: HarnessLog,
        checked: bool = False,
    ) -> int:
        """Write the log of a job, with the mark that says whether its handed texts were
        compared (``checked``: the registration carried a text for every reference of
        the hand-off, and each matched). A job that already has a stored log keeps it,
        mark included: registering it again never lifts the mark."""
        held = conn.execute(
            "SELECT log_id FROM harness_log WHERE job_name = ?", (log.job_name,)
        ).fetchone()
        if held is not None:
            return held[0]  # checked equal before any row of this write went in
        log_id = conn.execute(
            "INSERT INTO harness_log (job_name, log_digest, model, harness, hand_off_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            (log.job_name, log.log_digest, log.model, log.harness, int(bool(checked))),
        ).lastrowid
        # The hand-off keeps its order: a record handed twice is one row, at the
        # position of its first hand-off.
        handed = dict.fromkeys(resolver.find(r.instance_id, r.locator) for r in log.handed)
        conn.executemany(
            "INSERT INTO harness_handed (log_id, occurrence_id, position) VALUES (?, ?, ?)",
            [(log_id, occurrence_id, position) for position, occurrence_id in enumerate(handed)],
        )
        reads = []
        for seq, read in enumerate(log.reads):
            ref = read.occurrence
            occurrence_id = None if ref is None else resolver.find(ref.instance_id, ref.locator)
            outside = read.outside_ref
            if occurrence_id is None and outside is None:
                outside = ref.key
            reads.append((log_id, seq, occurrence_id, outside, int(bool(read.refused))))
        conn.executemany(
            "INSERT INTO harness_read (log_id, seq, occurrence_id, outside_ref, refused) "
            "VALUES (?, ?, ?, ?, ?)",
            reads,
        )
        return log_id

    def _write_edges(
        self,
        conn: sqlite3.Connection,
        resolver: OccurrenceResolver,
        instance: str,
        edges: Sequence[EdgeRecord],
    ) -> None:
        written = 0
        # The texts the registration carries, per job: those of every edge of the job and of
        # its log (they were all compared before anything was written). A log is marked
        # compared when they cover every reference of its hand-off.
        shown: dict[str, set[OccurrenceRef]] = {}
        for edge in edges:
            log = edge.harness_log
            if log is not None:
                refs = shown.setdefault(log.job_name, set())
                refs.update(ref for ref, _ in edge.handed_texts)
                refs.update(ref for ref, _ in log.handed_texts)
        for edge in edges:
            if self._row_hook is not None:
                self._row_hook("edge", written)
            log_id = None
            bound = 0
            if edge.harness_log is not None:
                log = edge.harness_log
                covered = bool(log.handed) and set(log.handed) <= shown[log.job_name]
                log_id = self._write_log(conn, resolver, log, covered)
                bound = int(log.bound_stated())  # the edge's own field: what the log says
            edge_id = conn.execute(
                "INSERT INTO edge (child_id, transform_name, transform_version, param_digest, "
                "parent_mode, derivation, verification, label, is_generation, job_name, log_id, "
                "bound_established) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    resolver.find(instance, edge.child_locator),
                    edge.transform.name, edge.transform.version, edge.transform.param_digest,
                    edge.parent_mode, edge.derivation, edge.verification, edge.label,
                    int(bool(edge.is_generation)), edge.job_name, log_id, bound,
                ),
            ).lastrowid
            conn.executemany(
                "INSERT INTO edge_input (edge_id, position, occurrence_id, used, selected) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (edge_id, position, resolver.find(i.parent.instance_id, i.parent.locator),
                     int(bool(i.used)), int(bool(i.selected)))
                    for position, i in enumerate(edge.inputs)
                ],
            )
            written += 1
        if self._row_hook is not None and edges:
            self._row_hook("edge", written)

    def _plan_drops(
        self,
        conn: sqlite3.Connection,
        resolver: OccurrenceResolver,
        drops: Sequence[DropRecord],
        output: str,
    ) -> list[tuple]:
        """Check drops against the store and return the rows still to write. A drop
        is one drop per dropped record, step name and output artifact: the same
        step name can drop one record for each artifact it writes."""
        rows: dict[tuple, tuple] = {}
        for drop in drops:
            where = f"drop of {drop.input.key}"
            artifact_id = resolver.artifact_id(drop.input.instance_id)
            _need(artifact_id is not None, f"{where}: its artifact is not registered")
            _need(resolver.find(drop.input.instance_id, drop.input.locator) is not None,
                  f"{where}: not a record of the store")
            key = (artifact_id, drop.input.locator, drop.step_name, output)
            row = (*key[:2], drop.reason, drop.step_name, output, drop.group_key)
            if key in rows:
                _need(rows[key] == row, f"{where}: named twice by step {drop.step_name!r} with other fields")
                continue
            held = conn.execute(
                "SELECT reason, group_key FROM drop_record WHERE artifact_id = ? "
                "AND locator = ? AND step_name = ? AND output_instance_id = ?",
                key,
            ).fetchone()
            if held is not None:
                _need(
                    tuple(held) == (drop.reason, drop.group_key),
                    f"{where}: step {drop.step_name!r} already dropped this record for this "
                    "output with another reason or group key; the store holds one drop per "
                    "record, step name and output",
                    step=drop.step_name, output=output,
                )
                continue
            rows[key] = row
        return list(rows.values())

    def _write_drops(self, conn: sqlite3.Connection, rows: list[tuple]) -> None:
        if not rows:
            return
        conn.executemany(
            "INSERT INTO drop_record (artifact_id, locator, reason, step_name, "
            "output_instance_id, group_key) VALUES (?, ?, ?, ?, ?, ?)",
            self._feed("drop_record", rows),
        )

    # ------------------------------------------------------------------------------
    # JSONL artifacts
    # ------------------------------------------------------------------------------

    def register_jsonl(
        self,
        path,
        *,
        name: str,
        role: str,
        step: StepRef | None,
        member_name: str | None,
        inputs: Sequence[str],
        occurrences: Sequence[OccurrenceRecord],
        edges: Sequence[EdgeRecord] = (),
        drops: Sequence[DropRecord] = (),
        declared_count: int,
        declared_digest: str,
    ) -> str:
        """Register a JSONL artifact with its occurrences, edges and drops, in one
        transaction, and return its instance id.

        The file is read first: the declared line count and digest must match it.
        Then the edge digest is taken over the stated edges, then the instance id,
        and only then are the occurrences keyed and stored.

        A file with no producing step (``step`` None) is registered under a member
        name and may carry any role: "source" for an ingest, "import" for a
        text-only import, and "intermediate" or "release" for a pipeline file
        found afterwards, which keeps its place in the pipeline. A file a step
        produced is an "intermediate" or a "release".

        A parent of a stated edge lies in an artifact named in ``inputs`` or in the
        artifact itself (instance id ``OWN_ARTIFACT``), else the record is refused
        (DECISIONS.md D-15). An edge that carries the texts its job was handed
        (``EdgeRecord.handed_texts`` or ``HarnessLog.handed_texts``) is registered only
        when each text is the record its reference names, compared by fingerprint after the store's normaliser;
        otherwise the write is refused whole (``_check_handed_texts``). The same
        history registered again returns the same id
        and writes nothing but a new name; when it differs from the stored instance
        in a field the id does not cover it is refused (D-14, see
        ``_same_registration``)."""
        self._require_writable()
        _need(_is_text(name), "an artifact is registered under a name")
        _need(role in v.JSONL_ROLES, f"unknown role {role!r}")
        inputs = _check_inputs(inputs)
        edges = tuple(edges)
        drops = tuple(drops)
        for edge in edges:
            _check_edge(edge)
        for drop in drops:
            _check_drop(drop)
        stated = [e for e in edges if e.derivation == v.DERIVED_STATED]
        if step is None:
            # No recorded producing step: an ingested source file, a text-only
            # import, or a pipeline file found afterwards, which keeps the role it
            # has in the pipeline ("intermediate" or "release"). Its id comes from
            # its content digest and the member name it is registered under.
            _need(_is_text(member_name),
                  "an artifact with no producing step is registered under a member name")
            _need(not inputs and not stated and not drops,
                  "an artifact with no producing step has no inputs, stated edges or drops")
        else:
            _check_step(step, f"artifact {name!r}")
            _need(member_name is None, "an artifact a step produced is not registered under a member")
            _need(role in (v.ROLE_INTERMEDIATE, v.ROLE_RELEASE),
                  f"role {role!r} is for a file with no producing step")
        _need(_is_int(declared_count) and declared_count >= 0, "the declared count is a number of lines")
        _need(isinstance(declared_digest, str), "the declared digest is text")
        children: set[str] = set()
        for edge in edges:
            _need(edge.child_locator not in children,
                  f"two edges are given for {edge.child_locator!r}; a record has one edge")
            children.add(edge.child_locator)
        for edge in stated:
            _check_stated_parents(edge, inputs)

        # The artifact on disk (plan 3.4).
        file = Path(path)
        _need(file.is_file(), f"no file at {file}")
        on_disk = artifact_files.read_jsonl(file)
        if declared_count != on_disk.line_count:
            raise DeclaredCountMismatch(
                f"{name!r}: {declared_count} lines declared, the file holds {on_disk.line_count}",
                declared=declared_count, on_disk=on_disk.line_count,
            )
        if declared_digest != on_disk.digest:
            raise DeclaredDigestMismatch(
                f"{name!r}: digest {declared_digest} declared, the file has {on_disk.digest}",
                declared=declared_digest, on_disk=on_disk.digest,
            )
        occurrences = tuple(occurrences)
        if len(occurrences) != declared_count:
            raise DeclaredCountMismatch(
                f"{name!r}: {declared_count} lines declared, {len(occurrences)} occurrences handed",
                declared=declared_count, handed=len(occurrences),
            )
        unset = object()
        fingerprints: list = [unset] * declared_count
        for occurrence in occurrences:
            _need(isinstance(occurrence, OccurrenceRecord), "expected an occurrence record")
            line = _line_number(occurrence.locator, f"artifact {name!r}")
            _need(line <= declared_count, f"{name!r}: {occurrence.locator} is past the last line")
            _need(fingerprints[line - 1] is unset, f"{name!r}: {occurrence.locator} is given twice")
            _check_fingerprint(occurrence.fingerprint, f"{name!r} {occurrence.locator}")
            fingerprints[line - 1] = occurrence.fingerprint
        for edge in edges:
            _need(_line_number(edge.child_locator, "edge") <= declared_count,
                  f"edge of {edge.child_locator!r}: the file has no such line")

        # Identity: content digest, then edge digest, then instance id (plan 3.1).
        edge_digest = ids.edge_digest(_edge_identity(e) for e in stated)
        if step is None:
            instance = ids.ingest_instance_id(on_disk.digest, member_name)
            step_fields = None
        else:
            instance = ids.instance_id(
                on_disk.digest, step.name, step.version, step.param_digest, inputs, edge_digest
            )
            step_fields = (step.name, step.version, step.param_digest)
        edges = tuple(_bind_edge(e, instance) for e in edges)

        with self._transaction() as conn:
            held = self._artifact_row(instance)
            if held is not None:
                self._same_registration(
                    conn, held, role=role, fingerprints=fingerprints, edges=edges, drops=drops,
                    lines=declared_count,
                )
                self._add_name(conn, name, held["artifact_id"])
                return instance
            resolver = OccurrenceResolver(conn)
            self._check_edges_against_store(
                conn, resolver, edges, instance, declared_count, fingerprints
            )
            drop_rows = self._plan_drops(conn, resolver, drops, instance)

            artifact_id = self._insert_artifact(
                conn, instance=instance, kind=v.KIND_JSONL, role=role,
                content_digest=on_disk.digest, path=self._stored_path(file), step=step_fields,
                member_name=member_name, edge_digest=edge_digest, name=name, inputs=inputs,
            )
            binding_id = conn.execute(
                "INSERT INTO binding (artifact_id, namespace, declared_count, declared_digest) "
                "VALUES (?, ?, ?, ?)",
                (artifact_id, v.NAMESPACE_CAPTURE, declared_count, declared_digest),
            ).lastrowid
            conn.executemany(
                "INSERT INTO occurrence (artifact_id, binding_id, locator, fingerprint) "
                "VALUES (?, ?, ?, ?)",
                self._feed(
                    "occurrence",
                    (
                        (artifact_id, binding_id, ids.line_locator(line), fingerprint)
                        for line, fingerprint in enumerate(fingerprints, start=1)
                    ),
                ),
            )
            resolver.forget(instance)
            self._write_edges(conn, resolver, instance, edges)
            self._write_drops(conn, drop_rows)
        return instance

    def _same_registration(
        self,
        conn: sqlite3.Connection,
        held: sqlite3.Row,
        *,
        role: str,
        fingerprints: list,
        edges: Sequence[EdgeRecord],
        drops: Sequence[DropRecord],
        lines: int,
    ) -> None:
        """A history whose instance id the store already holds is registered again
        (DECISIONS.md D-14). The id covers the content, the step, the inputs, the
        parents of the stated edges and the selected parent, so these agree
        already. Every other field handed again must equal what is stored, else
        the registration is refused: nothing is changed and nothing is dropped
        silently. Compared here: the kind and the role, each line's fingerprint,
        and for every edge handed its transform, parent mode, derivation,
        verification, label, generation mark, the name of its job (D-24), its
        inputs with the records that are in scope only, and its harness log whole
        (job name, digest, model, hand-off, reads); and every drop handed. The path is where the file lay,
        not a field of the record, and is not compared."""
        instance = held["instance_id"]

        def differs(what: str, **context: object) -> InvalidRecord:
            return InvalidRecord(
                f"instance {instance} is registered already and this registration differs "
                f"from it in {what}; a field the instance id does not cover is not changed "
                "by registering again",
                instance=instance, **context,
            )

        _need(held["kind"] == v.KIND_JSONL, f"instance {instance} is registered as a {held['kind']}")
        if held["role"] != role:
            raise differs(f"the role (stored {held['role']!r}, handed {role!r})")
        self._same_fingerprints(held["artifact_id"], instance, fingerprints)

        resolver = OccurrenceResolver(conn)
        stored_edges = {edge.child.locator: edge for edge in self.edges(instance)}
        for edge in edges:
            # The same history handed again with a text that is not its record is
            # refused as a first registration of it would be.
            self._check_handed_texts(conn, resolver, edge, None, 0, ())
            stored = stored_edges.get(edge.child_locator)
            if stored is None:
                raise differs(
                    f"the edge of {edge.child_locator}, which the stored instance does not "
                    "hold (an edge found after registration is added with add_recovered_edges)",
                    child=edge.child_locator,
                )
            if not self._same_edge(stored, edge):
                raise differs(f"the edge of {edge.child_locator}", child=edge.child_locator)
            log = edge.harness_log
            if log is not None and not self._log_matches(
                self._log_form(resolver, log, instance, lines), stored.harness_log
            ):
                raise differs(
                    f"the harness log of job {log.job_name!r} behind the edge of "
                    f"{edge.child_locator} (its digest, its model entry, its hand-off or its reads)",
                    child=edge.child_locator, job=log.job_name,
                )
        for row in self._plan_drops(conn, resolver, drops, instance):
            raise differs(
                f"the drop of line {row[1]} of an input by step {row[3]!r}, which the stored "
                "instance does not hold (a drop found after registration is added with add_drops)",
                step=row[3],
            )

    def _same_fingerprints(self, artifact_id: int, instance: str, fingerprints: list) -> None:
        """A history registered again must describe the same records. The stated
        edges are held by the id itself; the fingerprints are compared here."""
        stored = {
            row[0]: row[1]
            for row in self._conn.execute(
                "SELECT locator, fingerprint FROM occurrence WHERE artifact_id = ?", (artifact_id,)
            )
        }
        for line, fingerprint in enumerate(fingerprints, start=1):
            locator = ids.line_locator(line)
            _need(
                locator in stored and stored[locator] == fingerprint,
                f"instance {instance} is registered with another fingerprint for {locator}; "
                "the same history must be handed with the same normaliser and text field",
                instance=instance, locator=locator,
            )

    def add_recovered_edges(self, instance_id: str, edges: Sequence[EdgeRecord]) -> None:
        """Add edges derived after registration (plan 3.3). They never change an
        instance id, so none may be "stated", and their parents need not lie in
        the inputs of the artifact's step: a recovered edge may name any
        registered record (DECISIONS.md D-15). A child that has an edge keeps it:
        an equal edge is passed over, another one is refused."""
        edges = tuple(edges)
        children: set[str] = set()
        for edge in edges:
            _check_edge(edge)
            _need(edge.derivation != v.DERIVED_STATED,
                  f"edge of {edge.child_locator!r}: a stated edge is part of an artifact's "
                  "identity and is written when the artifact is registered")
            _need(edge.child_locator not in children,
                  f"two edges are given for {edge.child_locator!r}; a record has one edge")
            children.add(edge.child_locator)
        edges = tuple(_bind_edge(e, instance_id) for e in edges)
        with self._transaction() as conn:
            held = self._artifact_row(instance_id)
            _need(held is not None, f"artifact {instance_id} is not registered")
            _need(held["kind"] == v.KIND_JSONL, "an edge belongs to a record of a JSONL artifact")
            resolver = OccurrenceResolver(conn)
            self._check_edges_against_store(conn, resolver, edges, None, 0)
            fresh = []
            for edge in edges:
                ref = OccurrenceRef(instance_id, edge.child_locator)
                _need(resolver.find(instance_id, edge.child_locator) is not None,
                      f"edge of {ref.key}: the child is not a record of the store")
                stored = self.edge_of(ref)
                if stored is None:
                    fresh.append(edge)
                    continue
                _need(self._same_edge(stored, edge),
                      f"{ref.key} already has an edge ({stored.derivation}); it is not replaced",
                      child=ref.key)
            self._write_edges(conn, resolver, instance_id, fresh)

    @staticmethod
    def _same_edge(stored: StoredEdge, edge: EdgeRecord) -> bool:
        """Does a stored edge equal one handed for the same child. The job name
        the edge keeps is compared here (DECISIONS.md D-24). Of the harness log
        only the job name is compared here; the caller holds the handed log
        against the stored log of that job name."""
        if (stored.transform, stored.parent_mode, stored.derivation, stored.verification,
                stored.label, stored.is_generation, stored.job_name, stored.inputs) != (
                edge.transform, edge.parent_mode, edge.derivation, edge.verification,
                edge.label, bool(edge.is_generation), edge.job_name, tuple(edge.inputs)):
            return False
        if (stored.harness_log is None) != (edge.harness_log is None):
            return False
        if edge.harness_log is None:
            return True
        return stored.harness_log.job_name == edge.harness_log.job_name

    def add_drops(self, instance_id: str, drops: Sequence[DropRecord]) -> None:
        """Record drops of the step that wrote a registered artifact, after the
        artifact was registered: the lines a tokenizer gave no tokens for, or the
        skips a recovery read from a sanitizer's log. ``instance_id`` is the
        artifact the step wrote. A drop is one drop per dropped record, step name
        and output: one already stored with the same fields is passed over, and
        the same three with another reason or group key are refused. Adding a
        drop never changes an instance id."""
        drops = tuple(drops)
        for drop in drops:
            _check_drop(drop)
        with self._transaction() as conn:
            _need(self._artifact_row(instance_id) is not None,
                  f"artifact {instance_id} is not registered")
            resolver = OccurrenceResolver(conn)
            self._write_drops(conn, self._plan_drops(conn, resolver, drops, instance_id))

    def add_source_link(
        self,
        instance_id: str,
        locator: str | None,
        member: str,
        label: str,
        evidence: str,
        derivation: str,
        verification: str | None = None,
    ) -> None:
        """Link one record (or with ``locator`` None every record of the file) to a
        source member with no parent record behind it: file-level evidence, or a
        publisher's pointer that names a source.

        The link stores how it was derived and the result of its verification, and
        the label rule holds for it as for an edge and a map (plan 3.2, 2.2(f)): a
        "file-level" link has the derivation "file"; a "mapped" link needs the
        derivation "pointer" with verification "passed"; a pointer that is not
        verified, or the derivation "none", can only be "unknown". The same link
        again changes nothing."""
        _need(derivation in _SOURCE_LINK_DERIVATIONS,
              f"a source link is derived by {', '.join(_SOURCE_LINK_DERIVATIONS)}, "
              f"not {derivation!r}")
        _need(verification is None or verification in v.VERIFICATIONS,
              f"unknown verification result {verification!r}")
        _check_label(label, derivation, verification, "source link")
        _need(_is_text(evidence), "a source link says what it rests on")
        if locator is not None:
            _line_number(locator, "source link")
        with self._transaction() as conn:
            artifact_id = self._artifact_id(instance_id)
            _need(artifact_id is not None, f"artifact {instance_id} is not registered")
            row = conn.execute("SELECT member_id FROM member WHERE name = ?", (member,)).fetchone()
            _need(row is not None, f"member {member!r} is not registered", member=member)
            member_id = row[0]
            occurrence_id = None
            if locator is not None:
                found = conn.execute(
                    "SELECT occurrence_id FROM occurrence WHERE artifact_id = ? AND locator = ?",
                    (artifact_id, locator),
                ).fetchone()
                _need(found is not None, f"{instance_id}:{locator} is not a record of the store")
                occurrence_id = found[0]
            fields = (artifact_id, occurrence_id, member_id, label, evidence, derivation, verification)
            same = conn.execute(
                "SELECT 1 FROM source_link WHERE artifact_id = ? AND occurrence_id IS ? "
                "AND member_id = ? AND label = ? AND evidence = ? AND derivation = ? "
                "AND verification IS ?",
                fields,
            ).fetchone()
            if same is None:
                conn.execute(
                    "INSERT INTO source_link (artifact_id, occurrence_id, member_id, label, "
                    "evidence, derivation, verification) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    fields,
                )

    # ------------------------------------------------------------------------------
    # Shards and their maps
    # ------------------------------------------------------------------------------

    def register_shard(self, prefix, *, name: str, step: StepRef, inputs: Sequence[str]) -> str:
        """Register a tokenized shard (``<prefix>.bin`` and ``<prefix>.idx``): the
        artifact row only. Its map is written by ``write_map``; until then the
        shard has no record of its own (plan 2.1). A tokenization link is never
        part of a shard's identity."""
        self._require_writable()
        _need(_is_text(name), "an artifact is registered under a name")
        _check_step(step, f"shard {name!r}")
        inputs = _check_inputs(inputs)
        for file in artifact_files.shard_paths(prefix):
            _need(file.is_file(), f"no file at {file}")
        on_disk = artifact_files.read_shard(prefix)
        edge_digest = ids.no_edges_digest()
        instance = ids.instance_id(
            on_disk.digest, step.name, step.version, step.param_digest, inputs, edge_digest
        )
        with self._transaction() as conn:
            if self._existing(conn, instance, v.KIND_SHARD, name):
                return instance
            self._insert_artifact(
                conn, instance=instance, kind=v.KIND_SHARD, role=None,
                content_digest=on_disk.digest, path=self._stored_path(os.fspath(prefix)),
                step=(step.name, step.version, step.param_digest), member_name=None,
                edge_digest=edge_digest, name=name, inputs=inputs,
            )
        return instance

    def write_map(
        self,
        shard_instance_id: str,
        map: MapRecord,
        *,
        declared_count: int,
        declared_digest: str,
        drops: Sequence[DropRecord] = (),
    ) -> None:
        """Write a shard's tokenization map in one transaction: the binding, all
        its rows and the drops of the same step, or nothing (plan 3.4).

        The shard's files are read first; the declared document count and digest
        must match them. ``drops`` are the drops that belong to the map: the lines
        the tokenizer gave no tokens for, and the skips a recovery read from a
        sanitizer's log. They are recorded for the shard, in the same
        transaction, so a process killed at any point leaves the map with its
        drops or neither.

        A namespace and derivation of a shard hold at most one map whose
        verification passed (DECISIONS.md D-20). A map that is held is never
        overwritten: the same map again changes nothing, a verified map is final
        for its namespace and derivation, and a map that did not verify can be
        followed by another map of the same namespace and derivation, written
        beside it. A reader takes the verified one."""
        self.write_maps(shard_instance_id, (map,), declared_count=declared_count,
                        declared_digest=declared_digest, drops=drops)

    def write_maps(
        self,
        shard_instance_id: str,
        maps: Sequence[MapRecord],
        *,
        declared_count: int,
        declared_digest: str,
        drops: Sequence[DropRecord] = (),
    ) -> None:
        """``write_map`` for every map one derivation run gave for one shard (a
        position map and a token-matching map), with the drops of that run, in
        one transaction: all of it or nothing. A map the store refuses refuses
        the whole call."""
        self._require_writable()
        maps = tuple(maps)
        _need(maps, "at least one map is written")
        slots: set[tuple[str, str]] = set()
        for map in maps:
            _need(isinstance(map, MapRecord), "expected a map record")
            _need(map.namespace in v.NAMESPACES, f"unknown namespace {map.namespace!r}")
            _need(map.derivation in _MAP_DERIVATIONS,
                  f"a map is derived by position or token matching, not {map.derivation!r}")
            _need(map.verification in v.VERIFICATIONS,
                  f"unknown verification result {map.verification!r}")
            _check_label(map.label, map.derivation, map.verification, "map")
            _need(isinstance(map.append_eod, bool) and isinstance(map.eod_in_span, bool),
                  "a map records the append-eod setting and the end-of-document convention")
            slot = (map.namespace, map.derivation)
            _need(slot not in slots,
                  f"two maps are given for the namespace {map.namespace!r} and the "
                  f"derivation {map.derivation!r}")
            slots.add(slot)
        drops = tuple(drops)
        for drop in drops:
            _check_drop(drop)
        _need(_is_int(declared_count) and declared_count >= 0, "the declared count is a number of documents")
        _need(isinstance(declared_digest, str), "the declared digest is text")

        shard = self._artifact_row(shard_instance_id)
        _need(shard is not None, f"shard {shard_instance_id} is not registered")
        _need(shard["kind"] == v.KIND_SHARD, f"{shard_instance_id} is a {shard['kind']}, not a shard")
        disk = look_at(v.KIND_SHARD, self._disk_path(shard["path"]))
        _need(disk.state == FILE_PRESENT, f"the shard's two files are not at {disk.path}")
        if disk.count is None:
            if declared_digest != disk.digest:
                raise DeclaredDigestMismatch(
                    f"digest {declared_digest} declared, the shard on disk has {disk.digest}",
                    declared=declared_digest, on_disk=disk.digest,
                )
            raise UnsupportedFormat(f"the shard's index cannot be read: {disk.unreadable}")
        if declared_count != disk.count:
            raise DeclaredCountMismatch(
                f"{declared_count} documents declared, the shard on disk holds {disk.count}",
                declared=declared_count, on_disk=disk.count,
            )
        if declared_digest != disk.digest:
            raise DeclaredDigestMismatch(
                f"digest {declared_digest} declared, the shard on disk has {disk.digest}",
                declared=declared_digest, on_disk=disk.digest,
            )
        if disk.digest != shard["content_digest"]:
            raise ArtifactDigestMismatch(
                f"shard {shard_instance_id} is registered with digest {shard['content_digest']} "
                f"and its files have {disk.digest}",
                instance=shard_instance_id, registered=shard["content_digest"], on_disk=disk.digest,
            )
        planned = [(map, self._ordered_rows(map, declared_count)) for map in maps]

        with self._transaction() as conn:
            artifact_id = self._artifact_id(shard_instance_id)
            _need(artifact_id is not None, f"shard {shard_instance_id} is not registered")
            resolver = OccurrenceResolver(conn)
            fresh = []
            for map, ordered in planned:
                if not self._map_held(conn, artifact_id, shard_instance_id, map, ordered,
                                      declared_count, declared_digest):
                    fresh.append(self._map_values(resolver, map, ordered))
            drop_rows = self._plan_drops(conn, resolver, drops, shard_instance_id)
            for map, row_values, candidate_values in fresh:
                binding_id = conn.execute(
                    "INSERT INTO binding (artifact_id, namespace, declared_count, declared_digest, "
                    "derivation, verification, label, append_eod, eod_in_span) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (artifact_id, map.namespace, declared_count, declared_digest, map.derivation,
                     map.verification, map.label, int(map.append_eod), int(map.eod_in_span)),
                ).lastrowid
                conn.executemany(
                    "INSERT INTO map_row (binding_id, ordinal, parent_occurrence_id, fingerprint, "
                    "token_start, token_end, label) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    self._feed("map_row", ((binding_id, *values) for values in row_values)),
                )
                conn.executemany(
                    "INSERT INTO map_candidate (binding_id, ordinal, occurrence_id) VALUES (?, ?, ?)",
                    self._feed("map_candidate", ((binding_id, *values) for values in candidate_values)),
                )
            self._write_drops(conn, drop_rows)

    @staticmethod
    def _ordered_rows(map: MapRecord, declared_count: int) -> list[MapRow]:
        """The rows of a map in ordinal order, checked against the record model:
        exactly one row per document, each with the fields its label asks for."""
        rows = tuple(map.rows)
        if len(rows) != declared_count:
            raise DeclaredCountMismatch(
                f"{declared_count} documents declared, the map holds {len(rows)} rows",
                declared=declared_count, handed=len(rows),
            )
        ordered: list[MapRow | None] = [None] * declared_count
        for row in rows:
            _need(isinstance(row, MapRow), "expected a map row")
            _need(_is_int(row.ordinal) and 0 <= row.ordinal < declared_count,
                  f"map row: no document {row.ordinal!r} in a shard of {declared_count}")
            what = f"map row D{row.ordinal}"
            _need(ordered[row.ordinal] is None, f"{what} is given twice")
            _need(_is_int(row.token_start) and _is_int(row.token_end)
                  and 0 <= row.token_start <= row.token_end, f"{what}: not a token span")
            _check_label(row.label, map.derivation, map.verification, what)
            _check_fingerprint(row.fingerprint, what)
            if row.parent is not None:
                _check_ref(row.parent, what)
            for ref in row.candidates:
                _check_ref(ref, what)
            if row.label == v.RECOVERED:
                _need(row.parent is not None and not row.candidates,
                      f'{what}: a "recovered" row names one parent and no candidate set')
            elif row.label == v.CANDIDATES:
                _need(row.parent is None and len(set(row.candidates)) >= 2,
                      f'{what}: a "candidates" row names two or more candidates and no parent')
            else:
                _need(row.parent is None and not row.candidates,
                      f"{what}: a row labelled {row.label!r} names no record")
            ordered[row.ordinal] = row
        return ordered  # type: ignore[return-value]

    def _map_held(
        self,
        conn: sqlite3.Connection,
        artifact_id: int,
        shard_instance_id: str,
        map: MapRecord,
        ordered: list[MapRow],
        declared_count: int,
        declared_digest: str,
    ) -> bool:
        """True when the store holds this very map already. Refuses a map whose
        namespace and derivation hold a verified map with other contents: a
        verified map is final. A map is written beside the maps of its namespace
        and derivation that did not verify."""
        held = conn.execute(
            "SELECT * FROM binding WHERE artifact_id = ? AND namespace = ? AND derivation = ? "
            "ORDER BY binding_id",
            (artifact_id, map.namespace, map.derivation),
        ).fetchall()
        head = (declared_count, declared_digest, map.verification, map.label,
                int(map.append_eod), int(map.eod_in_span))
        for binding in held:
            if (binding["declared_count"], binding["declared_digest"], binding["verification"],
                    binding["label"], binding["append_eod"], binding["eod_in_span"]) != head:
                continue
            stored = self._map_rows_of(shard_instance_id, binding["binding_id"], 0, 1 << 62)
            if [_row_form(r) for r in stored] == [_row_form(r) for r in ordered]:
                return True
        for binding in held:
            if binding["verification"] == v.VERIFIED_PASSED:
                raise InvalidRecord(
                    f"shard {shard_instance_id} already has a verified {map.namespace} map "
                    f"derived by {map.derivation} with other contents; a map is never "
                    "overwritten, and a verified map is final for its namespace and derivation",
                    instance=shard_instance_id, namespace=map.namespace,
                    derivation=map.derivation,
                )
        return False

    @staticmethod
    def _map_values(
        resolver: OccurrenceResolver, map: MapRecord, ordered: list[MapRow]
    ) -> tuple[MapRecord, list[tuple], list[tuple]]:
        row_values = []
        candidate_values = []
        for row in ordered:
            what = f"map row D{row.ordinal}"
            parent_id = None
            if row.parent is not None:
                parent_id = resolver.find(row.parent.instance_id, row.parent.locator)
                _need(parent_id is not None, f"{what}: {row.parent.key} is not a record of the store")
            for ref in set(row.candidates):
                occurrence_id = resolver.find(ref.instance_id, ref.locator)
                _need(occurrence_id is not None, f"{what}: {ref.key} is not a record of the store")
                candidate_values.append((row.ordinal, occurrence_id))
            row_values.append(
                (row.ordinal, parent_id, row.fingerprint, row.token_start, row.token_end, row.label)
            )
        return map, row_values, candidate_values

    # ------------------------------------------------------------------------------
    # Mixtures, runs, traces, checkpoints
    # ------------------------------------------------------------------------------

    def register_mixture(self, *, name: str, shards: Sequence[SnapshotShard], settings: dict) -> str:
        """Register a mixture snapshot: the shards the trainer was handed, in its
        order, each with its digest (plan Decision 27), and the settings the
        trainer used (a JSON object, kept whole). A shard is named by value and
        need not be registered. Returns the mixture's instance id, which is
        ``ids.mixture_instance_id`` of the shards' instance ids and digests in
        order and of the settings. The same mixture again returns the same id; a
        path is where a shard lay and is not part of the id."""
        self._require_writable()
        _need(_is_text(name), "an artifact is registered under a name")
        _need(isinstance(settings, dict), "a mixture's settings are a JSON object")
        listed = sorted(shards, key=lambda s: s.position if isinstance(s, SnapshotShard) else 0)
        _need(listed, "a mixture holds at least one shard")
        for position, shard in enumerate(listed):
            _need(isinstance(shard, SnapshotShard), "expected a snapshot shard")
            _need(shard.position == position,
                  "the shards of a snapshot are numbered 0, 1, 2 and so on, each once")
            _need(_is_text(shard.shard_instance_id) and _is_text(shard.path)
                  and _is_text(shard.content_digest),
                  f"shard {position} of the snapshot needs an instance id, a path and a digest")
        try:
            settings_json = ids.canonical(settings).decode("ascii")
            param_digest = ids.param_digest(settings)
        except (TypeError, ValueError) as exc:
            raise InvalidRecord(f"the mixture's settings are not plain JSON values: {exc}") from exc
        shard_ids = [s.shard_instance_id for s in listed]
        digests = [s.content_digest for s in listed]
        instance = mixture_instance_id(shard_ids, digests, settings)
        with self._transaction() as conn:
            if self._existing(conn, instance, v.KIND_MIXTURE, name):
                return instance
            for shard in listed:
                row = self._artifact_row(shard.shard_instance_id)
                if row is None:
                    continue
                _need(row["kind"] == v.KIND_SHARD,
                      f"shard {shard.position} of the snapshot names a {row['kind']}")
                _need(row["content_digest"] == shard.content_digest,
                      f"shard {shard.position} of the snapshot carries digest {shard.content_digest} "
                      f"and its instance is registered with {row['content_digest']}")
            artifact_id = self._insert_artifact(
                conn, instance=instance, kind=v.KIND_MIXTURE, role=None,
                content_digest=ids.mixture_content_digest(digests), path=None,
                step=(*ids.MIXTURE_STEP, param_digest), member_name=None,
                edge_digest=ids.no_edges_digest(), name=name, inputs=(),
            )
            conn.executemany(
                "INSERT INTO mixture_shard (artifact_id, position, shard_instance_id, path, "
                "content_digest) VALUES (?, ?, ?, ?, ?)",
                [(artifact_id, s.position, s.shard_instance_id, s.path, s.content_digest) for s in listed],
            )
            conn.execute(
                "INSERT INTO mixture_settings (artifact_id, settings) VALUES (?, ?)",
                (artifact_id, settings_json),
            )
        return instance

    @staticmethod
    def _segment_from(
        run, segment, position, mixture_instance_id, parent_checkpoint, start_kind, run_record,
        steps, trace_route, trace_cause, trace_cause_detail,
    ) -> Segment:
        """Check what a segment registration was handed and put it in the form the
        store holds and reads back."""
        _need(_is_text(run) and _is_text(segment), "a run and a segment have names")
        _need(_is_int(position) and position >= 0, "a segment has a position inside its run")
        _need(mixture_instance_id is None or _is_text(mixture_instance_id),
              "a mixture is named by its instance id, or None when there is no snapshot")
        _need(parent_checkpoint is None or _is_text(parent_checkpoint),
              "a parent checkpoint is named by its instance id")
        _need(start_kind in v.START_KINDS, f"unknown start kind {start_kind!r}")
        if start_kind == v.START_FRESH:
            _need(parent_checkpoint is None, "a fresh start has no parent checkpoint")
        else:
            _need(parent_checkpoint is not None,
                  f"a {start_kind} names the checkpoint it starts from; without it the "
                  "history would end here and read as complete")
        _need(trace_route in v.TRACE_ROUTES, f"unknown trace route {trace_route!r}")
        if trace_route == v.ROUTE_NONE:
            _need(trace_cause in _TRACE_CAUSES,
                  f"a segment with no trace names one of the causes a missing data order has "
                  f"({', '.join(repr(c) for c in v.TRACE_CAUSES)}); {trace_cause!r} is not one of them")
        else:
            _need(trace_cause is None and trace_cause_detail is None,
                  "a trace cause is given only when the trace route is \"none\"")
        _need(trace_cause_detail is None or isinstance(trace_cause_detail, str),
              "a trace cause detail is text")
        if trace_cause_detail is not None:
            # A reason stands beside "data-order trace missing" only, and is one of the
            # six words of the vocabulary (plan 3.5; docs/interfaces.md section 2); a
            # caller that does not know the reason stores none.
            _need(trace_cause == v.NAMES_TRACE_MISSING and trace_cause_detail in _TRACE_MISSING_REASONS,
                  f"a trace cause detail stands beside {v.NAMES_TRACE_MISSING!r} and is one of "
                  f"{', '.join(repr(r) for r in v.TRACE_MISSING_STORED_REASONS)}; "
                  f"{trace_cause_detail!r} beside {trace_cause!r} is not")

        listed = []
        seen: set[int] = set()
        for item in steps:
            item = tuple(item)
            _need(len(item) == 3, f"a step is (step, batch size, tokens read of the last sample or None), got {item!r}")
            step, batch, last = item
            _need(_is_int(step) and step >= 0 and _is_int(batch) and batch >= 0,
                  f"step {step!r}: the step number and the batch size are numbers")
            _need(last is None or (_is_int(last) and last >= 0),
                  f"step {step}: the tokens read of the last sample is a number or None")
            _need(step not in seen, f"step {step} is given twice")
            seen.add(step)
            listed.append(SegmentStep(step, batch, last))
        listed.sort(key=lambda s: s.step)

        record = None
        digest = None
        fields: dict[str, object] = dict.fromkeys(
            ("sampler_type", "sampler_class", "data_sharding", "micro_batch_size",
             "global_batch_size", "sequence_length", "add_extra_token", "random_seed",
             "megatron_core_version", "megatron_bridge_version", "torch_version", "release_tag")
        )
        launches: list[Launch] = []
        cache: list[CacheDigest] = []
        shuffle: list[ShuffleDigest] = []
        listed_shards: list = []
        if run_record is not None:
            _need(isinstance(run_record, dict), "a run record is a JSON object")
            try:
                text = ids.canonical(run_record)
            except (TypeError, ValueError) as exc:
                raise InvalidRecord(f"the run record is not plain JSON: {exc}") from exc
            record = json.loads(text)
            # The digest is taken over the text the store keeps: the canonical text of
            # the record as read back. For a record whose keys are all text this is
            # the text just made; for a number used as a key the two orders of the
            # keys can differ, and the stored text is the one a check can hold the
            # digest against (``check_integrity``, check 6).
            digest = ids.bytes_digest(ids.canonical(record))
            listed_shards = record.get("shards") or []
            _need(isinstance(listed_shards, list), "run record: shards is a list")
            _need(record.get("format") == RUN_RECORD_FORMAT,
                  f"run record: format {record.get('format')!r}, expected {RUN_RECORD_FORMAT!r}")
            for key, value in (("run", run), ("segment", segment), ("start_kind", start_kind)):
                _need(record.get(key) is None or record[key] == value,
                      f"run record: its {key} is {record.get(key)!r}, the registration says {value!r}")
            settings = record.get("settings") or {}
            versions = record.get("versions") or {}
            _need(isinstance(settings, dict) and isinstance(versions, dict),
                  "run record: settings and versions are objects")
            fields.update(
                sampler_type=_opt(settings, "dataloader_type", str, "settings"),
                sampler_class=_opt(settings, "sampler_class", str, "settings"),
                data_sharding=_opt(settings, "data_sharding", bool, "settings"),
                micro_batch_size=_opt(settings, "micro_batch_size", int, "settings"),
                global_batch_size=_opt(settings, "global_batch_size", int, "settings"),
                sequence_length=_opt(settings, "sequence_length", int, "settings"),
                add_extra_token=_opt(settings, "add_extra_token_to_sequence", bool, "settings"),
                random_seed=_opt(settings, "random_seed", int, "settings"),
                megatron_core_version=_opt(versions, "megatron_core", str, "versions"),
                megatron_bridge_version=_opt(versions, "megatron_bridge", str, "versions"),
                torch_version=_opt(versions, "torch", str, "versions"),
                release_tag=_opt(versions, "release_tag", str, "versions"),
            )
            launch_keys = ("position", "world_size", "data_parallel_size", "consumed_at_start",
                           "step_at_start", "first_step", "last_step")
            for item in record.get("launches") or []:
                _need(isinstance(item, dict) and all(_is_int(item.get(k)) for k in launch_keys),
                      f"run record: a launch needs the numbers {', '.join(launch_keys)}; got {item!r}")
                launches.append(Launch(*(item[k] for k in launch_keys)))
            launches.sort(key=lambda x: x.position)
            _need(len({x.position for x in launches}) == len(launches),
                  "run record: two launches share a position")
            for item in record.get("cache_digests") or []:
                _need(isinstance(item, dict), "run record: a cache digest is an object")
                kind, shard, value = item.get("kind"), item.get("shard"), item.get("digest")
                _need(kind in _CACHE_KINDS, f"run record: unknown cache kind {kind!r}")
                _need(_is_text(value), f"run record: the cache digest of {kind} is text")
                if kind in v.CACHE_KINDS_PER_SHARD:
                    _need(_is_int(shard), f"run record: a {kind} digest names its shard position")
                else:
                    _need(shard is None, f"run record: a {kind} digest belongs to the blend, not a shard")
                _need(all((c.kind, c.shard_position) != (kind, shard) for c in cache),
                      f"run record: two digests for {kind} of shard {shard!r}")
                cache.append(CacheDigest(kind, shard, value))
            for item in record.get("shuffle_digests") or []:
                _need(isinstance(item, dict)
                      and all(_is_int(item.get(k)) for k in ("launch", "epoch", "rank"))
                      and _is_text(item.get("digest")),
                      f"run record: a shuffle digest needs launch, epoch, rank and digest; got {item!r}")
                shuffle.append(ShuffleDigest(item["launch"], item["epoch"], item["rank"], item["digest"]))
            shuffle.sort(key=lambda x: (x.launch, x.epoch, x.rank))
            _need(len({(x.launch, x.epoch, x.rank) for x in shuffle}) == len(shuffle),
                  "run record: two shuffle digests for one launch, epoch and rank")
        if trace_route != v.ROUTE_NONE:
            # A route follows from the run record, whoever calls the store: a replay
            # counts only against the digests recorded at training time (plan 3.5), and
            # kept cache files count only against the same digests (a decision stricter
            # than plan 3.5 and 4.6, DECISIONS.md D-41), so a record that holds no digest
            # of an index array the segment needs supports neither (D-25, D-36).
            have = {(c.kind, c.shard_position) for c in cache}
            count = v.shard_count_of(len(listed_shards), (c.shard_position for c in cache))
            absent = [f"{kind} of shard {at}" if at is not None else f"{kind} of the blend"
                      for kind, at in v.cache_keys(count) if (kind, at) not in have]
            _need(count > 0 and not absent,
                  f"segment {segment!r}: the route {trace_route!r} needs the run record to hold the "
                  "digest of every index array the segment has, and "
                  + ("it holds no run record" if run_record is None else
                     "it holds no digest of an index array" if count == 0 else
                     "it lacks " + ", ".join(absent)),
                  segment=segment, route=trace_route)
        if fields["sampler_type"] == v.SAMPLER_CYCLIC and not shuffle:
            # A segment read with the cyclic sampler whose run record holds no
            # shuffle digest supports no trace (plan 3.5, 2.4): the order of its
            # samples is unknown. The route follows from the record, not from the
            # caller (D-26).
            _need(trace_route == v.ROUTE_NONE,
                  f"segment {segment!r} was read with the cyclic sampler and its run record holds "
                  f"no shuffle digest, so it has no trace route; {trace_route!r} was given",
                  segment=segment)
        if trace_route != v.ROUTE_NONE and run_record is not None:
            # Nothing says which step comes first or where the counter of the segment
            # starts, so no step of it is known to be whole: a run record that holds no
            # launch (empty, null or absent) supports no trace route, whoever calls the
            # store (plan 2.2(e), 3.1, 3.5; DECISIONS.md D-25). A call with no run record
            # at all is refused by the digest rule above.
            _need(bool(launches),
                  f"segment {segment!r}: the route {trace_route!r} needs the run record to hold "
                  "a launch, and it holds none",
                  segment=segment, route=trace_route)
        return Segment(
            run=run, name=segment, position=position, mixture_instance_id=mixture_instance_id,
            parent_checkpoint=parent_checkpoint, start_kind=start_kind, run_record=record,
            run_record_digest=digest, trace_route=trace_route, trace_cause=trace_cause,
            trace_cause_detail=trace_cause_detail, launches=tuple(launches),
            cache_digests=tuple(cache), shuffle_digests=tuple(shuffle), steps=tuple(listed),
            **fields,
        )

    def register_run_segment(
        self,
        *,
        run: str,
        segment: str,
        position: int,
        mixture_instance_id: str | None,
        parent_checkpoint: str | None,
        start_kind: str,
        run_record: dict | None,
        steps: Sequence[tuple[int, int, int | None]],
        trace_route: str,
        trace_cause: str | None = None,
        trace_cause_detail: str | None = None,
        traces: Sequence[StepTrace] = (),
        checkpoints: Sequence[tuple[str, int]] = (),
    ) -> None:
        """Register a run segment: its mixture and parent checkpoint (both by
        value), the run record whole, the columns read from it, its launches and
        digests, its completed steps with their batch sizes, the step traces the
        exposure join resolved and the checkpoints at its steps, all in one
        transaction: a stop leaves nothing of it written. A field the run record
        does not hold stays empty; nothing is guessed.

        ``traces`` are checked as ``write_step_traces`` checks them, against the
        steps given here. The same segment again changes nothing, and what the call adds to it (a trace
        of a step that has none, a checkpoint that is not there yet) is written; a
        segment name with other contents is refused, and so is another trace for
        a step that has one."""
        self._require_writable()
        wanted = self._segment_from(
            run, segment, position, mixture_instance_id, parent_checkpoint, start_kind,
            run_record, steps, trace_route, trace_cause, trace_cause_detail,
        )
        fresh_traces = self._traces_from(traces)
        named = []
        for item in checkpoints:
            item = tuple(item)
            _need(len(item) == 2 and _is_text(item[0]) and _is_int(item[1]) and item[1] >= 0,
                  f"a checkpoint is (name, step), got {item!r}")
            named.append(item)
        with self._transaction() as conn:
            held = self.segment(segment)
            if held is not None:
                _need(held == wanted,
                      f"segment {segment!r} is already registered with other contents", segment=segment)
                segment_id = self._segment_row(segment)["segment_id"]
            else:
                segment_id = self._insert_segment(conn, wanted, run, segment, position,
                                                  mixture_instance_id, parent_checkpoint)
            self._write_traces(conn, segment, segment_id, fresh_traces)
            for name, step in named:
                self._register_checkpoint(conn, name=name, segment=segment, step=step)

    def _insert_segment(self, conn, w: Segment, run: str, segment: str, position: int,
                        mixture_instance_id: str | None, parent_checkpoint: str | None) -> int:
        """The rows of a segment that is not yet in the store, inside the caller's
        transaction. Returns its id."""
        if mixture_instance_id is not None:
            row = self._artifact_row(mixture_instance_id)
            _need(row is None or row["kind"] == v.KIND_MIXTURE,
                  f"segment {segment!r}: {mixture_instance_id} is not a mixture")
        if parent_checkpoint is not None:
            row = self._artifact_row(parent_checkpoint)
            _need(row is None or row["kind"] == v.KIND_CHECKPOINT,
                  f"segment {segment!r}: {parent_checkpoint} is not a checkpoint")
            self._refuse_loop(segment, parent_checkpoint)
        run_row = conn.execute("SELECT run_id FROM run WHERE name = ?", (run,)).fetchone()
        if run_row is None:
            run_id = conn.execute("INSERT INTO run (name) VALUES (?)", (run,)).lastrowid
        else:
            run_id = run_row[0]
            taken = conn.execute(
                "SELECT name FROM run_segment WHERE run_id = ? AND position = ?", (run_id, position)
            ).fetchone()
            _need(taken is None,
                  f"position {position} of run {run!r} is taken"
                  + ("" if taken is None else f" by segment {taken[0]!r}"))
        segment_id = conn.execute(
            "INSERT INTO run_segment (run_id, name, position, mixture_instance_id, "
            "parent_checkpoint, start_kind, run_record, run_record_digest, sampler_type, "
            "sampler_class, data_sharding, micro_batch_size, global_batch_size, sequence_length, "
            "add_extra_token, random_seed, megatron_core_version, megatron_bridge_version, "
            "torch_version, release_tag, trace_route, trace_cause, trace_cause_detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, w.name, w.position, w.mixture_instance_id, w.parent_checkpoint,
                w.start_kind,
                None if w.run_record is None else ids.canonical(w.run_record).decode("ascii"),
                w.run_record_digest, w.sampler_type, w.sampler_class,
                None if w.data_sharding is None else int(w.data_sharding),
                w.micro_batch_size, w.global_batch_size, w.sequence_length,
                None if w.add_extra_token is None else int(w.add_extra_token),
                w.random_seed, w.megatron_core_version, w.megatron_bridge_version,
                w.torch_version, w.release_tag, w.trace_route, w.trace_cause,
                w.trace_cause_detail,
            ),
        ).lastrowid
        conn.executemany(
            "INSERT INTO segment_launch (segment_id, position, world_size, data_parallel_size, "
            "consumed_at_start, step_at_start, first_step, last_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(segment_id, *dataclasses.astuple(x)) for x in w.launches],
        )
        conn.executemany(
            "INSERT INTO segment_cache_digest (segment_id, kind, shard_position, digest) "
            "VALUES (?, ?, ?, ?)",
            [(segment_id, *dataclasses.astuple(x)) for x in w.cache_digests],
        )
        conn.executemany(
            "INSERT INTO segment_shuffle_digest (segment_id, launch_position, epoch, rank, digest) "
            "VALUES (?, ?, ?, ?, ?)",
            [(segment_id, *dataclasses.astuple(x)) for x in w.shuffle_digests],
        )
        conn.executemany(
            "INSERT INTO segment_step (segment_id, step, batch_size, last_sample_tokens_read) "
            "VALUES (?, ?, ?, ?)",
            self._feed("segment_step", ((segment_id, *dataclasses.astuple(x)) for x in w.steps)),
        )
        return segment_id

    def _refuse_loop(self, segment: str, parent_checkpoint: str) -> None:
        """A segment may not start from a checkpoint whose own history holds the
        segment: a history is a chain that ends."""
        seen: set[str] = set()
        current: str | None = parent_checkpoint
        while current is not None and current not in seen:
            seen.add(current)
            mark = self._checkpoint_by_instance(current)
            if mark is None:
                return
            _need(mark.segment != segment,
                  f"segment {segment!r} would start from a checkpoint of its own history",
                  segment=segment, checkpoint=current)
            row = self._segment_row(mark.segment)
            current = None if row is None else row["parent_checkpoint"]

    def write_step_traces(self, segment: str, traces: Sequence[StepTrace]) -> None:
        """Write the traces of completed steps of a registered segment, once (plan
        3.5, Decision 19). A trace must count exactly its step's registered batch
        size; a sample the join could not resolve whole is written with a lowered
        coverage mark, never left out. A step that has a trace keeps it: the same
        trace again changes nothing, another is refused."""
        self._require_writable()
        wanted = self._traces_from(traces)
        with self._transaction() as conn:
            row = self._segment_row(segment)
            _need(row is not None, f"segment {segment!r} is not registered", segment=segment)
            self._write_traces(conn, segment, row["segment_id"], wanted)

    @staticmethod
    def _traces_from(traces: Sequence[StepTrace]) -> list[StepTrace]:
        """Check what was handed as step traces and put them in the form the store
        holds: samples in order, rows as tuples."""
        wanted: list[StepTrace] = []
        seen: set[int] = set()
        for trace in traces:
            _need(isinstance(trace, StepTrace), "expected a step trace")
            what = f"trace of step {trace.step!r}"
            _need(_is_int(trace.step) and _is_int(trace.consumed_samples), f"{what}: numbers needed")
            _need(trace.step not in seen, f"{what} is given twice")
            seen.add(trace.step)
            samples = sorted(trace.samples, key=lambda s: s.sample_pos)
            _need(len(samples) == trace.consumed_samples,
                  f"{what}: it counts {trace.consumed_samples} samples and holds {len(samples)}")
            for position, sample in enumerate(samples):
                _need(isinstance(sample, TraceSample) and sample.sample_pos == position,
                      f"{what}: its samples are numbered 0, 1, 2 and so on, each once")
                _need(all(_is_int(x) for x in (sample.global_index, sample.rank,
                                               sample.sample_tokens, sample.covered_tokens)),
                      f"{what}, sample {position}: numbers needed")
                _need(0 <= sample.covered_tokens <= sample.sample_tokens,
                      f"{what}, sample {position}: the coverage mark lies inside the sample")
                for row in sample.rows:
                    _need(isinstance(row, TraceRow)
                          and all(_is_int(x) for x in dataclasses.astuple(row))
                          and row.shard_position >= 0 and row.ordinal >= 0
                          and 0 <= row.token_start <= row.token_end,
                          f"{what}, sample {position}: not a trace row: {row!r}")
            wanted.append(StepTrace(trace.step, trace.consumed_samples,
                                    tuple(dataclasses.replace(s, rows=tuple(s.rows)) for s in samples)))
        return wanted

    def _write_traces(self, conn, segment: str, segment_id: int, wanted: Sequence[StepTrace]) -> None:
        """Write checked traces of a registered segment inside the caller's
        transaction: each against its step's registered batch size, a trace the
        step already has only when equal. A segment whose route is "none" holds no
        trace: the route says no step of it has a trace the proof can use, and a stored
        one would say otherwise (plan 3.5, 2.4). This is the one place a step trace is
        written, so the refusal holds for every caller."""
        if wanted:
            route = conn.execute(
                "SELECT trace_route FROM run_segment WHERE segment_id = ?", (segment_id,)
            ).fetchone()[0]
            _need(route != v.ROUTE_NONE,
                  f"segment {segment!r} has the trace route {route!r}, so it holds no trace",
                  segment=segment)
        batch = dict(conn.execute(
            "SELECT step, batch_size FROM segment_step WHERE segment_id = ?", (segment_id,)
        ).fetchall())
        fresh = []
        for trace in wanted:
            what = f"trace of step {trace.step} of segment {segment!r}"
            _need(trace.step in batch, f"{what}: the step is not among the segment's registered steps")
            _need(trace.consumed_samples == batch[trace.step],
                  f"{what}: it counts {trace.consumed_samples} samples and the step's batch "
                  f"size is {batch[trace.step]}",
                  segment=segment, step=trace.step)
            held = self.step_trace(segment, trace.step)
            if held is None:
                fresh.append(trace)
            else:
                _need(held == trace, f"{what}: the step already has another trace; a trace is written once")
        for trace in fresh:
            trace_id = conn.execute(
                "INSERT INTO step_trace (segment_id, step, consumed_samples) VALUES (?, ?, ?)",
                (segment_id, trace.step, trace.consumed_samples),
            ).lastrowid
            conn.executemany(
                "INSERT INTO trace_sample (trace_id, sample_pos, global_index, rank, "
                "sample_tokens, covered_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                self._feed("trace_sample", (
                    (trace_id, s.sample_pos, s.global_index, s.rank, s.sample_tokens, s.covered_tokens)
                    for s in trace.samples
                )),
            )
            conn.executemany(
                "INSERT INTO trace_row (trace_id, sample_pos, row_pos, shard_position, ordinal, "
                "token_start, token_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
                self._feed("trace_row", (
                    (trace_id, s.sample_pos, row_pos, r.shard_position, r.ordinal,
                     r.token_start, r.token_end)
                    for s in trace.samples for row_pos, r in enumerate(s.rows)
                )),
            )

    def register_checkpoint(self, *, name: str, segment: str, step: int) -> str:
        """Register a checkpoint: a marker at a completed step of a segment. The
        segment is named by value and need not be registered; when it is, the step
        must be one of its registered steps. Returns the instance id, which is
        ``ids.checkpoint_instance_id(segment, step)``."""
        self._require_writable()
        with self._transaction() as conn:
            return self._register_checkpoint(conn, name=name, segment=segment, step=step)

    def _register_checkpoint(self, conn, *, name: str, segment: str, step: int) -> str:
        """``register_checkpoint`` inside the caller's transaction."""
        _need(_is_text(name) and _is_text(segment), "a checkpoint has a name and names its segment")
        _need(_is_int(step) and step >= 0, "a checkpoint marks a step number")
        instance = checkpoint_instance_id(segment, step)
        if self._existing(conn, instance, v.KIND_CHECKPOINT, name):
            return instance
        row = self._segment_row(segment)
        if row is not None:
            known = conn.execute(
                "SELECT 1 FROM segment_step WHERE segment_id = ? AND step = ?",
                (row["segment_id"], step),
            ).fetchone()
            _need(known is not None,
                  f"step {step} is not among the registered steps of segment {segment!r}",
                  segment=segment, step=step)
            # Would a segment of this one's history start from this very checkpoint?
            seen: set[str] = set()
            current = row
            while current is not None and current["name"] not in seen:
                seen.add(current["name"])
                parent = current["parent_checkpoint"]
                _need(parent != instance,
                      f"checkpoint {name!r} would lie in its own history", segment=segment)
                mark = None if parent is None else self._checkpoint_by_instance(parent)
                current = None if mark is None else self._segment_row(mark.segment)
        artifact_id = self._insert_artifact(
            conn, instance=instance, kind=v.KIND_CHECKPOINT, role=None,
            content_digest=ids.checkpoint_content_digest(segment, step), path=None,
            step=(*ids.CHECKPOINT_STEP, ids.param_digest(ids.checkpoint_params(segment, step))),
            member_name=None, edge_digest=ids.no_edges_digest(), name=name, inputs=(),
        )
        conn.execute(
            "INSERT INTO checkpoint (artifact_id, segment_name, step) VALUES (?, ?, ?)",
            (artifact_id, segment, step),
        )
        return instance
