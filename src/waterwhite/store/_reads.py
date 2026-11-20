"""The store's read methods (documented one by one in docs/store.md).

Every method returns typed results (``views`` and ``waterwhite.records``), never a
raw row. A method that can meet a large table returns an iterator and reads as it
goes: occurrences, edges, map rows, trace rows. Do not write to the store while
such an iterator is open.

Plan sections: 2.1 and Decision 21 (own record or only named), 2.2 (what the proof
walks), 2.4 (history), 3.1 (records), 3.4 (computed on demand, no second ledger).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import vocabulary as v
from ..errors import AmbiguousArtifact, EdgeParentMissing, IntegrityFault, WriteRefused
from ..harness import MODEL_NONE
from ..records import (
    EdgeInput,
    HarnessLog,
    HarnessRead,
    MapRow,
    OccurrenceRef,
    SnapshotShard,
    StepRef,
    StepTrace,
    TraceRow,
    TraceSample,
)
from ._core import StoreCore
from .views import (
    REL_ARTIFACT_INPUT,
    REL_CHECKPOINT_SEGMENT,
    REL_MIXTURE_SHARD,
    REL_SEGMENT_MIXTURE,
    REL_SEGMENT_PARENT,
    Artifact,
    Binding,
    CacheDigest,
    Checkpoint,
    ConsumedDocument,
    History,
    HistorySegment,
    Launch,
    MapLink,
    Member,
    Occurrence,
    Reference,
    Segment,
    SegmentStep,
    ShuffleDigest,
    SourceLink,
    StoredDrop,
    StoredEdge,
    TraceHit,
    TraceState,
)


@dataclass(frozen=True)
class ModelRef:
    """One model a stored harness log names, as ``ReadMixin.resolve_model`` reads it:
    ``none`` for the text "none", ``checkpoint`` for a registered checkpoint, else neither,
    and ``kind`` says what the text names when it names an artifact that is not a checkpoint."""

    text: str
    none: bool = False
    checkpoint: Artifact | None = None
    kind: str | None = None


def model_elements(text: str) -> tuple[str, ...]:
    """The models a log's model text names: the elements of the list form (two or more
    distinct non-empty identifiers, none of them "none", in order of first call), else the
    text itself as one model."""
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except (ValueError, RecursionError):
            # Not a list the harness wrote (a text nested past the parser's limit included):
            # one text, which is not a registered checkpoint.
            return (text,)
        if (
            isinstance(parsed, list)
            and len(parsed) >= 2
            and all(isinstance(item, str) and item != "" and item != MODEL_NONE for item in parsed)
            and len(set(parsed)) == len(parsed)
        ):
            return tuple(parsed)
    return (text,)


def parent_missing(child: str, what: str, occurrence_id: int) -> EdgeParentMissing:
    """The one wording of the loud fault "edge parent missing" (plan 2.2(c), 3.4)."""
    return EdgeParentMissing(
        f"{what} of {child} names occurrence row {occurrence_id}, which is gone",
        child=child,
        occurrence_row=occurrence_id,
    )


def _bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)


_EDGE_SELECT = (
    "SELECT e.edge_id, e.transform_name, e.transform_version, e.param_digest, "
    "e.parent_mode, e.derivation, e.verification, e.label, e.is_generation, e.job_name, "
    "e.log_id, e.bound_established, c.locator AS child_locator, a.instance_id AS child_instance "
    "FROM edge e JOIN occurrence c ON c.occurrence_id = e.child_id "
    "JOIN artifact a ON a.artifact_id = c.artifact_id "
)

_LINK_SELECT = (
    "SELECT a.instance_id, o.locator, m.name AS member, s.name AS source, l.label, l.evidence, "
    "l.derivation, l.verification "
    "FROM source_link l JOIN artifact a ON a.artifact_id = l.artifact_id "
    "JOIN member m ON m.member_id = l.member_id JOIN source s ON s.source_id = m.source_id "
    "LEFT JOIN occurrence o ON o.occurrence_id = l.occurrence_id "
)

_DROP_SELECT = (
    "SELECT a.instance_id, d.locator, d.reason, d.step_name, d.output_instance_id, d.group_key "
    "FROM drop_record d JOIN artifact a ON a.artifact_id = d.artifact_id "
)


def _reader_order(alias: str) -> str:
    """The order in which a reader takes an artifact's bindings (DECISIONS.md D-20
    as amended by D-24; the same three rules are written at the ``binding`` table
    in schema.sql): a verified map before one that is not, across both namespaces;
    then the capture namespace before the recovery namespace; then the position
    mode before token matching. So among verified maps a reader gets the capture
    map; else the position-mode recovery map; else the token-matching recovery
    map. Only a shard with no verified map gives a reader one that is not
    verified. That map is still the shard's own record, and its label can only be
    "unknown". A JSONL artifact has one binding and the order is idle."""
    return (
        f"CASE WHEN {alias}.verification = 'passed' THEN 0 ELSE 1 END, "
        f"CASE {alias}.namespace WHEN 'capture' THEN 0 ELSE 1 END, "
        f"CASE WHEN {alias}.derivation = 'token-matching' THEN 1 ELSE 0 END, "
        f"{alias}.binding_id"
    )


# True for the one binding of its artifact that a reader gets. ``b`` is the binding
# a row belongs to.
_READER_BINDING = (
    "b.binding_id = (SELECT c.binding_id FROM binding c WHERE c.artifact_id = b.artifact_id "
    f"ORDER BY {_reader_order('c')} LIMIT 1)"
)


class ReadMixin(StoreCore):
    # ------------------------------------------------------------------------------
    # The store itself
    # ------------------------------------------------------------------------------

    def meta(self) -> dict[str, str]:
        """The ``meta`` rows: ``schema_version`` and ``normaliser``."""
        return {row[0]: row[1] for row in self._conn.execute("SELECT key, value FROM meta")}

    @property
    def normaliser(self) -> str:
        """The named normalisation the store's fingerprints were taken after."""
        return self.meta()["normaliser"]

    @property
    def schema_version(self) -> int:
        return int(self.meta()["schema_version"])

    @property
    def path(self) -> Path:
        """The store file."""
        return self._path

    def disk_path(self, instance_id: str) -> Path | None:
        """Where a registered artifact's file lies now: the path the store holds
        for it, read from the folder of the store file when it is relative
        (docs/store.md, "Opening a store"). For a shard it is the prefix, without
        ``.bin`` and ``.idx``. None when the store has no row for the artifact or
        holds no path for it (a mixture, a checkpoint). The file need not be
        there: ``file_state`` says whether it is."""
        row = self._artifact_row(instance_id)
        return None if row is None else self._disk_path(row["path"])

    def state_mark(self) -> tuple[int, int]:
        """A mark of the store as it is now, for a reader that keeps what it learned
        (a coverage session, an integrity pass, a proof). The mark differs after any
        write: one made through this store, or one committed by another connection to
        the file (SQLite's data version, which another connection's commit changes,
        and this connection's own count of changed rows, which its own writes change).
        Reads change neither. A closed store has no mark and refuses."""
        if self._closed:
            raise WriteRefused("the store is closed")
        return (self._conn.execute("PRAGMA data_version").fetchone()[0], self._conn.total_changes)

    def verified_joins(self) -> tuple[tuple[str, str, str], ...]:
        """The (release tag, sampler type, sampler class) rows the join was verified
        for (plan F6), as written when the store was created."""
        return tuple(
            (row[0], row[1], row[2])
            for row in self._conn.execute(
                "SELECT release_tag, sampler_type, sampler_class FROM verified_join "
                "ORDER BY release_tag, sampler_type, sampler_class"
            )
        )

    # ------------------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------------------

    def _artifact_view(self, row: sqlite3.Row) -> Artifact:
        artifact_id = row["artifact_id"]
        names = tuple(
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM artifact_name WHERE artifact_id = ? ORDER BY name",
                (artifact_id,),
            )
        )
        inputs = tuple(
            r[0]
            for r in self._conn.execute(
                "SELECT input_instance_id FROM artifact_input WHERE artifact_id = ? "
                "ORDER BY position",
                (artifact_id,),
            )
        )
        step = None
        if row["step_name"] is not None:
            step = StepRef(row["step_name"], row["step_version"], row["param_digest"])
        return Artifact(
            instance_id=row["instance_id"],
            kind=row["kind"],
            role=row["role"],
            content_digest=row["content_digest"],
            path=row["path"],
            step=step,
            member_name=row["member_name"],
            edge_digest=row["edge_digest"],
            names=names,
            inputs=inputs,
        )

    def artifact(self, ref: str) -> Artifact | None:
        """The artifact with this name, or else with this instance id. None when the
        store has no row for it (see ``references_to`` to tell "record missing" from
        "not registered")."""
        row = self._conn.execute(
            "SELECT a.* FROM artifact_name n JOIN artifact a ON a.artifact_id = n.artifact_id "
            "WHERE n.name = ?",
            (ref,),
        ).fetchone()
        if row is None:
            row = self._artifact_row(ref)
        return None if row is None else self._artifact_view(row)

    def artifacts(self, kind: str | None = None) -> Iterator[Artifact]:
        """Every registered artifact, in registration order; one kind when given."""
        if kind is None:
            rows = self._conn.execute("SELECT * FROM artifact ORDER BY artifact_id")
        else:
            rows = self._conn.execute(
                "SELECT * FROM artifact WHERE kind = ? ORDER BY artifact_id", (kind,)
            )
        for row in rows.fetchall():
            yield self._artifact_view(row)

    def artifacts_by_digest(self, content_digest: str) -> tuple[Artifact, ...]:
        """Every instance with this content digest. Two instances can share their
        bytes (plan Decision 29), so this can hold more than one."""
        rows = self._conn.execute(
            "SELECT * FROM artifact WHERE content_digest = ? ORDER BY artifact_id",
            (content_digest,),
        ).fetchall()
        return tuple(self._artifact_view(r) for r in rows)

    def artifact_by_digest(self, content_digest: str) -> Artifact | None:
        """The one instance with this content digest; ``AmbiguousArtifact`` when
        several share it; None when none has it."""
        return self._only(self.artifacts_by_digest(content_digest), "content digest", content_digest)

    def artifacts_by_path(self, path) -> tuple[Artifact, ...]:
        """Every instance registered at this path. A path is tried as given, as a
        path under the store's folder, and as a shard file with its ``.bin`` or
        ``.idx`` ending removed."""
        text = str(path)
        forms = {text, self._stored_path(text), self._stored_path(self._dir / text)}
        for form in list(forms):
            for ending in (".bin", ".idx"):
                if form.endswith(ending):
                    forms.add(form[: -len(ending)])
        marks = ",".join("?" for _ in forms)
        rows = self._conn.execute(
            f"SELECT * FROM artifact WHERE path IN ({marks}) ORDER BY artifact_id",
            tuple(sorted(forms)),
        ).fetchall()
        return tuple(self._artifact_view(r) for r in rows)

    def artifact_by_path(self, path) -> Artifact | None:
        """The one instance registered at this path; ``AmbiguousArtifact`` when
        several are; None when none is."""
        return self._only(self.artifacts_by_path(path), "path", str(path))

    def resolve(self, ref: str) -> Artifact | None:
        """How a query names an artifact (docs/interfaces.md section 3): a name, an
        instance id, a content digest, or a path, tried in that order. A digest or
        a path that matches several instances is refused with ``AmbiguousArtifact``."""
        found = self.artifact(ref)
        if found is None:
            found = self.artifact_by_digest(ref)
        if found is None:
            found = self.artifact_by_path(ref)
        return found

    def shard_files_state(self, instance_id: str) -> str | None:
        """What a check of the shard's files found: "passed", "failed", or None when none
        was made (a shard registered under an earlier schema and not checked since; schema 5,
        ``shard_files_check``)."""
        row = self._conn.execute(
            "SELECT c.state FROM shard_files_check c JOIN artifact a ON a.artifact_id = c.artifact_id "
            "WHERE a.instance_id = ?", (instance_id,)).fetchone()
        return None if row is None else row[0]

    def resolve_model(self, text: str) -> tuple[ModelRef, ...]:
        """The one reading of the model entry of a stored harness log (plan 2.2(h)): what
        each model the text names resolves to. The generating-model rule and the closure of the
        loud checks both call this, so no second reading exists.

        A text that is the list form the harness writes (``model_field``: a JSON list of two or
        more distinct non-empty identifiers, none of them "none") names each listed model, in
        listed order. Any other text, one that starts with "[" included, names itself alone. A
        model is "none" (the log shows no model call), a registered checkpoint (the artifact the
        text names as one of its own names has the kind checkpoint; an instance id, a digest or
        a path does not find a model), or neither: then ``kind`` is the kind of the artifact the
        text names, when it names one that is not a checkpoint."""
        return tuple(self._model_ref(element) for element in model_elements(text))

    def _model_ref(self, text: str) -> ModelRef:
        if text == MODEL_NONE:
            return ModelRef(text, none=True)
        found = self.artifact(text)
        if found is not None and found.kind == v.KIND_CHECKPOINT and text in found.names:
            return ModelRef(text, checkpoint=found)
        # An instance id, a digest or a path that finds a checkpoint is no registered name: it reads as not registered, never as "a checkpoint, not a checkpoint".
        return ModelRef(text, kind=None if found is None or found.kind == v.KIND_CHECKPOINT else found.kind)

    @staticmethod
    def _only(found: tuple[Artifact, ...], what: str, value: str) -> Artifact | None:
        if not found:
            return None
        if len(found) > 1:
            raise AmbiguousArtifact(
                f"{len(found)} instances match the {what} {value}; name one by a name "
                "or by its instance id",
                instances=[a.instance_id for a in found],
                names=[list(a.names) for a in found],
            )
        return found[0]

    # ------------------------------------------------------------------------------
    # Named by another record (plan 2.1, Decisions 21 and 26)
    # ------------------------------------------------------------------------------

    _REFERENCE_QUERIES = (
        (
            REL_MIXTURE_SHARD,
            "SELECT a.instance_id AS referrer, m.shard_instance_id AS referred, m.position AS pos, "
            "EXISTS (SELECT 1 FROM artifact x WHERE x.instance_id = m.shard_instance_id) AS reg "
            "FROM mixture_shard m JOIN artifact a ON a.artifact_id = m.artifact_id",
            "m.shard_instance_id",
            "a.artifact_id, m.position",
        ),
        (
            REL_SEGMENT_MIXTURE,
            "SELECT s.name AS referrer, s.mixture_instance_id AS referred, NULL AS pos, "
            "EXISTS (SELECT 1 FROM artifact x WHERE x.instance_id = s.mixture_instance_id) AS reg "
            "FROM run_segment s",
            "s.mixture_instance_id",
            "s.segment_id",
        ),
        (
            REL_CHECKPOINT_SEGMENT,
            "SELECT a.instance_id AS referrer, c.segment_name AS referred, NULL AS pos, "
            "EXISTS (SELECT 1 FROM run_segment x WHERE x.name = c.segment_name) AS reg "
            "FROM checkpoint c JOIN artifact a ON a.artifact_id = c.artifact_id",
            "c.segment_name",
            "a.artifact_id",
        ),
        (
            REL_SEGMENT_PARENT,
            "SELECT s.name AS referrer, s.parent_checkpoint AS referred, NULL AS pos, "
            "EXISTS (SELECT 1 FROM artifact x WHERE x.instance_id = s.parent_checkpoint) AS reg "
            "FROM run_segment s",
            "s.parent_checkpoint",
            "s.segment_id",
        ),
        (
            REL_ARTIFACT_INPUT,
            "SELECT a.instance_id AS referrer, i.input_instance_id AS referred, i.position AS pos, "
            "EXISTS (SELECT 1 FROM artifact x WHERE x.instance_id = i.input_instance_id) AS reg "
            "FROM artifact_input i JOIN artifact a ON a.artifact_id = i.artifact_id",
            "i.input_instance_id",
            "a.artifact_id, i.position",
        ),
    )

    def _references(self, referred: str | None, missing_only: bool) -> tuple[Reference, ...]:
        found: list[Reference] = []
        for relation, select, column, order in self._REFERENCE_QUERIES:
            where = [f"{column} IS NOT NULL"]
            params: tuple = ()
            if referred is not None:
                where.append(f"{column} = ?")
                params = (referred,)
            sql = f"{select} WHERE {' AND '.join(where)} ORDER BY {order}"
            for row in self._conn.execute(sql, params):
                if missing_only and row["reg"]:
                    continue
                found.append(
                    Reference(relation, row["referrer"], row["referred"], row["pos"], bool(row["reg"]))
                )
        return tuple(found)

    def named_without_record(self) -> tuple[Reference, ...]:
        """Every place where a record names an artifact, or a checkpoint names a
        segment, that has no row of its own: a snapshot's shard, a segment's
        mixture, a checkpoint's segment, a segment's parent checkpoint, an
        artifact's input. A listing shows these as "not covered", "record missing"
        (plan 2.1, Decision 26)."""
        return self._references(None, True)

    def references_to(self, referred: str) -> tuple[Reference, ...]:
        """Every record that names this instance id (or, for a checkpoint's segment,
        this segment name), whether it is registered or not. An unregistered
        artifact with at least one reference is "record missing"; with none it is
        "not registered"."""
        return self._references(referred, False)

    # ------------------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------------------

    @staticmethod
    def _binding_view(row: sqlite3.Row) -> Binding:
        return Binding(
            instance_id=row["instance_id"],
            namespace=row["namespace"],
            declared_count=row["declared_count"],
            declared_digest=row["declared_digest"],
            derivation=row["derivation"],
            verification=row["verification"],
            label=row["label"],
            append_eod=_bool(row["append_eod"]),
            eod_in_span=_bool(row["eod_in_span"]),
        )

    def _binding_row(
        self, instance_id: str, namespace: str | None, derivation: str | None = None,
        verification: str | None = None,
    ) -> sqlite3.Row | None:
        """The binding a reader gets: the first, in the reader's order
        (``_reader_order``), among those of the named namespace, derivation and
        verification result (None: any)."""
        sql = (
            "SELECT b.*, a.instance_id, a.kind FROM binding b "
            "JOIN artifact a ON a.artifact_id = b.artifact_id WHERE a.instance_id = ? "
        )
        params: list = [instance_id]
        if namespace is not None:
            sql += "AND b.namespace = ? "
            params.append(namespace)
        if derivation is not None:
            sql += "AND b.derivation = ? "
            params.append(derivation)
        if verification is not None:
            sql += "AND b.verification = ? "
            params.append(verification)
        return self._conn.execute(
            sql + f"ORDER BY {_reader_order('b')} LIMIT 1", params
        ).fetchone()

    def bindings(self, instance_id: str) -> tuple[Binding, ...]:
        """Every binding of an artifact, in the reader's order: the first is the
        one a reader gets. Empty for an artifact with no record of its own (a
        shard with no map)."""
        rows = self._conn.execute(
            "SELECT b.*, a.instance_id FROM binding b "
            "JOIN artifact a ON a.artifact_id = b.artifact_id WHERE a.instance_id = ? "
            f"ORDER BY {_reader_order('b')}",
            (instance_id,),
        ).fetchall()
        return tuple(self._binding_view(r) for r in rows)

    def binding(
        self, instance_id: str, namespace: str | None = None, *, derivation: str | None = None,
        verification: str | None = None,
    ) -> Binding | None:
        """One binding. With no namespace and no derivation: the one a reader gets
        (among verified maps the capture map, else the position-mode recovery map,
        else the token-matching one; with no verified map, one that is not
        verified, in the same order). Naming a namespace, a derivation, a
        verification result or several of them narrows the choice; the same order
        picks among what is left, so of a namespace and derivation that hold a
        verified map beside one that did not verify, the verified one is read
        unless the other's result is named. None when there is none."""
        row = self._binding_row(instance_id, namespace, derivation, verification)
        return None if row is None else self._binding_view(row)

    def stored_row_count(
        self, instance_id: str, namespace: str | None = None, *, derivation: str | None = None,
        verification: str | None = None,
    ) -> int | None:
        """How many rows the store holds under a binding: occurrence rows of a JSONL
        artifact, map rows of a shard. None when there is no such binding."""
        row = self._binding_row(instance_id, namespace, derivation, verification)
        if row is None:
            return None
        return self._rows_under(row["kind"], row["artifact_id"], row["binding_id"])

    def _rows_under(self, kind: str, artifact_id: int, binding_id: int) -> int:
        if kind == v.KIND_JSONL:
            sql = "SELECT COUNT(*) FROM occurrence WHERE artifact_id = ? AND binding_id = ?"
            return self._conn.execute(sql, (artifact_id, binding_id)).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM map_row WHERE binding_id = ?", (binding_id,)
        ).fetchone()[0]

    # ------------------------------------------------------------------------------
    # Occurrences
    # ------------------------------------------------------------------------------

    def occurrence(self, ref: OccurrenceRef) -> Occurrence | None:
        """One record of a JSONL artifact, or None when the store has no row for it."""
        row = self._conn.execute(
            "SELECT o.fingerprint FROM occurrence o JOIN artifact a ON a.artifact_id = o.artifact_id "
            "WHERE a.instance_id = ? AND o.locator = ?",
            (ref.instance_id, ref.locator),
        ).fetchone()
        return None if row is None else Occurrence(ref, row[0])

    def occurrences(self, instance_id: str) -> Iterator[Occurrence]:
        """The records of a JSONL artifact in line order, read as the caller goes."""
        artifact_id = self._artifact_id(instance_id)
        if artifact_id is None:
            return
        for row in self._conn.execute(
            "SELECT locator, fingerprint FROM occurrence WHERE artifact_id = ? "
            "ORDER BY occurrence_id",
            (artifact_id,),
        ):
            yield Occurrence(OccurrenceRef(instance_id, row[0]), row[1])

    def occurrence_count(self, instance_id: str) -> int:
        """How many occurrence rows an artifact has."""
        artifact_id = self._artifact_id(instance_id)
        if artifact_id is None:
            return 0
        return self._conn.execute(
            "SELECT COUNT(*) FROM occurrence WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()[0]

    def occurrences_by_fingerprint(self, fingerprint: bytes) -> Iterator[Occurrence]:
        """Every record with this fingerprint, in any artifact. Equal fingerprints
        mean equal after the normalisation and nothing about origin (plan 3.1)."""
        for row in self._conn.execute(
            "SELECT a.instance_id, o.locator, o.fingerprint FROM occurrence o "
            "JOIN artifact a ON a.artifact_id = o.artifact_id WHERE o.fingerprint = ? "
            "ORDER BY o.occurrence_id",
            (fingerprint,),
        ):
            yield Occurrence(OccurrenceRef(row[0], row[1]), row[2])

    # ------------------------------------------------------------------------------
    # Edges and harness logs
    # ------------------------------------------------------------------------------

    def _harness_log(self, log_id: int | None) -> HarnessLog | None:
        if log_id is None:
            return None
        head = self._conn.execute(
            "SELECT job_name, log_digest, model, harness, hand_off_checked FROM harness_log "
            "WHERE log_id = ?",
            (log_id,),
        ).fetchone()
        if head is None:
            return None  # the log was removed: the bound cannot be derived (2.2(g))
        # A handed row whose occurrence is gone is left out. That can only shrink the
        # hand-off, so it can only make the bound "not established", never the reverse.
        # The hand-off is read by its position, the order it was handed in. A log
        # written before the store kept the position (NULL) has no order to give back
        # and is read in the order of its occurrence ids.
        handed = tuple(
            OccurrenceRef(r[0], r[1])
            for r in self._conn.execute(
                "SELECT a.instance_id, o.locator FROM harness_handed h "
                "JOIN occurrence o ON o.occurrence_id = h.occurrence_id "
                "JOIN artifact a ON a.artifact_id = o.artifact_id WHERE h.log_id = ? "
                "ORDER BY h.position IS NULL, h.position, h.occurrence_id",
                (log_id,),
            )
        )
        reads = []
        for r in self._conn.execute(
            "SELECT r.occurrence_id, r.outside_ref, r.refused, a.instance_id, o.locator "
            "FROM harness_read r LEFT JOIN occurrence o ON o.occurrence_id = r.occurrence_id "
            "LEFT JOIN artifact a ON a.artifact_id = o.artifact_id WHERE r.log_id = ? "
            "ORDER BY r.seq",
            (log_id,),
        ):
            occurrence = None if r["locator"] is None else OccurrenceRef(r["instance_id"], r["locator"])
            outside = r["outside_ref"]
            if occurrence is None and outside is None and r["occurrence_id"] is not None:
                outside = f"occurrence row {r['occurrence_id']} (gone)"
            reads.append(HarnessRead(occurrence, outside, bool(r["refused"])))
        return HarnessLog(head["job_name"], head["log_digest"], handed, tuple(reads),
                          head["model"], head["harness"], hand_off_checked=bool(head["hand_off_checked"]))

    def harness_log(self, job_name: str) -> HarnessLog | None:
        """The stored log of a generation job, by the job's name."""
        row = self._conn.execute(
            "SELECT log_id FROM harness_log WHERE job_name = ?", (job_name,)
        ).fetchone()
        return None if row is None else self._harness_log(row[0])

    def _edge_view(self, row: sqlite3.Row, logs: dict[int, HarnessLog | None]) -> StoredEdge:
        child = OccurrenceRef(row["child_instance"], row["child_locator"])
        inputs = []
        for r in self._conn.execute(
            "SELECT i.position, i.occurrence_id, i.used, i.selected, o.locator, a.instance_id "
            "FROM edge_input i LEFT JOIN occurrence o ON o.occurrence_id = i.occurrence_id "
            "LEFT JOIN artifact a ON a.artifact_id = o.artifact_id WHERE i.edge_id = ? "
            "ORDER BY i.position",
            (row["edge_id"],),
        ):
            if r["locator"] is None:
                raise parent_missing(child.key, f"input {r['position']} of the edge", r["occurrence_id"])
            inputs.append(
                EdgeInput(OccurrenceRef(r["instance_id"], r["locator"]), bool(r["used"]), bool(r["selected"]))
            )
        log_id = row["log_id"]
        if log_id is not None and log_id not in logs:
            logs[log_id] = self._harness_log(log_id)
        return StoredEdge(
            child=child,
            inputs=tuple(inputs),
            transform=StepRef(row["transform_name"], row["transform_version"], row["param_digest"]),
            parent_mode=row["parent_mode"],
            derivation=row["derivation"],
            verification=row["verification"],
            label=row["label"],
            is_generation=bool(row["is_generation"]),
            harness_log=None if log_id is None else logs[log_id],
            bound_field=bool(row["bound_established"]),
            job_name=row["job_name"],
        )

    def edge_of(self, ref: OccurrenceRef) -> StoredEdge | None:
        """The edge whose child is this occurrence, with its inputs and its harness
        log. None when the occurrence has no edge: that is a missing link the proof
        reads (plan 2.2(a)), not a fault. A stored input whose occurrence row is
        gone raises ``EdgeParentMissing``."""
        row = self._conn.execute(
            _EDGE_SELECT + "WHERE a.instance_id = ? AND c.locator = ?",
            (ref.instance_id, ref.locator),
        ).fetchone()
        return None if row is None else self._edge_view(row, {})

    def edges(self, instance_id: str) -> Iterator[StoredEdge]:
        """Every edge whose child lies in this artifact, in line order of the child."""
        logs: dict[int, HarnessLog | None] = {}
        for row in self._conn.execute(
            _EDGE_SELECT + "WHERE a.instance_id = ? ORDER BY e.child_id", (instance_id,)
        ):
            yield self._edge_view(row, logs)

    def edges_naming(self, ref: OccurrenceRef) -> Iterator[StoredEdge]:
        """The walk downward: every edge that names this occurrence among its
        inputs, as a parent or in scope only (each edge says which)."""
        logs: dict[int, HarnessLog | None] = {}
        for row in self._conn.execute(
            _EDGE_SELECT + "WHERE e.edge_id IN (SELECT i.edge_id FROM edge_input i "
            "JOIN occurrence p ON p.occurrence_id = i.occurrence_id "
            "JOIN artifact pa ON pa.artifact_id = p.artifact_id "
            "WHERE pa.instance_id = ? AND p.locator = ?) ORDER BY e.edge_id",
            (ref.instance_id, ref.locator),
        ):
            yield self._edge_view(row, logs)

    def documents_naming(self, ref: OccurrenceRef) -> Iterator[MapLink]:
        """The walk downward into shards: every document whose map row names this
        occurrence as its parent or holds it in its candidate set. Of each shard
        only the map a reader gets is searched (see ``binding``); a second map of
        the same shard is read by naming it to ``map_rows``."""
        occ = self._conn.execute(
            "SELECT o.occurrence_id FROM occurrence o JOIN artifact a ON a.artifact_id = o.artifact_id "
            "WHERE a.instance_id = ? AND o.locator = ?",
            (ref.instance_id, ref.locator),
        ).fetchone()
        if occ is None:
            return
        for via, sql in (
            (
                "parent",
                "SELECT a.instance_id, b.namespace, m.ordinal, m.label, b.derivation FROM map_row m "
                "JOIN binding b ON b.binding_id = m.binding_id "
                "JOIN artifact a ON a.artifact_id = b.artifact_id "
                f"WHERE m.parent_occurrence_id = ? AND {_READER_BINDING} "
                "ORDER BY a.artifact_id, m.ordinal",
            ),
            (
                "candidate",
                "SELECT a.instance_id, b.namespace, m.ordinal, m.label, b.derivation "
                "FROM map_candidate k "
                "JOIN map_row m ON m.binding_id = k.binding_id AND m.ordinal = k.ordinal "
                "JOIN binding b ON b.binding_id = m.binding_id "
                "JOIN artifact a ON a.artifact_id = b.artifact_id "
                f"WHERE k.occurrence_id = ? AND {_READER_BINDING} "
                "ORDER BY a.artifact_id, m.ordinal",
            ),
        ):
            for row in self._conn.execute(sql, (occ[0],)):
                yield MapLink(row[0], row[1], row[2], row[3], via, row[4])

    # ------------------------------------------------------------------------------
    # Tokenization maps
    # ------------------------------------------------------------------------------

    def map_rows(
        self,
        shard_instance_id: str,
        namespace: str | None = None,
        *,
        first_ordinal: int = 0,
        last_ordinal: int | None = None,
        derivation: str | None = None,
        verification: str | None = None,
    ) -> Iterator[MapRow]:
        """The rows of a shard's map in ordinal order, read as the caller goes.

        With no namespace and no derivation the rows are those of the map a reader
        gets (see ``binding``); name a namespace, a derivation, a verification
        result or several of them to read another map of the shard. A shard with
        no map gives no rows: ask ``binding`` first to tell "no map" apart. A row
        whose parent occurrence row is gone raises ``EdgeParentMissing``."""
        binding = self._binding_row(shard_instance_id, namespace, derivation, verification)
        if binding is None:
            return
        upper = (1 << 62) if last_ordinal is None else last_ordinal
        yield from self._map_rows_of(shard_instance_id, binding["binding_id"], first_ordinal, upper)

    def _map_rows_of(
        self, shard_instance_id: str, binding_id: int, first_ordinal: int, upper: int
    ) -> Iterator[MapRow]:
        """The rows under one binding, by its row id."""
        candidates = self._conn.execute(
            "SELECT k.ordinal, k.occurrence_id, o.locator, a.instance_id FROM map_candidate k "
            "LEFT JOIN occurrence o ON o.occurrence_id = k.occurrence_id "
            "LEFT JOIN artifact a ON a.artifact_id = o.artifact_id "
            "WHERE k.binding_id = ? AND k.ordinal BETWEEN ? AND ? ORDER BY k.ordinal, k.occurrence_id",
            (binding_id, first_ordinal, upper),
        )
        pending = candidates.fetchone()
        for row in self._conn.execute(
            "SELECT m.ordinal, m.token_start, m.token_end, m.label, m.fingerprint, "
            "m.parent_occurrence_id, o.locator, a.instance_id FROM map_row m "
            "LEFT JOIN occurrence o ON o.occurrence_id = m.parent_occurrence_id "
            "LEFT JOIN artifact a ON a.artifact_id = o.artifact_id "
            "WHERE m.binding_id = ? AND m.ordinal BETWEEN ? AND ? ORDER BY m.ordinal",
            (binding_id, first_ordinal, upper),
        ):
            ordinal = row["ordinal"]
            document = f"{shard_instance_id}:D{ordinal}"
            parent = None
            if row["parent_occurrence_id"] is not None:
                if row["locator"] is None:
                    raise parent_missing(document, "the map row", row["parent_occurrence_id"])
                parent = OccurrenceRef(row["instance_id"], row["locator"])
            twins = []
            while pending is not None and pending["ordinal"] <= ordinal:
                if pending["ordinal"] == ordinal:
                    if pending["locator"] is None:
                        raise parent_missing(
                            document, "the candidate set of the map row", pending["occurrence_id"]
                        )
                    twins.append(OccurrenceRef(pending["instance_id"], pending["locator"]))
                pending = candidates.fetchone()
            yield MapRow(
                ordinal=ordinal,
                token_start=row["token_start"],
                token_end=row["token_end"],
                label=row["label"],
                parent=parent,
                candidates=tuple(twins),
                fingerprint=row["fingerprint"],
            )

    def map_row(
        self,
        shard_instance_id: str,
        ordinal: int,
        namespace: str | None = None,
        *,
        derivation: str | None = None,
    ) -> MapRow | None:
        """One document's map row, from the map ``map_rows`` would read."""
        for row in self.map_rows(
            shard_instance_id, namespace, first_ordinal=ordinal, last_ordinal=ordinal,
            derivation=derivation,
        ):
            return row
        return None

    # ------------------------------------------------------------------------------
    # Sources, members, source links
    # ------------------------------------------------------------------------------

    def sources(self) -> tuple[str, ...]:
        """The names of the registered sources."""
        return tuple(r[0] for r in self._conn.execute("SELECT name FROM source ORDER BY name"))

    def _member_view(self, row: sqlite3.Row) -> Member:
        parts = tuple(
            (r[0], r[1])
            for r in self._conn.execute(
                "SELECT a.instance_id, p.locator FROM member_part p "
                "JOIN artifact a ON a.artifact_id = p.artifact_id WHERE p.member_id = ? "
                "ORDER BY p.artifact_id, p.locator",
                (row["member_id"],),
            )
        )
        return Member(row["name"], row["source"], row["revision"], row["part_kind"], parts)

    _MEMBER_SELECT = (
        "SELECT m.member_id, m.name, m.revision, m.part_kind, s.name AS source "
        "FROM member m JOIN source s ON s.source_id = m.source_id "
    )

    def members(self, source: str | None = None) -> tuple[Member, ...]:
        """The members registered under a source (every member when None), each
        with its parts. A withdrawal with no member named expands to these."""
        if source is None:
            rows = self._conn.execute(self._MEMBER_SELECT + "ORDER BY m.member_id").fetchall()
        else:
            rows = self._conn.execute(
                self._MEMBER_SELECT + "WHERE s.name = ? ORDER BY m.member_id", (source,)
            ).fetchall()
        return tuple(self._member_view(r) for r in rows)

    def member(self, name: str) -> Member | None:
        row = self._conn.execute(self._MEMBER_SELECT + "WHERE m.name = ?", (name,)).fetchone()
        return None if row is None else self._member_view(row)

    def member_occurrences(self, member_name: str) -> Iterator[OccurrenceRef]:
        """The occurrences a member covers: every record of a whole-file part, the
        one record of a record part. Read as the caller goes."""
        found = self.member(member_name)
        if found is None:
            return
        for instance_id, locator in found.parts:
            if locator:
                yield OccurrenceRef(instance_id, locator)
            else:
                for occurrence in self.occurrences(instance_id):
                    yield occurrence.ref

    def members_covering(self, ref: OccurrenceRef) -> tuple[Member, ...]:
        """The members whose parts cover this occurrence: where a chain back to an
        ingested revision ends (plan 2.2(a))."""
        rows = self._conn.execute(
            self._MEMBER_SELECT + "WHERE m.member_id IN (SELECT p.member_id FROM member_part p "
            "JOIN artifact a ON a.artifact_id = p.artifact_id "
            "WHERE a.instance_id = ? AND p.locator IN ('', ?)) ORDER BY m.member_id",
            (ref.instance_id, ref.locator),
        ).fetchall()
        return tuple(self._member_view(r) for r in rows)

    _MEMBER_OF_PART = (
        "SELECT m.name, s.name FROM member_part p JOIN member m ON m.member_id = p.member_id "
        "JOIN source s ON s.source_id = m.source_id "
        "JOIN artifact a ON a.artifact_id = p.artifact_id "
    )

    def file_members(self, instance_id: str) -> tuple[tuple[str, str], ...]:
        """The (member name, source name) pairs of the members that hold the whole
        file of this artifact, in the order the members were registered. A walk
        reads this once per artifact, so that a record of a source file that a
        whole-file member covers costs no query of its own."""
        rows = self._conn.execute(
            self._MEMBER_OF_PART + "WHERE a.instance_id = ? AND p.locator = '' ORDER BY m.member_id",
            (instance_id,),
        )
        return tuple((row[0], row[1]) for row in rows)

    def has_record_parts(self, instance_id: str) -> bool:
        """Whether any member names single records of this artifact. When none
        does, ``record_members`` is empty for every record of it."""
        row = self._conn.execute(
            "SELECT 1 FROM member_part p JOIN artifact a ON a.artifact_id = p.artifact_id "
            "WHERE a.instance_id = ? AND p.locator <> '' LIMIT 1",
            (instance_id,),
        ).fetchone()
        return row is not None

    def record_members(self, ref: OccurrenceRef) -> tuple[tuple[str, str], ...]:
        """The (member name, source name) pairs of the members that name this one
        record as a part. The members that hold its whole file are not among
        them: see ``file_members``. Unlike ``members_covering`` it loads no
        member's list of parts."""
        rows = self._conn.execute(
            self._MEMBER_OF_PART + "WHERE a.instance_id = ? AND p.locator = ? ORDER BY m.member_id",
            (ref.instance_id, ref.locator),
        )
        return tuple((row[0], row[1]) for row in rows)

    def sources_of_artifact(self, instance_id: str) -> tuple[str, ...]:
        """The names, sorted, of the sources that have a member with a part in this
        artifact (its whole file or single records of it), or that a source link
        of the artifact names. It names the sources recorded for the artifact's
        own records and follows no edge."""
        rows = self._conn.execute(
            "SELECT s.name FROM member_part p JOIN member m ON m.member_id = p.member_id "
            "JOIN source s ON s.source_id = m.source_id "
            "JOIN artifact a ON a.artifact_id = p.artifact_id WHERE a.instance_id = ? "
            "UNION SELECT s.name FROM source_link l JOIN member m ON m.member_id = l.member_id "
            "JOIN source s ON s.source_id = m.source_id "
            "JOIN artifact a ON a.artifact_id = l.artifact_id WHERE a.instance_id = ?",
            (instance_id, instance_id),
        )
        return tuple(sorted(row[0] for row in rows))

    @staticmethod
    def _link_view(row: sqlite3.Row) -> SourceLink:
        return SourceLink(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])

    def source_links(self, ref: OccurrenceRef) -> tuple[SourceLink, ...]:
        """The source links that hold for one occurrence: its own, then those of
        its whole file (``locator`` None). A file-level link to one source stands
        beside any stronger link to another (plan 2.2(b), Decision 3)."""
        rows = self._conn.execute(
            _LINK_SELECT + "WHERE a.instance_id = ? AND (o.locator = ? OR l.occurrence_id IS NULL) "
            "ORDER BY l.occurrence_id IS NULL, l.link_id",
            (ref.instance_id, ref.locator),
        ).fetchall()
        return tuple(self._link_view(r) for r in rows)

    def file_source_links(self, instance_id: str) -> tuple[SourceLink, ...]:
        """The links that tie every record of an artifact's file to a member."""
        rows = self._conn.execute(
            _LINK_SELECT + "WHERE a.instance_id = ? AND l.occurrence_id IS NULL ORDER BY l.link_id",
            (instance_id,),
        ).fetchall()
        return tuple(self._link_view(r) for r in rows)

    def record_source_links(self, instance_id: str) -> Iterator[SourceLink]:
        """Every link that ties one record of this artifact to a member."""
        for row in self._conn.execute(
            _LINK_SELECT + "WHERE a.instance_id = ? AND l.occurrence_id IS NOT NULL "
            "ORDER BY l.occurrence_id, l.link_id",
            (instance_id,),
        ):
            yield self._link_view(row)

    # ------------------------------------------------------------------------------
    # Drops
    # ------------------------------------------------------------------------------

    @staticmethod
    def _drop_view(row: sqlite3.Row) -> StoredDrop:
        return StoredDrop(OccurrenceRef(row[0], row[1]), row[2], row[3], row[4], row[5])

    def drops_from(self, input_instance_id: str) -> Iterator[StoredDrop]:
        """The records of this artifact that some step removed."""
        for row in self._conn.execute(
            _DROP_SELECT + "WHERE a.instance_id = ? ORDER BY d.drop_id", (input_instance_id,)
        ):
            yield self._drop_view(row)

    def drops_of(self, ref: OccurrenceRef) -> tuple[StoredDrop, ...]:
        """The drops of one record (one per step that removed it)."""
        rows = self._conn.execute(
            _DROP_SELECT + "WHERE a.instance_id = ? AND d.locator = ? ORDER BY d.drop_id",
            (ref.instance_id, ref.locator),
        ).fetchall()
        return tuple(self._drop_view(r) for r in rows)

    def drops_in_group(self, group_key: str) -> tuple[StoredDrop, ...]:
        """The drops that carry this group key: the losers of one dedup group."""
        rows = self._conn.execute(
            _DROP_SELECT + "WHERE d.group_key = ? ORDER BY d.drop_id", (group_key,)
        ).fetchall()
        return tuple(self._drop_view(r) for r in rows)

    def drops_by_output(self, output_instance_id: str) -> Iterator[StoredDrop]:
        """The drops the step that wrote this artifact recorded."""
        for row in self._conn.execute(
            _DROP_SELECT + "WHERE d.output_instance_id = ? ORDER BY d.drop_id",
            (output_instance_id,),
        ):
            yield self._drop_view(row)

    # ------------------------------------------------------------------------------
    # Mixtures
    # ------------------------------------------------------------------------------

    def mixture_shards(self, instance_id: str) -> tuple[SnapshotShard, ...]:
        """A mixture's snapshot: its shards in the trainer's order. A shard is named
        by value and may have no row of its own."""
        rows = self._conn.execute(
            "SELECT m.position, m.shard_instance_id, m.path, m.content_digest FROM mixture_shard m "
            "JOIN artifact a ON a.artifact_id = m.artifact_id WHERE a.instance_id = ? "
            "ORDER BY m.position",
            (instance_id,),
        ).fetchall()
        return tuple(SnapshotShard(r[0], r[1], r[2], r[3]) for r in rows)

    def mixture_settings(self, instance_id: str) -> object | None:
        """The settings a mixture was registered with, as the JSON value that was
        handed in (their digest is part of the mixture's instance id). None when
        the store holds no settings row for this instance id: it is not a
        registered mixture."""
        row = self._conn.execute(
            "SELECT s.settings FROM mixture_settings s "
            "JOIN artifact a ON a.artifact_id = s.artifact_id WHERE a.instance_id = ?",
            (instance_id,),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def mixtures_holding(self, shard_instance_id: str) -> tuple[str, ...]:
        """The instance ids of the mixtures whose snapshot names this shard."""
        rows = self._conn.execute(
            "SELECT DISTINCT a.instance_id FROM mixture_shard m "
            "JOIN artifact a ON a.artifact_id = m.artifact_id WHERE m.shard_instance_id = ? "
            "ORDER BY a.artifact_id",
            (shard_instance_id,),
        ).fetchall()
        return tuple(r[0] for r in rows)

    # ------------------------------------------------------------------------------
    # Runs, segments, traces
    # ------------------------------------------------------------------------------

    def runs(self) -> tuple[str, ...]:
        return tuple(r[0] for r in self._conn.execute("SELECT name FROM run ORDER BY run_id"))

    def segments(self, run: str | None = None) -> tuple[str, ...]:
        """Segment names, in run order and then in the order inside the run."""
        sql = "SELECT s.name FROM run_segment s JOIN run r ON r.run_id = s.run_id "
        if run is None:
            rows = self._conn.execute(sql + "ORDER BY r.run_id, s.position")
        else:
            rows = self._conn.execute(sql + "WHERE r.name = ? ORDER BY s.position", (run,))
        return tuple(r[0] for r in rows)

    def segments_using(self, mixture_instance_id: str) -> tuple[str, ...]:
        """The segments that trained on this mixture."""
        rows = self._conn.execute(
            "SELECT name FROM run_segment WHERE mixture_instance_id = ? ORDER BY segment_id",
            (mixture_instance_id,),
        )
        return tuple(r[0] for r in rows)

    def segments_started_from(self, checkpoint_instance_id: str) -> tuple[str, ...]:
        """The segments whose parent checkpoint is this one (its descendants)."""
        rows = self._conn.execute(
            "SELECT name FROM run_segment WHERE parent_checkpoint = ? ORDER BY segment_id",
            (checkpoint_instance_id,),
        )
        return tuple(r[0] for r in rows)

    def segment(self, name: str) -> Segment | None:
        """A run segment with its launches, cache and shuffle digests and steps."""
        row = self._segment_row(name)
        if row is None:
            return None
        sid = row["segment_id"]
        launches = tuple(
            Launch(*r)
            for r in self._conn.execute(
                "SELECT position, world_size, data_parallel_size, consumed_at_start, "
                "step_at_start, first_step, last_step FROM segment_launch "
                "WHERE segment_id = ? ORDER BY position",
                (sid,),
            )
        )
        cache = tuple(
            CacheDigest(*r)
            for r in self._conn.execute(
                "SELECT kind, shard_position, digest FROM segment_cache_digest "
                "WHERE segment_id = ? ORDER BY rowid",
                (sid,),
            )
        )
        shuffle = tuple(
            ShuffleDigest(*r)
            for r in self._conn.execute(
                "SELECT launch_position, epoch, rank, digest FROM segment_shuffle_digest "
                "WHERE segment_id = ? ORDER BY launch_position, epoch, rank",
                (sid,),
            )
        )
        steps = tuple(
            SegmentStep(*r)
            for r in self._conn.execute(
                "SELECT step, batch_size, last_sample_tokens_read FROM segment_step "
                "WHERE segment_id = ? ORDER BY step",
                (sid,),
            )
        )
        record = None if row["run_record"] is None else json.loads(row["run_record"])
        return Segment(
            run=row["run_name"],
            name=row["name"],
            position=row["position"],
            mixture_instance_id=row["mixture_instance_id"],
            parent_checkpoint=row["parent_checkpoint"],
            start_kind=row["start_kind"],
            run_record=record,
            run_record_digest=row["run_record_digest"],
            sampler_type=row["sampler_type"],
            sampler_class=row["sampler_class"],
            data_sharding=_bool(row["data_sharding"]),
            micro_batch_size=row["micro_batch_size"],
            global_batch_size=row["global_batch_size"],
            sequence_length=row["sequence_length"],
            add_extra_token=_bool(row["add_extra_token"]),
            random_seed=row["random_seed"],
            megatron_core_version=row["megatron_core_version"],
            megatron_bridge_version=row["megatron_bridge_version"],
            torch_version=row["torch_version"],
            release_tag=row["release_tag"],
            trace_route=row["trace_route"],
            trace_cause=row["trace_cause"],
            trace_cause_detail=row["trace_cause_detail"],
            launches=launches,
            cache_digests=cache,
            shuffle_digests=shuffle,
            steps=steps,
        )

    def trace_states(self, segment: str, *, upto_step: int | None = None) -> Iterator[TraceState]:
        """One row per registered step of a segment, in step order: is there a
        trace, what does it count, how many of its samples have an uncovered range.
        This is what condition 2.2(e) reads; it loads no trace row into memory (the
        sum of a sample's row lengths is taken in the query)."""
        row = self._segment_row(segment)
        if row is None:
            return
        upper = (1 << 62) if upto_step is None else upto_step
        for r in self._conn.execute(
            "SELECT s.step, s.batch_size, s.last_sample_tokens_read, t.trace_id, t.consumed_samples, "
            "(SELECT COUNT(*) FROM trace_sample x WHERE x.trace_id = t.trace_id) AS sample_rows, "
            "(SELECT COUNT(*) FROM trace_sample x WHERE x.trace_id = t.trace_id "
            " AND x.covered_tokens < x.sample_tokens) AS uncovered, "
            "(SELECT COUNT(*) FROM trace_sample x WHERE x.trace_id = t.trace_id "
            " AND x.covered_tokens > COALESCE((SELECT SUM(w.token_end - w.token_start) "
            "  FROM trace_row w WHERE w.trace_id = x.trace_id AND w.sample_pos = x.sample_pos), 0)"
            ") AS unbacked "
            "FROM segment_step s LEFT JOIN step_trace t "
            "ON t.segment_id = s.segment_id AND t.step = s.step "
            "WHERE s.segment_id = ? AND s.step <= ? ORDER BY s.step",
            (row["segment_id"], upper),
        ):
            yield TraceState(
                step=r["step"],
                batch_size=r["batch_size"],
                last_sample_tokens_read=r["last_sample_tokens_read"],
                traced=r["trace_id"] is not None,
                consumed_samples=r["consumed_samples"],
                sample_rows=r["sample_rows"],
                uncovered_samples=r["uncovered"],
                unbacked_samples=r["unbacked"],
            )

    def _trace_view(self, trace_id: int, step: int, consumed: int) -> StepTrace:
        rows: dict[int, list[TraceRow]] = {}
        for r in self._conn.execute(
            "SELECT sample_pos, shard_position, ordinal, token_start, token_end FROM trace_row "
            "WHERE trace_id = ? ORDER BY sample_pos, row_pos",
            (trace_id,),
        ):
            rows.setdefault(r[0], []).append(TraceRow(r[1], r[2], r[3], r[4]))
        samples = tuple(
            TraceSample(r[0], r[1], r[2], r[3], r[4], tuple(rows.get(r[0], ())))
            for r in self._conn.execute(
                "SELECT sample_pos, global_index, rank, sample_tokens, covered_tokens "
                "FROM trace_sample WHERE trace_id = ? ORDER BY sample_pos",
                (trace_id,),
            )
        )
        return StepTrace(step, consumed, samples)

    def step_trace(self, segment: str, step: int) -> StepTrace | None:
        """The trace of one step with its samples and rows. None when the step has
        no trace: an untraced step, not a fault (plan 2.2(e))."""
        row = self._conn.execute(
            "SELECT t.trace_id, t.step, t.consumed_samples FROM step_trace t "
            "JOIN run_segment s ON s.segment_id = t.segment_id WHERE s.name = ? AND t.step = ?",
            (segment, step),
        ).fetchone()
        return None if row is None else self._trace_view(row[0], row[1], row[2])

    def step_traces(
        self, segment: str, *, first_step: int | None = None, last_step: int | None = None
    ) -> Iterator[StepTrace]:
        """The traces a segment holds, in step order, one step loaded at a time."""
        lower = -(1 << 62) if first_step is None else first_step
        upper = (1 << 62) if last_step is None else last_step
        for row in self._conn.execute(
            "SELECT t.trace_id, t.step, t.consumed_samples FROM step_trace t "
            "JOIN run_segment s ON s.segment_id = t.segment_id "
            "WHERE s.name = ? AND t.step BETWEEN ? AND ? ORDER BY t.step",
            (segment, lower, upper),
        ):
            yield self._trace_view(row[0], row[1], row[2])

    def trace_documents(
        self, segment: str, *, upto_step: int | None = None
    ) -> Iterator[ConsumedDocument]:
        """The distinct (shard position, ordinal) documents the traced steps of a
        segment read, up to a step, each with the first step that read it. A
        document counts when any of its tokens is in a row (plan Decision 18)."""
        upper = (1 << 62) if upto_step is None else upto_step
        for r in self._conn.execute(
            "SELECT r.shard_position, r.ordinal, MIN(t.step) FROM step_trace t "
            "JOIN run_segment s ON s.segment_id = t.segment_id "
            "JOIN trace_row r ON r.trace_id = t.trace_id "
            "WHERE s.name = ? AND t.step <= ? GROUP BY r.shard_position, r.ordinal "
            "ORDER BY r.shard_position, r.ordinal",
            (segment, upper),
        ):
            yield ConsumedDocument(r[0], r[1], r[2])

    def trace_hits(
        self, segment: str, shard_position: int, ordinal: int, *, upto_step: int | None = None
    ) -> Iterator[TraceHit]:
        """Every trace row of a segment that reads one document, in step order, with
        the coverage mark of its sample.

        The statement starts from the segment and finds the rows of the document in it through
        the index on (segment, shard position, ordinal) (schema 5), so the cost of a call
        grows with neither the steps of the segment nor the rows that read the same document in
        other segments. The tables are joined in the order written (CROSS JOIN): left to
        choose, SQLite started at the segment's steps and looked the document up once per step,
        which made a walk over every document of a checkpoint cost documents times steps. The
        order is complete: a step number is one trace of a segment, and a row is one place in
        one sample, so no two rows tie. A store of an earlier schema is not read (it must be
        migrated first)."""
        upper = (1 << 62) if upto_step is None else upto_step
        for r in self._conn.execute(
            "SELECT t.step, r.sample_pos, r.row_pos, r.token_start, r.token_end, "
            "x.sample_tokens, x.covered_tokens FROM run_segment s "
            "CROSS JOIN trace_row r CROSS JOIN step_trace t CROSS JOIN trace_sample x "
            "WHERE s.name = ? AND r.segment_id = s.segment_id "
            "AND r.shard_position = ? AND r.ordinal = ? "
            "AND t.trace_id = r.trace_id AND t.step <= ? "
            "AND x.trace_id = r.trace_id AND x.sample_pos = r.sample_pos "
            "ORDER BY t.step, r.sample_pos, r.row_pos",
            (segment, shard_position, ordinal, upper),
        ):
            yield TraceHit(*r)

    # ------------------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------------------

    _CHECKPOINT_SELECT = (
        "SELECT a.artifact_id, a.instance_id, c.segment_name, c.step FROM checkpoint c "
        "JOIN artifact a ON a.artifact_id = c.artifact_id "
    )

    def _checkpoint_view(self, row: sqlite3.Row) -> Checkpoint:
        names = tuple(
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM artifact_name WHERE artifact_id = ? ORDER BY name", (row[0],)
            )
        )
        return Checkpoint(row[1], names, row[2], row[3])

    def checkpoint(self, ref: str) -> Checkpoint | None:
        """A checkpoint by name or instance id. None when it is not registered."""
        row = self._conn.execute(
            self._CHECKPOINT_SELECT + "JOIN artifact_name n ON n.artifact_id = a.artifact_id "
            "WHERE n.name = ?",
            (ref,),
        ).fetchone()
        if row is None:
            return self._checkpoint_by_instance(ref)
        return self._checkpoint_view(row)

    def _checkpoint_by_instance(self, instance_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            self._CHECKPOINT_SELECT + "WHERE a.instance_id = ?", (instance_id,)
        ).fetchone()
        return None if row is None else self._checkpoint_view(row)

    def checkpoints_of(self, segment: str) -> tuple[Checkpoint, ...]:
        """The checkpoints registered at steps of a segment, in step order."""
        rows = self._conn.execute(
            self._CHECKPOINT_SELECT + "WHERE c.segment_name = ? ORDER BY c.step, a.artifact_id",
            (segment,),
        ).fetchall()
        return tuple(self._checkpoint_view(r) for r in rows)

    def checkpoint_history(self, ref: str) -> History | None:
        """A checkpoint's history through its ancestors (plan 2.4): for the
        checkpoint's own segment the registered steps up to its step, then the same
        for the segment its parent checkpoint lies in, and so on, returned oldest
        first. None when the checkpoint is not registered. A history that stops at
        a missing record says so (``complete`` False, ``missing`` set)."""
        current = self.checkpoint(ref)
        if current is None:
            return None
        start = current.instance_id
        parts: list[HistorySegment] = []
        seen: set[str] = set()
        missing: Reference | None = None
        complete = False
        while True:
            if current.segment in seen:
                raise IntegrityFault(
                    f"the history of checkpoint {start} loops at segment {current.segment}",
                    checkpoint=start,
                    segment=current.segment,
                )
            seen.add(current.segment)
            row = self._segment_row(current.segment)
            if row is None:
                missing = Reference(
                    REL_CHECKPOINT_SEGMENT, current.instance_id, current.segment, None, False
                )
                break
            steps = tuple(
                r[0]
                for r in self._conn.execute(
                    "SELECT step FROM segment_step WHERE segment_id = ? AND step <= ? ORDER BY step",
                    (row["segment_id"], current.step),
                )
            )
            parts.append(
                HistorySegment(
                    segment=current.segment,
                    checkpoint=current.instance_id,
                    upto_step=current.step,
                    steps=steps,
                    upto_registered=bool(steps) and steps[-1] == current.step,
                )
            )
            parent = row["parent_checkpoint"]
            if parent is None:
                complete = True
                break
            following = self._checkpoint_by_instance(parent)
            if following is None:
                missing = Reference(REL_SEGMENT_PARENT, current.segment, parent, None, False)
                break
            current = following
        parts.reverse()
        return History(start, tuple(parts), complete, missing)
