"""The loud checks, implemented once (plan 3.4, the closing paragraph of 2.2).

A loud fault stops a query: a named error, a nonzero exit code, no answer, whatever
the answer would have been. What each check compares (DECISIONS.md D-18), in the
order in which they raise:

1. ``MapCountMismatch``: a binding (a tokenization map, or a JSONL artifact's
   occurrence record) whose declared count differs from the number of rows the
   store holds under it (the half of the count comparison that needs no file;
   the other half runs in check 5);
2.``MapDigestMismatch``: a binding whose declared digest differs from the digest
   its artifact is registered with;
3. ``EdgeParentMissing``: an edge input, a map row or a candidate entry that names
   an occurrence row which is gone;
4. ``TraceSampleCountMismatch``: a present step trace whose consumed-sample count
   differs from its step's registered batch size, or from its own number of sample
   rows, or whose step has no registered batch size;
5. the files, each artifact in turn. ``ArtifactDigestMismatch``: an artifact
   whose file on disk no longer has the registered digest; a shard with one of its
   two files gone; a mixture snapshot entry whose digest differs from the digest
   its shard is registered with. Then, only for a file that is present and has
   its registered digest, ``MapCountMismatch``: a binding whose declared count
   differs from the number of records that file holds (the lines of a JSONL
   file, the documents of a shard's index);
6. ``StoredDigestMismatch``: a segment's run record whose text no longer has the
   digest registered with it;
7. ``StoredDigestMismatch``: a mixture whose snapshot entries or settings no longer
   give its content digest, parameter digest and instance id (a mixture with no
   snapshot entry at all is a record that is gone, which the proof reads);
8. ``LocatorDomainMismatch``: a binding that holds a record whose number lies
   outside the count the binding declares (L1 to Ln, ordinals 0 to n-1), for an
   artifact whose file is present;
9. ``CacheDigestConflict``: a segment with two recorded digests for one array of
   the blend.

Checks 6 to 9 compare records of the store with a digest or a count the store
holds of them, and never open a file. They find a record altered or removed
without the digest or the count that was recorded with it; they do not defeat a
change that rewrites every commitment as well (docs/store.md "The loud checks").

The store's checks 1 to 4 compare records of the store with other records of the
store and never open an artifact's file. Of this list, check 5 alone reads files:
each file once, and the read that gives the digest gives the record count too.
So a file that differs from the registered one, by one byte or by a line, with
the store untouched, raises "artifact digest mismatch" and no map fault. A count
or digest edited in the store raises its map fault: at check 1 or 2 when the
store's own records then disagree, and at check 5 when rows were removed
together with a lowered count, so that the store agrees with itself and
disagrees with the file. The two kinds of fault can be told apart, and no record
of a file that is present is invisible to the count (plan 2.2(d), 2.3).

States that are NOT faults, because the proof reads them and answers from them
(plan 2.2): a step with no trace, a trace with an uncovered range, an artifact
with no binding, an occurrence with no edge, and an artifact whose file is no
longer at its recorded place (``missing_files`` lists those): the records still
answer. For a file that is gone the declared count is compared with the store's
rows only.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .. import ids
from .. import vocabulary as v
from ..errors import (
    ArtifactDigestMismatch,
    CacheDigestConflict,
    LocatorDomainMismatch,
    MapCountMismatch,
    MapDigestMismatch,
    StoredDigestMismatch,
    TraceRowSegmentMismatch,
    TraceSampleCountMismatch,
)
from . import artifact_files
from ._core import DiskState
from ._reads import ReadMixin, parent_missing
from .views import FILE_ABSENT, FILE_NONE, FILE_PARTIAL, FILE_PRESENT, MissingFile

_FILE_KINDS = (v.KIND_JSONL, v.KIND_SHARD)


def _read_once(kind: str, path: Path) -> DiskState:
    """The state of the artifact at ``path`` and, when its file is present, its
    content digest and the number of records it holds, from one read of each
    file: the lines of a JSONL file as ``ids.split_lines`` defines them, the
    documents of a shard's index. A shard's index that cannot be read as an
    index gives no count; the digest is taken all the same."""
    if kind == v.KIND_JSONL:
        if not path.is_file():
            return DiskState(FILE_ABSENT, str(path))
        found = artifact_files.read_jsonl(path)
        return DiskState(FILE_PRESENT, str(path), found.line_count, found.digest)
    have = [p.is_file() for p in artifact_files.shard_paths(path)]
    if not any(have):
        return DiskState(FILE_ABSENT, str(path))
    if not all(have):
        return DiskState(FILE_PARTIAL, str(path))
    shard = artifact_files.read_shard_once(path)
    return DiskState(FILE_PRESENT, str(path), shard.document_count, shard.digest, shard.unreadable)


class IntegrityMixin(ReadMixin):
    # ------------------------------------------------------------------------------
    # Scope: a set of artifacts and their ancestor closure
    # ------------------------------------------------------------------------------

    def _closure(
        self, instance_ids: Iterable[str], models: bool = False
    ) -> tuple[list[sqlite3.Row], dict[int, int]]:
        """The artifact rows of the ancestor closure of the given instances, and for
        each run segment in a checkpoint's history the last step that lies in it.

        Ancestors of a JSONL artifact: its step's inputs and the artifacts that hold
        the records its edges and harness logs name. Of a shard: its step's inputs
        and the artifacts its map rows name. Of a mixture: its shards. Of a
        checkpoint: its segment's mixture and the checkpoint the segment started
        from. An instance id with no row has no closure.

        ``models`` (the generating-model extension is on, plan 2.2(h), 3.3 "Generation"):
        a registered checkpoint that a generation log of a JSONL artifact in the closure
        names as its model is an ancestor too, with its own ancestry, and so on to a fixpoint
        (the walk below visits each artifact once). The model text is resolved by
        ``resolve_model``, the one reading the rule itself uses. Off (the default) the
        closure is the one above and no model is looked at."""
        conn = self._conn
        found: dict[int, sqlite3.Row] = {}
        segments: dict[int, int] = {}
        queue: list[sqlite3.Row] = []
        for instance_id in instance_ids:
            row = self._artifact_row(instance_id)
            if row is not None:
                queue.append(row)

        per_kind = {
            v.KIND_JSONL: (
                "SELECT DISTINCT p.artifact_id FROM occurrence c "
                "JOIN edge e ON e.child_id = c.occurrence_id "
                "JOIN edge_input i ON i.edge_id = e.edge_id "
                "JOIN occurrence p ON p.occurrence_id = i.occurrence_id WHERE c.artifact_id = ?",
                "SELECT DISTINCT p.artifact_id FROM occurrence c "
                "JOIN edge e ON e.child_id = c.occurrence_id "
                "JOIN harness_handed h ON h.log_id = e.log_id "
                "JOIN occurrence p ON p.occurrence_id = h.occurrence_id WHERE c.artifact_id = ?",
            ),
            v.KIND_SHARD: (
                "SELECT DISTINCT p.artifact_id FROM binding b "
                "JOIN map_row m ON m.binding_id = b.binding_id "
                "JOIN occurrence p ON p.occurrence_id = m.parent_occurrence_id "
                "WHERE b.artifact_id = ?",
                "SELECT DISTINCT p.artifact_id FROM binding b "
                "JOIN map_candidate k ON k.binding_id = b.binding_id "
                "JOIN occurrence p ON p.occurrence_id = k.occurrence_id WHERE b.artifact_id = ?",
            ),
            v.KIND_MIXTURE: (
                "SELECT x.artifact_id FROM mixture_shard m "
                "JOIN artifact x ON x.instance_id = m.shard_instance_id WHERE m.artifact_id = ?",
            ),
        }
        inputs_sql = (
            "SELECT x.artifact_id FROM artifact_input i "
            "JOIN artifact x ON x.instance_id = i.input_instance_id WHERE i.artifact_id = ?"
        )

        while queue:
            row = queue.pop()
            artifact_id = row["artifact_id"]
            if artifact_id in found:
                continue
            found[artifact_id] = row
            ancestors: set[int] = set()
            for sql in (inputs_sql, *per_kind.get(row["kind"], ())):
                ancestors.update(r[0] for r in conn.execute(sql, (artifact_id,)))
            by_value: list[str] = []
            if models and row["kind"] == v.KIND_JSONL:
                texts = conn.execute(
                    "SELECT DISTINCT l.model FROM occurrence c "
                    "JOIN edge e ON e.child_id = c.occurrence_id "
                    "JOIN harness_log l ON l.log_id = e.log_id "
                    "WHERE c.artifact_id = ? AND l.model IS NOT NULL ORDER BY l.model",
                    (artifact_id,),
                ).fetchall()
                for (text,) in texts:
                    by_value += [m.checkpoint.instance_id for m in self.resolve_model(text)
                                 if m.checkpoint is not None]
            if row["kind"] == v.KIND_CHECKPOINT:
                mark = conn.execute(
                    "SELECT segment_name, step FROM checkpoint WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
                segment = None if mark is None else self._segment_row(mark["segment_name"])
                if segment is not None:
                    sid = segment["segment_id"]
                    segments[sid] = max(segments.get(sid, mark["step"]), mark["step"])
                    by_value += [segment["mixture_instance_id"], segment["parent_checkpoint"]]
            for instance_id in by_value:
                if instance_id is not None:
                    hit = self._artifact_row(instance_id)
                    if hit is not None and hit["artifact_id"] not in found:
                        queue.append(hit)
            for ancestor in ancestors - found.keys():
                hit = conn.execute(
                    "SELECT * FROM artifact WHERE artifact_id = ?", (ancestor,)
                ).fetchone()
                if hit is not None:
                    queue.append(hit)
        return [found[k] for k in sorted(found)], segments

    def _scope(
        self, instance_ids: Iterable[str] | None, models: bool = False
    ) -> tuple[list[sqlite3.Row], dict[int, int] | None]:
        if instance_ids is None:
            rows = self._conn.execute("SELECT * FROM artifact ORDER BY artifact_id").fetchall()
            return rows, None
        return self._closure(instance_ids, models)

    # ------------------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------------------

    def file_state(self, instance_id: str) -> str:
        """Where an artifact's file is: "present", "absent" (not at its recorded
        place: archived or moved), "partial" (a shard with one of its two files
        gone) or "no file" (a mixture, a checkpoint, an unregistered id). This looks
        and does not read the file."""
        row = self._artifact_row(instance_id)
        if row is None or row["kind"] not in _FILE_KINDS or row["path"] is None:
            return FILE_NONE
        path = self._disk_path(row["path"])
        if row["kind"] == v.KIND_JSONL:
            return FILE_PRESENT if path.is_file() else FILE_ABSENT
        have = [p.is_file() for p in artifact_files.shard_paths(path)]
        if all(have):
            return FILE_PRESENT
        return FILE_PARTIAL if any(have) else FILE_ABSENT

    def check_trace_row_segments(self) -> None:
        """One statement over the trace rows: every row's stored segment is the segment of its trace
        (``TraceRowSegmentMismatch`` names the first that is not). Run by ``check`` alone, not by
        ``check_integrity``, which a query runs: it reads every trace row of the store."""
        row = self._conn.execute(
            "SELECT r.trace_id, r.sample_pos, r.row_pos, r.segment_id, t.segment_id, t.step, s.name "
            "FROM trace_row r JOIN step_trace t ON t.trace_id = r.trace_id "
            "LEFT JOIN run_segment s ON s.segment_id = t.segment_id "
            "WHERE r.segment_id IS NOT t.segment_id ORDER BY r.trace_id, r.sample_pos, r.row_pos LIMIT 1"
        ).fetchone()
        if row is not None:
            raise TraceRowSegmentMismatch(
                f"a trace row of step {row[5]} of {row[6]} (trace {row[0]}, sample {row[1]}, row {row[2]}) "
                f"stores the segment {row[3]!r}, and its trace belongs to segment {row[4]}",
                trace=row[0], sample=row[1], row=row[2], stored=row[3], segment=row[6], step=row[5])

    def missing_files(
        self, instance_ids: Iterable[str] | None = None, *, models: bool = False
    ) -> tuple[MissingFile, ...]:
        """The artifacts, in the ancestor closure of the given instances or in the
        whole store, whose file is not at its recorded place. For these the digest
        checks cannot be made; an answer that rests on them can say so. ``models`` adds
        the ancestry of the generating models to the closure (``_closure``)."""
        rows, _ = self._scope(instance_ids, models)
        found = []
        for row in rows:
            state = self.file_state(row["instance_id"])
            if state in (FILE_ABSENT, FILE_PARTIAL):
                found.append(
                    MissingFile(row["instance_id"], row["kind"], str(self._disk_path(row["path"])), state)
                )
        return tuple(found)

    # ------------------------------------------------------------------------------
    # The checks
    # ------------------------------------------------------------------------------

    def check_integrity(
        self, instance_ids: Iterable[str] | None = None, *, models: bool = False
    ) -> None:
        """Run the loud checks over the ancestor closure of the given instances, or
        over the whole store when None. Returns None when there is no fault. ``models``
        adds the ancestry of the generating models to the closure (``_closure``); it is
        passed only with the generating-model switch on.

        The first fault found is raised. The kinds are tried in a fixed order: map
        count against the store's rows, map digest, edge parent, trace count, the
        files, run record digest, mixture identity, record numbers, blend digests.
        Inside one kind the artifacts are taken in registration order (for
        the trace check: the segments in registration order, each one's steps in
        step order), and an artifact's bindings in the order they were written.
        Inside the file check each artifact is looked at once, in registration
        order, and its file is read once: for a file, one of a shard's two files
        gone, then the file's digest, then, when the digest is the registered
        one, each binding's declared count against the records the file holds;
        for a mixture, its snapshot entries by position. The last four checks
        read the segments of the checkpoints' histories in the closure (every
        segment, for the whole store), the mixtures and the bindings of the
        artifacts in the closure."""
        artifacts, segments = self._scope(instance_ids, models)
        conn = self._conn

        bound: list[tuple[sqlite3.Row, sqlite3.Row]] = []
        bindings_of: dict[int, list[sqlite3.Row]] = {}
        for artifact in artifacts:
            for binding in conn.execute(
                "SELECT * FROM binding WHERE artifact_id = ? ORDER BY binding_id",
                (artifact["artifact_id"],),
            ).fetchall():
                bound.append((artifact, binding))
                bindings_of.setdefault(artifact["artifact_id"], []).append(binding)

        def name(binding: sqlite3.Row) -> str:
            mode = "" if binding["derivation"] is None else f" {binding['derivation']}"
            return f"{binding['namespace']}{mode}"

        # 1. map counts: the declared count against the rows the store holds (the
        #    declared count against the file is part of 5, where the file is read)
        for artifact, binding in bound:
            declared = binding["declared_count"]
            stored = self._rows_under(artifact["kind"], artifact["artifact_id"], binding["binding_id"])
            if stored != declared:
                raise MapCountMismatch(
                    f"the {name(binding)} binding of {artifact['instance_id']} declares "
                    f"{declared} records and the store holds {stored} rows under it",
                    declared=declared, stored=stored, instance=artifact["instance_id"],
                    namespace=binding["namespace"], derivation=binding["derivation"],
                )

        # 2. map digests: the declared digest against the artifact's registered digest
        for artifact, binding in bound:
            declared = binding["declared_digest"]
            if declared != artifact["content_digest"]:
                raise MapDigestMismatch(
                    f"the {name(binding)} binding of {artifact['instance_id']} declares the "
                    f"digest {declared} and its artifact is registered with "
                    f"{artifact['content_digest']}",
                    declared=declared, registered=artifact["content_digest"],
                    instance=artifact["instance_id"], namespace=binding["namespace"],
                    derivation=binding["derivation"],
                )

        # 3. parents (plan 2.2(c))
        self._check_parents(None if instance_ids is None else [a["artifact_id"] for a in artifacts])

        # 4. trace counts (plan 2.2(e), 3.4)
        self._check_traces(segments)

        # 5. the files: the registered digest against the file on disk, then a
        #    binding's declared count against the records of a file that has its
        #    registered digest; and a snapshot entry's digest against its shard's
        #    registered digest. One read of a file gives its digest and its count.
        for artifact in artifacts:
            if artifact["kind"] == v.KIND_MIXTURE:
                self._check_snapshot(artifact)
                continue
            if artifact["kind"] not in _FILE_KINDS or artifact["path"] is None:
                continue
            disk = _read_once(artifact["kind"], self._disk_path(artifact["path"]))
            if disk.state == FILE_PARTIAL:
                raise ArtifactDigestMismatch(
                    f"one of the two files of shard {artifact['instance_id']} is gone "
                    f"from {disk.path}",
                    instance=artifact["instance_id"], path=disk.path,
                )
            if disk.state == FILE_PRESENT and disk.digest != artifact["content_digest"]:
                raise ArtifactDigestMismatch(
                    f"artifact {artifact['instance_id']} is registered with digest "
                    f"{artifact['content_digest']} and its file has {disk.digest}",
                    instance=artifact["instance_id"], registered=artifact["content_digest"],
                    on_disk=disk.digest, path=disk.path,
                )
            if disk.state != FILE_PRESENT:
                continue  # gone from its recorded place: check 1 compared the store's rows
            # The file is the registered one: every binding declares what it holds.
            unit = "records" if artifact["kind"] == v.KIND_JSONL else "documents"
            for binding in bindings_of.get(artifact["artifact_id"], ()):
                declared = binding["declared_count"]
                if disk.count is None:
                    raise MapCountMismatch(
                        f"the {name(binding)} binding of {artifact['instance_id']} declares "
                        f"{declared} {unit} and the index of its shard at {disk.path} cannot "
                        f"be read to count them: {disk.unreadable}",
                        declared=declared, on_disk=None, instance=artifact["instance_id"],
                        path=disk.path, namespace=binding["namespace"],
                        derivation=binding["derivation"], unreadable=disk.unreadable,
                    )
                if declared != disk.count:
                    raise MapCountMismatch(
                        f"the {name(binding)} binding of {artifact['instance_id']} declares "
                        f"{declared} {unit} and its file at {disk.path} holds {disk.count}",
                        declared=declared, on_disk=disk.count, instance=artifact["instance_id"],
                        path=disk.path, namespace=binding["namespace"],
                        derivation=binding["derivation"],
                    )

        # 6 to 9: what the store recorded of its own records, held against them. None
        #    of these opens a file, and each comes after the five checks above.
        self._check_run_records(segments)
        for artifact in artifacts:
            if artifact["kind"] == v.KIND_MIXTURE:
                self._check_mixture_identity(artifact)
        for artifact, binding in bound:
            # A file that is gone from its recorded place is no fault (DECISIONS.md D-18):
            # an artifact whose file is not there is not looked at here either.
            if self.file_state(artifact["instance_id"]) == FILE_PRESENT:
                self._check_record_numbers(artifact, binding, name(binding))
        self._check_blend_digests(segments)

    # ------------------------------------------------------------------------------
    # Checks 6 to 9
    # ------------------------------------------------------------------------------

    def _check_run_records(self, segments: dict[int, int] | None) -> None:
        """A segment keeps its run record whole, as the canonical text it was
        registered with, and the digest of that text. The two must agree. A
        segment registered with no run record has neither. Only the text is held
        against its digest: the columns beside it (the sampler, the batch sizes,
        the route and its cause) are read from the record at registration and are
        not compared with it again."""
        select = "SELECT segment_id, name, run_record, run_record_digest FROM run_segment"
        if segments is None:
            rows = self._conn.execute(f"{select} ORDER BY segment_id").fetchall()
        else:
            rows = [
                self._conn.execute(f"{select} WHERE segment_id = ?", (segment_id,)).fetchone()
                for segment_id in sorted(segments)
            ]
        for row in rows:
            if row is None:
                continue
            text, registered = row["run_record"], row["run_record_digest"]
            computed = None if text is None else ids.bytes_digest(text.encode("utf-8"))
            if computed == registered:
                continue
            if computed is None:
                said = f"holds no run record and is registered with the digest {registered}"
            elif registered is None:
                said = f"holds a run record with the digest {computed} and is registered with none"
            else:
                said = f"holds a run record with the digest {computed} and is registered with {registered}"
            raise StoredDigestMismatch(
                f"segment {row['name']} {said}",
                record="run record", segment=row["name"], registered=registered, computed=computed,
            )

    def _check_mixture_identity(self, mixture: sqlite3.Row) -> None:
        """A mixture's content digest follows from the digests of its snapshot
        entries in position order, its parameter digest from its settings, and its
        instance id from its entries' shard instance ids, those two digests and
        nothing else (``ids.mixture_instance_id``). A mixture with no snapshot
        entry at all has no record of its own left: that is a state the proof
        reads ("not covered", "record missing"), and nothing is compared."""
        entries = self._conn.execute(
            "SELECT shard_instance_id, content_digest FROM mixture_shard "
            "WHERE artifact_id = ? ORDER BY position",
            (mixture["artifact_id"],),
        ).fetchall()
        if not entries:
            return
        context = {"instance": mixture["instance_id"]}
        content = ids.mixture_content_digest(e["content_digest"] for e in entries)
        if content != mixture["content_digest"]:
            raise StoredDigestMismatch(
                f"the snapshot of mixture {mixture['instance_id']} gives the content digest "
                f"{content} and the mixture is registered with {mixture['content_digest']}",
                record="mixture snapshot", registered=mixture["content_digest"], computed=content,
                **context,
            )
        held = self._conn.execute(
            "SELECT settings FROM mixture_settings WHERE artifact_id = ?", (mixture["artifact_id"],)
        ).fetchone()
        # The settings are held against the parameter digest when their text is the
        # canonical text of what it parses to, as every document with text keys is:
        # the digest was taken over the settings as given, and a text that does not
        # parse back to itself (a number used as a key) cannot be tied to it.
        if held is not None:
            try:
                value = json.loads(held["settings"])
                canonical = ids.canonical(value).decode("ascii") == held["settings"]
            except ValueError:  # text that is not JSON, or JSON that is not plain
                value, canonical = None, True
            params = None if value is None else ids.param_digest(value)
            if canonical and params != mixture["param_digest"]:
                raise StoredDigestMismatch(
                    f"the settings of mixture {mixture['instance_id']} give the parameter digest "
                    f"{params} and the mixture is registered with {mixture['param_digest']}",
                    record="mixture settings", registered=mixture["param_digest"], computed=params,
                    **context,
                )
        instance = ids.instance_id(
            content, *ids.MIXTURE_STEP, mixture["param_digest"],
            [e["shard_instance_id"] for e in entries], ids.no_edges_digest(),
        )
        if instance != mixture["instance_id"]:
            raise StoredDigestMismatch(
                f"the snapshot of mixture {mixture['instance_id']} gives the instance id {instance}",
                record="mixture snapshot", registered=mixture["instance_id"], computed=instance,
                **context,
            )

    def _check_record_numbers(self, artifact: sqlite3.Row, binding: sqlite3.Row, mode: str) -> None:
        """The records of a binding that declares n are numbered by the write: the
        lines L1 to Ln of a JSONL artifact, the ordinals 0 to n-1 of a shard's map.
        Check 1 has shown that the store holds n rows under the binding and the
        locator and the ordinal are unique, so a row outside the range means that
        a number was changed. A caller looks at an artifact whose file is present
        only (a gone file is no fault, DECISIONS.md D-18)."""
        declared = binding["declared_count"]
        if artifact["kind"] == v.KIND_JSONL:
            row = self._conn.execute(
                "SELECT locator FROM occurrence WHERE artifact_id = ? AND binding_id = ? AND ("
                "'L' || CAST(SUBSTR(locator, 2) AS INTEGER) != locator "
                "OR CAST(SUBSTR(locator, 2) AS INTEGER) < 1 "
                "OR CAST(SUBSTR(locator, 2) AS INTEGER) > ?) ORDER BY occurrence_id LIMIT 1",
                (artifact["artifact_id"], binding["binding_id"], declared),
            ).fetchone()
            if row is not None:
                raise LocatorDomainMismatch(
                    f"the {mode} binding of {artifact['instance_id']} declares {declared} records "
                    f"and the store holds the row {row['locator']}, which is not one of L1 to L{declared}",
                    instance=artifact["instance_id"], namespace=binding["namespace"],
                    derivation=binding["derivation"], declared=declared, locator=row["locator"],
                )
        elif artifact["kind"] == v.KIND_SHARD:
            row = self._conn.execute(
                "SELECT ordinal FROM map_row WHERE binding_id = ? AND (ordinal < 0 OR ordinal >= ?) "
                "ORDER BY ordinal DESC LIMIT 1",
                (binding["binding_id"], declared),
            ).fetchone()
            if row is not None:
                raise LocatorDomainMismatch(
                    f"the {mode} binding of {artifact['instance_id']} declares {declared} documents "
                    f"and the store holds a map row for the document {row['ordinal']}, which is "
                    f"not one of 0 to {declared - 1}",
                    instance=artifact["instance_id"], namespace=binding["namespace"],
                    derivation=binding["derivation"], declared=declared, ordinal=row["ordinal"],
                )

    def _check_blend_digests(self, segments: dict[int, int] | None) -> None:
        """One array of the blend (shard position NULL) has one recorded digest. The
        schema's uniqueness does not hold for a NULL position, so the check does."""
        select = (
            "SELECT s.name, d.kind, COUNT(*) AS held FROM segment_cache_digest d "
            "JOIN run_segment s ON s.segment_id = d.segment_id WHERE d.shard_position IS NULL"
        )
        group = " GROUP BY d.segment_id, d.kind HAVING COUNT(*) > 1 ORDER BY d.segment_id, d.kind LIMIT 1"
        if segments is None:
            rows = [self._conn.execute(select + group).fetchone()]
        else:
            rows = [
                self._conn.execute(select + " AND d.segment_id = ?" + group, (segment_id,)).fetchone()
                for segment_id in sorted(segments)
            ]
        for row in rows:
            if row is not None:
                raise CacheDigestConflict(
                    f"segment {row['name']} holds {row['held']} digests for its {row['kind']} array",
                    segment=row["name"], kind=row["kind"], held=row["held"],
                )

    def _check_snapshot(self, mixture: sqlite3.Row) -> None:
        """A snapshot entry carries the digest of the shard file the trainer was
        handed. When the shard it names is registered, the two digests must agree.
        A shard the store was never told of is named by value and is no fault."""
        row = self._conn.execute(
            "SELECT m.position, m.shard_instance_id, m.content_digest, x.content_digest AS registered "
            "FROM mixture_shard m JOIN artifact x ON x.instance_id = m.shard_instance_id "
            "WHERE m.artifact_id = ? AND m.content_digest != x.content_digest "
            "ORDER BY m.position LIMIT 1",
            (mixture["artifact_id"],),
        ).fetchone()
        if row is not None:
            raise ArtifactDigestMismatch(
                f"entry {row['position']} of the snapshot of mixture {mixture['instance_id']} "
                f"carries the digest {row['content_digest']} and its shard "
                f"{row['shard_instance_id']} is registered with {row['registered']}",
                instance=mixture["instance_id"], position=row["position"],
                shard=row["shard_instance_id"], snapshot=row["content_digest"],
                registered=row["registered"],
            )

    _PARENT_QUERIES = (
        (
            "SELECT a.instance_id || ':' || c.locator, 'input ' || i.position || ' of the edge', "
            "i.occurrence_id FROM edge_input i JOIN edge e ON e.edge_id = i.edge_id "
            "JOIN occurrence c ON c.occurrence_id = e.child_id "
            "JOIN artifact a ON a.artifact_id = c.artifact_id "
            "LEFT JOIN occurrence p ON p.occurrence_id = i.occurrence_id "
            "WHERE p.occurrence_id IS NULL",
            "c.artifact_id",
            "e.edge_id, i.position",
        ),
        (
            "SELECT a.instance_id || ':D' || m.ordinal, 'the map row', m.parent_occurrence_id "
            "FROM map_row m JOIN binding b ON b.binding_id = m.binding_id "
            "JOIN artifact a ON a.artifact_id = b.artifact_id "
            "LEFT JOIN occurrence p ON p.occurrence_id = m.parent_occurrence_id "
            "WHERE m.parent_occurrence_id IS NOT NULL AND p.occurrence_id IS NULL",
            "b.artifact_id",
            "m.binding_id, m.ordinal",
        ),
        (
            "SELECT a.instance_id || ':D' || k.ordinal, 'the candidate set of the map row', "
            "k.occurrence_id FROM map_candidate k JOIN binding b ON b.binding_id = k.binding_id "
            "JOIN artifact a ON a.artifact_id = b.artifact_id "
            "LEFT JOIN occurrence p ON p.occurrence_id = k.occurrence_id "
            "WHERE p.occurrence_id IS NULL",
            "b.artifact_id",
            "k.binding_id, k.ordinal, k.occurrence_id",
        ),
    )

    def _check_parents(self, artifact_ids: list[int] | None) -> None:
        for select, column, order in self._PARENT_QUERIES:
            if artifact_ids is None:
                row = self._conn.execute(f"{select} ORDER BY {order} LIMIT 1").fetchone()
                if row is not None:
                    raise parent_missing(row[0], row[1], row[2])
                continue
            for artifact_id in artifact_ids:
                row = self._conn.execute(
                    f"{select} AND {column} = ? ORDER BY {order} LIMIT 1", (artifact_id,)
                ).fetchone()
                if row is not None:
                    raise parent_missing(row[0], row[1], row[2])

    _TRACE_SELECT = (
        "SELECT s.name, t.step, t.consumed_samples, st.batch_size, "
        "(SELECT COUNT(*) FROM trace_sample x WHERE x.trace_id = t.trace_id) AS sample_rows "
        "FROM step_trace t JOIN run_segment s ON s.segment_id = t.segment_id "
        "LEFT JOIN segment_step st ON st.segment_id = t.segment_id AND st.step = t.step "
        "WHERE (st.batch_size IS NULL OR t.consumed_samples != st.batch_size OR "
        "t.consumed_samples != (SELECT COUNT(*) FROM trace_sample x WHERE x.trace_id = t.trace_id))"
    )

    def _check_traces(self, segments: dict[int, int] | None) -> None:
        """A present trace must count as many samples as its step's registered batch
        size, in its own count and in its sample rows. A step with no trace is not
        looked at here: it is untraced, which the proof reads."""
        if segments is None:
            rows = [self._conn.execute(self._TRACE_SELECT + " ORDER BY t.segment_id, t.step LIMIT 1").fetchone()]
        else:
            rows = [
                self._conn.execute(
                    self._TRACE_SELECT + " AND t.segment_id = ? AND t.step <= ? ORDER BY t.step LIMIT 1",
                    (segment_id, segments[segment_id]),
                ).fetchone()
                for segment_id in sorted(segments)
            ]
        for row in rows:
            if row is None:
                continue
            where = f"the trace of step {row['step']} of segment {row['name']}"
            context = {"segment": row["name"], "step": row["step"]}
            if row["batch_size"] is None:
                raise TraceSampleCountMismatch(
                    f"{where} counts {row['consumed_samples']} samples and the step has no "
                    "registered batch size",
                    consumed=row["consumed_samples"], **context,
                )
            if row["consumed_samples"] != row["batch_size"]:
                raise TraceSampleCountMismatch(
                    f"{where} counts {row['consumed_samples']} samples and the step's batch "
                    f"size is {row['batch_size']}",
                    consumed=row["consumed_samples"], batch_size=row["batch_size"], **context,
                )
            raise TraceSampleCountMismatch(
                f"{where} counts {row['consumed_samples']} samples and holds "
                f"{row['sample_rows']} sample rows",
                consumed=row["consumed_samples"], sample_rows=row["sample_rows"], **context,
            )
