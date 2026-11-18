"""OriginBlame together with the trace (plan 5.5 Check 10; table README, section "Check 10":
the subject ``originblame+trace``, "The OriginBlame tie by token range" and "Readings chosen
here" 7 and 8; the rows ``c10.<group>.originblame+trace`` of ``check10.toml``).

The tool half is a ``baselines.originblame.OriginBlame`` as built (its shards, its requests, its
runner, its fault putter), the caller having called its ``track_all``. The trace half is a
document ``{"segments": [...]}`` in the form ``baselines.trace.export_segment`` returns, given
by ``trace_of(listed_row)`` (a document, or the failed ``Attempt`` of its export; a fault of the
trace half is already in the document it gives). ``place_of(listed_row)`` is the one
``baselines.trace.TraceSubject`` takes. Neither half is changed, and nothing is started here:
the tool is started only through the steps of the tool half.

What each question is read from:

- A row that records none is the reading with ``raw`` None, and nothing is called for it.
- The trace half alone (``c01.positions``, ``c01.step-facts``,
  ``c05.unregistered-checkpoint``): what ``TraceSubject`` reads from the document. A failed
  export is that ``Attempt``.
- The tool half alone (``c03.spans``, ``c04.interrupted-write``, ``c04.count-499``,
  ``c05.unmutated.shards``, ``c05.withdraw-C.unmutated``, ``c05.withdraw-C-on`` (the row of the
  generating-model extension: the trace holds no record about its artifact), ``c06.downstream``,
  ``c09.artifact-actions``, ``c09.successors``) and a shard of the four groups that put a
  fault: the tool half's own reading, a failed ``Attempt`` of it and its ``NotAttempted``
  included. A row that names two layouts is read by ``read_by_layout`` only. Where ``served``
  is given and gives the plain line's own reading (or failed ``Attempt``) of the same listed
  row and size, that raw reading is the combination's: the tool half is called for nothing and
  the combination's own row grades it. ``was_served(question)`` says so for the last reading.
- A source file, a generation output, a release, a mixture or the listing in one of the four
  fault groups: ``cannot_answer`` with the missing text of the combination's own row, and only
  where the tool ran on the store the question is asked on (the tool half's own gate: where a
  shard failed, its failed ``Attempt``; where none was given or tracked, ``NotAttempted``).
- A checkpoint (the walk): the history of the checkpoint comes from the document
  (``baselines.history.history_rows``). Each row is tied to the token-index entry of its
  document by RANGE: the document with ordinal n covers the range the shard's own index gives
  it (the sizes of the documents of the ``ShardSpec``), entry k of the store that tracked the
  shard covers the tokens from the sum of the counts of entries 0 to k-1 up to that sum plus
  its own count, and a row is tied to the entry whose range equals its document's. The row's
  own token range is a range inside its document and plays no part. The bit of an entry is read
  from the forget set the tool makes after the by-sources call for the files of the members the
  request names, on a fresh copy of the store of each shard the history reads, called once per
  copy. The reading is ``contains`` where a tied entry's bit is 1, ``does not contain`` where
  the history has rows, at least one is tied and no tied entry's bit is 1, and ``not covered``
  where the trace holds no such checkpoint or no row for any step of its history, and where
  rows exist and none is tied to an entry (the table names no other reading for it). A step
  with no row, an uncovered range and a row with no entry add nothing. The tie is valid only
  where the run of every shard the history uses says one process in packed order; else the
  reading is ``cannot_answer`` with the ``otherwise`` text of the tie of the question's row.
- A fault of the tool half (``f.map-deleted``, ``f.digest``) is put by the putter of the tool
  half, once, on the fresh copy of the shard the fault names and only where the history reads
  that shard, before the by-sources call; the two steps that read the copy are told it is
  faulted, and on such a copy the tool's own end with an error is ``{"stopped": True}`` while
  any other end is the failed ``Attempt`` of the step. A shard read before an ``Attempt`` came
  does not change the ``Attempt``: it wins over ``stopped``, which wins over the bits.
- ``c08.step-8-holds-no-T``: ``rows_that_name_T`` is the number of rows of the named sample
  tied to an entry whose sources name the section registered for the file of ``gen/T``, and
  ``first_exposure`` is the segment and step of the first step of the history of the
  checkpoint (oldest segment first) that has such a row. It is read as it stands from the
  tracked stores; no copy is made and no revoke is called.
- ``c09.exposure-free-restart``: the checkpoints of the run in step order, each read as above;
  the reading is the latest one read ``does not contain`` with every earlier one read so, and
  None where the first is not. A failed ``Attempt``, or a checkpoint read ``cannot_answer``,
  is that ``Attempt`` or that reading for the row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from . import history, record, tokenindex
from . import originblame as tool_half
from .interfaces import Attempt, Question, Reading
from .ob_driver import FAULTED_FLAG, META_FILE
from .trace import TraceSubject

SUBJECT = "originblame+trace"

# The routed groups, by what each is read from.
TRACE_GROUPS = ("c01.positions", "c01.step-facts", "c05.unregistered-checkpoint")
TOOL_GROUPS = ("c03.spans", "c04.interrupted-write", "c04.count-499", "c05.unmutated.shards",
               "c05.withdraw-C.unmutated", "c05.withdraw-C-on", "c06.downstream", "c09.artifact-actions",
               "c09.successors")
WALK_GROUPS = ("c01.E@9", "c02.checkpoints", "c04.trace-count", "c04.not-a-stop",
               "c05.unmutated.checkpoints", "c08.checkpoints")
KIND_GROUPS = ("c05.f.trace-deleted", "c05.f.trace-half", "c05.f.map-deleted", "c05.f.digest")
STEP_GROUP = "c08.step-8-holds-no-T"
RESTART_GROUP = "c09.exposure-free-restart"

# The row whose missing text a kind of artifact reads, in the combination's own rows.
MISSING_ROWS = {
    "files": "c10.c05.unmutated.files.originblame+trace",
    "mixtures": "c10.c05.unmutated.mixtures.originblame+trace",
    "listing": "c10.c05.listing.map-deleted.originblame+trace",
}

# The file of the generation output T, whose registered section names its entries.
T_FILE = "gen/T.jsonl"

CONTAINS = tool_half.CONTAINS
DOES_NOT_CONTAIN = tool_half.DOES_NOT_CONTAIN
NOT_COVERED = tool_half.NOT_COVERED

_STOPPED = object()


class _TwoEntries(ValueError):
    """Two entries of one store hold the range of a document: the tie names no one entry."""


class OriginBlameTrace:
    """The ``originblame+trace`` subject over a tool half and a trace.

    ``originblame`` is an ``OriginBlame`` whose ``track_all`` was called. ``trace_of(listed_row)``
    gives the document ``{"segments": [...]}`` the question's starting state reads, or the failed
    ``Attempt`` of its export. ``place_of(listed_row)`` is the one ``TraceSubject`` takes.
    """

    name = SUBJECT

    def __init__(self, originblame, *, trace_of, place_of, served=None):
        self._ob = originblame
        self._trace_of = trace_of
        self._place_of = place_of
        self._served = served
        self._served_ids: set[str] = set()

    # -- what the plain line serves ------------------------------------------------------------

    def was_served(self, question: Question) -> bool:
        """Whether the last reading of ``question`` was the plain line's own (its twin's) and the
        tool half was called for nothing."""
        return question.id in self._served_ids

    def _from_tool_half(self, question: Question) -> Reading | Attempt:
        """The reading of a row the tool half alone reads: the raw reading (or failed attempt) of
        the plain line of the same row where ``served`` gives one, graded as this question's own;
        else the tool half's own read."""
        self._served_ids.discard(question.id)
        found = self._served(question) if self._served is not None else None
        if isinstance(found, Attempt):
            self._served_ids.add(question.id)
            return found
        if isinstance(found, Reading):
            self._served_ids.add(question.id)
            return Reading(question_id=question.id, raw=found.raw, route=question.route)
        return self._ob.read(question)

    # -- reading a question ----------------------------------------------------------------

    def read(self, question: Question) -> Reading | Attempt:
        """A Reading, or the failed Attempt. A row that records none has the reading ``raw``
        None, and nothing is called for it. A row that names two layouts is read by
        ``read_by_layout``."""
        if question.route is None:
            return Reading(question_id=question.id, raw=None, route=None)
        if record.layouts_of(question) is not None:
            raise ValueError(f"{question.listed_row} names two layouts: use read_by_layout")
        group = question.group
        if group in TRACE_GROUPS:
            return self._read_trace(question)
        if group in TOOL_GROUPS:
            return self._from_tool_half(question)
        if group in WALK_GROUPS:
            return self._read_walk(question)
        if group in KIND_GROUPS:
            return self._read_by_kind(question)
        if group == STEP_GROUP:
            return self._read_step_8(question)
        if group == RESTART_GROUP:
            return self._read_restart(question)
        raise ValueError(f"the combination has no reading for {question.listed_row}")

    def read_by_layout(self, question: Question) -> dict[str, Reading | Attempt | None]:
        """The reading of a row that names two layouts, one per layout, as the tool half gives
        it: a layout none of whose shards was given or tracked is None (not attempted), never
        "cannot answer"."""
        if record.layouts_of(question) is None:
            raise ValueError(f"{question.listed_row} names no layouts")
        if question.route is None:
            return self._ob.read_by_layout(question)
        if question.group in TOOL_GROUPS:
            self._served_ids.discard(question.id)
            found = self._served(question) if self._served is not None else None
            if isinstance(found, Mapping):
                # The plain line's reading of each layout, graded as this question's own.
                self._served_ids.add(question.id)
                return {layout: (Reading(question_id=question.id, raw=one.raw, route=question.route)
                                 if isinstance(one, Reading) else one)
                        for layout, one in found.items()}
            return self._ob.read_by_layout(question)
        raise ValueError(f"{question.listed_row} is not read per layout by the combination")

    @staticmethod
    def _reading(question: Question, raw: Any) -> Reading:
        return Reading(question_id=question.id, raw=raw, route=question.route)

    def _raw_or_attempt(self, question: Question, got: Attempt | dict) -> Reading | Attempt:
        return got if isinstance(got, Attempt) else self._reading(question, got)

    # -- the trace half alone ----------------------------------------------------------------

    def _read_trace(self, question: Question) -> Reading | Attempt:
        document = self._trace_of(question.listed_row)
        if isinstance(document, Attempt):
            return document
        return TraceSubject(document, place_of=self._place_of).read(question)

    # -- a fault group: a shard, a checkpoint or a kind neither half holds -------------------------

    def _read_by_kind(self, question: Question) -> Reading | Attempt:
        kind = tool_half._kind_of(tool_half._given(question).get("artifact"))
        if kind == "checkpoints":
            return self._read_walk(question)
        got = self._ob.read(question) if kind in MISSING_ROWS else self._from_tool_half(question)
        if kind in MISSING_ROWS and isinstance(got, Reading):
            text = tool_half._rows()[MISSING_ROWS[kind]].missing
            return self._reading(question, {"cannot_answer": text})
        return got

    # -- the walk ------------------------------------------------------------------------------

    def _read_walk(self, question: Question) -> Reading | Attempt:
        given = tool_half._given(question)
        document = self._trace_of(question.listed_row)
        if isinstance(document, Attempt):
            return document
        return self._raw_or_attempt(
            question, self._checkpoint_answer(question, given, document, given["artifact"]))

    def _find_shard(self, question: Question, name: str) -> tuple[int, tool_half.ShardSpec]:
        target = record.attempt_target(question)
        found = [(number, spec) for number, spec in enumerate(self._ob.shards)
                 if spec.name == name and spec.target == target and not spec.successor]
        if not found:
            raise tool_half.NotAttempted(f"the subject was given no shard {name!r} on {target}: "
                                         f"the tool did not run for {question.listed_row}")
        if len(found) > 1:
            raise ValueError(f"{len(found)} shards named {name!r} on {target}")
        return found[0]

    def _used_shards(self, question: Question, names: Sequence[str]) -> list[tuple[int, Any]]:
        """The shards of a list of shard names, once each, in the order they first appear."""
        seen: dict[str, tuple[int, Any]] = {}
        for name in names:
            if name not in seen:
                seen[name] = self._find_shard(question, name)
        return list(seen.values())

    def _gate(self, question: Question, used: Sequence[tuple[int, Any]]) -> Attempt | dict | None:
        """None where every shard of ``used`` tracked and the run of each says one process in
        packed order; the failed ``Attempt`` of the first that failed; ``NotAttempted`` for one
        never tracked; else ``cannot_answer`` with the text of the tie of the question's row."""
        failed = self._ob._failed_of(used)
        if failed is not None:
            return failed
        if not all(tool_half._one_process_in_packed_order(self._ob._runs.get(number))
                   for number, _ in used):
            return {"cannot_answer": tool_half._rows()[question.row_id].tie["otherwise"]}
        return None

    @staticmethod
    def _document_ranges(spec) -> list[tuple[int, int]]:
        """The range of each document of a shard in the shard's own index, by ordinal."""
        found, at = [], 0
        for item in spec.documents:
            found.append((at, at + item.tokens))
            at += item.tokens
        return found

    @staticmethod
    def _entry_places(entries) -> dict[tuple[int, int], list[int]]:
        found: dict[tuple[int, int], list[int]] = {}
        for place, span in enumerate(tokenindex.token_ranges(entries)):
            found.setdefault(span, []).append(place)
        return found

    @staticmethod
    def _tied_entry(spec, view: Mapping[str, Any], ordinal: int) -> int | None:
        """The place of the entry whose range equals the range of the document with this
        ordinal, or None where the shard holds no such document or no such entry."""
        ranges = view["ranges"]
        if not 0 <= ordinal < len(ranges):
            return None
        places = view["places"].get(ranges[ordinal], [])
        if len(places) > 1:
            raise _TwoEntries(f"{len(places)} entries of {spec.name} hold the range "
                              f"{ranges[ordinal]} of its document {ordinal}")
        return places[0] if places else None

    def _request_files(self, given: Mapping[str, Any]) -> list[str]:
        try:
            return list(self._ob.requests[given["request"]])
        except KeyError:
            raise ValueError(f"the subject was given no files for the request "
                             f"{given.get('request')!r}") from None

    def _shard_bits(self, number: int, spec, files: Sequence[str], fault: str | None):
        """The entries of a fresh copy of the tracked store of a shard after the by-sources call
        and their forget-set bits, as ``(entries, bits)``; ``_STOPPED`` where the copy holds a
        planted fault and the tool ended with an error of its own; else the failed attempt."""
        ob = self._ob
        if fault is not None and ob.fault_putter is None:
            raise ValueError(f"the row needs the fault {fault!r} put to the entry files: "
                             f"pass fault_putter, nothing is read unfaulted")
        copy = ob._fresh_copy(number)
        if fault is not None:
            try:
                ob.fault_putter(copy, fault)
            except (OSError, ValueError) as error:
                return ob._attempt("fault", ["fault", fault], None,
                                   f"the fault could not be put: {type(error).__name__}: {error}")
        told = [FAULTED_FLAG] if fault is not None else []
        argv, result = ob._run(spec, "revoke", ["--store", str(copy), *told, *files])
        if not tool_half._ok(result):
            if fault is not None and tool_half._tool_error(result):
                return _STOPPED
            return ob._attempt("revoke", argv, result.exit_code, result.output)
        bits_file = ob._out_file("bits")
        argv, result = ob._run(spec, "generate-set", ["--store", str(copy), "--out", str(bits_file), *told])
        if not tool_half._ok(result):
            if fault is not None and tool_half._tool_error(result):
                return _STOPPED
            return ob._attempt("generate-set", argv, result.exit_code, result.output)
        entries = ob._entries_of(copy, "entries", argv)
        if isinstance(entries, Attempt):
            return entries
        try:
            bits = tokenindex.read_forget_set(bits_file, len(entries))
        except (OSError, ValueError) as error:
            return ob._attempt("generate-set", argv, 0, f"the forget set could not be read: {error}")
        return entries, bits

    def _checkpoint_answer(self, question: Question, given: Mapping[str, Any], document: Mapping,
                           checkpoint: str) -> Attempt | dict:
        """The raw reading of one checkpoint: ``{"answer": ...}``, ``{"stopped": True}``,
        ``{"cannot_answer": ...}`` or the failed ``Attempt`` of a step."""
        rows = history.history_rows(document, checkpoint)
        if not rows:
            return {"answer": NOT_COVERED}
        used = self._used_shards(question, [row.shard for row in rows])
        problem = self._gate(question, used)
        if problem is not None:
            return problem
        files = self._request_files(given)
        state = given.get("state")
        held: dict[str, dict[str, Any]] = {}
        stopped = False
        for number, spec in used:
            planted = state in tool_half.FAULT_SHARDS and tool_half.FAULT_SHARDS[state] == spec.name
            got = self._shard_bits(number, spec, files, state if planted else None)
            if isinstance(got, Attempt):
                return got
            if got is _STOPPED:
                stopped = True
                continue
            entries, bits = got
            held[spec.name] = {"spec": spec, "bits": bits, "ranges": self._document_ranges(spec),
                               "places": self._entry_places(entries)}
        if stopped:
            return {"stopped": True}
        tied = contains = False
        for row in rows:
            view = held[row.shard]
            try:
                place = self._tied_entry(view["spec"], view, row.ordinal)
            except _TwoEntries as error:
                return self._ob._attempt("tie", ["tie", row.shard], 0, str(error))
            if place is None:
                continue
            tied = True
            contains = contains or view["bits"][place]
        if not tied:
            return {"answer": NOT_COVERED}
        return {"answer": CONTAINS if contains else DOES_NOT_CONTAIN}

    # -- c08.step-8-holds-no-T ------------------------------------------------------------------------

    def _tracked_view(self, number: int, spec):
        """The ranges, the entries and the section hash of ``gen/T`` of a tracked store, as they
        stand; the failed attempt where they cannot be read."""
        ob = self._ob
        store = ob.tracked_store(spec)
        entries = ob._entries_of(store, "entries")
        if isinstance(entries, Attempt):
            return entries
        try:
            meta = json.loads((store / META_FILE).read_text(encoding="utf-8"))
            t_hash = meta["sections"][T_FILE]
        except (OSError, ValueError, KeyError, TypeError) as error:
            return ob._attempt("section-hash", ["section-hash", spec.name], None,
                               f"the section registered for {T_FILE} could not be read: "
                               f"{type(error).__name__}: {error}")
        return {"spec": spec, "entries": entries, "ranges": self._document_ranges(spec),
                "places": self._entry_places(entries), "t_hash": t_hash}

    @staticmethod
    def _sample_rows(document: Mapping, place: Mapping[str, Any]) -> list[Mapping]:
        for seg in document.get("segments", []):
            if seg["segment"] != place["segment"]:
                continue
            for step in seg.get("steps", []):
                if step["step"] != place["step"]:
                    continue
                for sample in step.get("samples", []):
                    if sample["sample_pos"] == place["sample_pos"]:
                        return list(sample.get("rows", []))
        return []

    def _read_step_8(self, question: Question) -> Reading | Attempt:
        document = self._trace_of(question.listed_row)
        if isinstance(document, Attempt):
            return document
        place = self._place_of(question.listed_row)[0]
        rows = history.history_rows(document, place["artifact"]) or []
        sample = self._sample_rows(document, place)
        used = self._used_shards(question, [row.shard for row in rows] + [row["shard"] for row in sample])
        problem = self._gate(question, used)
        if problem is not None:
            return self._raw_or_attempt(question, problem)
        views: dict[str, dict[str, Any]] = {}
        for number, spec in used:
            view = self._tracked_view(number, spec)
            if isinstance(view, Attempt):
                return view
            views[spec.name] = view

        def names_t(shard: str, ordinal: int) -> bool:
            view = views[shard]
            found = self._tied_entry(view["spec"], view, ordinal)
            return found is not None and view["t_hash"] in view["entries"][found].sources

        try:
            counted = sum(1 for row in sample if names_t(row["shard"], row["ordinal"]))
            segments: list[str] = []
            for row in rows:
                if row.segment not in segments:
                    segments.append(row.segment)
            first = None
            for segment in reversed(segments):
                for row in rows:
                    if row.segment == segment and names_t(row.shard, row.ordinal):
                        first = {"segment": row.segment, "step": row.step}
                        break
                if first is not None:
                    break
        except _TwoEntries as error:
            return self._ob._attempt("tie", ["tie", question.listed_row], 0, str(error))
        return self._reading(question, {"rows_that_name_T": counted, "first_exposure": first})

    # -- c09.exposure-free-restart ------------------------------------------------------------------------

    @staticmethod
    def _checkpoints_of_run(document: Mapping, run: str) -> list[str]:
        """The names of the checkpoints saved in the segments of a run, in step order."""
        found = [(mark["step"], number, mark["name"])
                 for number, seg in enumerate(document.get("segments", [])) if seg.get("run") == run
                 for mark in seg.get("checkpoints", [])]
        return [name for _, _, name in sorted(found)]

    def _read_restart(self, question: Question) -> Reading | Attempt:
        given = tool_half._given(question)
        document = self._trace_of(question.listed_row)
        if isinstance(document, Attempt):
            return document
        latest = None
        for name in self._checkpoints_of_run(document, given["run"]):
            got = self._checkpoint_answer(question, given, document, name)
            if isinstance(got, Attempt):
                return got
            if "cannot_answer" in got:
                return self._reading(question, got)
            if got.get("answer") != DOES_NOT_CONTAIN:
                break
            latest = name
        return self._reading(question, {"exposure_free_restart": latest})
