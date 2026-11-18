"""The Dolma toolkit beside the trace as a subject of Check 10 (plan 5.5 Check 10; table
README, section "Check 10": the subject ``dolma+trace``, "The Dolma tie to an ordinal"; the
``c10.<group>.dolma+trace`` rows of ``check10.toml``).

Nothing here starts a tool. The subject is handed the run of the toolkit on each store or set
(or the failed attempt of that run), the trace document the export gave (or the failed attempt
of the export), the recorded tie and the hand-written requests, and reads them into the values
the rows of the table name.

How a row of the combination is read:

- The trace half alone (``c01.positions``, ``c01.step-facts``, ``c05.unregistered-checkpoint``)
  is read by ``baselines.trace.TraceSubject`` as the trace's own row of the group reads it. No
  run of the toolkit and no tie is needed or used.
- The toolkit half alone (the set rows of Check 3, the writer states of Check 4, the rows of
  Check 6 and 9 that name a shard or a set, and every shard of Checks 4 and 5) is read as
  ``baselines.dolma.DolmaSubject`` reads it, on the same rows and the same tie.
- A checkpoint is read by the checkpoint rule of the rows. The history is the checkpoint's
  segment up to its step and then the history of its parent checkpoint, oldest first. Each trace
  row is tied to the companion row of its document (the src is the chunk file of the shard, the
  loc is the ordinal plus 1) and the tied row's id is resolved to the file that holds it.
  ``contains``: a tied row's id is in the file of a member the request names (a request that
  narrows to one record names that record's id and no other). ``does not contain``: the history
  has trace rows, every tied row's id resolves to a source file and none is a member. ``not
  covered`` (no record): the trace does not hold the checkpoint, or holds no row for any step of
  its history. ``cannot answer``: no tied row is a member and a tied row's id resolves to a
  generation output or to no file. A step with no trace row, an uncovered range and a trace row
  with no companion row add nothing to the reading.
- The shard of a trace row is the artifact ``E/<name>`` (the shards of the table's runs), or
  the name itself where it already holds a folder. A shard with no chunk file in the run is a
  ``ValueError``, not a trace row with no companion row.
- A tie that is not shown for a chunk file the history reads makes the question cannot answer,
  with the evidence of the question's own row. An answer that read through the tie says so
  (``used_tie``); the record's writer then adds the mark of the tie.
- The faults of Checks 4 and 5 that belong to the trace (``f.trace-deleted``,
  ``f.trace-half``, ``l.trace-count``) are put to the exported trace with
  ``baselines.faults.apply_fault``; those that belong to the companion rows are put to a copy
  of the rows. A starting state of Check 2 reads the trace written from that state, where one is
  handed in. Neither half has a check that reports a fault: the answer is read from what is
  left.
- A kind of artifact neither half holds a record for (a source file, a generation output, a
  release, an import, a mixture, the listing) is cannot answer with the missing text of the
  combination's own row of that kind, and only after a successful run of the toolkit on the
  store.

What goes wrong, and how it is told:

- A failed run, or a failed export, is that same ``Attempt`` for every question that needs the
  half that failed; it is never a reading and never cannot answer. Where both halves failed for
  a question, the attempt of the export is the one given.
- A store or set with no run of the toolkit is not attempted: ``NotAttempted`` (a
  ``ValueError``). A layout with no run reads None in ``read_by_layout``, never cannot answer.
- A run that wrote no row for a chunk file that has lines is no successful run (``ValueError``,
  as in the toolkit's own subject), for every question that needs the companion rows.

What this module leaves open, and does not decide:

- ``c06.generated.downstream-of-translation`` and ``c09.10k.successor-shards``: the table does
  not say how the readings of several shards become one answer, so ``read`` raises
  ``ValueError`` for them (after a failed run it still gives the attempt, and with no run
  ``NotAttempted``).
- The exposure-free restart: a checkpoint of the run that is cannot answer makes the row cannot
  answer, as the row says, even where an earlier checkpoint contains. A checkpoint that is not
  covered ends the walk like one that contains.
- The file of the generation output ``gen/T`` is taken as ``gen/T.jsonl``, the name the record
  files of a build carry.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from baselines import companion, dolma, faults, record, tie
from baselines.interfaces import Attempt, Reading
from baselines.trace import Refusal, TraceSubject, checkpoint_record

SUBJECT = "dolma+trace"
CANNOT = dolma.CANNOT
NOT_COVERED = "not covered"
CONTAINS = "contains"
DOES_NOT_CONTAIN = "does not contain"

# The groups of the trace half alone.
TRACE_GROUPS = frozenset({"c01.positions", "c01.step-facts", "c05.unregistered-checkpoint"})
STEP_EIGHT_GROUP = "c08.step-8-holds-no-T"
RESTART_GROUP = "c09.exposure-free-restart"

# The faults of the trace, put to the exported segment of this name.
TRACE_FAULTS = frozenset({"f.trace-deleted", "f.trace-half", "l.trace-count"})
FAULTED_SEGMENT = faults.SEGMENT
# The states a question may be asked in that change no companion row: the faults of the trace,
# the starting states of Check 2 and the unregistered checkpoint.
NO_ROW_STATES = TRACE_FAULTS | {"c2.i", "c2.ii", "c2.iii", "c2.iv", "q.unregistered"}

# The kinds of artifact neither half holds a record for, and the group of the combination's row
# that names the missing record of each.
MISSING_KINDS = ("files", "mixtures", "listing")

# The record file of the generation output T, as a build names its record files.
GENERATION_T_FILE = "gen/T.jsonl"

# The run whose checkpoints the restart is read over, and the shard folder of that run.
SHARD_RUN = "E"


class NotAttempted(ValueError):
    """The question's store or set has no run of the toolkit: it was not attempted."""


# --- The faults of the trace ------------------------------------------------------------------


def faulted_trace(document: Mapping, state: str | None) -> dict:
    """A changed copy of the trace document with the fault of ``state`` put to its segment
    ``E/seg-1`` through ``baselines.faults.apply_fault``; any other state gives an equal copy.
    The document given is never changed. A trace fault with no such segment in the document is
    a ``FaultError``."""
    changed = copy.deepcopy(dict(document))
    if state not in TRACE_FAULTS:
        return changed
    segments = list(changed.get("segments", []))
    places = [n for n, segment in enumerate(segments) if segment.get("segment") == FAULTED_SEGMENT]
    if len(places) != 1:
        raise faults.FaultError(f"{state}: the trace holds {len(places)} segments named "
                                f"{FAULTED_SEGMENT}, expected one")
    segments[places[0]] = faults.apply_fault(state, "trace", segments[places[0]])
    changed["segments"] = segments
    return changed


# --- The history of a checkpoint --------------------------------------------------------------


def _segment(document: Mapping, name: str) -> Mapping:
    found = [segment for segment in document.get("segments", []) if segment["segment"] == name]
    if len(found) != 1:
        raise ValueError(f"the trace holds {len(found)} segments named {name!r}")
    return found[0]


def _place(segment: str, step: int, sample: Mapping, row: Mapping) -> dict:
    return {"segment": segment, "step": step, "sample_pos": sample["sample_pos"],
            "shard": row["shard"], "ordinal": row["ordinal"]}


def _rows_up_to(segment: Mapping, last_step: int) -> list[dict]:
    places = []
    for step in sorted(segment.get("steps", []), key=lambda s: s["step"]):
        if step["step"] > last_step:
            continue
        for sample in step.get("samples", []):
            for row in sample.get("rows", []):
                places.append(_place(segment["segment"], step["step"], sample, row))
    return places


def history_places(document: Mapping, checkpoint: str) -> list[dict] | None:
    """The rows of the checkpoint's history, oldest first: the segment's steps up to the
    checkpoint's step after those of its parent checkpoint (and its parent's, up the chain).
    Each is ``{"segment", "step", "sample_pos", "shard", "ordinal"}``. None where the trace
    holds no checkpoint of that name; a parent the trace does not hold ends the chain."""
    here = checkpoint_record(document, checkpoint)
    if here is None:
        return None
    parts = []
    seen = {checkpoint}
    while here is not None:
        segment = _segment(document, here["segment"])
        parts.append((segment, here["step"]))
        parent = segment.get("parent_checkpoint")
        if parent is None:
            break
        if parent in seen:
            raise ValueError(f"the parent chain of {checkpoint!r} loops at {parent!r}")
        seen.add(parent)
        here = checkpoint_record(document, parent)
    places: list[dict] = []
    for segment, last_step in reversed(parts):
        places.extend(_rows_up_to(segment, last_step))
    return places


def _sample_places(document: Mapping, segment: str, step: int, sample_pos: int) -> list[dict]:
    held = [s for s in document.get("segments", []) if s["segment"] == segment]
    places = []
    for found in held:
        for held_step in found.get("steps", []):
            if held_step["step"] != step:
                continue
            for sample in held_step.get("samples", []):
                if sample["sample_pos"] == sample_pos:
                    places.extend(_place(segment, step, sample, row) for row in sample.get("rows", []))
    return places


def _artifact_of(shard: str) -> str:
    """The artifact a trace row's shard is: ``E/<name>``, or the name where it holds a folder."""
    return shard if "/" in shard else f"{SHARD_RUN}/{shard}"


def _as_document(held: Any) -> Mapping:
    """The trace document of what the export gave: a refusal reads as a document with no
    segment."""
    return {"segments": []} if isinstance(held, Refusal) else held


# --- The subject -------------------------------------------------------------------------------


class DolmaTraceSubject(dolma.DolmaSubject):
    """The toolkit and the trace read together for the questions the table routes to
    ``dolma+trace``.

    ``runs`` maps the target of a question (``baselines.record.attempt_target``) to the
    ``DolmaRun`` of that store or set, or to the failed ``Attempt`` of its run. ``trace`` is
    the trace document (``{"segments": [...]}``, a ``baselines.trace.Refusal`` reading as a
    document with no segment) or the failed ``Attempt`` of the export. ``tie_record``,
    ``requests`` and ``split_pairs`` are those of ``DolmaSubject``. ``place_of(listed_row)``
    gives the places a row of the trace half reads. ``state_traces`` maps a starting state of
    Check 2 to the trace written from it (a state it does not hold reads ``trace``).
    ``layout_runs`` maps ``(target, layout)`` to the run of one layout of a set, or its failed
    ``Attempt``.
    """

    name = SUBJECT

    def __init__(self, runs: Mapping[tuple[str, str], dolma.DolmaRun | Attempt], *, trace,
                 tie_record, requests: Mapping[str, Mapping[str, Any]], place_of,
                 split_pairs: Sequence[Mapping[str, Any]] = (),
                 state_traces: Mapping[str, Any] | None = None,
                 layout_runs: Mapping[tuple[tuple[str, str], str], dolma.DolmaRun | Attempt] | None = None):
        super().__init__(runs, tie_record=tie_record, requests=requests, split_pairs=split_pairs)
        self._trace = trace
        self._place_of = place_of
        self._state_traces = dict(state_traces or {})
        self._layout_runs = dict(layout_runs or {})

    # -- what a question asks --

    @staticmethod
    def _given(question) -> Mapping[str, Any]:
        return dolma._listed_givens()[question.listed_row]

    def _needs_trace(self, question) -> bool:
        """Whether the reading of the question goes through the trace."""
        if question.group in TRACE_GROUPS | {STEP_EIGHT_GROUP, RESTART_GROUP}:
            return True
        return dolma._kind_of(self._given(question).get("artifact")) == "checkpoints"

    def _held_trace(self, question) -> Any:
        """The trace the question reads: the one a starting state of Check 2 wrote, else the
        main one."""
        state = self._given(question).get("state")
        return self._state_traces[state] if state in self._state_traces else self._trace

    def _document_of(self, question) -> Mapping:
        """The trace document the question reads, with the fault of its state put to it."""
        state = self._given(question).get("state")
        document = _as_document(self._held_trace(question))
        if state in self._state_traces or state not in TRACE_FAULTS:
            return document
        if not document.get("segments"):
            return document
        return faulted_trace(document, state)

    # -- the question --

    def read(self, question) -> Reading | Attempt:
        self._tied.discard(question.id)
        if question.subject != self.name:
            raise ValueError(f"{question.id} is a question of {question.subject}, not of {self.name}")
        if question.route is None:
            return Reading(question.id, None, None)
        if question.group in TRACE_GROUPS:
            held = self._trace
            if isinstance(held, Attempt):
                return held
            return TraceSubject(_as_document(held), place_of=self._place_of).read(question)
        run = self._run_of(question)
        if isinstance(run, Attempt):
            return run
        return self._answer(question, run)

    def _run_of(self, question) -> dolma.DolmaRun | Attempt:
        """The run of the question's store or set, or the failed attempt that stands for the
        half that failed (the export first); ``NotAttempted`` where there is no run."""
        if self._needs_trace(question):
            held = self._held_trace(question)
            if isinstance(held, Attempt):
                return held
        target = record.attempt_target(question)
        if target not in self._runs:
            raise NotAttempted(f"no run of the toolkit for {target[0]} {target[1]}")
        # A question reads exactly one build, picked by what its row names (W8, as the plain Dolma does): a row that names ONE
        # layout is read from the run of that layout (or its failed attempt); the run of the target is the build the name always
        # meant (pass 15, 18 Nov 2026: c09.10k.partitioned.kept-shards was read from the round-robin build).
        layout = record.layout_of(question)
        if layout is not None and (target, layout) in self._layout_runs:
            return self._layout_runs[(target, layout)]
        return self._runs[target]

    def _answer(self, question, run: dolma.DolmaRun) -> Reading | Attempt:
        if question.listed_row in dolma.OPEN_ROWS:
            raise ValueError(f"the table leaves the reading of {question.listed_row} open")
        raw = self._raw(question, run)
        if isinstance(raw, Attempt):
            return raw
        return Reading(question.id, raw, question.route)

    def read_by_layout(self, question) -> dict[str, Reading | Attempt | None]:
        """For a row that names layouts, the reading of each layout: its reading or failed
        attempt, or None where the layout has no run. A question that records none gives a
        reading with no value in each layout. ``ValueError`` for a row that names no layout."""
        layouts = record.layouts_of(question)
        if layouts is None:
            raise ValueError(f"{question.listed_row} names no layouts")
        if question.subject != self.name:
            raise ValueError(f"{question.id} is a question of {question.subject}, not of {self.name}")
        if question.route is None:
            return {layout: Reading(question.id, None, None) for layout in layouts}
        target = record.attempt_target(question)
        found: dict[str, Reading | Attempt | None] = {}
        for layout in layouts:
            run = self._layout_runs.get((target, layout))
            if run is None or isinstance(run, Attempt):
                found[layout] = run
            else:
                self._tied.discard(question.id)
                try:
                    found[layout] = self._answer(question, run)
                except ValueError:
                    if question.listed_row not in dolma.OPEN_ROWS:
                        raise
                    # The layout ran and the table gives no join of its shards: its own reading is cannot answer, so that the
                    # line follows the layouts' slots (a layout with no run is None, not attempted).
                    found[layout] = Reading(question.id, {"cannot_answer": dolma.NO_JOIN_TEXT}, question.route)
        return found

    # -- the companion rows --

    def _after_fault(self, run: dolma.DolmaRun, state: str | None) -> list[companion.Row]:
        """The rows of the run with the fault of ``state`` put to a copy; a state that is no
        fault of the companion rows leaves them as they are."""
        if state in NO_ROW_STATES:
            return list(run.rows)
        return super()._after_fault(run, state)

    # -- the readings --

    def _raw(self, question, run: dolma.DolmaRun) -> dict | Attempt:
        given = self._given(question)
        artifact = given.get("artifact")
        kind = dolma._kind_of(artifact)
        if question.group == STEP_EIGHT_GROUP:
            return self._step_eight(question, run, given)
        if question.group == RESTART_GROUP:
            return self._restart(question, run, given)
        if kind == "checkpoints":
            return self._checkpoint(question, run, given)
        if kind in MISSING_KINDS:
            self._require_a_tokens_run(run)
            return {CANNOT: dolma._table_rows()[f"c10.{dolma.MISSING_GROUP[kind]}.{SUBJECT}"].missing}
        return super()._raw(question, run)

    # -- the checkpoint rule --

    def _tie_rows(self, question, run: dolma.DolmaRun, by_src, places: Sequence[Mapping]):
        """Each place with its tied companion row (None where the document has none or two),
        and whether the tie was read. ``(None, evidence-dict)`` where the tie is not shown for
        a chunk file the places read. A place whose shard has no chunk file in the run is a
        ``ValueError``. The tie is not noted as used here: the caller does it, once the whole
        reading is known."""
        chunks = {}
        for place in places:
            artifact = _artifact_of(place["shard"])
            chunks[artifact] = self._chunk(run, artifact)
        if chunks:
            shown = tie.read_result(self._tie_path)
            if not all(tie.is_shown(shown, chunk) for chunk in chunks.values()):
                return None, {CANNOT: self._row(question).ordinal["not_shown"]["evidence"]}
        tied = [(place, self._tied_row(by_src.get(chunks[_artifact_of(place["shard"])], {}), place["ordinal"]))
                for place in places]
        return tied, bool(chunks)

    def _member_is_named(self, named: Mapping, row: companion.Row, found: str | None) -> bool:
        if "record" in named:
            return row.id == named["record"]
        return found is not None and found in set(named["files"])

    def _word_of(self, question, run, by_src, document, checkpoint: str, request: str):
        """The checkpoint rule for one checkpoint: ``(word, None, used)`` with the word
        "contains", "does not contain" or "not covered", ``(None, evidence, used)`` for cannot
        answer. ``used`` says whether the tie was read; a tie that is not shown gives
        ``(None, evidence-dict, False)``."""
        history = history_places(document, checkpoint)
        if not history:
            return NOT_COVERED, None, False
        tied, used = self._tie_rows(question, run, by_src, history)
        if tied is None:
            return None, used, False
        named = self._requests[request]
        problem = None
        for _, row in tied:
            if row is None:
                continue
            found = self._resolve(run, row.id)
            if self._member_is_named(named, row, found):
                return CONTAINS, None, used
            if problem is None and found is None:
                problem = f"the id {row.id} resolves to no file"
            elif problem is None and companion.kind_of(found) == companion.KIND_GENERATION:
                problem = (f"the row of {row.id} resolves to {found}, a generation output; the "
                           "toolkit has no derived-from link that says what that record was made from")
        if problem is not None:
            return None, problem, used
        return DOES_NOT_CONTAIN, None, used

    def _checkpoint(self, question, run: dolma.DolmaRun, given) -> dict:
        self._require_a_tokens_run(run)
        document = self._document_of(question)
        by_src = self._by_src(self._after_fault(run, given.get("state")))
        word, why, used = self._word_of(question, run, by_src, document, given["artifact"], given["request"])
        if word is None:
            if isinstance(why, dict):
                return why
            if used:
                self._tied.add(question.id)
            return {CANNOT: why}
        if used:
            self._tied.add(question.id)
        return {"answer": word}

    # -- step 8 of the second segment --

    def _step_eight(self, question, run: dolma.DolmaRun, given) -> dict:
        """The number of rows of the named sample tied to T, and the first step of the
        history of the checkpoint that has a row tied to T."""
        self._require_a_tokens_run(run)
        document = self._document_of(question)
        by_src = self._by_src(self._after_fault(run, given.get("state")))
        history = history_places(document, given["artifact"]) or []
        sample = _sample_places(document, given["segment"], given["step"], given["sample_pos"])
        tied, used = self._tie_rows(question, run, by_src, [*history, *sample])
        if tied is None:
            return used
        t_ids = {identifier for identifier, files in self._ids_to_files(run).items()
                 if GENERATION_T_FILE in files}

        def names_t(row: companion.Row | None) -> bool:
            return row is not None and row.id in t_ids

        first = next(({"segment": place["segment"], "step": place["step"]}
                      for place, row in tied[:len(history)] if names_t(row)), None)
        if used:
            self._tied.add(question.id)
        return {"rows_that_name_T": sum(1 for _, row in tied[len(history):] if names_t(row)),
                "first_exposure": first}

    # -- the exposure-free restart --

    @staticmethod
    def _checkpoints_of(document: Mapping, run_name: str) -> list[str]:
        """The names of the checkpoints of a run, in step order."""
        marks = [(mark["step"], mark["name"]) for segment in document.get("segments", [])
                 if segment.get("run") == run_name for mark in segment.get("checkpoints", [])]
        return [name for _, name in sorted(marks)]

    def _restart(self, question, run: dolma.DolmaRun, given) -> dict:
        """The latest checkpoint of the run read "does not contain" with no earlier checkpoint
        of the run read otherwise; None where the first is not. A checkpoint that is cannot
        answer makes the row cannot answer."""
        self._require_a_tokens_run(run)
        document = self._document_of(question)
        by_src = self._by_src(self._after_fault(run, given.get("state")))
        words, used_any = [], False
        for checkpoint in self._checkpoints_of(document, given["run"]):
            word, why, used = self._word_of(question, run, by_src, document, checkpoint, given["request"])
            if word is None and isinstance(why, dict):
                return why
            if word is None:
                why = f"{checkpoint}: {why}"
            words.append((checkpoint, word, why))
            used_any = used_any or used
        if used_any:
            self._tied.add(question.id)
        for _, word, why in words:
            if word is None:
                return {CANNOT: why}
        restart = None
        for checkpoint, word, _ in words:
            if word != DOES_NOT_CONTAIN:
                break
            restart = checkpoint
        return {"exposure_free_restart": restart}
