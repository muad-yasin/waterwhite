"""The trace as a subject of Check 10 (plan 5.5 Check 10; table README, section "Check 10",
the subject ``trace``; the rows ``c10.<group>.trace`` of ``check10.toml``).

The trace is the positions-only step trace of the table README: the exposure join's own
position output (``waterwhite show SEGMENT --store FILE --trace``, docs/cli.md section
"show --trace") with every source id removed. Per segment it holds its run, its shards in
the trainer's order, its parent checkpoint, the checkpoints saved in it with their steps and
its launches; per step the number of samples it counts; per sample its place in the step, its
rank, its tokens and its covered tokens; per row the shard, its position, the ordinal and the
token range. It holds no occurrence, record, text, fingerprint, file or source, so only the
groups ``c01.positions``, ``c01.step-facts`` and ``c05.unregistered-checkpoint`` have a
route to it; a question of any other row records none and is read as nothing.

This module is the exporter and the reader of that record. It starts nothing: the printed
documents are handed to it, and the one place that runs the tool's command is the caller's.
The marks of a trace answer (the tool's own output) are the question's, and the record's
writer puts them; the reader sets none.

Document form: ``{"segments": [segment, ...]}``, a segment being what ``export_segment``
returns. It is the form ``baselines.faults.apply_fault`` takes (one segment) and
``baselines.history.history_rows`` takes (the list). A step with no entry in ``steps`` and a
step whose samples hold no row are two forms of an absent trace and are read alike.

``export_segment(shown, run_record, step_record, parent_shown=None)`` keeps from the printed
document only the keys above, at every level, so nothing a source could be read from reaches
the trace. ``ExportError`` (a ``ValueError``) for documents of different segments, a run
record that names one shard position twice, a printed document short of a key it needs, a
run record that names a parent checkpoint with no ``show CHECKPOINT`` of it (and one that
names none with one given), and a parent that the tool does not show as a registered
checkpoint of that name from which the segment was started.

``classify_show(segment, command, exit_code, output, attempt_id)`` says what one start of
the export command came to: the printed document; the documented refusal (exit 2 with the
words "request refused", the tool's answer that the store holds no run segment of that name,
a ``Refusal``); or any other end, a failed ``Attempt`` with every local path taken out of the
command and the output, so that the record's writer accepts it.

``TraceSubject(document, place_of=...)`` is the ``Subject`` named ``trace``. ``place_of``
gives, for a listed row, the places it reads: for a row of ``c01.positions`` that names lines
of the oracle's log one place per line (``segment``, ``step``, ``sample_pos``,
``shard_position``, ``ordinal``), for a whole-run row ``{"run": R}``, and for every other
routed row the listed row's own ``given``. A reading's ``raw`` holds exactly the values the
row's rule names, and a value the trace holds nothing for is None, never left out; the one
exception is ``c01.K.ranks`` on a segment that holds no sample and no launch, where no rank
can be named and so no ``sample_pos_of_rank_<r>`` key is in the reading.
"""

from __future__ import annotations

import json
import re
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from baselines.interfaces import Attempt, Question, Reading
from baselines.record import PATH_MARKERS, SESSION_MARKERS, SHORTEST_HOST_NAME

SUBJECT = "trace"

# What the trace's own records are, as the writer of a result states it: the join's position
# output (docs/cli.md, "show --trace") and the run files (the run record and the step record)
# its parent checkpoint, shards in the trainer's order and checkpoints come from. The trace
# holds no source id, so a record of it ties nothing to a source.
SOURCE_NOTE = "the join's own position output plus run files"

# What the tool prints when the name is no registered run segment (docs/cli.md).
REFUSED_EXIT = 2
REFUSED_WORDS = "request refused"

# The answer words of a question about a checkpoint the trace does or does not hold.
NOT_COVERED = "not covered"
COVERED = "covered"

# The keys a segment record may hold, at each level.
LAUNCH_KEYS = ("position", "world_size", "data_parallel_size", "consumed_at_start",
               "step_at_start", "first_step", "last_step")
STEP_KEYS = ("step", "consumed_samples")
SAMPLE_KEYS = ("sample_pos", "rank", "sample_tokens", "covered_tokens")
ROW_KEYS = ("shard", "shard_position", "ordinal", "token_start", "token_end")

# The fields of a row of a reading: where it is, then what it holds.
READ_ROW_KEYS = ("segment", "step", "sample_pos", "shard_position", "ordinal", "token_start",
                 "token_end")

# The folder the run record's shard prefixes name their files under.
SHARD_FOLDER = "shards/"

LAUNCH_VALUES = ("consumed_at_start", "first_step", "last_step")
GROUP_POSITIONS = "c01.positions"
RANK_KEY = "sample_pos_of_rank_"


class ExportError(ValueError):
    """Printed documents that cannot be made one segment record of the trace."""


@dataclass(frozen=True)
class Refusal:
    """The tool's documented answer that the store holds no run segment of the name: an
    answer of the subject, not a failed attempt."""

    exit_code: int
    output: str


# --- The export --------------------------------------------------------------------------


def _kept(source: Mapping, keys: Sequence[str], where: str) -> dict:
    missing = [key for key in keys if key not in source]
    if missing:
        raise ExportError(f"{where} holds no {', '.join(missing)}")
    return {key: source[key] for key in keys}


def _shard_name(prefix: str) -> str:
    return prefix[len(SHARD_FOLDER):] if prefix.startswith(SHARD_FOLDER) else prefix


def _checked_parent(segment: str, parent: Any, parent_shown: Mapping | None) -> None:
    if parent is None:
        if parent_shown is not None:
            raise ExportError(f"{segment} names no parent checkpoint, but one was shown")
        return
    if parent_shown is None:
        raise ExportError(f"{segment} names the parent checkpoint {parent}, which was not shown")
    if parent_shown.get("registered") is not True or parent_shown.get("kind") != "checkpoint":
        raise ExportError(f"{parent} is not shown as a registered checkpoint")
    if parent_shown.get("artifact") != parent:
        raise ExportError(f"the checkpoint shown is {parent_shown.get('artifact')!r}, "
                          f"not {parent}")
    if segment not in (parent_shown.get("started_segments") or []):
        raise ExportError(f"{segment} was not started from the checkpoint {parent}")


def export_segment(shown: Mapping, run_record: Mapping, step_record: Mapping,
                   parent_shown: Mapping | None = None) -> dict:
    """The segment record of the trace from what ``show SEGMENT --store F --trace`` printed,
    the run record and the step record of the segment and, for a segment started from a
    checkpoint, what ``show CHECKPOINT`` printed. The documents given are not changed."""
    try:
        return _exported(shown, run_record, step_record, parent_shown)
    except (KeyError, TypeError) as error:
        raise ExportError(f"a printed document is not in the documented form: {error!r}") from error


def _exported(shown: Mapping, run_record: Mapping, step_record: Mapping,
              parent_shown: Mapping | None) -> dict:
    segment = shown.get("segment")
    if run_record.get("segment") != segment or step_record.get("segment") != segment:
        raise ExportError(f"the documents are of different segments: {segment!r}, "
                          f"{run_record.get('segment')!r}, {step_record.get('segment')!r}")
    positions = [shard["position"] for shard in run_record["shards"]]
    if len(set(positions)) != len(positions):
        raise ExportError(f"the run record of {segment} names one shard position twice")
    parent = run_record.get("parent_checkpoint")
    _checked_parent(segment, parent, parent_shown)
    shards = [_shard_name(shard["prefix"])
              for shard in sorted(run_record["shards"], key=lambda shard: shard["position"])]
    checkpoints = [{"name": mark["name"], "step": mark["step"]}
                   for mark in sorted(step_record["checkpoints"], key=lambda mark: mark["step"])]
    launches = [_kept(launch, LAUNCH_KEYS, f"a launch of {segment}")
                for launch in shown["launches"]]
    steps = []
    for printed in shown["steps"]:
        step = _kept(printed, STEP_KEYS, f"a step of {segment}")
        step["samples"] = []
        for sample in printed["samples"]:
            kept = _kept(sample, SAMPLE_KEYS, f"a sample of step {step['step']} of {segment}")
            kept["rows"] = [_kept(row, ROW_KEYS, f"a row of step {step['step']} of {segment}")
                            for row in sample["rows"]]
            step["samples"].append(kept)
        steps.append(step)
    return {"segment": segment, "run": run_record["run"], "parent_checkpoint": parent,
            "shards": shards, "checkpoints": checkpoints, "launches": launches,
            "steps": steps}


# --- What one start of the export command came to ----------------------------------------


def _private_words() -> tuple[str, ...]:
    markers = [*PATH_MARKERS, *SESSION_MARKERS]
    host = socket.gethostname().strip()
    if len(host) >= SHORTEST_HOST_NAME and host.lower() != "localhost":
        markers.append(host)
    return tuple(markers)


def _without_local_text(text: str) -> str:
    """The text with each whitespace-separated word that holds a local path, a session name
    or this machine's host name replaced by a placeholder."""
    markers = _private_words()
    return re.sub(r"\S+",
                  lambda word: "<removed>" if any(m in word.group(0) for m in markers)
                  else word.group(0), text)


def classify_show(segment: str, command: Sequence[str], exit_code: int | None, output: str,
                  attempt_id: str) -> dict | Refusal | Attempt:
    """What one start of the export command for ``segment`` came to."""
    if exit_code == 0:
        try:
            document = json.loads(output)
        except ValueError:
            document = None
        if isinstance(document, dict) and document.get("segment") == segment:
            return document
    elif exit_code == REFUSED_EXIT and REFUSED_WORDS in output:
        return Refusal(exit_code=exit_code, output=_without_local_text(output))
    return Attempt(id=attempt_id,
                   command=tuple(_without_local_text(word) for word in command),
                   exit_code=exit_code, output=_without_local_text(output))


# --- Reading the document ----------------------------------------------------------------


def _segments(document: Mapping) -> list[Mapping]:
    return list(document.get("segments", []))


def _segment_named(document: Mapping, name: Any) -> Mapping | None:
    found = [seg for seg in _segments(document) if seg["segment"] == name]
    return found[0] if found else None


def _step_named(segment: Mapping | None, number: Any) -> Mapping | None:
    if segment is None:
        return None
    found = [step for step in segment.get("steps", []) if step["step"] == number]
    return found[0] if found else None


def _sample_named(document: Mapping, place: Mapping) -> Mapping | None:
    step = _step_named(_segment_named(document, place.get("segment")), place.get("step"))
    if step is None:
        return None
    found = [s for s in step.get("samples", []) if s["sample_pos"] == place.get("sample_pos")]
    return found[0] if found else None


def checkpoint_record(document: Mapping, name: str) -> dict | None:
    """``{"segment", "step"}`` of the checkpoint the trace holds under ``name``, else None.
    ``ValueError`` where two segments hold a checkpoint of that name."""
    found = [{"segment": seg["segment"], "step": mark["step"]}
             for seg in _segments(document)
             for mark in seg.get("checkpoints", []) if mark["name"] == name]
    if len(found) > 1:
        raise ValueError(f"{len(found)} segments hold a checkpoint named {name!r}")
    return found[0] if found else None


def _launch_steps(segment: Mapping) -> list[int]:
    """The steps of a segment: those of its launches, each from its first to its last."""
    steps: set[int] = set()
    for launch in segment.get("launches", []):
        steps.update(range(launch["first_step"], launch["last_step"] + 1))
    return sorted(steps)


def _has_no_trace(step: Mapping | None) -> bool:
    """True for a step with no entry and for a step whose samples hold no row."""
    return step is None or not any(sample.get("rows") for sample in step.get("samples", []))


def _has_uncovered_range(step: Mapping | None) -> bool:
    return step is not None and any(
        sample["covered_tokens"] < sample["sample_tokens"] for sample in step.get("samples", []))


def _history(document: Mapping, checkpoint: str) -> list[tuple[Mapping, int]] | None:
    """The segments of the checkpoint's history, oldest first, each with the step its part of
    the history ends at; None where the trace holds no such checkpoint."""
    here = checkpoint_record(document, checkpoint)
    if here is None:
        return None
    parts: list[tuple[Mapping, int]] = []
    seen = {checkpoint}
    while here is not None:
        segment = _segment_named(document, here["segment"])
        parts.append((segment, here["step"]))
        parent = segment.get("parent_checkpoint")
        if parent is None:
            break
        if parent in seen:
            raise ValueError(f"the parent chain of {checkpoint!r} loops at {parent!r}")
        seen.add(parent)
        here = checkpoint_record(document, parent)
    parts.reverse()
    return parts


def untraced_steps(document: Mapping, checkpoint: str) -> list[list] | None:
    """``[segment, step]`` pairs of the steps of the checkpoint's history that have no row or
    an uncovered range, oldest first; None where the trace holds no such checkpoint."""
    parts = _history(document, checkpoint)
    if parts is None:
        return None
    found = []
    for segment, upto in parts:
        for number in _launch_steps(segment):
            if number > upto:
                continue
            step = _step_named(segment, number)
            if _has_no_trace(step) or _has_uncovered_range(step):
                found.append([segment["segment"], number])
    return found


def _rows_of(segment: Mapping):
    for step in segment.get("steps", []):
        for sample in step.get("samples", []):
            for row in sample.get("rows", []):
                yield {"segment": segment["segment"], "step": step["step"],
                       "sample_pos": sample["sample_pos"],
                       "shard_position": row["shard_position"], "ordinal": row["ordinal"],
                       "token_start": row["token_start"], "token_end": row["token_end"]}


def _rows_of_place(document: Mapping, place: Mapping) -> list[dict]:
    if "run" in place and "segment" not in place:
        return [row for seg in _segments(document) if seg.get("run") == place["run"]
                for row in _rows_of(seg)]
    return [row for seg in _segments(document) for row in _rows_of(seg)
            if all(row[key] == value for key, value in place.items() if key in READ_ROW_KEYS)]


def _read_positions(document: Mapping, places: Sequence[Mapping]) -> dict:
    return {"rows": [row for place in places for row in _rows_of_place(document, place)]}


def _read_uncovered(document: Mapping, place: Mapping) -> dict:
    sample = _sample_named(document, place)
    if sample is None:
        return {"uncovered_in_sample": None, "sample_tokens": None, "covered_tokens": None}
    covered, tokens = sample["covered_tokens"], sample["sample_tokens"]
    return {"uncovered_in_sample": [covered, tokens] if covered < tokens else None,
            "sample_tokens": tokens, "covered_tokens": covered}


def _read_counted(document: Mapping, place: Mapping) -> dict:
    step = _step_named(_segment_named(document, place.get("segment")), place.get("step"))
    return {"consumed_samples": None if step is None else step["consumed_samples"]}


def _read_fully_traced(document: Mapping, place: Mapping) -> dict:
    uncovered, without = [], []
    for seg in _segments(document):
        if seg.get("run") != place.get("run"):
            continue
        for number in _launch_steps(seg):
            step = _step_named(seg, number)
            if _has_no_trace(step):
                without.append([seg["segment"], number])
            elif _has_uncovered_range(step):
                uncovered.append([seg["segment"], number])
    return {"steps_with_uncovered_range": uncovered, "steps_without_trace": without}


def _read_launch(document: Mapping, place: Mapping) -> dict:
    segment = _segment_named(document, place.get("segment"))
    found = [] if segment is None else [launch for launch in segment.get("launches", [])
                                        if launch["position"] == place.get("launch")]
    if not found:
        return {key: None for key in LAUNCH_VALUES}
    return {key: found[0][key] for key in LAUNCH_VALUES}


def _agreed(values: list) -> Any:
    """The one value every entry holds, None where they differ or there is none."""
    if not values or any(value != values[0] for value in values):
        return None
    return values[0]


def _read_ranks(document: Mapping, place: Mapping) -> dict:
    segment = _segment_named(document, place.get("segment"))
    steps = [] if segment is None else list(segment.get("steps", []))
    ranks_of = [sorted({sample["rank"] for sample in step.get("samples", [])}) for step in steps]
    named = sorted({rank for ranks in ranks_of for rank in ranks})
    if segment is not None:
        for launch in segment.get("launches", []):
            named = sorted(set(named) | set(range(launch["data_parallel_size"])))
    per_rank = [[sum(1 for sample in step.get("samples", []) if sample["rank"] == rank)
                 for rank in ranks] for step, ranks in zip(steps, ranks_of)]
    reading = {
        "samples_per_step": _agreed([len(step.get("samples", [])) for step in steps]),
        "samples_per_rank": _agreed([_agreed(counts) for counts in per_rank]),
        "ranks": _agreed(ranks_of),
    }
    for rank in named:
        reading[f"{RANK_KEY}{rank}"] = _agreed([
            sorted(sample["sample_pos"] for sample in step.get("samples", [])
                   if sample["rank"] == rank) for step in steps])
    return reading


def _read_unregistered(document: Mapping, place: Mapping) -> dict:
    held = checkpoint_record(document, place["artifact"])
    return {"answer": NOT_COVERED if held is None else COVERED}


class TraceSubject:
    """The trace as a ``Subject``: read from a document, nothing started, nothing changed."""

    name = SUBJECT

    def __init__(self, document: Mapping, *, place_of) -> None:
        self._document = document
        self._place_of = place_of

    def read(self, question: Question) -> Reading:
        if question.route is None:
            return Reading(question_id=question.id, raw=None, route=None)
        listed = question.listed_row
        places = self._place_of(listed)
        document = self._document
        if question.group == GROUP_POSITIONS:
            raw = _read_positions(document, places)
        else:
            place = places[0]
            if listed == "c01.E.step8.uncovered":
                raw = _read_uncovered(document, place)
            elif listed == "c01.E.step8.counted":
                raw = _read_counted(document, place)
            elif listed == "c01.E.step8.untraced":
                raw = {"untraced_steps": untraced_steps(document, place["artifact"])}
            elif listed == "c01.E-rr.fully-traced":
                raw = _read_fully_traced(document, place)
            elif listed in ("c01.K.warm-start", "c01.K.resume"):
                raw = _read_launch(document, place)
            elif listed == "c01.K.ranks":
                raw = _read_ranks(document, place)
            elif listed == "c05.unregistered-checkpoint":
                raw = _read_unregistered(document, place)
            else:
                raise ValueError(f"the trace has no reading for {listed}")
        return Reading(question_id=question.id, raw=raw, route=question.route)
