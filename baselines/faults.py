"""The fault applicator on a baseline's own records (plan 5.5 Check 10; table README,
section "The faults of Checks 4 and 5", and the ``fault`` text of each row of
``check10.toml``).

``apply_fault(fault_id, kind, record, chunk_file=None)`` returns a changed copy of a
record; the record it is given is never changed. ``kind`` names the subject whose own
record it is: ``manifest``, ``trace``, ``dolma`` or ``originblame`` (a combination
puts the half of a fault that belongs to a record of one of these). Every target the
table names is fixed here; only the chunk file of the Dolma rows is an argument,
because its name depends on the layout.

What each fault does, by the kind of record:

- manifest: ``l.edge-parent`` replaces the entry ``src/A`` of the ``built_from`` list of
  ``gen/P`` by a name no row has; ``f.edge-deleted`` removes that entry and keeps the
  row; ``f.map-deleted`` deletes the row of ``E/shard-1`` and leaves the names of it in
  other rows.
- trace (segment ``E/seg-1``): ``f.trace-deleted`` takes step 3 out of ``steps`` (the
  form docs/cli.md documents: ``steps`` holds one entry per traced step);
  ``f.trace-half`` cuts the sample at position 3 of step 4 at 16 tokens;
  ``l.trace-count`` deletes the sample at position 3 of step 5 and sets that step's
  count to 3. A step, a sample and a row are found by their own values, never by their
  place in a list.
- dolma (rows of the companion file, in any order): ``f.map-deleted`` deletes every row
  of the chunk file; ``f.digest`` changes the first digit of ``end`` of the row at loc 1
  of the chunk file; ``l.count-499`` deletes the row with the greatest loc of the chunk
  file.
- originblame (``entries`` in the order they were tracked): ``f.map-deleted`` deletes
  every entry and keeps the sections; ``f.digest`` changes the first digit of
  ``token_count`` of the first entry; ``l.count-499`` deletes the last entry.

A fault that no record of the kind has a counterpart for, a fault that is no change to
a record (the writer faults, the fault on a file), an id the table does not hold and a
target the record does not hold are each a ``FaultError``, never a silent no-op. The
message names the fault id.

For ``f.trace-half`` a row that holds tokens from the cut on but starts at or after
the cut is removed with the later rows, so no row is left with an empty range.
"""

from __future__ import annotations

import copy

SEGMENT = "E/seg-1"
EDGE_ARTIFACT = "gen/P"
EDGE_PARENT = "src/A"
MAP_SHARD = "E/shard-1"
CUT_TOKENS = 16

# The faults the applicator holds, by the kind of record that has a counterpart.
_FAULTS = {
    "manifest": ("l.edge-parent", "f.edge-deleted", "f.map-deleted"),
    "trace": ("f.trace-deleted", "f.trace-half", "l.trace-count"),
    "dolma": ("f.map-deleted", "f.digest", "l.count-499"),
    "originblame": ("f.map-deleted", "f.digest", "l.count-499"),
}
# The faults of the table that this module does not put to a record, with why.
_NOT_A_RECORD = {
    "w.kill-after-100": "a stop of the tool's own writer, not a change to a record",
    "w.retry": "a run of the tool's own writer again, not a change to a record",
    "l.artifact-digest": "a change to a file on disk, not to a record",
}


class FaultError(Exception):
    """A fault that cannot be put: no counterpart in the record, no such fault, or a
    target the record does not hold. The message names the fault id."""


def _error(fault_id: str, text: str) -> FaultError:
    return FaultError(f"{fault_id}: {text}")


def _only(fault_id: str, items: list, what: str):
    """The one item of a list of matches, or an error that says there is none or many."""
    if not items:
        raise _error(fault_id, f"the record holds no {what}")
    if len(items) > 1:
        raise _error(fault_id, f"the record holds {len(items)} of {what}, expected one")
    return items[0]


def _first_digit_changed(fault_id: str, value) -> int:
    """The number with its first digit replaced by another digit that is not 0; the
    rest of the digits and the length stay."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(fault_id, f"{value!r} is not a count whose first digit can change")
    text = str(value)
    return int(str((int(text[0]) % 9) + 1) + text[1:])


# --- manifest ----------------------------------------------------------------------------


def _manifest_row(fault_id: str, manifest: list, name: str) -> dict:
    return _only(fault_id, [r for r in manifest if r["name"] == name], f"row {name}")


def _manifest_edge_entry(fault_id: str, manifest: list) -> tuple[dict, int]:
    row = _manifest_row(fault_id, manifest, EDGE_ARTIFACT)
    places = [n for n, entry in enumerate(row["built_from"]) if entry == EDGE_PARENT]
    if not places:
        raise _error(fault_id, f"the row {EDGE_ARTIFACT} has no entry {EDGE_PARENT} in "
                               f"built_from")
    return row, places[0]


def _edge_parent(fault_id: str, manifest: list) -> list:
    row, place = _manifest_edge_entry(fault_id, manifest)
    names = {r["name"] for r in manifest} | set(row["built_from"])
    replacement = EDGE_PARENT + ".absent"
    while replacement in names:
        replacement += "_"
    row["built_from"][place] = replacement
    return manifest


def _edge_deleted(fault_id: str, manifest: list) -> list:
    row, place = _manifest_edge_entry(fault_id, manifest)
    del row["built_from"][place]
    return manifest


def _manifest_map_deleted(fault_id: str, manifest: list) -> list:
    row = _manifest_row(fault_id, manifest, MAP_SHARD)
    return [r for r in manifest if r is not row]


# --- trace -------------------------------------------------------------------------------


def _trace_step(fault_id: str, trace: dict, step: int) -> dict:
    if trace["segment"] != SEGMENT:
        raise _error(fault_id, f"the trace is of segment {trace['segment']!r}, the fault "
                               f"names {SEGMENT}")
    return _only(fault_id, [s for s in trace["steps"] if s["step"] == step],
                 f"step {step}")


def _trace_last_sample(fault_id: str, step: dict) -> dict:
    return _only(fault_id, [s for s in step["samples"] if s["sample_pos"] == 3],
                 f"sample at position 3 in step {step['step']}")


def _trace_deleted(fault_id: str, trace: dict) -> dict:
    # The documented form: `steps` holds one entry per traced step, so the step whose
    # trace is deleted has no entry (a step with an empty `samples` is not that form).
    gone = _trace_step(fault_id, trace, 3)
    trace["steps"] = [s for s in trace["steps"] if s is not gone]
    return trace


def _trace_half(fault_id: str, trace: dict) -> dict:
    sample = _trace_last_sample(fault_id, _trace_step(fault_id, trace, 4))
    # A row's token range lies inside its document, so the cut is counted in the tokens of
    # the sample: the lengths of the rows, in reading order (docs/interfaces.md 8.3).
    kept = []
    before = 0
    for row in sample["rows"]:
        left = CUT_TOKENS - before
        if left <= 0:
            break
        length = row["token_end"] - row["token_start"]
        kept.append(row if length <= left
                    else {**row, "token_end": row["token_start"] + left})
        before += length
    sample["rows"] = kept
    sample["covered_tokens"] = CUT_TOKENS
    return trace


def _trace_count(fault_id: str, trace: dict) -> dict:
    step = _trace_step(fault_id, trace, 5)
    last = _trace_last_sample(fault_id, step)
    step["samples"] = [s for s in step["samples"] if s is not last]
    step["consumed_samples"] = 3
    return trace


# --- dolma -------------------------------------------------------------------------------


def _dolma_rows(fault_id: str, rows: list, chunk_file) -> list:
    if chunk_file is None:
        raise _error(fault_id, "a Dolma fault names the chunk file it is put to")
    held = [r for r in rows if r["src"] == chunk_file]
    if not held:
        raise _error(fault_id, f"the rows hold no row of the chunk file {chunk_file}")
    return held


def _dolma_map_deleted(fault_id: str, rows: list, chunk_file) -> list:
    held = _dolma_rows(fault_id, rows, chunk_file)
    return [r for r in rows if not any(r is h for h in held)]


def _dolma_digest(fault_id: str, rows: list, chunk_file) -> list:
    held = _dolma_rows(fault_id, rows, chunk_file)
    target = _only(fault_id, [r for r in held if r["loc"] == 1],
                   f"row at loc 1 of the chunk file {chunk_file}")
    target["end"] = _first_digit_changed(fault_id, target["end"])
    return rows


def _dolma_count(fault_id: str, rows: list, chunk_file) -> list:
    held = _dolma_rows(fault_id, rows, chunk_file)
    greatest = max(r["loc"] for r in held)
    target = _only(fault_id, [r for r in held if r["loc"] == greatest],
                   f"row at the greatest loc ({greatest}) of the chunk file {chunk_file}")
    return [r for r in rows if r is not target]


# --- originblame -------------------------------------------------------------------------


def _entries(fault_id: str, store: dict) -> list:
    if not store["entries"]:
        raise _error(fault_id, "the store holds no token-index entry")
    return store["entries"]


def _ob_map_deleted(fault_id: str, store: dict) -> dict:
    _entries(fault_id, store)
    store["entries"] = []
    return store


def _ob_digest(fault_id: str, store: dict) -> dict:
    first = _entries(fault_id, store)[0]
    first["token_count"] = _first_digit_changed(fault_id, first["token_count"])
    return store


def _ob_count(fault_id: str, store: dict) -> dict:
    entries = _entries(fault_id, store)
    store["entries"] = entries[:-1]
    return store


_APPLY = {
    ("manifest", "l.edge-parent"): _edge_parent,
    ("manifest", "f.edge-deleted"): _edge_deleted,
    ("manifest", "f.map-deleted"): _manifest_map_deleted,
    ("trace", "f.trace-deleted"): _trace_deleted,
    ("trace", "f.trace-half"): _trace_half,
    ("trace", "l.trace-count"): _trace_count,
    ("originblame", "f.map-deleted"): _ob_map_deleted,
    ("originblame", "f.digest"): _ob_digest,
    ("originblame", "l.count-499"): _ob_count,
}
_APPLY_DOLMA = {
    "f.map-deleted": _dolma_map_deleted,
    "f.digest": _dolma_digest,
    "l.count-499": _dolma_count,
}


def apply_fault(fault_id: str, kind: str, record, chunk_file: str | None = None):
    """A changed copy of ``record``, the record of the subject ``kind``, with the fault
    put to it. ``chunk_file`` names the chunk file of a Dolma fault and is ignored for
    every other kind."""
    if fault_id in _NOT_A_RECORD:
        raise _error(fault_id, _NOT_A_RECORD[fault_id])
    if kind not in _FAULTS:
        raise _error(fault_id, f"no record of the kind {kind!r}")
    if fault_id not in _FAULTS[kind]:
        raise _error(fault_id, f"no counterpart in the record of {kind}")
    changed = copy.deepcopy(record)
    try:
        if kind == "dolma":
            return _APPLY_DOLMA[fault_id](fault_id, changed, chunk_file)
        return _APPLY[(kind, fault_id)](fault_id, changed)
    except (KeyError, TypeError, AttributeError) as error:
        raise _error(fault_id, f"the {kind} record is not in the shape the fault needs "
                               f"({error!r})") from error
