"""The drivers of the six subjects of Check 10 and the order the runner drives them in
(plan 5.5 Check 10; table README, section "Check 10": the six subjects, "What a run hands a
subject, and its conditions"; docs/baselines.md, the section on the subjects).

The runner drives every subject through a factory that is given a ``SubjectContext`` and gives
back an object with ``read(question)``. This module holds what is the same for all of them:

- ``DRIVER_ORDER`` and ``ordered``: the one order the drivers are built and run in, whatever order
  ``--subjects`` names them: plain ``dolma``, plain ``originblame``, then ``manifest``,
  ``trace``, ``dolma+trace`` and ``originblame+trace``. ``missing_half`` names a combination
  whose plain half is not in the pass.
- ``twin_id``: for a question of ``originblame+trace`` that the combination reads from its tool
  half alone, the id of the question of the plain ``originblame`` that serves it (the same listed
  row and size), where the plain row holds a record.
- The factories of the four subjects the tools do not run: ``trace_factory``,
  ``manifest_factory``, ``dolma_trace_factory`` and ``originblame_trace_factory``. Each takes
  documents that are already loaded, never a path; ``load_subject_inputs`` reads them from the
  ``subject_inputs`` of the world file and refuses a file whose sha256 is not the one the world
  names. A combination reuses the runs of its plain half (the plain subject object the context
  hands it): no tool runs again for its line.
- ``table_place_of``: the places a row of the trace reads, from the table and the fixture spec.

A driver adds no reading rule: where the table is silent for a question, the driver gives no
line for it (the question stays not attempted) and keeps the fact in ``subject-facts.jsonl``.
Nothing here starts a command.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from baselines import combo_dolma, combo_originblame, dolma, faults, manifest, record, table10
from baselines import originblame as tool_half
from baselines.interfaces import Attempt, Question, Reading
from baselines.trace import Refusal, TraceSubject

# --- The order ----------------------------------------------------------------------------------

# The plain tools first, then the other four in the order of the table's list of subjects.
DRIVER_ORDER = ("dolma", "originblame", "manifest", "trace", "dolma+trace", "originblame+trace")

# The plain half of each combination: the subject whose runs it reuses.
PLAIN_HALF = {"dolma+trace": "dolma", "originblame+trace": "originblame"}

# The subjects that run a tool of their own.
PLAIN_TOOLS = ("dolma", "originblame")


def ordered(names: Sequence[str]) -> tuple[str, ...]:
    """The subjects of ``names`` in the one order the drivers are built and run in; a name no
    driver is known for stands after them, in the order it came."""
    named = list(dict.fromkeys(names))
    known = [name for name in DRIVER_ORDER if name in named]
    return (*known, *(name for name in named if name not in DRIVER_ORDER))


def missing_half(names: Sequence[str]) -> str | None:
    """The words that say which combination is named without its plain half, else None."""
    for combination, plain in PLAIN_HALF.items():
        if combination in names and plain not in names:
            return (f"{combination} reads the runs of {plain} and starts no tool of its own: "
                    f"{plain} has to be in the same pass")
    return None


def owners_of(subject: str) -> tuple[str | None, ...]:
    """The drivers whose shared steps (``StepLog.shared``) a line of ``subject`` rests on: the
    steps no driver owns, then the two plain tools (the steps of a store or set are the steps of
    every tool run on it, as they have always been for the plain subjects), and for any other
    subject its own. A step a later driver logs is therefore in its own lines and in no plain
    line."""
    if subject in PLAIN_TOOLS:
        return (None, *PLAIN_TOOLS)
    if subject in PLAIN_HALF:
        return (None, *PLAIN_TOOLS, subject)
    return (None, subject)


# --- Keys of a target ------------------------------------------------------------------------------


def target_key(target: Sequence[str]) -> str:
    """``<kind>-<name>`` of a store or set, the key of the world's files for it."""
    return "-".join(target)


def repaired_key(rest: str) -> str:
    """The key of a repaired side in ``repaired`` from its name in the world (``repaired-<rest>``): ``set-10k`` and
    ``set-10k-partitioned`` name a generated set (and its added layout), any other name a store (``main``: ``store-main``)."""
    return rest if rest.startswith("set-") else f"store-{rest}"


def keyed_by_target(mapping: Mapping[str, Any], what: str) -> dict[tuple[str, str], Any]:
    """``mapping`` by the target of each key; ``ValueError`` where two keys name one target (a key of a build ADDED beside the
    target's own, ``set-10k-partitioned``, names the same target as ``set-10k``: only the manifest holds one, and a map that
    would let the second overwrite the first is refused, never silently read)."""
    found: dict[tuple[str, str], Any] = {}
    for key, value in mapping.items():
        target = target_of_key(key)
        if target in found:
            raise ValueError(f"the {what} of {target[0]} {target[1]} is given twice (key {key!r})")
        found[target] = value
    return found


LAYOUT_SUFFIXES = ("partitioned", "round-robin")


def layout_of_key(key: str) -> str | None:
    """The layout a key of a build ADDED beside the target's own ends in (``set-10k-partitioned``), else None (W8, 18 Nov 2026)."""
    for layout in LAYOUT_SUFFIXES:
        if key.endswith("-" + layout):
            return layout
    return None


def target_of_key(key: str) -> tuple[str, str]:
    """The target of a key ``store-main`` or ``set-small``: the first hyphen splits the kind from
    the name (a name may hold hyphens). The layout a key of an ADDED build ends in (``set-10k-partitioned``) is not part of the
    target: the two builds are one target."""
    kind, _, name = key.partition("-")
    layout = layout_of_key(key)
    if layout is not None:
        name = name[: -(len(layout) + 1)]
    if kind not in ("store", "set") or not name:
        raise ValueError(f"{key!r} is no key of a store or set (store-NAME or set-SIZE)")
    return (kind, name)


# --- The twin of a question of originblame+trace ---------------------------------------------------


@lru_cache(maxsize=1)
def _plain_originblame_questions() -> Mapping[tuple[str, str | None], Question]:
    found = {}
    for question in table10.expand(table10.load_table10()):
        if question.subject == "originblame":
            found[(question.listed_row, question.size)] = question
    return found


def served_kind(question: Question) -> bool:
    """Whether the combination reads this question from its tool half alone: a row of one of the
    eight tool groups, or the shard row of one of the four fault groups."""
    if question.subject != "originblame+trace":
        return False
    if question.group in combo_originblame.TOOL_GROUPS:
        return True
    if question.group in combo_originblame.KIND_GROUPS:
        given = record._listed_givens()[question.listed_row]
        return tool_half._kind_of(given.get("artifact")) == "shard"
    return False


def twin_id(question: Question) -> str | None:
    """The id of the question of the plain ``originblame`` that serves ``question`` (a question of
    ``originblame+trace`` the combination reads from the tool half), computed from the table: the
    plain row of the same listed row and size. None for any other question, and for a row whose
    plain row holds no record: it has no twin and is read by the tool half."""
    if not served_kind(question):
        return None
    twin = _plain_originblame_questions().get((question.listed_row, question.size))
    if twin is None or twin.route is None:
        return None
    return twin.id


# --- The places a row of the trace reads -------------------------------------------------------------

PLACE_KEYS = ("segment", "step", "sample_pos", "shard_position", "ordinal")
POSITIONS_PREFIX = "pending:positions."
TRACE_POSITIONS_GROUP = "c01.positions"
FIXTURE_SPEC = table10.REPO / "fixture" / "spec" / "fixture_spec.toml"


def table_place_of(spec_path: Path | str | None = None) -> Callable[[str], list[dict]]:
    """``place_of(listed_row)`` for the trace: for a row of ``c01.positions`` that names lines of
    the oracle's log one place per line (read from the fixture spec), for a whole-run row
    ``{"run": R}``, and for every other routed row the listed row's own ``given``. Everything is
    read from the table and the spec, both frozen files of the repository."""
    acceptance = table10._acceptance_table()
    rows = {row.id: row for number in table10.LISTED_CHECKS
            for row in acceptance.load_check(number, table10.DEFAULT_PATH.parent)}
    group_of = {listed: group.id for group in table10.load_table10().groups for listed in group.rows}
    held: dict[str, Any] = {}

    def spec() -> Mapping[str, Any]:
        if "spec" not in held:
            with open(spec_path or FIXTURE_SPEC, "rb") as handle:
                held["spec"] = tomllib.load(handle)
        return held["spec"]

    def place_of(listed_row: str) -> list[dict]:
        row = rows[listed_row]
        if group_of[listed_row] == TRACE_POSITIONS_GROUP:
            named = row.expect.get("rows")
            if isinstance(named, str) and named.startswith(POSITIONS_PREFIX):
                lines = [json.loads(text) for text in spec()["positions"][named[len(POSITIONS_PREFIX):]]["log_lines"]]
                return [{key: line[key] for key in PLACE_KEYS} for line in lines]
            return [{"run": row.given["run"]}]
        return [dict(row.given)]

    return place_of


# --- What a world file names -----------------------------------------------------------------------------


class SubjectInputsError(ValueError):
    """A world file whose ``subject_inputs`` cannot drive a subject: a key missing, a file that
    is not there, or one whose sha256 differs from the one the world names."""


def _checked_key(key: str, where: str) -> None:
    """A key of the world's subject files is ``store-NAME`` or ``set-SIZE`` (a manifest also has
    ``repaired-NAME`` for the repaired side)."""
    if where == "manifest" and key.startswith("repaired-") and len(key) > len("repaired-"):
        return
    try:
        target_of_key(key)
    except ValueError as error:
        raise SubjectInputsError(f"{where}.{key}: {error}") from None


def _file_of(entry: Any, folder: Path, where: str) -> Path:
    if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
        raise SubjectInputsError(f"{where} names no file as {{path, sha256}}")
    path = Path(folder) / entry["path"]
    if not path.is_file():
        raise SubjectInputsError(f"{where}: {entry['path']} is no file of the world")
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != entry["sha256"]:
        raise SubjectInputsError(f"{where}: {entry['path']} has the sha256 {found}, the world names {entry['sha256']}")
    return path


def _document_at(entry: Any, folder: Path, where: str) -> dict:
    path = _file_of(entry, folder, where)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise SubjectInputsError(f"{where}: {entry['path']} is not JSON ({error})") from None


def _trace_value(value: Any, folder: Path, key: str) -> tuple[Any, dict[str, Any]]:
    """The trace of a store or set: its document (or ``{"none": reason}``) and, for the four
    starting states of Check 2, the document each wrote."""
    if not isinstance(value, Mapping):
        raise SubjectInputsError(f"trace.{key} is neither a document nor a none")
    states: dict[str, Any] = {}
    for state, item in (value.get("state_traces") or {}).items():
        states[state] = (dict(item) if "none" in item and "document" not in item
                         else _document_at(item["document"], folder, f"trace.{key}.state_traces.{state}"))
    if "document" in value:
        return _document_at(value["document"], folder, f"trace.{key}"), states
    if "none" in value:
        return {"none": str(value["none"])}, states
    raise SubjectInputsError(f"trace.{key} names neither a document nor a none")


def load_subject_inputs(world: Mapping[str, Any], folder: Path | str) -> dict[str, Any]:
    """The inputs of the four subjects the tools do not run, read from ``world["subject_inputs"]``
    (paths relative to ``folder``, each file checked against the sha256 the world names).

    ``traces``: ``<kind>-<name>`` to a document or ``{"none": reason}``; ``state_traces``: the
    same for the starting states of Check 2, by target key and state; ``manifests``: key to the
    rows of ``built-from.jsonl`` or None where a build made without capture wrote no file;
    ``repaired``: the rows of the repaired side's file, by key. ``SubjectInputsError`` where the
    world holds no ``subject_inputs``."""
    inputs = world.get("subject_inputs")
    if not isinstance(inputs, Mapping):
        raise SubjectInputsError("the world file holds no subject_inputs: the subjects manifest, trace, "
                                 "dolma+trace and originblame+trace need the files it names")
    folder = Path(folder)
    traces: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, str]] = {}
    for key, value in (inputs.get("trace") or {}).items():
        _checked_key(key, "trace")
        traces[key], held = _trace_value(value, folder, key)
        if held:
            states[key] = held
        if value.get("failed"):
            failed[key] = {str(segment): str(why) for segment, why in value["failed"].items()}
    manifests: dict[str, Any] = {}
    repaired: dict[str, Any] = {}
    for key, value in (inputs.get("manifest") or {}).items():
        _checked_key(key, "manifest")
        if not isinstance(value, Mapping):
            raise SubjectInputsError(f"manifest.{key} is neither a file nor a none")
        rows = None
        if "built_from" in value:
            path = _file_of(value["built_from"], folder, f"manifest.{key}")
            try:
                rows = manifest.load_manifest(path.parent)
            except manifest.ManifestError as error:
                raise SubjectInputsError(f"manifest.{key}: {error}") from None
        if key.startswith("repaired-"):
            # ``repaired-main`` is the repaired side of the store main; ``repaired-set-10k`` (and ``repaired-set-10k-partitioned`` for
            # a build added in a layout) that of a generated set.
            repaired[repaired_key(key[len("repaired-"):])] = rows
        else:
            manifests[key] = rows
    return {"traces": traces, "state_traces": states, "failed_exports": failed, "manifests": manifests,
            "repaired": repaired}


# --- The trace ------------------------------------------------------------------------------------------


def _held_outcome(held: Any) -> tuple[str, str | None]:
    """What a kept fact says of one export: its word and the id of the failed attempt."""
    if isinstance(held, Attempt):
        return "failed", held.id
    if isinstance(held, Refusal):
        return "refusal", None
    if isinstance(held, Mapping) and "none" in held and "segments" not in held:
        return f"none: {held['none']}", None
    return "document", None


def as_document(held: Any) -> Any:
    """A trace the way a subject reads it: a refusal, or a store with no run registered, reads as a
    document with no segment; a failed export stays its ``Attempt``."""
    if isinstance(held, Refusal) or (isinstance(held, Mapping) and "none" in held and "segments" not in held):
        return {"segments": []}
    return held


class TraceDriver:
    """The ``trace`` subject over the exported documents of the stores and sets: a routed row is read
    by ``TraceSubject`` from the document of the question's own store, and a store whose export
    failed gives that failed ``Attempt``. A store the driver holds no document for is not
    attempted; a routed row on a store with no run registered cannot be answered from nothing and
    is not attempted either (the table puts no routed row of the trace on such a store)."""

    name = "trace"

    def __init__(self, documents: Mapping[str, Any], *, place_of, context,
                 failed: Mapping[str, Mapping[str, str]] | None = None):
        self._documents = keyed_by_target(documents, "trace document")
        self._place_of = place_of
        for key, segments in (failed or {}).items():
            # A segment whose export gave no document: kept as a fact of the store, not an attempt.
            for segment, why in sorted(segments.items()):
                context.keep_fact("export", target_of_key(key), f"segment {segment}: none ({why})")
        for target, held in self._documents.items():
            word, attempt_id = _held_outcome(held)
            context.keep_fact("export", target, word, attempt_id=attempt_id)
            if isinstance(held, Attempt):
                context.failed_runs[target] = held
            else:
                context.ran_on.add(target)

    def read(self, question: Question) -> Reading | Attempt:
        if question.route is None:
            return Reading(question_id=question.id, raw=None, route=None)
        target = tuple(record.attempt_target(question))
        if target not in self._documents:
            raise combo_dolma.NotAttempted(f"the trace was exported for no {target[0]} {target[1]}")
        held = self._documents[target]
        if isinstance(held, Attempt):
            return held
        if isinstance(held, Refusal) or (isinstance(held, Mapping) and "none" in held and "segments" not in held):
            raise combo_dolma.NotAttempted(f"the {target[0]} {target[1]} holds no registered run: "
                                           f"no document to read {question.listed_row} from")
        return TraceSubject(held, place_of=self._place_of).read(question)


def trace_factory(documents: Mapping[str, Any], *, place_of,
                  failed: Mapping[str, Mapping[str, str]] | None = None) -> Callable:
    """The factory of the ``trace`` subject. ``documents`` maps the key ``<kind>-<name>`` of a
    store or set (``store-main``, ``set-small``) to its exported trace document, to its failed
    ``Attempt``, or to ``{"none": reason}`` for a store that has no run registered; the driver has
    run on each store it holds one for (a failed export is the failed run of its store)."""
    def factory(context):
        return TraceDriver(documents, place_of=place_of, context=context, failed=failed)

    return factory


# --- The manifest ----------------------------------------------------------------------------------------

# The groups the driver reads, each by the reader it names (``baselines.manifest``). The groups of
# the manifest the table routes that this driver does not read are listed in docs/baselines.md:
# their arguments (the artifacts of a list, the file family of an import, the joined rows of a
# repaired side) are not named by the listed row.
WALK_GROUPS = tuple(group for group, reader in manifest.GROUP_READINGS.items() if reader == "read_artifact"
                    and group not in ("c09.successors", "c09.history"))
LIST_GROUP = "c09.mixture-successors"
RESTART_GROUP = "c09.exposure-free-restart"
# Read since 17 Nov 2026, by the manifest's own walk rule
# of their rows in check10.toml: the actions (at fixture size), the successors (in the manifest of the repaired
# side) and the history (the rows of both builds joined).
ACTIONS_GROUP = "c09.artifact-actions"
SUCCESSORS_GROUP = "c09.successors"
HISTORY_GROUP = "c09.history"
COUNTS_ROW = "c09.fixture.shards.counts"
KEPT_ROW = "c09.10k.partitioned.kept-shards"
ALL_REWRITE_ROW = "c09.10k.round-robin.all-rewrite"


def shard_rows(rows) -> list[str]:
    """The names of the rows that are shards (docs/replica.md section 16: a shard's files are its ``.bin`` and ``.idx`` and its
    ``built_from`` is its one chunk file, ``<name>.input``; a mixture's files are shard files too, built from shards)."""
    return sorted(row["name"] for row in rows
                  if row["files"] and all(str(path).endswith((".bin", ".idx")) for path in row["files"])
                  and list(row["built_from"]) == [row["name"] + ".input"])


def shard_name_of(rows, name: str) -> str:
    """The shard as the counting of the plant list names it: by the file stem of its chunk file (the file of the row
    ``<name>.input``); the registered name where the chunk has no row or no file."""
    for row in rows:
        if row["name"] == name + ".input" and row["files"]:
            return Path(str(row["files"][0])).stem
    return name


def chunk_rows(rows) -> list[str]:
    """The names of the rows that are chunk files (section 16: ``<shard>.input``, the files its tokenizer input)."""
    return sorted(row["name"] for row in rows if row["name"].endswith(".input") and row["files"])
# And the two groups with a reading of their own, written from the rule text of their rows: the listing of the
# manifest after the fault ("list every name the manifest holds ... by kind of artifact; a name with no row of its own
# is listed as no record") and the file family of an import ("the reading is the files that list names").
LISTING_GROUP = "c05.listing.map-deleted"
IMPORT_GROUP = "c06.import.file-family"
READ_GROUPS = (*WALK_GROUPS, LIST_GROUP, RESTART_GROUP, ACTIONS_GROUP, SUCCESSORS_GROUP, HISTORY_GROUP,
               LISTING_GROUP, IMPORT_GROUP)


def checkpoints_of_run(document: Mapping, run: str) -> list[str]:
    """The names of the checkpoints saved in the segments of ``run``, in step order, from the
    trace document (the run's record)."""
    found = [(mark["step"], number, mark["name"])
             for number, seg in enumerate(document.get("segments", [])) if seg.get("run") == run
             for mark in seg.get("checkpoints", [])]
    return [name for _, _, name in sorted(found)]


class ManifestDriver:
    """The ``manifest`` subject over the rows of ``built-from.jsonl`` of each build (None for a
    build made without capture).

    A request's files are turned into artifact names by the manifest's own rows: the names of the
    rows whose ``files`` hold a file the request names. A request no row names a file of is not
    read (a manifest cannot read a walk against no member). The checkpoints of a run for the
    restart are those the trace document of the store holds for it, in step order. A group
    this driver does not read, and a question it cannot give the arguments of from its listed row,
    is not attempted and leaves a fact.
    """

    name = "manifest"

    def __init__(self, manifests: Mapping[str, Any], *, requests: Mapping[str, Mapping[str, Any]],
                 traces: Mapping[str, Any], context, repaired: Mapping[str, Any] | None = None):
        # The manifest of a build ADDED beside the target's own (key ``set-10k-partitioned``) is kept under its target and layout:
        # a question that names that layout reads it; any other question reads the target's own.
        self._rows = {target_of_key(key): rows for key, rows in manifests.items() if layout_of_key(key) is None}
        self._layered = {(target_of_key(key), layout_of_key(key)): rows for key, rows in manifests.items()
                         if layout_of_key(key) is not None}
        self._repaired = keyed_by_target({k: v for k, v in (repaired or {}).items() if layout_of_key(k) is None},
                                         "manifest of the repaired side")
        self._repaired_layered = {(target_of_key(k), layout_of_key(k)): v for k, v in (repaired or {}).items()
                                  if layout_of_key(k) is not None}
        self._requests = requests
        self._traces = keyed_by_target(traces, "trace")
        self._context = context
        self._said: set[tuple[str, str]] = set()
        for target, rows in self._rows.items():
            context.ran_on.add(target)
            context.keep_fact("export", target, "manifest absent" if rows is None else f"{len(rows)} rows")
        for (target, layout), rows in self._layered.items():
            context.keep_fact("export", (*target, layout), "manifest absent" if rows is None else f"{len(rows)} rows")

    def _not_read(self, question: Question, target: Sequence[str], why: str):
        if (question.group, why) not in self._said:
            self._said.add((question.group, why))
            self._context.keep_fact("half", target, f"not read: {question.group}: {why}")
        raise combo_dolma.NotAttempted(f"{question.listed_row}: {why}")

    def _fault_state(self, given: Mapping[str, Any]) -> str | None:
        state = given.get("state")
        return state if state in faults._FAULTS["manifest"] else None

    def _targets(self, question: Question, target, rows, given) -> set[str]:
        word = given.get("request")
        files = (self._requests.get(word) or {}).get("files") if word is not None else None
        if not files:
            self._not_read(question, target, f"the request {word!r} names no file")
        named = {row["name"] for row in rows if set(row["files"]) & set(files)}
        if not named:
            self._not_read(question, target, f"no row of the manifest holds a file of the request {word!r}")
        return named

    def read(self, question: Question) -> Reading | Attempt:
        return self._read(question, record.layout_of(question))

    def read_by_layout(self, question: Question) -> dict[str, Reading | Attempt | None]:
        """For a row that names layouts, the reading of each layout from the manifest of that layout: its reading, or None where the
        world holds no manifest of the layout (or the row is not read there), so that the line follows the layouts' slots. The rows
        with no join (``dolma.OPEN_ROWS``) read cannot answer in each layout the subject read, as every per-layout reader does."""
        layouts = record.layouts_of(question)
        if layouts is None:
            raise ValueError(f"{question.listed_row} names no layouts")
        if question.route is None:
            return {layout: Reading(question.id, None, None) for layout in layouts}
        target = tuple(record.attempt_target(question))
        found: dict[str, Reading | Attempt | None] = {}
        for layout in layouts:
            try:
                if question.listed_row in dolma.OPEN_ROWS:
                    if self._rows_of(target, layout) is None:
                        raise combo_dolma.NotAttempted(f"no manifest of the {layout} build of {target[0]} {target[1]}")
                    found[layout] = Reading(question.id, {"cannot_answer": dolma.NO_JOIN_TEXT}, question.route)
                else:
                    found[layout] = self._read(question, layout)
            except combo_dolma.NotAttempted:
                found[layout] = None
        return found

    def _rows_of(self, target, layout):
        """The rows of the manifest of the build in ``layout`` of ``target``: the build added in that layout, else the target's own
        manifest, which is the round-robin build of a generated set (and the build of any layout of a store)."""
        if (target, layout) in self._layered:
            return self._layered[(target, layout)]
        if layout is None or target[0] != "set" or layout == "round-robin":
            return self._rows.get(target)
        return None

    def _read(self, question: Question, layout: str | None) -> Reading | Attempt:
        if question.route is None:
            return Reading(question_id=question.id, raw=None, route=None)
        target = tuple(record.attempt_target(question))
        if target not in self._rows:
            raise combo_dolma.NotAttempted(f"no manifest was read for {target[0]} {target[1]}")
        if question.group not in READ_GROUPS:
            self._not_read(question, target, "the listed row names no arguments for this group")
        given = record._listed_givens()[question.listed_row]
        if question.listed_row in dolma.OPEN_ROWS:
            # The table names no fold of many shards into one answer for these rows: no subject folds (D-SUBJ-11); the line is
            # the subject's cannot answer with the table's words, as for every other subject, but only where the subject read
            # the manifest of EACH layout the row names (the cannot answer holds on a layout the tool ran on).
            for named in record.layouts_of(question) or ():
                # The target's own manifest is the build the name always meant: the round-robin build of a generated set.
                held = self._layered.get((target, named)) if (target, named) in self._layered else (
                    self._rows.get(target) if target[0] == "set" and named == "round-robin" else None)
                if held is None:
                    self._not_read(question, target, f"the world holds no manifest of the {named} build of this target")
            raise ValueError(dolma.NO_JOIN_TEXT)
        rows = self._layered.get((target, layout), self._rows[target]) if layout is not None else self._rows[target]
        if rows is None:
            return Reading(question.id, self._absent(question), question.route)
        fault = self._fault_state(given)
        if fault is not None:
            rows = faults.apply_fault(fault, "manifest", rows)
        group = question.group
        if group == LISTING_GROUP:
            return Reading(question.id, self._read_listing(rows), question.route)
        if group == IMPORT_GROUP:
            return Reading(question.id, self._read_import(question, target, rows, given), question.route)
        if group in (SUCCESSORS_GROUP, HISTORY_GROUP):
            # The successors are the artifacts Check 9 builds from the tool's rebuild report, and the subject's records
            # for them come from its own run on that build: the repaired side's manifest. The historical R1 is read
            # after the rows of the successor build have been added to the manifest: both builds' rows, joined.
            repaired = self._repaired_layered.get((target, layout), self._repaired.get(target)) if layout is not None else self._repaired.get(target)
            if layout is not None and layout != "round-robin" and (target, layout) not in self._repaired_layered and target[0] == "set":
                repaired = None   # the repaired side of a set is that of the layout the row names
            if repaired is None:
                self._not_read(question, target, "the world holds no manifest of the repaired side of this store")
            try:
                rows = repaired if group == SUCCESSORS_GROUP else manifest.join_manifests(rows, repaired)
            except manifest.ManifestError as error:
                self._not_read(question, target, f"the rows of the two builds cannot be joined: {error}")
        targets = self._targets(question, target, rows, given)
        if group == ACTIONS_GROUP:
            raw = self._read_actions(question, target, rows, given, targets)
        elif group == RESTART_GROUP:
            document = self._traces.get(target)
            if not isinstance(document, Mapping) or "segments" not in document:
                self._not_read(question, target, "the store holds no trace to order the checkpoints of the run by")
            checkpoints = checkpoints_of_run(document, given["run"])
            raw = {"exposure_free_restart": manifest.read_group(group, rows, targets=targets, checkpoints=checkpoints)}
        elif group == LIST_GROUP:
            names = [row["name"] for row in rows if "/mix-" in row["name"]]
            raw = manifest.read_group(group, rows, targets=targets, artifacts=names)
        else:
            artifact = given.get("artifact")
            if artifact is None or str(artifact).startswith("each "):
                self._not_read(question, target, "the listed row names no one artifact")
            raw = {"answer": manifest.read_group(group, rows, targets=targets, artifact=artifact)}
        return Reading(question.id, raw, question.route)

    @staticmethod
    def _read_listing(rows) -> dict:
        """Every name the manifest holds (after the fault), by kind, and the names with no row of their own as
        ``not_covered``. The manifest holds no run, so ``runs`` stays empty: nothing is made up and no reason is read."""
        listed = manifest.read_group(LISTING_GROUP, rows)
        return {"covers": {kind: names for kind, names in listed.items() if kind != "no_record"},
                "not_covered": [{"artifact": name} for name in listed["no_record"]]}

    def _read_import(self, question: Question, target, rows, given: Mapping[str, Any]) -> dict:
        """The files the ``built_from`` list of the row of the import names (each name of the list through its own row's
        files); a name with no row names no file."""
        artifact = given.get("artifact")
        if artifact is None:
            self._not_read(question, target, "the listed row names no artifact")
        names = manifest.read_group(IMPORT_GROUP, rows, artifact=artifact)
        if not isinstance(names, list):
            return {"answer": names}
        files_of = {row["name"]: list(row["files"]) for row in rows}
        files = [path for name in names for path in files_of.get(name, ())]
        return {"link": {"files": files}}

    def _read_actions(self, question: Question, target, rows, given: Mapping[str, Any], targets) -> dict:
        """The actions of Check 9 by the plan's own definition (4.1): the walk of an artifact under the request,
        contains read as rewrite and does not contain as keep. One artifact gives its ``action``; the fixture's
        three shards of the run (the rows of the manifest named RUN/shard-N) give the lists and the counts. The
        rows of the 10k set: the shards read keep (``c09.10k.partitioned.kept-shards``) and the number of chunks read rewrite
        (``c09.10k.round-robin.all-rewrite``), each over the rows the manifest itself names as shards or chunk files."""
        listed = question.listed_row
        group = question.group
        if listed.endswith(".action"):
            artifact = given["artifact"]
            found = manifest.read_group(group, rows, targets=targets, artifacts=[artifact])
            for action in (manifest.KEEP, manifest.REWRITE):
                if artifact in found[action]:
                    return {"action": action}
            return {"answer": manifest.NO_RECORD}
        if listed == COUNTS_ROW:
            run = given["run"]
            shards = sorted(row["name"] for row in rows if re.fullmatch(re.escape(run) + r"/shard-\d+", row["name"]))
            if not shards:
                self._not_read(question, target, f"the manifest holds no row of a shard of the run {run}")
            return dict(manifest.read_group(group, rows, targets=targets, artifacts=shards))
        if listed in (KEPT_ROW, ALL_REWRITE_ROW):
            # The artifacts are named by the manifest's own rows (docs/replica.md section 16, "Rows"): a SHARD is a row whose files are
            # its .bin and .idx; a CHUNK is the row of a chunk file, <shard>.input. The shards read keep (kept-shards), the number of
            # chunks read rewrite (all-rewrite): the table's rule text.
            if listed == KEPT_ROW and (tuple(target), "partitioned") not in self._layered:
                # The partitioned layout's shards are in the manifest of that build; the target's own manifest is the other layout's.
                self._not_read(question, target, "the world holds no manifest of the partitioned build of this set")
            wanted = shard_rows(rows) if listed == KEPT_ROW else chunk_rows(rows)
            if not wanted:
                self._not_read(question, target, "the manifest holds no row of a " + ("shard" if listed == KEPT_ROW else "chunk file"))
            held = manifest.read_group(group, rows, targets=targets, artifacts=wanted)
            if listed == KEPT_ROW:
                # Named by the file stem of the chunk file, as the plant list's counting names a shard; the registered names
                # are kept beside (a re-grade can read either).
                return {"keep": sorted(shard_name_of(rows, name) for name in held[manifest.KEEP]),
                        "registered_names": list(held[manifest.KEEP])}
            return {"keep": len(held[manifest.KEEP]), "rewrite": len(held[manifest.REWRITE])}
        self._not_read(question, target, "the table's rule text does not name the artifacts of this list")

    @staticmethod
    def _absent(question: Question) -> dict:
        return {"answer": manifest.ABSENT}


def manifest_factory(manifests: Mapping[str, Any], *, requests: Mapping[str, Mapping[str, Any]],
                     traces: Mapping[str, Any] | None = None,
                     repaired: Mapping[str, Any] | None = None) -> Callable:
    """The factory of the ``manifest`` subject. ``manifests`` maps the key of a build to the rows of
    its ``built-from.jsonl`` (None for a build made without capture); ``requests`` is the world's;
    ``repaired`` maps the key of a store to the rows of the manifest of its repaired side."""
    def factory(context):
        return ManifestDriver(manifests, requests=requests, traces=traces or {}, context=context,
                              repaired=repaired or {})

    return factory


# --- The combinations ------------------------------------------------------------------------------------


def _export_state(documents: Mapping[tuple[str, str], Any], states: Mapping[tuple[str, str], Mapping[str, Any]],
                  target, state):
    held = documents.get(tuple(target))
    if state is not None and state in states.get(tuple(target), {}):
        held = states[tuple(target)][state]
    return held


def _keyed(mapping: Mapping[str, Any]) -> dict[tuple[str, str], Any]:
    return keyed_by_target(mapping, "export")


def _not_a_run(held: Any) -> Any:
    """What the combinations take of a trace: a store with no run is a refusal (a document with no
    segment), a failed export stays its attempt."""
    if isinstance(held, Mapping) and "none" in held and "segments" not in held:
        return Refusal(exit_code=2, output=str(held["none"]))
    return held


class DolmaTraceDriver:
    """``dolma+trace`` by store or set: the toolkit's runs are the plain ``dolma`` subject's own, and
    each store or set is read by a ``DolmaTraceSubject`` of its own with its own trace."""

    name = "dolma+trace"

    def __init__(self, plain, *, traces, state_traces, place_of, context):
        self._by_target: dict[tuple[str, str], combo_dolma.DolmaTraceSubject] = {}
        keyed, states = _keyed(traces), {key: dict(value) for key, value in _keyed(state_traces).items()}
        # The runs of the plain tool by target: the run of the target's own build under the two-word key and, where the world
        # holds a build ADDED beside it, the run of each under (kind, name, layout) (W8, 18 Nov 2026); one subject reads them
        # all, a single-layout row by the run of the layout it names and a row of two layouts by each (``layout_runs``).
        grouped: dict[tuple[str, str], dict[tuple, Any]] = {}
        for key, run in plain._runs.items():
            grouped.setdefault(tuple(key[:2]), {})[tuple(key)] = run
        for target, held_runs in grouped.items():
            run = held_runs.get(target)
            if target not in keyed:
                context.keep_fact("half", target, "no trace was exported for this target: read as a store with no run")
            held = _not_a_run(keyed.get(target, Refusal(exit_code=2, output="no trace was exported")))
            for key, each in held_runs.items():
                context.keep_fact("half", key, f"dolma {'failed' if isinstance(each, Attempt) else 'ran'}",
                                  attempt_id=each.id if isinstance(each, Attempt) else None)
            if isinstance(held, Attempt):
                # The export is the attempt the combination gives where both halves failed.
                context.failed_runs[target] = held
                context.ran_on.discard(target)
                context.keep_fact("half", target, "trace export failed", attempt_id=held.id)
            layout_runs = {(target, layout): each for key, each in held_runs.items() if len(key) == 3
                           for layout in [key[2]]}
            if run is not None and not isinstance(run, Attempt):
                layout_runs.update({(target, layout): run for layout in run.layouts if (target, layout) not in layout_runs})
            self._by_target[target] = combo_dolma.DolmaTraceSubject(
                held_runs, trace=held, tie_record=plain._tie_path, requests=plain._requests,
                place_of=place_of, split_pairs=plain._split_pairs, layout_runs=layout_runs,
                state_traces={state: _not_a_run(doc) for state, doc in states.get(target, {}).items()})

    def _for(self, question: Question):
        target = tuple(record.attempt_target(question))
        if target not in self._by_target:
            raise combo_dolma.NotAttempted(f"no run of the toolkit for {target[0]} {target[1]}")
        return self._by_target[target]

    def read(self, question: Question) -> Reading | Attempt:
        return self._for(question).read(question)

    def read_by_layout(self, question: Question):
        return self._for(question).read_by_layout(question)

    def used_tie(self, question: Question) -> bool:
        target = tuple(record.attempt_target(question))
        found = self._by_target.get(target)
        return found is not None and found.used_tie(question)

    @property
    def unparseable_lines(self) -> dict[str, dict[str, int]]:
        found: dict[str, dict[str, int]] = {}
        for subject in self._by_target.values():
            found.update(subject.unparseable_lines)
        return found


def dolma_trace_factory(*, traces: Mapping[str, Any], state_traces: Mapping[str, Any] | None = None,
                        place_of) -> Callable:
    """The factory of ``dolma+trace``: the context hands it the plain ``dolma`` subject, whose runs,
    tie record, requests and split pairs it reads."""
    def factory(context):
        plain = context.plain_subject
        if plain is None:
            raise ValueError("dolma+trace needs the plain dolma subject of the same pass")
        return DolmaTraceDriver(plain, traces=traces, state_traces=state_traces or {}, place_of=place_of,
                                context=context)

    return factory


def originblame_trace_factory(*, traces: Mapping[str, Any], state_traces: Mapping[str, Any] | None = None,
                              place_of) -> Callable:
    """The factory of ``originblame+trace``: the context hands it the plain ``originblame`` subject
    (the tool half, its tracks already made) and ``served``, the function that gives the plain line
    of a question's twin."""
    def factory(context):
        plain = context.plain_subject
        if plain is None:
            raise ValueError("originblame+trace needs the plain originblame subject of the same pass")
        keyed, states = _keyed(traces), {key: dict(value) for key, value in _keyed(state_traces or {}).items()}
        for target in sorted({spec.target for spec in plain.shards}):
            held = keyed.get(target)
            if isinstance(held, Attempt):
                context.failed_runs[target] = held
                context.ran_on.discard(target)
                context.keep_fact("half", target, "trace export failed", attempt_id=held.id)
        for number, spec in enumerate(plain.shards):
            if number in plain._failed:
                context.keep_fact("shard", spec.target, "failed", attempt_id=plain._failed[number].id)

        def trace_of(listed_row: str):
            given = record._listed_givens()[listed_row]
            target = ("set", given["set"]) if "set" in given else ("store", given.get("store", "main"))
            state = given.get("state")
            held = _export_state(keyed, states, target, state)
            if held is None:
                held = Refusal(exit_code=2, output="no trace was exported")
            held = _not_a_run(held)
            if isinstance(held, Attempt):
                return held
            document = as_document(held)
            if (state in combo_dolma.TRACE_FAULTS and state not in states.get(tuple(target), {})
                    and document.get("segments")):
                return combo_dolma.faulted_trace(document, state)
            return document

        return combo_originblame.OriginBlameTrace(plain, trace_of=trace_of, place_of=place_of, served=context.served)

    return factory
