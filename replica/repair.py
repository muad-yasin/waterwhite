"""The repair driver: the replica as the team's builder for a repair (plan 4.4).

    python -m replica repair --report FILE --historical DIR --out DIR --store FILE
            (--spec FILE | --set DIR) [options]

A rebuild report that ``waterwhite report`` printed is read as a JSON file with the
``json`` module and nothing else: this module imports nothing of the tool and no
module of the tool's rebuild, coverage or answer code (docs/interfaces.md section 1).
From the report it makes three things:

* the repaired side: the same build as the historical one, with the records the
  report retires left out and the withdrawn parent of a re-sourced row swapped for
  the equivalent parent that remains, with capture on into the historical store and
  with the name prefix ``successor of ``, so the successors register beside the
  historical artifacts under the names of docs/interfaces.md section 9.2;
* the reference side: a build with capture off from the sources that never held the
  withdrawn content (a generated set's clean folder; for the fixture, the spec with
  everything derived from the withdrawn source left out);
* the sides file ``waterwhite verify`` reads (docs/interfaces.md section 7.2).

The report says what to do; the replica never decides what is in scope. A record is
told apart by its own id, found through the manifest that lies beside the record
file of the historical build, because the tool stores no id (plan Check 3).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import megatron_tokenizer
from .build import (
    HARNESS_IN_PROCESS,
    HARNESSES,
    BuildOptions,
    RecordChanges,
    build,
    check_build,
    require_new_folder,
)
from .errors import CaptureError, InputError, OutputError, ReplicaError, SpecError
from .plan import (
    LAYOUTS,
    BuildPlan,
    load_fixture_spec,
    load_set,
    load_tokenizer_settings,
)
from .shard_index import read_shard_index

# The words of the report and of the sides file. The replica does not import the
# tool's vocabulary outside its capture hooks (docs/interfaces.md section 1), so the
# words it reads are spelled here, and a test compares them with the vocabulary.
REPORT_FORM = "waterwhite.report/1"
SIDES_FORM = "waterwhite.sides/1"
SUCCESSOR_PREFIX = "successor of "

ACTION_KEEP = "keep"
ACTION_RE_SOURCE = "re-source"
ACTION_PROMOTE = "promote"
ACTION_RETIRE = "retire"
ACTION_UNRESOLVED = "unresolved"
RECORD_ACTIONS = (ACTION_KEEP, ACTION_RE_SOURCE, ACTION_PROMOTE, ACTION_RETIRE,
                  ACTION_UNRESOLVED)

ARTIFACT_KEEP = "keep"
ARTIFACT_REWRITE = "rewrite"
ARTIFACT_CLEAN_PLACE_NOT_STATED = "clean, place not stated"
ARTIFACT_UNRESOLVED = "unresolved"
ARTIFACT_ACTIONS = (ARTIFACT_KEEP, ARTIFACT_REWRITE, ARTIFACT_CLEAN_PLACE_NOT_STATED,
                    ARTIFACT_UNRESOLVED)
# What a build holds for an artifact of these actions is named in the sides file: the
# fourth word is among them, a shard of it is built by its layout like any other.
SIDES_ACTIONS = (ARTIFACT_KEEP, ARTIFACT_REWRITE, ARTIFACT_CLEAN_PLACE_NOT_STATED)

KIND_JSONL = "jsonl"
KIND_SHARD = "shard"

BOTH = "both"

_LOCATOR = re.compile(r"(L[1-9][0-9]*|D(?:0|[1-9][0-9]*))")


# --- Reading the report ---------------------------------------------------------------


def read_report(path: Path) -> dict:
    """The report as a JSON file, and nothing of it but what the replica reads:
    ``form``, ``artifacts`` and ``records``, and the shards named under
    ``groups.placements[NAME].landing`` for the sides file. A file that cannot be read, whose form
    is another, or whose entries lack a key the replica needs is refused."""
    try:
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
    except (OSError, ValueError) as exc:
        raise InputError(f"the report {path} cannot be read: {exc}") from None
    if not isinstance(document, dict) or document.get("form") != REPORT_FORM:
        found = document.get("form") if isinstance(document, dict) else type(document).__name__
        raise InputError(f"{path} is not a rebuild report: its form is {found!r}, the replica "
                         f"reads {REPORT_FORM!r}")
    for key in ("artifacts", "records"):
        if not isinstance(document.get(key), list) or not all(
                isinstance(entry, dict) for entry in document[key]):
            raise InputError(f"the report {path} has no list of entries under {key!r}")
    for entry in document["artifacts"]:
        _text(entry, "artifact", "an entry of artifacts")
        if entry.get("action") not in ARTIFACT_ACTIONS:
            raise InputError(f"artifact {entry.get('artifact')!r}: the action "
                             f"{entry.get('action')!r} is not one of {ARTIFACT_ACTIONS}")
    for entry in document["records"]:
        where = f"the record entry {entry.get('record')!r}"
        _text(entry, "record", "an entry of records")
        _text(entry, "artifact", where)
        if entry.get("action") not in RECORD_ACTIONS:
            raise InputError(f"{where}: the action {entry.get('action')!r} is not one of "
                             f"{RECORD_ACTIONS}")
        if entry["action"] == ACTION_RE_SOURCE:
            _text(entry, "withdrawn_parent", where)
            _text(entry, "new_parent", where)
    return document


def _text(entry: Mapping, key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or value == "":
        raise InputError(f"{where} has no text under {key!r}")
    return value


# --- The historical build: a record's place to its id ------------------------------------


class Historical:
    """The folder of the build the store's historical artifacts were made from.

    The report names a record as ``artifact:locator`` (``R1:L2``, ``src/B:L1``). The
    replica gives the record an id the tool never sees, and the manifest beside the
    file of the artifact says which: a source line, a release line and an import line
    carry the id of the record they copy in ``origin``, a generation output row
    carries its own id in its ``id`` field (its ``origin`` is what the job used)."""

    def __init__(self, folder: Path, name_prefix: str = "",
                 shard_prefixes: Mapping[str, str] | None = None) -> None:
        self.folder = Path(folder)
        self.name_prefix = name_prefix
        # The store-name prefix of each layout's shards ("E") to the layout's name
        # ("partitioned"): a shard's documents are found through its layout.
        self._shard_prefixes = dict(shard_prefixes or {})
        self._ids: dict[str, list[str | None]] = {}
        if not self.folder.is_dir():
            raise InputError(f"the historical build {folder} is not a folder")

    def file_of(self, artifact: str) -> str:
        """The record file of a registered artifact, relative to the build folder.
        The names are those of docs/interfaces.md section 9.2."""
        if not artifact.startswith(self.name_prefix):
            raise InputError(f"artifact {artifact!r} does not begin with the historical name "
                             f"prefix {self.name_prefix!r}")
        name = artifact[len(self.name_prefix):]
        if name.startswith("src/"):
            return f"sources/{name[4:]}.jsonl"
        if name.startswith("gen/"):
            return f"{name}.jsonl"
        if name.startswith("import/"):
            return f"imports/{name.split('/')[-1]}.jsonl"
        return f"releases/{name}.jsonl"

    def _ids_of(self, artifact: str) -> list[str | None]:
        if artifact in self._ids:
            return self._ids[artifact]
        relative = self.file_of(artifact)
        path = self.folder / relative
        if not path.is_file():
            raise InputError(f"the historical build holds no file {relative} for artifact "
                             f"{artifact!r}")
        if relative.startswith("gen/"):
            ids = [self._row_id(path, number, line)
                   for number, line in enumerate(_lines(path), 1)]
        else:
            manifest = Path(f"{path}.manifest.jsonl")
            if not manifest.is_file():
                raise InputError(f"{relative} has no manifest beside it")
            ids = []
            for number, line in enumerate(_lines(manifest), 1):
                row = _json(line, f"{manifest} line {number}")
                origin = row.get("origin") if isinstance(row, dict) else None
                ids.append(origin[0] if isinstance(origin, list) and len(origin) == 1
                           and isinstance(origin[0], str) else None)
            if len(ids) != sum(1 for _ in _lines(path)):
                raise InputError(f"{relative} and its manifest hold different numbers of lines")
        self._ids[artifact] = ids
        return ids

    @staticmethod
    def _row_id(path: Path, number: int, line: str) -> str | None:
        row = _json(line, f"{path} line {number}")
        value = row.get("id") if isinstance(row, dict) else None
        return value if isinstance(value, str) else None

    def set_shard_prefixes(self, shard_prefixes: Mapping[str, str]) -> None:
        self._shard_prefixes = dict(shard_prefixes)

    def shard_of(self, artifact: str) -> tuple[str, str] | None:
        """The layout and the file name of a shard by its store name, or None."""
        for prefix, layout in self._shard_prefixes.items():
            full = f"{self.name_prefix}{prefix}/"
            if artifact.startswith(full):
                return layout, artifact[len(full):]
        return None

    def _document_ids(self, artifact: str) -> list[str | None]:
        """The id of each document of a shard, in document order: the tokenizer input's
        manifest gives the record of each line, and the shard's index says which
        lines gave tokens (a line that gave none is no document)."""
        if artifact in self._ids:
            return self._ids[artifact]
        found = self.shard_of(artifact)
        if found is None:
            raise InputError(f"{artifact!r} is no shard of a layout of this build "
                             f"(layout prefixes: {sorted(self._shard_prefixes)})")
        layout, shard = found
        base = self.folder / "pipeline" / layout
        manifest = base / "tokenizer-input" / f"{shard}.jsonl.manifest.jsonl"
        index = base / "shards" / f"{shard}.idx"
        if not manifest.is_file() or not index.is_file():
            raise InputError(f"the historical build holds no tokenized shard {shard} of layout "
                             f"{layout} for artifact {artifact!r}")
        ids = []
        for number, line in enumerate(_lines(manifest), 1):
            row = _json(line, f"{manifest} line {number}")
            origin = row.get("origin") if isinstance(row, dict) else None
            ids.append(origin[0] if isinstance(origin, list) and len(origin) == 1
                       and isinstance(origin[0], str) else None)
        tokens = read_shard_index(index).document_tokens
        if len(tokens) != len(ids):
            raise InputError(f"shard {shard} holds {len(tokens)} documents and its tokenizer "
                             f"input {len(ids)} lines")
        documents = [i for i, count in zip(ids, tokens) if count]
        self._ids[artifact] = documents
        return documents

    def id_of(self, record: str) -> str:
        """The id of the record ``artifact:locator``: ``L<n>`` is line n of a record
        file, counted from 1, and ``D<n>`` is document n of a shard, counted from 0.
        A line or document the file does not have and a record with no single id are
        refused: nothing is guessed."""
        artifact, separator, locator = record.rpartition(":")
        if not separator or not artifact:
            raise InputError(f"{record!r} is not written ARTIFACT:LOCATOR")
        if _LOCATOR.fullmatch(locator) is None:
            raise InputError(f"{record!r}: the locator {locator!r} is neither a line (L<n>, "
                             "from 1) nor a document (D<n>, from 0)")
        number = int(locator[1:])
        if locator[0] == "D":
            ids, what, value = self._document_ids(artifact), "documents", number
        else:
            if self.shard_of(artifact) is not None:
                raise InputError(f"{record!r}: a shard holds documents (D<n>), not lines")
            ids, what, value = self._ids_of(artifact), "lines", number - 1
        if not 0 <= value < len(ids):
            raise InputError(f"{record!r}: {artifact} has {len(ids)} {what}")
        found = ids[value]
        if found is None:
            raise InputError(f"{record!r}: no single record id stands behind that place (a "
                             "line the build wrote into a chunk, or a row with no id)")
        return found

    def layouts(self) -> tuple[str, ...]:
        """The layouts the historical build made: those with a folder under
        ``pipeline/``."""
        return tuple(name for name in LAYOUTS if (self.folder / "pipeline" / name).is_dir())

    def tokenized(self) -> bool:
        """Whether the historical build tokenized its shards."""
        return any((self.folder / "pipeline" / name / "shards").is_dir() and any(
            (self.folder / "pipeline" / name / "shards").glob("*.bin")) for name in LAYOUTS)

    def releases(self) -> tuple[str, ...]:
        """The releases the historical build wrote, by name."""
        return tuple(sorted(
            path.name[:-len(".jsonl")] for path in (self.folder / "releases").glob("*.jsonl")
            if not path.name.endswith((".dedup.jsonl", ".manifest.jsonl"))))

    def has_imports(self, names: Sequence[str]) -> tuple[str, ...]:
        """Those of the import names of a plan whose file the historical build wrote."""
        return tuple(name for name in names
                     if (self.folder / "imports" / f"{name.split('/')[-1]}.jsonl").is_file())

    def generation_captured(self) -> bool:
        """Whether the historical build ran its generation jobs in a harness: it kept
        their logs. A build that wrote generation outputs and no log left that step
        uncaptured."""
        return (self.folder / "gen" / "harness").is_dir() or not any(
            path for path in (self.folder / "gen").glob("*.jsonl")
            if not path.name.endswith(".manifest.jsonl"))

    def harness(self) -> str:
        """The harness the historical build ran its generation jobs in, from the first
        job entry of a harness log (the word is in every log, DECISIONS.md D-30);
        in-process where the build kept no log."""
        for path in sorted((self.folder / "gen" / "harness").glob("*/*.log.jsonl")):
            lines = _lines(path)
            row = _json(lines[0], str(path)) if lines else None
            if isinstance(row, dict) and row.get("kind") == "job" and "harness" in row:
                return row["harness"]
        return HARNESS_IN_PROCESS


def _lines(path: Path) -> list[str]:
    """The lines of a file as the tool counts them: a line ends at a line feed only."""
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _json(text: str, where: str):
    try:
        return json.loads(text)
    except ValueError as exc:
        raise InputError(f"{where} is not JSON: {exc}") from None


# --- What the report asks of the build ------------------------------------------------------


@dataclass(frozen=True)
class Repair:
    """What a report asks the repaired build to do.

    ``leave_out`` are the ids of the records the report retires. ``swap`` maps the id
    of a withdrawn parent to the id of the equivalent parent that takes its place.
    ``unresolved`` counts the entries the report lists as unresolved: they change
    nothing in the build. ``promoted`` counts the promote entries: they need no
    parameter of their own."""

    leave_out: tuple[str, ...]
    swap: Mapping[str, str]
    unresolved: int = 0
    promoted: int = 0


def plan_repair(report: Mapping, historical: Historical) -> Repair:
    """Read the actions of the report's records, and only those: never the lists
    ``removed`` or the counts of an artifact, so that a report edited by hand drives
    the build as edited.

    * retire: ``--leave-out`` of the record's id;
    * re-source: ``--swap OLD=NEW``, the withdrawn parent for the new parent (where
      the build selects both, because a dedup group selects every copy of its text,
      the old one just leaves: docs/replica.md section 8);
    * promote, keep and unresolved: nothing. A promote is the other half of a
      re-source;
    * one id with the action retire in one entry and keep in another is refused.
    """
    retired: dict[str, str] = {}
    kept: dict[str, str] = {}
    swap: dict[str, str] = {}
    unresolved = promoted = 0
    for entry in report["records"]:
        action, record = entry["action"], entry["record"]
        if action == ACTION_RETIRE:
            retired.setdefault(historical.id_of(record), record)
        elif action == ACTION_KEEP:
            kept.setdefault(historical.id_of(record), record)
        elif action == ACTION_RE_SOURCE:
            old = historical.id_of(entry["withdrawn_parent"])
            new = historical.id_of(entry["new_parent"])
            if old == new:
                raise InputError(f"{record!r}: the withdrawn and the new parent are one record "
                                 f"({old})")
            if swap.setdefault(old, new) != new:
                raise InputError(f"{record!r}: record {old} is re-sourced to {swap[old]} and to "
                                 f"{new}")
        elif action == ACTION_PROMOTE:
            promoted += 1
        else:
            unresolved += 1
    both = sorted(set(retired) & set(kept))
    if both:
        raise InputError("the report contradicts itself: " + "; ".join(
            f"record {i} is retired at {retired[i]} and kept at {kept[i]}" for i in both))
    return Repair(tuple(sorted(retired)), dict(sorted(swap.items())), unresolved, promoted)


# --- The reference side ----------------------------------------------------------------------


def derived_from(plan: BuildPlan, withdrawn: str) -> tuple[str, ...]:
    """The ids of every record of the source ``withdrawn`` and of every generation
    output made, directly or through other outputs, from them, taken from the spec's
    own lists of what each output used. The report is never read for this: the
    reference side must stand on something the report cannot change."""
    ids = {record for source in plan.sources if withdrawn in (source.key, source.source)
           for record in source.records}
    if not ids:
        raise SpecError(f"the spec names no source {withdrawn!r}; its sources are "
                        f"{sorted(s.key for s in plan.sources)}")
    rows = [row for gfile in plan.generation for row in gfile.rows]
    grown = True
    while grown:
        grown = False
        for row in rows:
            if row.id not in ids and any(used in ids for used in row.used):
                ids.add(row.id)
                grown = True
    return tuple(sorted(ids))


# --- The driver ---------------------------------------------------------------------------------


def _plan(spec: Path | None, set_dir: Path | None, tokenizer_spec: Path | None,
          tokenize: bool, base: Path | None, what: str) -> BuildPlan:
    if (spec is None) == (set_dir is None):
        raise SpecError(f"{what}: give exactly one of a spec (the fixture) and a set folder")
    if spec is not None:
        return load_fixture_spec(Path(spec), None if base is None else Path(base))
    tokenizer = None
    if tokenize:
        if tokenizer_spec is None:
            raise SpecError("a generated set names no tokenizer: give a tokenizer spec "
                            "(a file with a [tokenizer] table, such as the fixture spec) or "
                            "build without tokenization")
        tokenizer = load_tokenizer_settings(Path(tokenizer_spec))
    return load_set(Path(set_dir), tokenizer)


def _layouts(layout: str) -> tuple[str, ...]:
    if layout == BOTH:
        return LAYOUTS
    if layout not in LAYOUTS:
        raise SpecError(f"layout {layout!r}; the layouts are {LAYOUTS} and {BOTH!r}")
    return (layout,)


def _inside(path: Path, folder: Path) -> bool:
    path, folder = Path(os.path.abspath(path)), Path(os.path.abspath(folder))
    return path == folder or folder in path.parents


def _require_a_store(store: Path) -> None:
    """A file that is not a store is the replica's own input error, and nothing is
    created or changed to find that out. The file is read through a connection that
    cannot write: one that does not look at a write-ahead log when none lies beside
    the file, so that no sibling file is made either. A store of any schema version
    passes (the store's own open migrates or refuses it); a file with no ``meta`` table
    that holds a schema version does not."""
    uri = Path(os.path.abspath(store)).as_uri()
    quiet = not Path(f"{os.path.abspath(store)}-wal").exists()
    connection = None
    try:
        connection = sqlite3.connect(f"{uri}?{'immutable=1' if quiet else 'mode=ro'}", uri=True)
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.Error as exc:
        raise InputError(f"the store {store} is not a store: {' '.join(str(exc).split())}") from None
    finally:
        if connection is not None:
            connection.close()
    if row is None:
        raise InputError(f"the store {store} is not a store: it holds no schema version")


def repair(
    report: Path,
    historical: Path,
    out: Path,
    store: Path,
    *,
    spec: Path | None = None,
    set_dir: Path | None = None,
    tokenizer_spec: Path | None = None,
    base: Path | None = None,
    layout: str | None = None,
    tokenize: bool | None = None,
    seeded: bool = True,
    shuffle_seeds: Mapping[str, int] | None = None,
    parallel: int = 1,
    tokenizer_timeout: float = 900.0,
    generation_harness: str | None = None,
    historical_name_prefix: str = "",
    reference_set: Path | None = None,
    reference_withdrawn: str | None = None,
    reference_out: Path | None = None,
    sides: Path | None = None,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> dict:
    """Build the repaired side of ``report`` into ``out``, with capture on into
    ``store`` (the historical store) under the name prefix ``successor of ``; build
    the reference side into ``reference_out``; write the sides file.

    ``store`` must be the historical store, a file that exists and is a store: a path
    that holds none, or a file that is not a store, is refused before anything is built
    or created, and the file is not changed. ``historical`` is the folder of the
    build the store's historical artifacts were made from, and ``spec`` (the fixture) or ``set_dir`` (a generated set) what that
    build was made from. The reference side needs ``reference_out`` and, for a set,
    ``reference_set``, the folder the generator wrote without the withdrawn content;
    for the fixture ``reference_withdrawn``, the name of the withdrawn source, given
    by the caller.

    The repaired side is the same build as the historical one: ``layout`` ("both" or
    one layout), ``tokenize``, the releases, the imports and whether the generation
    step is captured default to what the historical folder holds, and
    ``generation_harness`` to the one its harness logs name (the store refuses a job
    name whose stored log differs, and the harness word is in the log). A caller
    names them to build something else. Returns the summary the command prints."""
    report_path, out, store = Path(report), Path(out), Path(store)
    document = read_report(report_path)
    hist = Historical(Path(historical), historical_name_prefix)
    layouts = hist.layouts() if layout is None else _layouts(layout)
    if tokenize is None:
        tokenize = hist.tokenized()
    harness = generation_harness or hist.harness()
    if harness not in HARNESSES:
        raise SpecError(f"generation harness {harness!r}; the harnesses are {HARNESSES}")
    plan = _plan(spec, set_dir, tokenizer_spec, tokenize, base, "the repaired side")
    hist.set_shard_prefixes({layout_plan.store_prefix: layout_plan.name
                             for layout_plan in plan.layouts.values()})
    changes = plan_repair(document, hist)
    if sides is not None:
        # The refusal of a shard that lies in no layout reads the report and the layouts
        # only, so it comes before either side is built: a refused repair registers no
        # successor and writes no folder (D-120, D-149).
        refuse_shards_in_no_layout(document, Layouts(plan, hist.name_prefix))
    reference_plan = _reference_plan(plan, spec, set_dir, tokenizer_spec, tokenize, base,
                                     reference_set, reference_withdrawn, reference_out)

    # Everything the build would refuse is refused before the store or a folder is
    # touched: a refused repair leaves no store behind.
    if _inside(store, out) or (reference_out is not None and _inside(store, reference_out)):
        raise OutputError(f"the store file {store} lies inside a build folder; a build "
                          "folder holds the build's files only")
    if sides is not None:
        for folder in (out, reference_out):
            if folder is not None and _inside(sides, folder):
                raise OutputError(f"the sides file {sides} lies inside the build folder "
                                  f"{folder}")
        if Path(sides).exists():
            raise OutputError(f"the sides file {sides} exists already")
    if reference_out is not None:
        require_new_folder(reference_out, "the reference side")
    if megatron_env is None and tokenize:
        megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    seeds = dict(shuffle_seeds or {})
    releases = hist.releases()
    imports = hist.has_imports([imported.name for imported in plan.imports])
    options = BuildOptions(
        layouts=layouts, seeded=seeded, shuffle_seeds=seeds, tokenize=tokenize,
        parallel=parallel, tokenizer_timeout=tokenizer_timeout,
        changes=RecordChanges(leave_out=frozenset(changes.leave_out), swap=dict(changes.swap)),
        releases=releases, imports=imports, capture_generation=hist.generation_captured(),
        store_name_prefix=SUCCESSOR_PREFIX + historical_name_prefix,
        generation_harness=harness)
    check_build(plan, out, options, capturing=True, megatron_env=megatron_env)
    reference_options = None
    if reference_plan is not None:
        ref_changes = RecordChanges(
            leave_out=frozenset(derived_from(plan, reference_withdrawn))
            if reference_withdrawn is not None else frozenset())
        reference_options = BuildOptions(
            layouts=layouts, seeded=seeded, shuffle_seeds=seeds, tokenize=tokenize,
            parallel=parallel, tokenizer_timeout=tokenizer_timeout, changes=ref_changes,
            releases=releases, imports=imports)
        check_build(reference_plan, reference_out, reference_options, capturing=False,
                    megatron_env=megatron_env)

    # The last refusal before the store is opened: opening a path that holds no store
    # would create one, so a mistyped path is refused here and nothing is created.
    if not store.is_file():
        raise InputError(f"the store {store} holds no store: the successors are registered "
                         "in the historical store, and opening a path would create an empty one")
    _require_a_store(store)

    # The store, the capture wrappers and the harness are imported here and only
    # when capture is on.
    from .capture_hooks import StoreCapture

    capture = StoreCapture.open(store, normaliser=plan.normaliser)
    try:
        repaired = build(plan, out, options, capture=capture, megatron_env=megatron_env)
    except ReplicaError:
        raise
    except Exception as exc:
        if not type(exc).__module__.startswith("waterwhite"):
            raise
        # What the tool's store or wrappers refuse (a successor name that is taken by
        # an instance of another history, a second repair of one store) is a refused
        # repair, in one line, and not a traceback.
        raise CaptureError(f"the store refused the repaired side: {type(exc).__name__}: "
                           f"{exc}") from exc
    finally:
        capture.close()
    reference = None
    if reference_plan is not None:
        reference = build(reference_plan, Path(reference_out), reference_options,
                          megatron_env=megatron_env)

    printed = {
        "report": str(report_path),
        "left_out": list(changes.leave_out),
        "swapped": dict(changes.swap),
        "unresolved": changes.unresolved,
        "promoted": changes.promoted,
        "harness": harness,
        "name_prefix": options.store_name_prefix,
        **_registered(document, repaired.get("instances", {}), options.store_name_prefix),
        "repaired": repaired,
        "reference": reference,
        "sides": None,
    }
    if sides is not None:
        printed["sides"] = str(write_sides(Path(sides), document, hist, plan, out, reference_out))
    return printed


def _reference_plan(plan, spec, set_dir, tokenizer_spec, tokenize, base, reference_set,
                    reference_withdrawn, reference_out) -> BuildPlan | None:
    given = [x is not None for x in (reference_set, reference_withdrawn, reference_out)]
    if not any(given):
        return None
    if reference_out is None:
        raise SpecError("a reference side needs its own output folder")
    if spec is not None:
        if reference_withdrawn is None or reference_set is not None:
            raise SpecError("the reference side of the fixture is built from the spec with "
                            "everything derived from the withdrawn source left out: give the "
                            "name of the withdrawn source, and no reference set")
        return plan
    if reference_set is None or reference_withdrawn is not None:
        raise SpecError("the reference side of a generated set is built from the folder the "
                        "generator wrote without the withdrawn content: give that folder, and "
                        "no withdrawn source")
    return _plan(None, reference_set, tokenizer_spec, tokenize, base, "the reference side")


def _registered(document: Mapping, instances: Mapping[str, str], prefix: str) -> dict:
    """What the repaired build registered, by historical name: the successor of every
    artifact the report rewrites, and for every artifact it keeps whether the file is
    the same instance under its second name (DECISIONS.md D-14)."""
    successors: dict[str, str] = {}
    missing: list[str] = []
    kept: dict[str, dict] = {}
    for entry in document["artifacts"]:
        name, action = entry["artifact"], entry["action"]
        registered = SUCCESSOR_PREFIX + name
        if action == ARTIFACT_REWRITE:
            if registered in instances:
                successors[name] = registered
            else:
                missing.append(name)
        elif action == ARTIFACT_KEEP:
            found, recorded = instances.get(registered), entry.get("instance")
            kept[name] = {"registered_as": registered if found is not None else None,
                          "same_instance": None if found is None or recorded is None
                          else found == recorded}
    return {"successors": successors, "successors_not_registered": missing, "kept": kept}


# --- The sides file --------------------------------------------------------------------------------


class Layouts:
    """Where a shard of a build lies, by its store name."""

    def __init__(self, plan: BuildPlan, name_prefix: str) -> None:
        self._by_prefix = {name_prefix + layout.store_prefix + "/": layout.name
                           for layout in plan.layouts.values()}

    def prefixes(self) -> list[str]:
        return list(self._by_prefix)

    def prefix_of(self, artifact: str) -> str | None:
        """The shard's file prefix relative to a build folder, or None."""
        for store_prefix, layout in self._by_prefix.items():
            if artifact.startswith(store_prefix):
                return f"pipeline/{layout}/shards/{artifact[len(store_prefix):]}"
        return None


def _in_no_layout(name: str, layouts: Layouts) -> InputError:
    return InputError(f"shard {name!r} lies in no layout of the build "
                      f"(layout prefixes: {sorted(layouts.prefixes())})")


def refuse_shards_in_no_layout(document: Mapping, layouts: Layouts) -> None:
    """Refuse a shard of the report that lies in no layout of the build. It reads the
    report and the layouts and nothing a build makes, so ``repair`` asks it before either
    side is built (docs/interfaces.md 9.5); ``_side`` keeps its own refusal for a caller
    that writes a sides file from builds it made itself. The shards are those the sides
    file names a file for: an entry of kind "shard" that is kept, rewritten or reads the
    fourth word, and then each shard the report names under a landing, which is a shard
    the report names as well."""
    for entry in document["artifacts"]:
        if entry["action"] in SIDES_ACTIONS and entry.get("kind") == KIND_SHARD \
                and layouts.prefix_of(entry["artifact"]) is None:
            raise _in_no_layout(entry["artifact"], layouts)
    for name in landing_shards(document):
        if layouts.prefix_of(name) is None:
            raise _in_no_layout(name, layouts)


def landing_shards(document: Mapping) -> list[str]:
    """The shards the report names under ``groups.placements[NAME].landing``, in the
    report's order, each once. A report without the section names none."""
    placements = (document.get("groups") or {}).get("placements") or {}
    names: list[str] = []
    for section in placements.values():
        for item in section.get("landing") or []:
            shard = item.get("shard")
            if isinstance(shard, str) and shard and shard not in names:
                names.append(shard)
    return names


def _side(document: Mapping, hist: Historical, layouts: Layouts, folder: Path) -> dict[str, Path]:
    """The file or shard prefix a build holds for each artifact of the report that
    is kept or rewritten, as an absolute path; an artifact the build did not write
    (a shard no record reaches) is left out. An entry that names no ``kind`` is
    looked for as a file and then as a shard; a mixture and every other kind has no
    file of its own. After those, each shard the report names under
    ``groups.placements[NAME].landing`` that the build holds is named too, found by its
    layout prefix as a shard of the report is; a landing shard that the build does not hold
    is left out, and one that lies in no layout is refused as a shard of the report is."""
    named: dict[str, Path] = {}
    for entry in document["artifacts"]:
        name, kind = entry["artifact"], entry.get("kind")
        if entry["action"] not in SIDES_ACTIONS:
            continue
        if kind in (None, KIND_JSONL):
            path = folder / hist.file_of(name)
            if path.is_file():
                named[name] = path
                continue
        if kind in (None, KIND_SHARD):
            relative = layouts.prefix_of(name)
            if relative is None:
                if kind == KIND_SHARD:
                    raise _in_no_layout(name, layouts)
                continue
            prefix = folder / relative
            if Path(f"{prefix}.bin").is_file() and Path(f"{prefix}.idx").is_file():
                named[name] = prefix
    for name in landing_shards(document):
        if name in named:
            continue
        relative = layouts.prefix_of(name)
        if relative is None:
            raise _in_no_layout(name, layouts)
        prefix = folder / relative
        if Path(f"{prefix}.bin").is_file() and Path(f"{prefix}.idx").is_file():
            named[name] = prefix
    return named


def write_sides(path: Path, document: Mapping, hist: Historical, plan: BuildPlan,
                repaired_out: Path, reference_out: Path | None) -> Path:
    """Write the sides file of docs/interfaces.md section 7.2: for every artifact
    of the report that is kept or rewritten, and for every shard the report names under
    a layout's landing that the build holds, the file (or shard prefix) the repaired
    side holds and the one the reference side holds, relative to the folder of the
    sides file. A mixture has no file of its own and is not named."""
    layouts = Layouts(plan, hist.name_prefix)
    folder = Path(os.path.abspath(path)).parent

    def relative(paths: Mapping[str, Path]) -> dict[str, str]:
        return {name: os.path.relpath(os.path.abspath(p), folder) for name, p in paths.items()}

    repaired = _side(document, hist, layouts, Path(repaired_out))
    reference = {} if reference_out is None else _side(document, hist, layouts,
                                                       Path(reference_out))
    text = json.dumps({"form": SIDES_FORM, "repaired": relative(repaired),
                       "reference": relative(reference)}, indent=2) + "\n"
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except OSError as exc:
        raise OutputError(f"the sides file {path} cannot be written: {exc}") from None
    return Path(path)


__all__: Sequence[str] = (
    "Historical", "Repair", "derived_from", "plan_repair", "read_report", "refuse_shards_in_no_layout",
    "repair", "write_sides",
)
