"""One build of the replica: from source files to releases, shards and mixtures.

The folder a build writes (docs/replica.md gives every file):

    sources/<KEY>.jsonl                          the source files, copied byte for byte
    gen/<NAME>.jsonl                             the generation stand-in's fixed outputs
    releases/<NAME>.jsonl                        rows {"text": ...} only
    imports/<NAME>.jsonl                         a text-only import
    pipeline/<layout>/sanitizer-input/<shard>.jsonl   a chunk as the sanitizer reads it
    pipeline/<layout>/sanitizer.log.jsonl        one row per skipped line
    pipeline/<layout>/tokenizer-input/<shard>.jsonl   sanitized, surviving lines
    pipeline/<layout>/shards/<shard>.bin, .idx   Megatron's output
    mixtures/<name>/                             the shard files a mixture selects
    <file>.manifest.jsonl                        beside every record file above
    built-from.jsonl                             the file-level manifest (capture on only)

Capture is a switch. With ``capture`` None the build calls nothing of the tool but
its normaliser (DECISIONS.md D-17). With a capture object
(``replica.capture_hooks.StoreCapture``) the build calls it at the capture points
of plan 5.4: ingest, the generation jobs, the dedup selection, each tokenized shard
and each mixture; and at the text-only import, which plan 5.4 does not list: the
fixture holds it (plan 5.1) as a derivation that was never recorded (plan L1). The
copy steps and the sanitizer are recovery points (plan 5.4): the build states
nothing about the files they write, and their links are recovered from the files at
the tokenization capture point.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import megatron_tokenizer
from .built_from import BuiltFrom
from .errors import InputError, OutputError, ReplicaError, SpecError
from .plan import (
    LAYOUTS,
    MODE_FIXTURE,
    BuildPlan,
    LayoutPlan,
)
from .records import Record, parse_line, read_records
from .shard_files import shard_files
from .shard_index import read_shard_index
from .steps import (
    SeededShuffler,
    UnseededShuffler,
    dedup,
    normaliser,
    partitioned_shards,
    round_robin_chunks,
    sanitize,
)

REASON_DEDUP_LOSER = "dedup loser"

MIXTURE_COPY = "copy"
MIXTURE_SYMLINK = "symlink"

# How a captured generation job runs. The two words are spelled once more in
# ``replica.capture_hooks``, which a build with capture off never loads.
HARNESS_IN_PROCESS = "in-process"
HARNESS_CONFINED = "confined"
HARNESSES = (HARNESS_IN_PROCESS, HARNESS_CONFINED)

# The handle a manifest gives the line the spec writes into a chunk
# (``[sanitizer_skip]``): it is no record of any source.
HANDLE_INSERTED = "sanitizer_skip"

# Scratch of a build: the tokenizer's input copies and logs. Not part of the result
# and not reproducible byte for byte (the logs hold timings).
WORK_DIR = ".work"


@dataclass(frozen=True)
class RecordChanges:
    """Records a build leaves out or swaps, by id (plan 4.4).

    A record in ``leave_out`` is written to no release, no import and no shard
    input. ``swap`` maps an id to the id of the record that takes its place, and
    its group in a layout, wherever the first was selected. Source files and
    generation outputs are built as their inputs say."""

    leave_out: frozenset[str] = frozenset()
    swap: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for old, new in self.swap.items():
            if old in self.leave_out or new in self.leave_out:
                raise SpecError(f"record {old!r} or {new!r} is both swapped and left out")
            if new in self.swap:
                raise SpecError(f"record {new!r} is swapped in and swapped out")


@dataclass(frozen=True)
class BuildOptions:
    """How one build runs.

    ``seeded`` False draws every shuffle from the operating system's randomness.
    ``shuffle_seeds`` replaces a layout's seed from the spec. ``parallel`` is the
    number of tokenizer runs at a time. ``releases`` and ``imports`` name the
    releases and imports to build (None: every one the plan holds). A generated
    set's pipeline needs its release ``R1`` among them.
    ``store_name_prefix`` is put before every name the capture registers, so a
    second build can share a store. ``capture_generation`` False, with capture on,
    leaves the generation step uncaptured: the output files are written as with
    capture off, no job runs in the harness and no output file is registered, so
    the store holds nothing about them. A release that selects a generated record
    cannot be captured then and is refused. ``generation_harness`` says how a
    captured generation job runs: "in-process" (the default), or "confined", as a
    program of its own in the tool's sandbox, with ``generation_timeout`` seconds
    for one job; the confined harness needs capture and is refused where its
    sandbox cannot be built."""

    layouts: tuple[str, ...] = LAYOUTS
    seeded: bool = True
    shuffle_seeds: Mapping[str, int] = field(default_factory=dict)
    tokenize: bool = True
    parallel: int = 1
    tokenizer_timeout: float = 900.0
    changes: RecordChanges = field(default_factory=RecordChanges)
    mixture_files: str = MIXTURE_COPY
    releases: tuple[str, ...] | None = None
    imports: tuple[str, ...] | None = None
    store_name_prefix: str = ""
    capture_generation: bool = True
    generation_harness: str = HARNESS_IN_PROCESS
    generation_timeout: float = 120.0


def json_row(row: dict) -> str:
    """One JSON line as the replica writes it: ASCII only, so no character of a
    text can be taken for a line break by a later reader."""
    return json.dumps(row, ensure_ascii=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def tree_digests(out: Path) -> dict[str, str]:
    """SHA-256 of every file of a build, by its path relative to the build folder,
    without the scratch folder. Two seeded builds of one input give equal maps."""
    out = Path(out)
    digests = {}
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if relative.parts[0] == WORK_DIR or not path.is_file():
            continue
        digests[relative.as_posix()] = file_sha256(path)
    return digests


def build(
    plan: BuildPlan,
    out: Path,
    options: BuildOptions | None = None,
    *,
    capture=None,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> dict:
    """Run one build into ``out``, which must be new or empty. Returns a summary
    that can be written as JSON."""
    return _Build(plan, Path(out), options or BuildOptions(), capture, megatron_env).run()


def check_build(
    plan: BuildPlan,
    out: Path,
    options: BuildOptions | None = None,
    *,
    capturing: bool = False,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> None:
    """Refuse what ``build`` would refuse before it writes anything, and write
    nothing: an option the plan cannot carry out, a missing Megatron environment,
    an output folder that is not new or empty. A caller that creates a store for
    the build calls this first, so that a refused build leaves no store file
    behind. ``capturing`` says whether the build will run with capture on."""
    _Build(plan, Path(out), options or BuildOptions(), None, megatron_env).check(capturing)


def require_new_folder(folder: Path, what: str = "a build") -> None:
    """Refuse a folder that exists and is not an empty folder."""
    folder = Path(folder)
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise OutputError(f"{folder} exists and is not empty; {what} needs a new folder")


class _Build:
    def __init__(self, plan, out, options, capture, megatron_env) -> None:
        self.plan: BuildPlan = plan
        self.out: Path = out
        self.options: BuildOptions = options
        self.capture = capture
        self.megatron_env = megatron_env
        self.normalise = normaliser(plan.normaliser)
        self.index: dict[str, Record] = {}  # id -> the record at the place it was read
        self.file_of: dict[str, str] = {}  # id -> the key of its file ("A", "gen/P")
        self.release_lines: dict[str, list[tuple[str, Record]]] = {}
        self.summary: dict = {}
        # The file-level manifest: made with the build folder, and only when capture
        # is on. The chunk files lists the own files (source files and generation
        # outputs) of the records a chunk's tokenizer input holds, by layout and shard,
        # for the row made at the tokenization point.
        self.built_from: BuiltFrom | None = None
        self.chunk_files: dict[tuple[str, str], list[str]] = {}

    # --- Plumbing ----------------------------------------------------------------

    def name(self, store_name: str) -> str:
        return self.options.store_name_prefix + store_name

    def write(self, relative: str, lines: Sequence[str],
              manifest: Sequence[tuple[str, Sequence[str]]] | None) -> Path:
        """Write a JSONL file and, for a record file, the manifest beside it: one
        row per line with the line number, the handle and the origin ids."""
        path = self.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")
        if manifest is not None:
            if len(manifest) != len(lines):
                raise ReplicaError(f"{relative}: {len(lines)} lines, {len(manifest)} manifest rows")
            self.write_manifest(path, manifest)
        return path

    @staticmethod
    def write_manifest(path: Path, manifest: Sequence[tuple[str, Sequence[str]]]) -> None:
        with open(f"{path}.manifest.jsonl", "x", encoding="utf-8", newline="\n") as f:
            for number, (handle, origin) in enumerate(manifest, 1):
                f.write(json_row({"line": number, "handle": handle,
                                  "origin": list(origin)}) + "\n")

    def manifest_row(self, record: Record) -> tuple[str, list[str]]:
        if record.artifact is None:
            return (HANDLE_INSERTED, [])
        return (self.plan.handle_of(record.ref), [record.ref])

    def lookup(self, record_id: str, what: str) -> Record:
        try:
            return self.index[record_id]
        except KeyError:
            raise InputError(f"{what}: no record with id {record_id!r} in the sources or the "
                             "generation outputs") from None

    def add_to_index(self, record: Record, key: str, where: str) -> None:
        if not record.has_id:
            return
        if record.ref in self.index:
            raise InputError(f"{where}: id {record.ref!r} is used twice")
        self.index[record.ref] = record
        self.file_of[record.ref] = key

    def own_file(self, record: Record) -> str | None:
        """The path, relative to the build folder, of the source file or generation output
        the record's id was read from; None for a line with no id of the spec's (the line the
        spec writes into a chunk). A record of a generated set is re-anchored to the release
        it was cut from, so its own file is found by its id in the index and not by where it
        now lies."""
        original = self.index.get(record.ref) if record.has_id else None
        return None if original is None else original.artifact

    def select(self, items: Sequence[str], what: str) -> list[tuple[str, Record]]:
        """The records a list of ids names, each with the key of the file it came
        from. The changes of the build are applied here."""
        changes = self.options.changes
        out: list[tuple[str, Record]] = []
        seen: set[str] = set()
        listed = set(items)
        for record_id in items:
            record = self.lookup(record_id, what)
            if record.ref in changes.leave_out:
                continue
            if record.ref in changes.swap:
                new = self.lookup(changes.swap[record.ref], f"{what}: swap of {record.ref}")
                if new.ref in listed and self.equivalent(record, new):
                    # The equivalent copy is selected here already, at its own place
                    # (a dedup group selects every copy of its text): the old copy
                    # just leaves, which is what a build from sources that never held
                    # it selects. A swap onto a record that is not equivalent is
                    # refused below as a record selected twice.
                    continue
                record = new
            if record.ref in seen:
                raise ReplicaError(f"{what}: record {record.ref!r} is selected twice")
            seen.add(record.ref)
            out.append((self.file_of[record.ref], record))
        return out

    def equivalent(self, first: Record, second: Record) -> bool:
        """Whether two records have the same text after the build's normaliser, which
        is what puts copies in one dedup group. A record with no readable text has
        none."""
        return (first.readable and second.readable
                and self.normalise(first.text) == self.normalise(second.text))

    def chosen(self, items, wanted: tuple[str, ...] | None, what: str) -> list:
        """The releases or imports this build makes: all of them, or the named ones."""
        if wanted is None:
            return list(items)
        known = {item.name for item in items}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise SpecError(f"no {what} named {unknown}; defined: {sorted(known)}")
        return [item for item in items if item.name in wanted]

    # --- The steps -----------------------------------------------------------------

    def check(self, capturing: bool) -> None:
        """Everything a build refuses before it writes. Nothing is written here."""
        plan, options = self.plan, self.options
        for layout in options.layouts:
            if layout not in plan.layouts:
                raise SpecError(f"layout {layout!r} is not defined for this build; defined: "
                                f"{sorted(plan.layouts)}")
        if options.mixture_files not in (MIXTURE_COPY, MIXTURE_SYMLINK):
            raise SpecError(f"mixture files are made by copy or symlink, "
                            f"got {options.mixture_files!r}")
        if plan.generation_model != "none":
            raise SpecError("the generation stand-in calls no model; the generation model "
                            f"must be \"none\", got {plan.generation_model!r}")
        if options.generation_harness not in HARNESSES:
            raise SpecError(f"a generation job runs {' or '.join(HARNESSES)}, "
                            f"got {options.generation_harness!r}")
        releases = self.chosen(plan.releases, options.releases, "release")
        self.chosen(plan.imports, options.imports, "import")
        if plan.pipeline_release is not None and options.layouts and (
                plan.pipeline_release not in {r.name for r in releases}):
            raise SpecError(f"the pipeline chunks the rows of release {plan.pipeline_release}, "
                            "which this build was told to leave out")
        if options.tokenize:
            if plan.tokenizer is None:
                raise SpecError("this build names no tokenizer: a generated set carries none, "
                                "so give the build one (--tokenizer-spec) or build with "
                                "tokenization off")
            megatron_tokenizer.check_settings(plan.tokenizer)
            if capturing and options.layouts:
                # A captured shard gets its map, and a map records the
                # end-of-document convention: refuse a table that does not hold it.
                from .pipeline_links import map_settings

                map_settings(plan.tokenizer)
            if self.megatron_env is None:
                self.megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
            self.megatron_env.tokenizer_files(plan.tokenizer)
        if options.generation_harness == HARNESS_CONFINED:
            if not (capturing and options.capture_generation):
                raise SpecError("the confined harness runs a captured generation job; this "
                                "build does not capture its generation step")
            from .capture_hooks import require_sandbox

            require_sandbox()
        require_new_folder(self.out)

    def run(self) -> dict:
        plan, options = self.plan, self.options
        self.check(self.capture is not None)
        releases = self.chosen(plan.releases, options.releases, "release")
        imports = self.chosen(plan.imports, options.imports, "import")
        self.out.mkdir(parents=True, exist_ok=True)
        if self.capture is not None:
            self.built_from = BuiltFrom(self.out)

        self.summary = {
            "mode": plan.mode,
            "seeded": options.seeded,
            "capture": self.capture is not None,
            "tokenized": options.tokenize,
            "left_out": sorted(options.changes.leave_out),
            "swapped": dict(sorted(options.changes.swap.items())),
            "releases": {},
            "layouts": {},
            "mixtures": {},
            "segments": {},
        }
        self.build_sources()
        self.build_generation()
        for release in releases:
            self.build_release(release)
        for imported in imports:
            self.build_import(imported)
        jobs = []
        for layout in options.layouts:
            jobs.extend(self.build_layout_inputs(plan.layouts[layout]))
        if options.tokenize:
            self.tokenize(jobs)
            self.build_mixtures()
        if self.capture is not None:
            self.summary["instances"] = dict(self.capture.instances)
            self.summary["files"] = dict(self.capture.files)
        return self.summary

    def build_sources(self) -> None:
        fixture = self.plan.mode == MODE_FIXTURE
        for source in self.plan.sources:
            relative = f"sources/{source.key}.jsonl"
            if not source.path.is_file():
                raise InputError(f"source file {source.path} is missing")
            target = self.out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.path, target)
            records = read_records(target, artifact=relative, source=source.source,
                                   strict=fixture,
                                   refs=source.records if source.by_line else None)
            if [r.ref for r in records] != list(source.records):
                raise InputError(f"{source.path}: the ids do not match the records the spec "
                                 f"lists for source {source.key}")
            for record in records:
                self.add_to_index(record, source.key, str(source.path))
            self.write_manifest(target, [self.manifest_row(r) for r in records])
            if self.capture is not None:
                self.capture.ingest(target, relative, source, self.name(f"src/{source.key}"))
                self.built_from.add(self.name(f"src/{source.key}"), [relative], [])

    def build_generation(self) -> None:
        """The generation stand-in: every job emits its fixed output and calls no
        model. A job is handed records of the source files and of earlier outputs,
        never of a release (generation reads the source revision, plan 5.1)."""
        captured = self.capture is not None and self.options.capture_generation
        for gfile in self.plan.generation:
            key = f"gen/{gfile.name}"
            relative = f"{key}.jsonl"
            lines: list[str] = []
            manifest: list[tuple[str, Sequence[str]]] = []
            made = []
            for row in gfile.rows:
                what = f"generation output {row.id}: handed record"
                handed = [self.lookup(h, what) for h in row.handed]
                unreadable = [h.ref for h in handed if not h.readable]
                if unreadable:
                    raise InputError(f"{what}: {unreadable} hold no text")
                number = len(lines) + 1
                if captured:
                    # A new log file for every job of every build: a build folder
                    # is new, and a harness log is never written over.
                    log_path = (self.out / "gen" / "harness" / gfile.name
                                / f"{row.id}.log.jsonl")
                    text = self.capture.generate(
                        row, handed, log_path, harness=self.options.generation_harness,
                        timeout=self.options.generation_timeout)
                else:
                    text = row.text
                line = json_row({"id": row.id, "text": text})
                record = Record(ref=row.id, text=text, raw=line, artifact=relative,
                                line=number, source=None)
                self.add_to_index(record, key, relative)
                lines.append(line)
                manifest.append((self.plan.handle_of(row.id), list(row.used)))
                made.append((row, handed, record))
            if not lines:
                continue
            path = self.write(relative, lines, manifest)
            if captured:
                self.capture.record_generation(path, relative, self.name(key), made)
                # Built from the files of the records the jobs were handed.
                self.built_from.add_from_files(
                    self.name(key), [relative],
                    [h.artifact for _, handed, _ in made for h in handed
                     if h.artifact != relative])

    def build_release(self, release) -> None:
        """A release: the selected records, through the exact dedup. A line that
        does not parse, or whose text is empty after normalisation, is in no dedup
        group and stays a row until the sanitizer drops it."""
        what = f"release {release.name}"
        selected = self.select(release.inputs, what)
        kept = dedup([record for _, record in selected], self.normalise,
                     enabled=release.dedup, priority=release.priority)
        relative = f"releases/{release.name}.jsonl"
        lines = [json_row({"text": k.record.text}) if k.record.readable else k.record.raw
                 for k in kept]
        path = self.write(relative, lines, [self.manifest_row(k.record) for k in kept])
        dropped = 0
        log: list[str] = []
        for number, k in enumerate(kept, 1):
            if len(k.copies) == 1:
                continue
            dropped += len(k.losers)
            log.append(json_row({"line": number, "kept": k.record.ref,
                                 "dropped": [c.ref for c in k.losers], "rule": k.rule}))
        if release.dedup:
            self.write(f"releases/{release.name}.dedup.jsonl", log, None)
        if self.capture is not None:
            # The winner is left out of the parameters on purpose: two releases that
            # differ only in which copy dedup kept then differ in their edges alone
            # (plan Check 7, Decision 29).
            params = {"dedup": release.dedup, "normaliser": self.plan.normaliser}
            self.capture.record_release(path, relative, self.name(release.name), params, kept)
            # Built from the files of the copies kept. The file of a copy dedup dropped
            # did not go into the release, so it is not listed.
            self.built_from.add_from_files(
                self.name(release.name), [relative],
                [k.record.artifact for k in kept if k.record.artifact is not None])
        self.release_lines[release.name] = [
            (self.file_of[k.record.ref], k.record.at(relative, number))
            for number, k in enumerate(kept, 1)
        ]
        self.summary["releases"][release.name] = {
            "lines": len(lines), "dedup": release.dedup, "dropped": dropped}

    def build_import(self, imported) -> None:
        selected = [record for _, record in
                    self.select(imported.records, f"import {imported.name}")]
        if not selected:
            return
        relative = f"imports/{imported.name.split('/')[-1]}.jsonl"
        path = self.write(relative, [json_row({"text": r.text}) for r in selected],
                          [self.manifest_row(r) for r in selected])
        if self.capture is not None:
            # The member name carries no prefix: the same bytes imported under the
            # same member are the same instance, whichever build registers them.
            self.capture.import_text(path, relative, self.name(imported.name),
                                     imported.name, imported.family_member,
                                     imported.evidence)
            # Built from the file of the member its file family gives, which is not
            # the file the import's text came from.
            self.built_from.add_from_files(
                self.name(imported.name), [relative],
                [f"sources/{s.key}.jsonl" for s in self.plan.sources
                 if s.member == imported.family_member])

    def layout_groups(self, layout: LayoutPlan) -> list[tuple[str, list[Record]]]:
        """The records each group of a layout holds, before any shuffle."""
        what = f"layout {layout.name}"
        if self.plan.mode == MODE_FIXTURE:
            return [(name, [record for _, record in self.select(items, f"{what}, {name}")])
                    for name, items in layout.groups]
        # A generated set: the shards are built from the rows of one release,
        # grouped by the file each row came from.
        lines = self.release_lines[self.plan.pipeline_release]
        groups = []
        for name, files in layout.groups:
            wanted = set(files)
            groups.append((name, [record for file, record in lines if file in wanted]))
        return groups

    def with_inserted_lines(self, layout: LayoutPlan, shard: str,
                            records: Sequence[Record]) -> list[Record]:
        """The chunk as the sanitizer reads it: the shuffled records, and at its
        place the line the spec writes into this shard (``[sanitizer_skip]``)."""
        out = list(records)
        for inserted in self.plan.inserted_lines:
            if inserted.layout != layout.name or inserted.shard != shard:
                continue
            if inserted.line > len(out) + 1:
                raise ReplicaError(
                    f"layout {layout.name}, {shard}: the spec writes a line at line "
                    f"{inserted.line} and the chunk holds {len(out)} lines")
            out.insert(inserted.line - 1, parse_line(inserted.text, artifact=None,
                                                     line=inserted.line, source=None))
        return out

    def build_layout_inputs(self, layout: LayoutPlan) -> list[megatron_tokenizer.ShardJob]:
        options = self.options
        if options.seeded:
            seed = options.shuffle_seeds.get(layout.name, layout.shuffle_seed)
            shuffler = SeededShuffler(seed)
        else:
            seed = None
            shuffler = UnseededShuffler()
        groups = self.layout_groups(layout)
        if layout.chunks is not None:
            chunks = round_robin_chunks(groups, layout.chunks, shuffler)
        else:
            chunks = partitioned_shards(groups, shuffler)
        # The record files the copy steps list for each shard: every listed file
        # can reach every round-robin chunk, and a partitioned shard reads the
        # files of its own records. This is the pipeline's own configuration; a
        # recovery of the copy steps names these files as the possible inputs.
        files_of = {name: tuple(dict.fromkeys(r.artifact for r in records
                                              if r.artifact is not None))
                    for name, records in groups}
        every_file = tuple(dict.fromkeys(f for files in files_of.values() for f in files))

        base = f"pipeline/{layout.name}"
        log: list[str] = []
        shards = []
        jobs = []
        for shard, shuffled in chunks:
            store_name = self.name(f"{layout.store_prefix}/{shard}")
            if not shuffled:
                # No record reached the shard (every one was left out or lost a
                # dedup, or there are more chunks than records). No file is written;
                # the summary says so, and a mixture cannot name the shard.
                shards.append({"shard": shard, "store_name": store_name, "chunk_lines": 0,
                               "input_lines": 0, "skipped": 0, "empty": True})
                continue
            records = self.with_inserted_lines(layout, shard, shuffled)
            # The copy steps pass a line on byte for byte, readable or not.
            chunk = f"{base}/sanitizer-input/{shard}.jsonl"
            self.write(chunk, [r.raw for r in records], [self.manifest_row(r) for r in records])
            survivors, skips = sanitize(records, self.normalise)
            if self.built_from is not None:
                self.chunk_files[(layout.name, shard)] = [
                    own for own in map(self.own_file, survivors) if own is not None]
            for skip in skips:
                log.append(json_row({
                    "shard": shard,
                    "input": chunk,
                    "line": skip.position,
                    "id": skip.record.ref if skip.record.has_id else None,
                    "origin_file": skip.record.artifact,
                    "origin_line": None if skip.record.artifact is None else skip.record.line,
                    "reason": skip.reason,
                    "detail": skip.detail,
                }))
            listed = every_file if layout.chunks is not None else files_of[shard]
            entry = {"shard": shard, "store_name": store_name, "chunk_lines": len(records),
                     "input_lines": len(survivors), "skipped": len(skips),
                     "empty": not survivors, "listed": list(listed)}
            shards.append(entry)
            if not survivors:
                continue  # every line was skipped: nothing to tokenize
            relative = f"{base}/tokenizer-input/{shard}.jsonl"
            path = self.write(relative,
                              [json_row({"id": r.ref, "text": r.text}) for r in survivors],
                              [self.manifest_row(r) for r in survivors])
            jobs.append(megatron_tokenizer.ShardJob(
                name=shard,
                input_path=path,
                prefix=self.out / base / "shards" / shard,
                work_dir=self.out / WORK_DIR / "tokenize" / layout.name / shard,
            ))
        self.write(f"{base}/sanitizer.log.jsonl", log, None)
        self.summary["layouts"][layout.name] = {"shuffle_seed": seed, "chunks": layout.chunks,
                                                "shards": shards}
        return jobs

    def tokenize(self, jobs: Sequence[megatron_tokenizer.ShardJob]) -> None:
        env, settings = self.megatron_env, self.plan.tokenizer
        self.summary["tokenizer"] = {
            "type": settings.type,
            "name": settings.name,
            "revision": settings.revision,
            "vocab_file": settings.vocab_file,
            "merges_file": settings.merges_file,
            "append_eod": settings.append_eod,
            "eod_token_id": settings.eod_token_id,
            "eod_in_span": settings.eod_in_span,
            "partitions": settings.partitions,
            "workers": settings.workers,
        }
        megatron_tokenizer.preflight(env, settings, self.out / WORK_DIR / "tokenize" / "preflight")
        megatron_tokenizer.tokenize_shards(env, settings, jobs,
                                           parallel=self.options.parallel,
                                           timeout=self.options.tokenizer_timeout)
        if self.capture is None:
            self.check_shards(settings, None)
            return
        # The capture maps are verified by tokenizing every line again, with the
        # same tokenizer as a function of one text.
        with megatron_tokenizer.line_tokenizer(
                env, settings, self.out / WORK_DIR / "tokenize" / "line-tokenizer",
                timeout=self.options.tokenizer_timeout) as tokenize:
            self.check_shards(settings, tokenize)

    def check_shards(self, settings, tokenize) -> None:
        """Hold every shard against its input (plan F1) and, with capture on, hand
        it to the tokenization capture point."""
        for layout in self.options.layouts:
            base = self.out / "pipeline" / layout
            for entry in self.summary["layouts"][layout]["shards"]:
                if entry["empty"]:
                    continue
                prefix = base / "shards" / entry["shard"]
                bin_path, idx_path = Path(f"{prefix}.bin"), Path(f"{prefix}.idx")
                index = read_shard_index(idx_path)
                if index.documents != entry["input_lines"]:
                    raise megatron_tokenizer.TokenizerError(
                        f"{idx_path}: {index.documents} documents for "
                        f"{entry['input_lines']} input lines; one document per line is the "
                        "condition the layout rests on (plan F1)")
                if bin_path.stat().st_size != index.tokens * index.token_bytes:
                    raise megatron_tokenizer.TokenizerError(
                        f"{bin_path}: size does not match the {index.tokens} tokens of its index")
                entry.update({"documents": index.documents, "tokens": index.tokens,
                              "bin_sha256": file_sha256(bin_path),
                              "idx_sha256": file_sha256(idx_path)})
                if self.capture is not None:
                    files = shard_files(self.out, layout, entry)
                    entry["capture"] = self.capture.shard_tokenized(
                        files, settings, tokenize, self.plan.normaliser)
                    # The chunk file is built from the files of its records, the shard
                    # from the chunk file the tokenizer was handed.
                    self.built_from.add_from_files(
                        files.tokenizer_input_name, [files.tokenizer_input_relative],
                        self.chunk_files[(layout, entry["shard"])])
                    self.built_from.add(
                        files.store_name,
                        [f"{files.prefix_relative}.bin", f"{files.prefix_relative}.idx"],
                        [files.tokenizer_input_name])

    def build_mixtures(self) -> None:
        """Directory-selected mixtures: a mixture is the folder, and what it selects
        is the shard files that lie in it. The summary keeps the order the spec
        lists the shards in: the order the trainer is handed them (plan Decision 27)."""
        built: dict[str, Path] = {}
        for mixture in self.plan.mixtures:
            if mixture.layout not in self.options.layouts:
                continue
            layout = self.plan.layouts[mixture.layout]
            entries = {entry["shard"]: entry
                       for entry in self.summary["layouts"][layout.name]["shards"]}
            named_empty = [s for s in mixture.shards if entries[s]["empty"]]
            if named_empty:
                raise ReplicaError(f"mixture {mixture.name}: {named_empty} hold no line in "
                                   "this build, so there is no shard to select")
            folder = self.out / "mixtures" / mixture.name
            folder.mkdir(parents=True, exist_ok=False)
            for shard in mixture.shards:
                for suffix in (".bin", ".idx"):
                    source = self.out / "pipeline" / layout.name / "shards" / f"{shard}{suffix}"
                    target = folder / f"{shard}{suffix}"
                    if self.options.mixture_files == MIXTURE_SYMLINK:
                        os.symlink(os.path.relpath(source, folder), target)
                    else:
                        shutil.copyfile(source, target)
            store_names = [entries[s]["store_name"] for s in mixture.shards]
            self.summary["mixtures"][mixture.name] = {
                "layout": layout.name, "shards": list(mixture.shards),
                "store_names": store_names, "folder": f"mixtures/{mixture.name}"}
            built[mixture.name] = folder
            if self.capture is not None:
                self.capture.mixture_selected(
                    folder, f"mixtures/{mixture.name}", self.name(mixture.name),
                    [(name, folder / shard) for name, shard in zip(store_names, mixture.shards)])
                # Built from its shards, in the trainer's order.
                self.built_from.add(
                    self.name(mixture.name),
                    [f"mixtures/{mixture.name}/{shard}{suffix}"
                     for shard in mixture.shards for suffix in (".bin", ".idx")],
                    store_names)
        for segment in self.plan.segments:
            if segment.mixture not in built:
                continue
            self.summary["segments"][segment.name] = {
                "run": segment.run, "position": segment.position, "mixture": segment.mixture}
            if self.capture is not None:
                self.capture.run_segment_ready(segment, built[segment.mixture],
                                               self.name(segment.mixture))
