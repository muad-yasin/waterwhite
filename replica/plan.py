"""What one build is made of, read from the fixture spec or from a generated set.

Two input modes give the same plan type:

- the hand-checked fixture (plan 5.1): ``fixture/spec/fixture_spec.toml``;
- a generated set: the folder ``fixture/plant/generate.py`` writes, with its
  ``set.toml`` (docs/plant.md section 4).

Both select records by id. In the fixture an id is the row's ``id`` field; in a
generated set an id names a line of a file, whatever the line holds.

A reader takes the tables it uses and leaves every other table and key alone: the
fixture spec also holds the runs' settings, the seed search and the read positions,
which belong to the oracle and to the hand-written table.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .errors import InputError, SpecError
from .records import FAULT_BLANK, parse_line, split_lines
from .steps import chunk_names, expand_ids, normaliser

FIXTURE_FORMAT = "waterwhite.fixture-spec/1"
SET_FORMAT = "waterwhite.plant-set/1"

MODE_FIXTURE = "fixture"
MODE_SET = "set"

LAYOUT_PARTITIONED = "partitioned"
LAYOUT_ROUND_ROBIN = "round-robin"
LAYOUTS = (LAYOUT_PARTITIONED, LAYOUT_ROUND_ROBIN)

# The store names of run E's shards (docs/interfaces.md 9.2): "E/shard-1" in the
# partitioned layout, "E-rr/shard-1" in the round-robin variant.
FIXTURE_RUN = "E"
ROUND_ROBIN_SUFFIX = "-rr"

# A generated set names no run and no set name. Its shards are registered as
# "set/<shard>" (partitioned) and "set-rr/<shard>" (round-robin).
SET_PREFIX = "set"
# The release whose rows a generated set's pipeline chunks (docs/replica.md 4).
SET_PIPELINE_RELEASE = "R1"
# The partitioned layout of a generated set: one shard per source file, named
# after the source, and one shard for every generation output.
SET_GENERATION_SHARD = "shard-gen"

ALL_RECORDS = "all"


@dataclass(frozen=True)
class TokenizerSettings:
    """The conditions of one preprocessing run (plan F1), and the tokenizer files.

    ``revision`` is the hub revision of the tokenizer files. ``vocab_file`` and
    ``merges_file`` are paths relative to the Megatron environment folder; every
    preprocessing run is handed both, so that Megatron's tokenizer class has
    nothing to download.

    ``eod_token_id`` and ``eod_in_span`` are the end-of-document convention as the
    spec records it from the oracle's output (plan 3.3, F1): the token, and whether
    it is the last token of every document's own span. Both are None when the file
    does not record them; a tokenization map can then not be written."""

    revision: str
    vocab_file: str
    merges_file: str
    type: str = "GPT2BPETokenizer"
    name: str = "gpt2"
    append_eod: bool = True
    partitions: int = 1
    workers: int = 1
    eod_token_id: int | None = None
    eod_in_span: bool | None = None


@dataclass(frozen=True)
class SourceFile:
    """One source file. ``key`` names the file: ``sources/<key>.jsonl``.

    ``records`` are the ids the file must hold, in line order. ``by_line`` says how
    a line gets its id: False, from the row's ``id`` field (the fixture); True,
    from its place in the file (a generated set)."""

    key: str
    path: Path
    source: str
    member: str
    revision: str
    records: tuple[str, ...]
    by_line: bool = False


@dataclass(frozen=True)
class GenerationRow:
    """One fixed output. ``job_name`` names the harness run that makes it."""

    id: str
    job: str
    handed: tuple[str, ...]
    used: tuple[str, ...]
    text: str
    job_name: str


@dataclass(frozen=True)
class GenerationFile:
    """One generation output file: ``gen/<name>.jsonl``."""

    name: str
    rows: tuple[GenerationRow, ...]


@dataclass(frozen=True)
class ReleasePlan:
    """A release: the ids of its input records in order, whether the exact dedup
    runs, and the dedup priority (source names; the copy from the first listed
    source is kept)."""

    name: str
    inputs: tuple[str, ...]
    dedup: bool
    priority: tuple[str, ...]


@dataclass(frozen=True)
class ImportPlan:
    """A text-only import with a known file family (plan Check 6)."""

    name: str
    records: tuple[str, ...]
    family_member: str
    evidence: str


@dataclass(frozen=True)
class LayoutPlan:
    """One layout. ``groups`` are the shards with what they hold (partitioned), or
    the listed files that are shuffled and dealt (round-robin, with ``chunks``).
    A group's items are record ids in the fixture and file keys in a set."""

    name: str
    store_prefix: str
    shuffle_seed: int
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    chunks: int | None = None

    def shard_names(self) -> list[str]:
        if self.chunks is not None:
            return chunk_names(self.chunks)
        return [name for name, _ in self.groups]


@dataclass(frozen=True)
class InsertedLine:
    """A line the spec writes into the file the sanitizer reads for one shard,
    after the shuffle (the fixture's skipped sanitizer line, plan 5.1). ``line`` is
    its line there, counted from 1. It belongs to no source."""

    layout: str
    shard: str
    line: int
    text: str


@dataclass(frozen=True)
class MixturePlan:
    """A mixture: a folder that holds the shard files it selects. ``shards`` is the
    trainer's order: the place of a shard in it is its dataset number in the blend
    (plan Decision 27)."""

    name: str
    layout: str
    shards: tuple[str, ...]


@dataclass(frozen=True)
class SegmentPlan:
    """A run segment as far as the replica needs it: its run, its place in the run
    (from 1) and the mixture it trains on."""

    run: str
    name: str
    position: int
    mixture: str


@dataclass(frozen=True)
class BuildPlan:
    mode: str
    normaliser: str
    tokenizer: TokenizerSettings | None
    sources: tuple[SourceFile, ...]
    generation: tuple[GenerationFile, ...]
    generation_model: str
    handles: dict[str, str]  # record id -> handle
    releases: tuple[ReleasePlan, ...]
    imports: tuple[ImportPlan, ...]
    layouts: dict[str, LayoutPlan]
    mixtures: tuple[MixturePlan, ...]
    segments: tuple[SegmentPlan, ...] = ()
    inserted_lines: tuple[InsertedLine, ...] = ()
    pipeline_release: str | None = None  # set mode: the release the shards are built from
    notes: tuple[str, ...] = field(default_factory=tuple)

    def handle_of(self, ref: str) -> str:
        return self.handles.get(ref, ref)


# --- Small readers -----------------------------------------------------------------


def _table(parent: dict, key: str, where: str, *, required: bool = True) -> dict:
    value = parent.get(key)
    if value is None:
        if required:
            raise SpecError(f"{where}: table [{key}] is missing")
        return {}
    if not isinstance(value, dict):
        raise SpecError(f"{where}: [{key}] must be a table")
    return value


def _value(parent: dict, key: str, kind: type, where: str, default=None):
    if key not in parent:
        if default is not None:
            return default
        raise SpecError(f"{where}: {key} is missing")
    value = parent[key]
    if kind is int and isinstance(value, bool):
        raise SpecError(f"{where}: {key} must be an integer")
    if not isinstance(value, kind):
        raise SpecError(f"{where}: {key} must be {kind.__name__}, got {value!r}")
    return value


def _optional(parent: dict, key: str, kind: type, where: str):
    """A value the file may leave out: None then."""
    if key not in parent:
        return None
    value = parent[key]
    if (kind is int and isinstance(value, bool)) or not isinstance(value, kind):
        raise SpecError(f"{where}: {key} must be {kind.__name__}, got {value!r}")
    return value


def _chunks(parent: dict, where: str) -> int:
    chunks = _value(parent, "chunks", int, where)
    if chunks < 1:
        raise SpecError(f"{where}: chunks must be at least 1, got {chunks}")
    return chunks


def _strings(parent: dict, key: str, where: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SpecError(f"{where}: {key} must be a list of strings")
    return tuple(value)


def _priority(table: dict, where: str) -> tuple[str, ...]:
    """The dedup priority of a release. ``dedup_winner`` alone is a priority of one
    source; ``dedup_priority`` gives the whole order and must start with the
    winner when both are written."""
    winner = table.get("dedup_winner")
    if winner is not None and not isinstance(winner, str):
        raise SpecError(f"{where}: dedup_winner must name a source")
    if "dedup_priority" not in table:
        return () if winner is None else (winner,)
    priority = _strings(table, "dedup_priority", where)
    if len(set(priority)) != len(priority):
        raise SpecError(f"{where}: dedup_priority names a source twice")
    if winner is not None and (not priority or priority[0] != winner):
        raise SpecError(f"{where}: dedup_winner is {winner!r} and dedup_priority starts "
                        f"with {priority[:1]}")
    return priority


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise SpecError(f"no such file: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"{path}: {exc}") from exc


def _environment_path(table: dict, key: str, where: str) -> str:
    """A path the spec gives relative to the Megatron environment folder."""
    value = _value(table, key, str, where)
    path = PurePosixPath(value)
    if value == "" or path.is_absolute() or ".." in path.parts:
        raise SpecError(f"{where}: {key} is a path relative to the Megatron environment "
                        f"folder, got {value!r}")
    return value


def _tokenizer(doc: dict, where: str) -> TokenizerSettings:
    megatron = _table(doc, "megatron", where, required=False)
    table = _table(doc, "tokenizer", where)
    w = f"{where} [tokenizer]"
    settings = TokenizerSettings(
        revision=_value(table, "revision", str, w),
        vocab_file=_environment_path(table, "vocab_file", w),
        merges_file=_environment_path(table, "merges_file", w),
        type=_value(table, "type", str, w),
        name=_value(table, "name", str, w),
        append_eod=_value(table, "append_eod", bool, w),
        partitions=_value(megatron, "partitions", int, f"{where} [megatron]", 1),
        workers=_value(megatron, "workers", int, f"{where} [megatron]", 1),
        eod_token_id=_optional(table, "eod_token_id", int, w),
        eod_in_span=_optional(table, "eod_in_span", bool, w),
    )
    if settings.partitions != 1:
        # With more than one partition the lines are dealt over the partitions and
        # document order is no longer line order (plan F1).
        raise SpecError(f"{where}: the replica tokenizes in one partition, got "
                        f"{settings.partitions}")
    if settings.workers < 1:
        raise SpecError(f"{where}: workers must be at least 1")
    return settings


def load_tokenizer_settings(path: Path) -> TokenizerSettings:
    """The tokenizer a build runs, from the ``[tokenizer]`` and ``[megatron]``
    tables of a TOML file. The fixture spec is such a file; a generated set names
    no tokenizer, so its build is given one this way."""
    path = Path(path)
    return _tokenizer(_load_toml(path), str(path))


def read_generation_rows(path: Path) -> list[dict]:
    """The rows of a ``generation.jsonl``: ``{"id", "job", "handed", "used", "text"}``.
    Every used id must be among the handed ids."""
    try:
        data = Path(path).read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise InputError(f"no such file: {path}") from None
    rows = []
    seen: set[str] = set()
    for number, raw in enumerate(split_lines(data), 1):
        where = f"{path}: line {number}"
        try:
            row = json.loads(raw)
        except ValueError as exc:
            raise InputError(f"{where}: not JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise InputError(f"{where}: not a JSON object")
        for key in ("id", "job", "text"):
            if not isinstance(row.get(key), str) or row[key] == "":
                raise InputError(f"{where}: {key} must be a non-empty string")
        for key in ("handed", "used"):
            value = row.get(key)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise InputError(f"{where}: {key} must be a list of ids")
        if not row["handed"]:
            raise InputError(f"{where}: a job is handed at least one record")
        outside = [u for u in row["used"] if u not in row["handed"]]
        if outside:
            raise InputError(f"{where}: used but not handed: {outside}")
        if row["id"] in seen:
            raise InputError(f"{where}: id {row['id']!r} appears twice")
        seen.add(row["id"])
        rows.append(row)
    return rows


def _generation_rows(path: Path) -> list[GenerationRow]:
    rows = read_generation_rows(path)
    per_job: dict[str, int] = {}
    for row in rows:
        per_job[row["job"]] = per_job.get(row["job"], 0) + 1
    numbered: dict[str, int] = {}
    out = []
    for row in rows:
        # A harness run needs a name of its own. A job with one output keeps its
        # name; a job with many outputs is one run per output, named job/N where N
        # is the row's number in the job's output, counted from 1. The name never
        # holds the row's id: the store is not given the ids of the chunk files
        # (plan Check 3), and a job name is stored.
        numbered[row["job"]] = numbered.get(row["job"], 0) + 1
        job = row["job"]
        job_name = job if per_job[job] == 1 else f"{job}/{numbered[job]}"
        out.append(GenerationRow(row["id"], row["job"], tuple(row["handed"]),
                                 tuple(row["used"]), row["text"], job_name))
    return out


# --- The hand-checked fixture ----------------------------------------------------------


def _resolve_shard(entry: str, layouts: dict[str, LayoutPlan], where: str) -> tuple[str, str]:
    """The layout and the shard a mixture entry names. ``"shard-1"`` is a shard of
    the partitioned layout; ``"E-rr/shard-1"`` is a shard by its store name, here
    of the round-robin variant."""
    if "/" not in entry:
        layout = layouts[LAYOUT_PARTITIONED]
        shard = entry
    else:
        prefix, _, shard = entry.rpartition("/")
        found = [plan for plan in layouts.values() if plan.store_prefix == prefix]
        if not found:
            raise SpecError(f"{where}: {entry!r} names no layout of this spec; the store "
                            f"prefixes are {sorted(p.store_prefix for p in layouts.values())}")
        layout = found[0]
    if shard not in layout.shard_names():
        raise SpecError(f"{where}: {entry!r} names a shard the layout {layout.name!r} "
                        f"does not have: {layout.shard_names()}")
    return layout.name, shard


def _mixture_shards(entries: tuple[str, ...], layouts: dict[str, LayoutPlan],
                    where: str) -> tuple[str, tuple[str, ...]]:
    if not entries:
        raise SpecError(f"{where}: a mixture selects at least one shard")
    resolved = [_resolve_shard(entry, layouts, where) for entry in entries]
    names = {layout for layout, _ in resolved}
    if len(names) != 1:
        raise SpecError(f"{where}: the shards of one mixture lie in one layout, "
                        f"got {sorted(names)}")
    shards = tuple(shard for _, shard in resolved)
    if len(set(shards)) != len(shards):
        raise SpecError(f"{where}: a shard is listed twice")
    return resolved[0][0], shards


def _fixture_mixtures(doc: dict, layouts: dict[str, LayoutPlan], where: str) -> list[MixturePlan]:
    """``[mixtures]``: each name with the list of its shards in the trainer's order."""
    out = []
    table = _table(doc, "mixtures", where, required=False)
    for name in table:
        w = f"{where} [mixtures] {name!r}"
        layout, shards = _mixture_shards(_strings(table, name, f"{where} [mixtures]"),
                                         layouts, w)
        out.append(MixturePlan(name, layout, shards))
    return out


def _fixture_segments(doc: dict, mixtures: list[MixturePlan], layouts: dict[str, LayoutPlan],
                      where: str) -> list[SegmentPlan]:
    """The segments of ``[runs.<run>.segments]`` with the mixture each trains on.
    A segment lists its shards too; the two lists must say the same."""
    by_name = {m.name: m for m in mixtures}
    out = []
    for run, run_table in _table(doc, "runs", where, required=False).items():
        if not isinstance(run_table, dict):
            continue
        segments = run_table.get("segments", {})
        if not isinstance(segments, dict):
            raise SpecError(f"{where}: [runs.{run}.segments] must be a table")
        for position, (segment, table) in enumerate(segments.items(), 1):
            w = f"{where} [runs.{run}.segments.\"{segment}\"]"
            if not isinstance(table, dict):
                raise SpecError(f"{w} must be a table")
            mixture = by_name.get(_value(table, "mixture", str, w))
            if mixture is None:
                raise SpecError(f"{w}: mixture {table['mixture']!r} is not in [mixtures]")
            listed = _mixture_shards(_strings(table, "shards", w), layouts, w)
            if listed != (mixture.layout, mixture.shards):
                raise SpecError(f"{w}: the segment lists the shards {list(listed[1])} of layout "
                                f"{listed[0]!r}, and its mixture {mixture.name!r} lists "
                                f"{list(mixture.shards)} of layout {mixture.layout!r}")
            out.append(SegmentPlan(run=run, name=segment, position=position,
                                   mixture=mixture.name))
    return out


def _inserted_lines(doc: dict, layouts: dict[str, LayoutPlan], where: str) -> list[InsertedLine]:
    """``[sanitizer_skip]``: one line that does not parse, written into the file the
    sanitizer reads for one shard in each layout the table lists."""
    if "sanitizer_skip" not in doc:
        return []
    table = _table(doc, "sanitizer_skip", where)
    w = f"{where} [sanitizer_skip]"
    shard = _value(table, "shard", str, w)
    line = _value(table, "line", int, w)
    text = _value(table, "text", str, w)
    if line < 1:
        raise SpecError(f"{w}: line is counted from 1, got {line}")
    if "\n" in text:
        raise SpecError(f"{w}: text is one line")
    probe = parse_line(text, artifact=None, line=line, source=None)
    if probe.readable or probe.fault == FAULT_BLANK:
        # A readable record would survive and shift every later ordinal; a blank
        # line would be logged as an empty line. The spec means neither.
        raise SpecError(f"{w}: text must be a line that does not parse, so that the "
                        "sanitizer skips it with the reason \"sanitizer skip\"")
    out = []
    for layout in _strings(table, "layouts", w):
        if layout not in layouts:
            raise SpecError(f"{w}: layout {layout!r} is not defined")
        if shard not in layouts[layout].shard_names():
            raise SpecError(f"{w}: layout {layout!r} has no shard {shard!r}")
        out.append(InsertedLine(layout=layout, shard=shard, line=line, text=text))
    return out


def load_fixture_spec(spec_path: Path, base: Path | None = None) -> BuildPlan:
    """Read the fixture spec.

    The file paths in the spec are relative to ``base``. By default that is the
    folder three levels above the spec file, the repository root when the spec lies
    at ``fixture/spec/`` under it."""
    spec_path = Path(spec_path)
    where = str(spec_path)
    doc = _load_toml(spec_path)
    if doc.get("format") != FIXTURE_FORMAT:
        raise SpecError(f"{where}: format must be {FIXTURE_FORMAT!r}, got {doc.get('format')!r}")
    base = Path(base) if base is not None else spec_path.resolve().parent.parent.parent

    tokenizer = _tokenizer(doc, where)
    normaliser_name = _value(_table(doc, "normaliser", where), "name", str, f"{where} [normaliser]")
    normaliser(normaliser_name)
    if _table(doc, "release_rows", where).get("carry_source_id") is not False:
        raise SpecError(f"{where}: the replica writes release rows that hold the text only "
                        "(release_rows.carry_source_id must be false)")

    sources = []
    for key, table in _table(doc, "sources", where).items():
        w = f"{where} [sources.{key}]"
        if not isinstance(table, dict):
            raise SpecError(f"{w} must be a table")
        file = Path(_value(table, "file", str, w))
        if file.stem != key:
            raise SpecError(f"{w}: the file of source {key} must be named {key}.jsonl")
        sources.append(SourceFile(
            key=key,
            path=base / file,
            source=key,
            member=_value(table, "member", str, w),
            revision=_value(table, "revision", str, w),
            records=tuple(expand_ids(_strings(table, "records", w))),
        ))

    handles: dict[str, str] = {}
    for handle, record_id in _table(doc, "handles", where).items():
        if not isinstance(record_id, str):
            raise SpecError(f"{where} [handles]: {handle} must name a record id")
        if record_id in handles:
            raise SpecError(f"{where} [handles]: {record_id} has two handles")
        handles[record_id] = handle

    generation_table = _table(doc, "generation", where)
    rows = _generation_rows(base / _value(generation_table, "file", str, f"{where} [generation]"))
    generation = tuple(GenerationFile(row.id, (row,)) for row in rows)
    model = _value(generation_table, "model", str, f"{where} [generation]")

    releases = []
    for name, table in _table(doc, "releases", where).items():
        w = f"{where} [releases.{name}]"
        if not isinstance(table, dict):
            raise SpecError(f"{w} must be a table")
        releases.append(ReleasePlan(
            name=name,
            inputs=tuple(expand_ids(_strings(table, "inputs", w))),
            dedup=_value(table, "dedup", bool, w),
            priority=_priority(table, w),
        ))

    imports = []
    for name, table in _table(doc, "imports", where, required=False).items():
        w = f"{where} [imports.\"{name}\"]"
        if not isinstance(table, dict):
            raise SpecError(f"{w} must be a table")
        imports.append(ImportPlan(
            name=name,
            records=tuple(expand_ids(_strings(table, "records", w))),
            family_member=_value(table, "file_family", str, w),
            evidence=_value(table, "evidence", str, w),
        ))

    layouts_table = _table(doc, "layouts", where)
    partitioned = _table(layouts_table, LAYOUT_PARTITIONED, f"{where} [layouts]")
    shards_table = _table(partitioned, "shards", f"{where} [layouts.partitioned]")
    ws = f"{where} [layouts.partitioned.shards]"
    shards = tuple(
        (name, tuple(expand_ids(_strings(shards_table, name, ws)))) for name in shards_table
    )
    layouts = {
        LAYOUT_PARTITIONED: LayoutPlan(
            name=LAYOUT_PARTITIONED,
            store_prefix=FIXTURE_RUN,
            shuffle_seed=_value(partitioned, "shuffle_seed", int, f"{where} [layouts.partitioned]"),
            groups=shards,
        )
    }
    if LAYOUT_ROUND_ROBIN in layouts_table:
        rr = _table(layouts_table, LAYOUT_ROUND_ROBIN, f"{where} [layouts]")
        w = f"{where} [layouts.round-robin]"
        # The same records as the partitioned shards, as one list (the spec's words:
        # "dealt over three chunks after one shuffle of the whole list").
        everything = tuple(item for _, items in shards for item in items)
        layouts[LAYOUT_ROUND_ROBIN] = LayoutPlan(
            name=LAYOUT_ROUND_ROBIN,
            store_prefix=FIXTURE_RUN + ROUND_ROBIN_SUFFIX,
            shuffle_seed=_value(rr, "shuffle_seed", int, w),
            groups=((ALL_RECORDS, everything),),
            chunks=_chunks(rr, w),
        )

    mixtures = _fixture_mixtures(doc, layouts, where)
    return BuildPlan(
        mode=MODE_FIXTURE,
        normaliser=normaliser_name,
        tokenizer=tokenizer,
        sources=tuple(sources),
        generation=generation,
        generation_model=model,
        handles=handles,
        releases=tuple(releases),
        imports=tuple(imports),
        layouts=layouts,
        mixtures=tuple(mixtures),
        segments=tuple(_fixture_segments(doc, mixtures, layouts, where)),
        inserted_lines=tuple(_inserted_lines(doc, layouts, where)),
    )


# --- A generated set ---------------------------------------------------------------


def load_set(set_dir: Path, tokenizer: TokenizerSettings | None = None) -> BuildPlan:
    """Read ``<set_dir>/set.toml`` as ``fixture/plant/generate.py`` writes it.

    The set names no tokenizer, no shuffle seed, no partitioned layout and no
    mixture. ``tokenizer`` is the tokenizer of the build (None: the build cannot
    tokenize). The shuffle seed is the set's own seed. The partitioned layout is
    one shard per source file and one for the generation outputs. docs/replica.md
    section 4 says this in full."""
    set_dir = Path(set_dir)
    path = set_dir / "set.toml"
    where = str(path)
    doc = _load_toml(path)
    if doc.get("format") != SET_FORMAT:
        raise SpecError(f"{where}: format must be {SET_FORMAT!r}, got {doc.get('format')!r}")
    normaliser_name = _value(_table(doc, "normaliser", where), "name", str, f"{where} [normaliser]")
    normaliser(normaliser_name)
    seed = _value(_table(doc, "parameters", where), "seed", int, f"{where} [parameters]")

    file_of: dict[str, str] = {}  # record id -> the key of the file that holds it

    def place(record_id: str, key: str, w: str) -> None:
        if record_id in file_of:
            raise SpecError(f"{w}: id {record_id!r} is used twice")
        file_of[record_id] = key

    sources = []
    for key, table in _table(doc, "sources", where).items():
        w = f"{where} [sources.{key}]"
        if not isinstance(table, dict):
            raise SpecError(f"{w} must be a table")
        records = tuple(expand_ids(_strings(table, "records", w)))
        lines = _value(table, "lines", int, w)
        if len(records) != lines:
            raise SpecError(f"{w}: records lists {len(records)} ids and lines is {lines}; "
                            "an id names a line")
        for record_id in records:
            place(record_id, key, w)
        sources.append(SourceFile(
            key=key,
            path=set_dir / _value(table, "file", str, w),
            source=_value(table, "source", str, w, key),
            member=_value(table, "member", str, w),
            revision=_value(table, "revision", str, w),
            records=records,
            by_line=True,
        ))

    generation: list[GenerationFile] = []
    generation_table = _table(doc, "generation", where)
    w = f"{where} [generation]"
    model = _value(generation_table, "model", str, w)
    rows = _generation_rows(set_dir / _value(generation_table, "file", str, w))
    declared = _value(generation_table, "rows", int, w)
    if declared != len(rows):
        raise SpecError(f"{w}: rows is {declared} and the file holds {len(rows)}")
    jobs = _table(generation_table, "jobs", w)
    for job in jobs:
        listed = expand_ids(_strings(jobs, job, f"{w} jobs"))
        made = tuple(row for row in rows if row.job == job)
        if [row.id for row in made] != listed:
            raise SpecError(f"{w}: jobs.{job} lists {len(listed)} ids and the file holds "
                            f"{len(made)} rows of that job, or their order differs")
        if not made:
            continue
        for row in made:
            place(row.id, f"gen/{job}", w)
        generation.append(GenerationFile(job, made))
    stray = sorted({row.job for row in rows} - set(jobs))
    if stray:
        raise SpecError(f"{w}: the file holds rows of jobs that [generation.jobs] does "
                        f"not name: {stray}")

    releases = []
    for release_name, table in _table(doc, "releases", where).items():
        w = f"{where} [releases.{release_name}]"
        if not isinstance(table, dict):
            raise SpecError(f"{w} must be a table")
        inputs = tuple(expand_ids(_strings(table, "inputs", w)))
        unknown = [i for i in inputs if i not in file_of]
        if unknown:
            raise SpecError(f"{w}: {len(unknown)} inputs name no line of a source file and "
                            f"no generation row (first: {unknown[:3]})")
        releases.append(ReleasePlan(
            name=release_name,
            inputs=inputs,
            dedup=_value(table, "dedup", bool, w),
            priority=_priority(table, w),
        ))
    by_name = {r.name: r for r in releases}
    if SET_PIPELINE_RELEASE not in by_name:
        raise SpecError(f"{where}: the pipeline chunks the rows of release "
                        f"{SET_PIPELINE_RELEASE}, which the set does not define")

    w = f"{where} [layouts.\"round-robin\"]"
    chunks = _chunks(_table(_table(doc, "layouts", where), LAYOUT_ROUND_ROBIN,
                            f"{where} [layouts]"), w)
    # The listed files of the round-robin layout: the files the release's inputs
    # come from, in the order the inputs first name them.
    listed_files = tuple(dict.fromkeys(file_of[i] for i in by_name[SET_PIPELINE_RELEASE].inputs))
    by_source = [(f"shard-{s.key}", (s.key,)) for s in sources]
    if generation:
        by_source.append((SET_GENERATION_SHARD, tuple(f"gen/{g.name}" for g in generation)))
    layouts = {
        LAYOUT_PARTITIONED: LayoutPlan(
            name=LAYOUT_PARTITIONED, store_prefix=SET_PREFIX, shuffle_seed=seed,
            groups=tuple(by_source)),
        LAYOUT_ROUND_ROBIN: LayoutPlan(
            name=LAYOUT_ROUND_ROBIN, store_prefix=SET_PREFIX + ROUND_ROBIN_SUFFIX,
            shuffle_seed=seed, groups=tuple((f, (f,)) for f in listed_files), chunks=chunks),
    }

    return BuildPlan(
        mode=MODE_SET,
        normaliser=normaliser_name,
        tokenizer=tokenizer,
        sources=tuple(sources),
        generation=tuple(generation),
        generation_model=model,
        handles={},
        releases=tuple(releases),
        imports=(),
        layouts=layouts,
        mixtures=(),
        pipeline_release=SET_PIPELINE_RELEASE,
    )
