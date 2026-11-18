"""Config files of the oracle's tools: reading, checking and path rules.

Every tool takes one JSON file. A path inside it is resolved like this:

- a path that starts with ``$WATERWHITE_MEGATRON_ENV`` has that part replaced by
  the value of the environment variable (the Megatron environment folder);
- any other relative path is relative to the folder that holds the config file;
- an absolute path is taken as written.

A key the tool does not know is an error, so a misspelt key cannot pass as a
default. The formats are described in docs/oracle.md.

Standard library only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from oracle.errors import ConfigError, EnvironmentNotReady

ENV_VARIABLE = "WATERWHITE_MEGATRON_ENV"
_ENV_PREFIX = "$" + ENV_VARIABLE

DATALOADER_SINGLE = "single"
DATALOADER_CYCLIC = "cyclic"
DATALOADER_TYPES = (DATALOADER_SINGLE, DATALOADER_CYCLIC)

# The words docs/interfaces.md and the store schema use for how a segment starts.
START_KINDS = ("fresh", "resume", "warm-start")

OUTPUT_KEYS = ("digests", "log", "summary", "run_record", "step_record")


# --- Small typed readers -------------------------------------------------------------


def _where(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _take(d: dict, key: str, where: str, *, required: bool = True, default=None):
    if key not in d:
        if required:
            raise ConfigError(f"missing key {_where(where, key)!r}")
        return default
    return d[key]


def _int(d: dict, key: str, where: str, *, minimum: int | None = None, **kw) -> int | None:
    value = _take(d, key, where, **kw)
    if value is None and not kw.get("required", True):
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{_where(where, key)!r} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{_where(where, key)!r} must be at least {minimum}, got {value}")
    return value


def _bool(d: dict, key: str, where: str, **kw) -> bool:
    value = _take(d, key, where, **kw)
    if not isinstance(value, bool):
        raise ConfigError(f"{_where(where, key)!r} must be true or false, got {value!r}")
    return value


def _str(d: dict, key: str, where: str, **kw) -> str | None:
    value = _take(d, key, where, **kw)
    if value is None and not kw.get("required", True):
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{_where(where, key)!r} must be a non-empty string, got {value!r}")
    return value


def _dict(d: dict, key: str, where: str, **kw) -> dict:
    value = _take(d, key, where, **kw)
    if not isinstance(value, dict):
        raise ConfigError(f"{_where(where, key)!r} must be an object, got {value!r}")
    return value


def _list(d: dict, key: str, where: str, **kw) -> list:
    value = _take(d, key, where, **kw)
    if not isinstance(value, list):
        raise ConfigError(f"{_where(where, key)!r} must be a list, got {value!r}")
    return value


def _only(d: dict, allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in {where or 'the config'}; known: {list(allowed)}")


# --- Paths ---------------------------------------------------------------------------


def megatron_env() -> Path:
    """The Megatron environment folder, from the environment variable."""
    value = os.environ.get(ENV_VARIABLE)
    if not value:
        raise EnvironmentNotReady(f"the environment variable {ENV_VARIABLE} is not set")
    path = Path(value)
    if not path.is_dir():
        raise EnvironmentNotReady(f"{ENV_VARIABLE} names a folder that does not exist")
    return path


def resolve_path(text: str, base: Path) -> Path:
    """Resolve a path of a config file by the rules at the top of this module."""
    if text == _ENV_PREFIX or text.startswith(_ENV_PREFIX + "/"):
        return megatron_env() / text[len(_ENV_PREFIX) :].lstrip("/")
    if text.startswith("$"):
        raise ConfigError(f"only {_ENV_PREFIX} is expanded in a path, got {text!r}")
    path = Path(text)
    # normpath folds "a/../b" without following links, so a path reads the same in
    # Megatron's cache description however the config spelt it.
    return Path(os.path.normpath(path if path.is_absolute() else base / path))


def load_json(path: str | Path) -> tuple[dict, Path]:
    """Read a JSON config file. Returns the object and the folder that holds it."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file is not JSON: {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"config file must hold one JSON object: {path}")
    return data, path.resolve().parent


def new_output_path(path: Path) -> Path:
    """Check that an output file can be written: it must not exist yet.

    The oracle never overwrites a file, for the reason it never reuses a cache
    folder: an old file left in place could pass as the result of this run."""
    if path.exists():
        raise ConfigError(f"output file exists already, name a new one: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: object) -> None:
    """Write a JSON file with sorted keys, so two runs give the same bytes."""
    new_output_path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


# --- The parts of a segment config -----------------------------------------------------


@dataclass(frozen=True)
class TokenizerSpec:
    """Which tokenizer Megatron builds. Only GPT-2 BPE is supported (DECISIONS.md
    D-09), with its two files named, so Megatron never tries to fetch them."""

    type: str
    vocab_file: Path
    merges_file: Path


@dataclass(frozen=True)
class DataOrder:
    """Everything the index build depends on, and where the cache goes."""

    shards: tuple[Path, ...]  # shard prefixes, in the trainer's order
    tokenizer: TokenizerSpec
    sequence_length: int
    add_extra_token_to_sequence: bool
    random_seed: int
    split: str
    global_batch_size: int
    train_iters: int
    train_samples: int | None
    cache_dir: Path

    @property
    def requested_samples(self) -> int:
        """The number of training samples the trainer asks the dataset builder for:
        ``train_samples`` when it is set, else ``train_iters`` times the global batch
        size. Megatron fixes this in ``get_train_valid_test_num_samples``
        (Megatron-LM ``megatron/training/training.py``; Megatron-Bridge
        ``data/loaders.py``)."""
        if self.train_samples is not None:
            return self.train_samples
        return self.train_iters * self.global_batch_size

    @property
    def sample_tokens(self) -> int:
        """Tokens the item getter reads per sample: the sequence length, plus one
        when the extra token is on."""
        return self.sequence_length + (1 if self.add_extra_token_to_sequence else 0)


@dataclass(frozen=True)
class Launch:
    """One start of the trainer inside a segment, with its real sizes and the train
    state it starts from (plan 3.1, F4)."""

    position: int
    world_size: int
    data_parallel_size: int
    consumed_at_start: int
    step_at_start: int
    steps: int

    @property
    def first_step(self) -> int:
        return self.step_at_start + 1

    @property
    def last_step(self) -> int:
        return self.step_at_start + self.steps


@dataclass(frozen=True)
class Segment:
    """One run segment: one shard list, one set of data-order settings, one or more
    launches, and what the tools write for it."""

    run: str
    segment: str
    start_kind: str
    parent_checkpoint: str | None
    data: DataOrder
    dataloader_type: str
    data_sharding: bool
    micro_batch_size: int
    micro_batches_per_step: int | None
    sampler_file: Path | None
    launches: tuple[Launch, ...]
    lowered: tuple[tuple[int, int], ...]  # (step, tokens read of its last sample)
    checkpoints: tuple[tuple[str, int], ...]  # (name, step)
    outputs: dict[str, Path]
    # False: a sequential launch that asks for more steps than its dataset holds is
    # refused (``stepping.check_launches_fit``). True: the pass is started again.
    allow_sequential_restart: bool = False

    def output(self, key: str) -> Path:
        if key not in self.outputs:
            raise ConfigError(f"segment {self.segment!r}: missing key 'outputs.{key}'")
        return self.outputs[key]

    def steps(self) -> list[int]:
        """Every completed step of the segment, in order."""
        return [s for launch in self.launches for s in range(launch.first_step, launch.last_step + 1)]

    def micro_batches(self, launch: Launch) -> int:
        """Micro batches each rank reads in one step of a launch.

        A step reads ``global_batch_size`` samples. With data-parallel size n and
        micro batch size m, each rank reads k micro batches of m samples, and
        k = global_batch_size / (m * n), which must divide. Megatron fixes this in
        ``ConstantNumMicroBatchesCalculator`` (``megatron/core/
        num_microbatches_calculator.py``), and the trainer adds n * m * k to the
        consumed-sample counter after each step (``megatron/training/training.py``).
        The config may state k as ``micro_batches_per_step``; it must agree."""
        per_round = self.micro_batch_size * launch.data_parallel_size
        gbs = self.data.global_batch_size
        if gbs % per_round != 0:
            raise ConfigError(
                f"segment {self.segment!r}, launch {launch.position}: global_batch_size {gbs} is "
                f"not a multiple of micro_batch_size {self.micro_batch_size} times "
                f"data_parallel_size {launch.data_parallel_size}"
            )
        k = gbs // per_round
        if self.micro_batches_per_step is not None and self.micro_batches_per_step != k:
            raise ConfigError(
                f"segment {self.segment!r}, launch {launch.position}: micro_batches_per_step "
                f"{self.micro_batches_per_step} does not give global_batch_size {gbs} "
                f"(it needs {k})"
            )
        return k


_SETTINGS_KEYS = (
    "dataloader_type",
    "data_sharding",
    "micro_batch_size",
    "micro_batches_per_step",
    "global_batch_size",
    "sequence_length",
    "add_extra_token_to_sequence",
    "random_seed",
    "split",
    "train_iters",
    "train_samples",
)
_SEGMENT_KEYS = (
    "run",
    "segment",
    "start_kind",
    "parent_checkpoint",
    "shards",
    "tokenizer",
    "settings",
    "sampler_file",
    "launches",
    "cache_dir",
    "outputs",
    "step_record",
    "allow_sequential_restart",
)
_LAUNCH_KEYS = ("world_size", "data_parallel_size", "consumed_at_start", "step_at_start", "steps")


def parse_tokenizer(raw: dict, base: Path, where: str = "tokenizer") -> TokenizerSpec:
    _only(raw, ("type", "vocab_file", "merges_file"), where)
    kind = _str(raw, "type", where)
    if kind != "GPT2BPETokenizer":
        raise ConfigError(
            f"{where}.type {kind!r} is not supported; the oracle builds 'GPT2BPETokenizer' only"
        )
    vocab = resolve_path(_str(raw, "vocab_file", where), base)
    merges = resolve_path(_str(raw, "merges_file", where), base)
    for label, path in (("vocab_file", vocab), ("merges_file", merges)):
        if not path.is_file():
            raise ConfigError(f"{where}.{label} is not a file: {path}")
    return TokenizerSpec(type=kind, vocab_file=vocab, merges_file=merges)


def parse_data_order(raw: dict, base: Path) -> DataOrder:
    """Read the part of a segment config the index build depends on. The index
    build and the replay need no more than this."""
    shards_raw = _list(raw, "shards", "")
    if not shards_raw:
        raise ConfigError("'shards' must name at least one shard prefix")
    shards = []
    for i, item in enumerate(shards_raw):
        if not isinstance(item, str) or not item:
            raise ConfigError(f"shards[{i}] must be a shard prefix, got {item!r}")
        prefix = resolve_path(item, base)
        for suffix in (".bin", ".idx"):
            if not Path(str(prefix) + suffix).is_file():
                raise ConfigError(f"shards[{i}]: no file {prefix}{suffix}")
        shards.append(prefix)
    if len({str(p.resolve()) for p in shards}) != len(shards):
        raise ConfigError("'shards' names the same shard twice")

    settings = _dict(raw, "settings", "")
    _only(settings, _SETTINGS_KEYS, "settings")
    split = _str(settings, "split", "settings")
    cache_dir = resolve_path(_str(raw, "cache_dir", ""), base)
    # Megatron's default is true (GPTDatasetConfig.add_extra_token_to_sequence). With
    # it off, Megatron's item getter writes the tokenizer's pad token id into the
    # last label, and the GPT-2 tokenizer has none: the getter stops with a
    # TypeError (seen in a run, docs/oracle.md). So the oracle reads no such run.
    if not _bool(settings, "add_extra_token_to_sequence", "settings", required=False, default=True):
        raise ConfigError(
            "settings.add_extra_token_to_sequence must be true: the oracle reads runs with "
            "the extra token on only"
        )
    return DataOrder(
        shards=tuple(shards),
        tokenizer=parse_tokenizer(_dict(raw, "tokenizer", ""), base),
        sequence_length=_int(settings, "sequence_length", "settings", minimum=1),
        add_extra_token_to_sequence=True,
        random_seed=_int(settings, "random_seed", "settings", minimum=0),
        split=split,
        global_batch_size=_int(settings, "global_batch_size", "settings", minimum=1),
        train_iters=_int(settings, "train_iters", "settings", minimum=1),
        train_samples=_int(settings, "train_samples", "settings", minimum=1, required=False),
        cache_dir=cache_dir,
    )


def _parse_launches(raw_list: list, segment: str) -> tuple[Launch, ...]:
    if not raw_list:
        raise ConfigError(f"segment {segment!r}: 'launches' must hold at least one launch")
    launches = []
    for i, raw in enumerate(raw_list):
        where = f"launches[{i}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{where} must be an object")
        _only(raw, _LAUNCH_KEYS, where)
        launch = Launch(
            position=i,
            world_size=_int(raw, "world_size", where, minimum=1),
            data_parallel_size=_int(raw, "data_parallel_size", where, minimum=1),
            consumed_at_start=_int(raw, "consumed_at_start", where, minimum=0),
            step_at_start=_int(raw, "step_at_start", where, minimum=0),
            steps=_int(raw, "steps", where, minimum=1),
        )
        if launch.world_size % launch.data_parallel_size != 0:
            raise ConfigError(
                f"{where}: data_parallel_size {launch.data_parallel_size} does not divide "
                f"world_size {launch.world_size}"
            )
        if launches and launch.first_step <= launches[-1].last_step:
            raise ConfigError(
                f"{where}: its first step {launch.first_step} is not after the last step "
                f"{launches[-1].last_step} of the launch before it; a step is logged once"
            )
        launches.append(launch)
    return tuple(launches)


def parse_segment(raw: dict, base: Path) -> Segment:
    """Read and check one whole segment config."""
    _only(raw, _SEGMENT_KEYS, "the segment config")
    segment = _str(raw, "segment", "")
    data = parse_data_order(raw, base)
    settings = raw["settings"]

    start_kind = _str(raw, "start_kind", "")
    if start_kind not in START_KINDS:
        raise ConfigError(f"start_kind must be one of {list(START_KINDS)}, got {start_kind!r}")
    parent = _str(raw, "parent_checkpoint", "", required=False)
    if start_kind == "fresh" and parent is not None:
        raise ConfigError("a fresh segment has no parent_checkpoint")
    # A warm start loads a checkpoint by name, so it has one to state. A resume may
    # continue a run whose checkpoint at that step was never registered; its record
    # then holds null, and ``check_starts`` says that its counters were not checked.
    # No run of the fixture does: every resume there names a registered checkpoint.
    if start_kind == "warm-start" and parent is None:
        raise ConfigError("a warm-start segment must name its parent_checkpoint")

    dataloader_type = _str(settings, "dataloader_type", "settings")
    if dataloader_type not in DATALOADER_TYPES:
        raise ConfigError(
            f"settings.dataloader_type must be one of {list(DATALOADER_TYPES)}, "
            f"got {dataloader_type!r}"
        )

    sampler_raw = _str(raw, "sampler_file", "", required=False)
    sampler_file = None
    if sampler_raw is not None:
        sampler_file = resolve_path(sampler_raw, base)
        if not sampler_file.is_file():
            raise ConfigError(f"sampler_file is not a file: {sampler_file}")

    launches = _parse_launches(_list(raw, "launches", ""), segment)
    all_steps = {s for launch in launches for s in range(launch.first_step, launch.last_step + 1)}

    step_raw = _dict(raw, "step_record", "", required=False, default={})
    _only(step_raw, ("lowered", "checkpoints"), "step_record")
    lowered = []
    for i, item in enumerate(_list(step_raw, "lowered", "step_record", required=False, default=[])):
        where = f"step_record.lowered[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be an object")
        _only(item, ("step", "last_sample_tokens_read"), where)
        step = _int(item, "step", where, minimum=1)
        tokens = _int(item, "last_sample_tokens_read", where, minimum=0)
        if step not in all_steps:
            raise ConfigError(f"{where}: step {step} is not a step of segment {segment!r}")
        if tokens >= data.sample_tokens:
            raise ConfigError(
                f"{where}: {tokens} does not lower a sample of {data.sample_tokens} tokens"
            )
        if any(step == s for s, _ in lowered):
            raise ConfigError(f"{where}: step {step} is lowered twice")
        lowered.append((step, tokens))
    checkpoints = []
    for i, item in enumerate(
        _list(step_raw, "checkpoints", "step_record", required=False, default=[])
    ):
        where = f"step_record.checkpoints[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be an object")
        _only(item, ("name", "step"), where)
        name = _str(item, "name", where)
        step = _int(item, "step", where, minimum=1)
        if step not in all_steps:
            raise ConfigError(f"{where}: step {step} is not a step of segment {segment!r}")
        checkpoints.append((name, step))

    outputs_raw = _dict(raw, "outputs", "", required=False, default={})
    _only(outputs_raw, OUTPUT_KEYS, "outputs")
    outputs = {}
    for key, value in outputs_raw.items():
        if not isinstance(value, str) or not value:
            raise ConfigError(f"outputs.{key} must be a path")
        outputs[key] = resolve_path(value, base)

    parsed = Segment(
        run=_str(raw, "run", ""),
        segment=segment,
        start_kind=start_kind,
        parent_checkpoint=parent,
        data=data,
        dataloader_type=dataloader_type,
        # Megatron-LM's default is true (the --no-data-sharding switch turns it off).
        data_sharding=_bool(settings, "data_sharding", "settings", required=False, default=True),
        micro_batch_size=_int(settings, "micro_batch_size", "settings", minimum=1),
        micro_batches_per_step=_int(
            settings, "micro_batches_per_step", "settings", minimum=1, required=False
        ),
        sampler_file=sampler_file,
        launches=launches,
        lowered=tuple(lowered),
        checkpoints=tuple(checkpoints),
        outputs=outputs,
        allow_sequential_restart=_bool(
            raw, "allow_sequential_restart", "", required=False, default=False
        ),
    )
    for launch in launches:
        parsed.micro_batches(launch)  # raises when the sizes do not fit
    return parsed


def load_segment(path: str | Path) -> Segment:
    raw, base = load_json(path)
    return parse_segment(raw, base)


def load_data_order(path: str | Path) -> tuple[DataOrder, dict, Path]:
    """Read a config for the index build or the replay. Returns the data-order
    part, the raw object and the config's folder."""
    raw, base = load_json(path)
    return parse_data_order(raw, base), raw, base


def merge_defaults(defaults: dict, segment: dict) -> dict:
    """A segment of a several-segment file over the file's ``defaults``: a key of
    the segment replaces the default, and the objects ``settings`` and
    ``tokenizer`` are merged key by key."""
    merged = dict(defaults)
    for key, value in segment.items():
        if key in ("settings", "tokenizer") and isinstance(value, dict):
            inner = dict(defaults.get(key, {}))
            inner.update(value)
            merged[key] = inner
        else:
            merged[key] = value
    return merged


# --- Several segments in one file -----------------------------------------------------


def _consumed_after(segment: Segment, step: int) -> int:
    for launch in segment.launches:
        if launch.first_step <= step <= launch.last_step:
            return launch.consumed_at_start + (step - launch.step_at_start) * segment.data.global_batch_size
    raise ConfigError(f"step {step} is not a step of segment {segment.segment!r}")


def check_starts(segments: list[Segment]) -> dict[str, str]:
    """Check how each segment of a file starts against the other segments.

    - A fresh segment starts at consumed count 0 and step 0.
    - A resume whose parent checkpoint is named in this file (in a segment's
      ``step_record.checkpoints``) must start at that checkpoint's step and at the
      consumed count the parent segment had reached there. A real resume takes
      both from the checkpoint (plan F4).
    - A warm start states its own counters; on Megatron-Bridge's path they restart
      at 0 (plan F4). They are taken as stated.

    Returns, per segment, what was checked."""
    by_checkpoint: dict[str, tuple[Segment, int]] = {}
    names = set()
    for segment in segments:
        if segment.segment in names:
            raise ConfigError(f"segment {segment.segment!r} appears twice")
        names.add(segment.segment)
        for name, step in segment.checkpoints:
            if name in by_checkpoint:
                raise ConfigError(f"checkpoint {name!r} is named by two segments")
            by_checkpoint[name] = (segment, step)
    checked = {}
    for segment in segments:
        first = segment.launches[0]
        if segment.start_kind == "fresh":
            if first.consumed_at_start != 0 or first.step_at_start != 0:
                raise ConfigError(
                    f"segment {segment.segment!r} is fresh but starts at consumed count "
                    f"{first.consumed_at_start}, step {first.step_at_start}"
                )
            checked[segment.segment] = "fresh: starts at consumed count 0 and step 0"
        elif segment.start_kind == "resume":
            if segment.parent_checkpoint is None:
                checked[segment.segment] = (
                    "resume: no parent checkpoint is named; the stated counters were not checked"
                )
                continue
            if segment.parent_checkpoint not in by_checkpoint:
                checked[segment.segment] = (
                    "resume: the parent checkpoint is not named in this file; "
                    "the stated counters were not checked"
                )
                continue
            parent, step = by_checkpoint[segment.parent_checkpoint]
            consumed = _consumed_after(parent, step)
            if first.step_at_start != step or first.consumed_at_start != consumed:
                raise ConfigError(
                    f"segment {segment.segment!r} resumes from {segment.parent_checkpoint!r} "
                    f"(step {step}, consumed count {consumed} in {parent.segment!r}) but states "
                    f"step {first.step_at_start}, consumed count {first.consumed_at_start}"
                )
            checked[segment.segment] = (
                f"resume: starts at step {step} and consumed count {consumed}, "
                f"the state of {segment.parent_checkpoint!r}"
            )
        else:
            checked[segment.segment] = "warm-start: the stated counters were taken as given"
    return checked


def parse_segments(raw: dict, base: Path) -> list[Segment]:
    """Read a several-segment file: ``defaults`` and ``segments``."""
    unknown = sorted(set(raw) - {"defaults", "segments", "summary"})
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in the config; known: defaults, segments, summary")
    defaults = raw.get("defaults", {})
    items = raw.get("segments")
    if not isinstance(defaults, dict) or not isinstance(items, list) or not items:
        raise ConfigError("the config needs 'segments', a non-empty list, and 'defaults', an object")
    segments = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ConfigError(f"segments[{i}] must be an object")
        segments.append(parse_segment(merge_defaults(defaults, item), base))
    return segments
