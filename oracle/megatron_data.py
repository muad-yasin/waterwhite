"""The one module that imports Megatron (plan 3.5, 5.3).

It builds Megatron's datasets with Megatron's own builder classes, instruments the
lowest read the item getter makes, loads Megatron's sampler classes, and digests
the index arrays Megatron wrote. It runs Megatron's code and holds none of it.

Which Megatron code is driven, at release core_v0.20.0:

- ``megatron.core.datasets.blended_megatron_dataset_builder.
  BlendedMegatronDatasetBuilder.build`` with ``GPTDataset`` and ``GPTDatasetConfig``
  builds one ``GPTDataset`` per shard and, for two or more shards, a
  ``BlendedDataset`` over them. The blend carries no weights, so each shard is
  built for one pass and the blend's length is the smaller of the requested
  samples and the pool.
- ``GPTDataset.__getitem__`` reads a sample through
  ``IndexedDataset.get(document, offset=, length=)``, once per document the sample
  touches. That call is what ``ReadTap`` instruments.
- ``megatron.training.datasets.data_samplers.MegatronPretrainingSampler`` and
  ``MegatronPretrainingRandomSampler``, or the classes of the same names in
  Megatron-Bridge's ``data/samplers.py`` loaded by file path.
- ``megatron.core.tokenizers.utils.build_tokenizer.build_tokenizer`` with the
  argument defaults of ``megatron.training.arguments._add_tokenizer_args``, as
  ``tools/preprocess_data.py`` builds its tokenizer.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from oracle.config import DATALOADER_SINGLE, DataOrder, Launch, Segment, TokenizerSpec
from oracle.digests import BLEND_KINDS, SHARD_KINDS, array_digest
from oracle.errors import (
    CacheFolderNotEmpty,
    ConfigError,
    EnvironmentNotReady,
    LaunchOutrunsDataset,
    OracleError,
    ReadMismatch,
    SamplerRefused,
    UnexpectedCacheContent,
)


def _require_offline() -> None:
    """The tokenizer is built through the Hugging Face library, which asks the hub
    unless it is told not to, and writes under the home folder unless HF_HOME says
    otherwise. Both must be set before that library is imported."""
    if not os.environ.get("HF_HOME"):
        raise EnvironmentNotReady("HF_HOME is not set (point it at hf_home/ of the environment)")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise EnvironmentNotReady("HF_HUB_OFFLINE must be 1: the oracle uses no network")


_require_offline()

import numpy  # noqa: E402
import torch  # noqa: E402

try:
    # The compiled index helpers. Megatron's install marks them optional, so a
    # failed compile passes silently at install time; this import is the check.
    import megatron.core.datasets.helpers_cpp  # noqa: E402, F401
except ImportError as e:  # pragma: no cover - depends on the environment
    raise EnvironmentNotReady(f"Megatron's compiled index helpers do not import: {e}") from e

from megatron.core.datasets.blended_dataset import BlendedDataset  # noqa: E402
from megatron.core.datasets.blended_megatron_dataset_builder import (  # noqa: E402
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig  # noqa: E402
from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer  # noqa: E402
from megatron.training.arguments import _add_tokenizer_args  # noqa: E402
from megatron.training.datasets import data_samplers as _lm_samplers  # noqa: E402

# The dotted name Megatron-Bridge gives its sampler module. The file is loaded by
# path under this name, so a run record names the class as a real launch would.
BRIDGE_SAMPLER_MODULE = "megatron.bridge.data.samplers"
_BRIDGE_SAMPLER_PATH_END = ("megatron", "bridge", "data", "samplers.py")

_TAP_MARK = "_oracle_read_tap"


# --- Tokenizer -----------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def build_megatron_tokenizer(spec: TokenizerSpec):
    """Build the tokenizer with Megatron's own ``build_tokenizer``.

    The four values set after parsing are the ones ``tools/preprocess_data.py``
    sets before it builds its tokenizer; they size a padded vocabulary and do not
    change a token."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_tokenizer_args(parser)
    args = parser.parse_args(
        [
            "--tokenizer-type",
            spec.type,
            "--vocab-file",
            str(spec.vocab_file),
            "--merge-file",
            str(spec.merges_file),
        ]
    )
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    return build_tokenizer(args)


def open_shard(prefix: Path) -> IndexedDataset:
    """Megatron's reader of one tokenized shard (``.bin`` and ``.idx``)."""
    return IndexedDataset(str(prefix))


# --- Index build -----------------------------------------------------------------------


@dataclass
class Built:
    """The datasets of one segment as Megatron built them."""

    data: DataOrder
    top: object  # what the sampler indexes: the blend, or the one GPTDataset
    shards: list  # the GPTDataset of each shard, by position in the shard list
    blend: object | None

    tap: object | None = None  # the one ReadTap of these datasets, once attached

    @property
    def samples(self) -> int:
        return len(self.top)


def prepare_cache_dir(path: Path) -> None:
    """Make the cache folder, or accept it when it exists and is empty.

    A folder that holds anything is refused. Megatron loads index files it finds
    without checking them against the data (its cache key holds paths, not
    content), so an old file would be read back as if it had just been built."""
    if path.exists():
        if not path.is_dir():
            raise CacheFolderNotEmpty(f"cache_dir is not a folder: {path}")
        if any(path.iterdir()):
            raise CacheFolderNotEmpty(f"cache_dir must be new or empty, it holds files: {path}")
    else:
        path.mkdir(parents=True)


def build_train_dataset(data: DataOrder) -> Built:
    """Build the training dataset of a segment into its new, empty cache folder.

    ``blend=(prefixes, None)`` is a blend with no weights. Megatron builds a
    ``BlendedDataset`` for two or more prefixes and a single ``GPTDataset`` for
    one. Only a split that gives every document to training is accepted: the run
    record keys a cache digest by kind and shard and has no place for a split."""
    prepare_cache_dir(data.cache_dir)
    prefixes = [str(p) for p in data.shards]
    config = GPTDatasetConfig(
        random_seed=data.random_seed,
        sequence_length=data.sequence_length,
        blend=(prefixes, None),
        split=data.split,
        path_to_cache=str(data.cache_dir),
        tokenizer=build_megatron_tokenizer(data.tokenizer),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        add_extra_token_to_sequence=data.add_extra_token_to_sequence,
    )
    matrix = config.split_matrix
    if matrix[0] is None or matrix[1] is not None or matrix[2] is not None:
        raise ConfigError(
            f"settings.split {data.split!r} gives documents to validation or test; the oracle "
            "reads the training split of a split such as '1,0,0' only"
        )
    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset, [data.requested_samples, 0, 0], lambda: True, config
    ).build()
    top = datasets[0]
    if len(prefixes) == 1:
        if not isinstance(top, GPTDataset):
            raise OracleError(f"one shard should give a GPTDataset, got {type(top).__name__}")
        shards, blend = [top], None
    else:
        if not isinstance(top, BlendedDataset):
            raise OracleError(f"several shards should give a BlendedDataset, got {type(top).__name__}")
        shards, blend = list(top.datasets), top
    if len(shards) != len(prefixes):
        raise OracleError(f"{len(prefixes)} shards were named, Megatron built {len(shards)} datasets")
    for position, (dataset, prefix) in enumerate(zip(shards, prefixes)):
        built_from = os.path.realpath(dataset.dataset.path_prefix)
        if not isinstance(dataset, GPTDataset) or built_from != os.path.realpath(prefix):
            raise OracleError(
                f"the dataset at position {position} was not built from shard {position} of the list"
            )
    return Built(data=data, top=top, shards=shards, blend=blend)


def _one_file(names: list[str], owner_hash: str, ending: str, what: str) -> str:
    found = [n for n in names if n.startswith(owner_hash + "-") and n.endswith("-" + ending)]
    if len(found) != 1:
        raise UnexpectedCacheContent(f"expected one cache file for {what}, found {len(found)}")
    return found[0]


def cache_digests(built: Built) -> list[dict]:
    """Digest the index arrays Megatron wrote for a segment.

    One row per array: three kinds per shard, keyed by the shard's position, and
    two for the blend when there is one (``shard`` null). A file is tied to its
    dataset through the dataset's description hash, with which Megatron starts the
    file's name. Each array on disk must equal the array the dataset object holds,
    and the folder must hold these files and their description files only."""
    cache_dir = built.data.cache_dir
    names = sorted(os.listdir(cache_dir))
    used: set[str] = set()
    rows: list[dict] = []

    def digest(owner, kind: str, shard: int | None, what: str) -> None:
        name = _one_file(names, owner.unique_description_hash, kind + ".npy", what)
        on_disk = numpy.load(os.path.join(cache_dir, name), allow_pickle=False)
        in_memory = numpy.asarray(getattr(owner, kind))
        if on_disk.dtype != in_memory.dtype or not numpy.array_equal(on_disk, in_memory):
            raise UnexpectedCacheContent(f"the cache file for {what} differs from the built array")
        used.add(name)
        rows.append(
            {
                "kind": kind,
                "shard": shard,
                "digest": array_digest(
                    on_disk.dtype.str, on_disk.shape, numpy.ascontiguousarray(on_disk).tobytes()
                ),
                "dtype": on_disk.dtype.str,
                "shape": [int(n) for n in on_disk.shape],
            }
        )

    for position, dataset in enumerate(built.shards):
        for kind in SHARD_KINDS:
            digest(dataset, kind, position, f"{kind} of shard {position}")
        used.add(
            _one_file(
                names,
                dataset.unique_description_hash,
                "description.txt",
                f"the description of shard {position}",
            )
        )
    if built.blend is not None:
        for kind in BLEND_KINDS:
            digest(built.blend, kind, None, f"{kind} of the blend")
        used.add(
            _one_file(
                names,
                built.blend.unique_description_hash,
                "description.txt",
                "the description of the blend",
            )
        )
    extra = sorted(set(names) - used)
    if extra:
        raise UnexpectedCacheContent(f"the cache folder holds files the build should not write: {extra}")
    return rows


def record_rows(rows: list[dict]) -> list[dict]:
    """The rows as a run record holds them: kind, shard and digest."""
    return [{"kind": r["kind"], "shard": r["shard"], "digest": r["digest"]} for r in rows]


# --- The read tap ------------------------------------------------------------------------


@dataclass(frozen=True)
class Read:
    """One call of ``IndexedDataset.get`` made while a sample was fetched."""

    shard_position: int
    ordinal: int  # the document number the call named
    token_start: int  # the offset the call named
    token_end: int  # the offset plus the number of tokens the call returned
    tokens: object  # the returned tokens


class ReadTap:
    """Records the reads Megatron's item getter makes.

    Each shard's ``IndexedDataset`` gets its ``get`` method replaced, on the object,
    by a function that calls the original and notes the document number, the
    offset and the number of tokens returned. Nothing about a span is computed
    here: a row exists because Megatron made that call."""

    def __init__(self, built: Built) -> None:
        self._built = built
        self._open: list[Read] | None = None
        for position, dataset in enumerate(built.shards):
            self._instrument(position, dataset.dataset)

    @classmethod
    def attach(cls, built: Built) -> "ReadTap":
        """The tap of these datasets: made on the first call, the same one after,
        so a read is never recorded twice."""
        if built.tap is None:
            built.tap = cls(built)
        return built.tap

    def _instrument(self, position: int, low: IndexedDataset) -> None:
        if getattr(low, _TAP_MARK, False):
            raise OracleError("this dataset is instrumented already; build a new one per tap")
        original = low.get
        lengths = low.sequence_lengths

        def tapped_get(idx, offset=0, length=None):
            out = original(idx, offset=offset, length=length)
            if self._open is None:
                raise OracleError("the indexed dataset was read outside a sample fetch")
            if isinstance(out, tuple):
                raise OracleError("a multimodal shard is not supported by the oracle")
            start = int(offset)
            end = start + len(out)
            if length is not None and len(out) != int(length):
                raise ReadMismatch(
                    f"shard {position}, document {int(idx)}: {int(length)} tokens were asked "
                    f"for, {len(out)} came back"
                )
            if end > int(lengths[int(idx)]):
                raise ReadMismatch(
                    f"shard {position}, document {int(idx)}: the read ends at token {end}, past "
                    f"the document's {int(lengths[int(idx)])} tokens"
                )
            self._open.append(Read(position, int(idx), start, end, out))
            return out

        low.get = tapped_get
        setattr(low, _TAP_MARK, True)

    def fetch(self, index: int):
        """Ask Megatron's item getter for one sample. Returns the item and the
        reads it made, after checking them against each other."""
        if self._open is not None:
            raise OracleError("a sample fetch is open already")
        self._open = []
        try:
            item = self._built.top[index]
        finally:
            reads, self._open = self._open, None
        self._check(index, item, reads)
        return item, reads

    def _check(self, index: int, item: dict, reads: list[Read]) -> None:
        """The tokens the item getter returned must be the logged reads, in order.

        The extra token is on (the config refuses a run without it), so the item
        holds ``tokens`` (the sample without its last token) and ``labels`` (the
        sample without its first): the sample is ``tokens`` plus the last label.
        Padding, or a read the tap did not see, makes the two differ, and the run
        stops."""
        data = self._built.data
        returned = torch.cat([item["tokens"], item["labels"][-1:]]).numpy().astype(numpy.int64)
        if not reads:
            raise ReadMismatch(f"sample {index}: the item getter made no read")
        logged = numpy.concatenate([numpy.asarray(r.tokens).astype(numpy.int64) for r in reads])
        if len(returned) != data.sample_tokens:
            raise ReadMismatch(
                f"sample {index}: the item getter returned {len(returned)} tokens, "
                f"expected {data.sample_tokens}"
            )
        if len(logged) != len(returned) or not numpy.array_equal(logged, returned):
            raise ReadMismatch(
                f"sample {index}: the returned tokens are not the concatenation of the "
                f"{len(reads)} logged reads ({len(logged)} tokens logged)"
            )
        positions = {r.shard_position for r in reads}
        if len(positions) != 1:
            raise ReadMismatch(f"sample {index}: its reads touch several shards: {sorted(positions)}")
        if self._built.blend is not None and int(item["dataset_id"]) not in positions:
            raise ReadMismatch(
                f"sample {index}: the blend names dataset {int(item['dataset_id'])}, the reads "
                f"went to shard {sorted(positions)[0]}"
            )


# --- Samplers --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Samplers:
    """The two sampler classes of one code base."""

    sequential: type
    cyclic: type
    bridge_file: Path | None  # None: Megatron-LM's own classes

    def for_type(self, dataloader_type: str) -> type:
        return self.sequential if dataloader_type == DATALOADER_SINGLE else self.cyclic

    def class_name(self, dataloader_type: str) -> str:
        cls = self.for_type(dataloader_type)
        return f"{cls.__module__}.{cls.__qualname__}"


@functools.lru_cache(maxsize=None)
def load_samplers(bridge_file: Path | None) -> Samplers:
    """Megatron-LM's sampler classes, or Megatron-Bridge's when its file is named.

    Megatron-Bridge is not installed in the environment. Its sampler file imports
    the standard library, numpy and torch only, so it is executed from its path."""
    if bridge_file is None:
        return Samplers(
            sequential=_lm_samplers.MegatronPretrainingSampler,
            cyclic=_lm_samplers.MegatronPretrainingRandomSampler,
            bridge_file=None,
        )
    if tuple(bridge_file.parts[-4:]) != _BRIDGE_SAMPLER_PATH_END:
        raise ConfigError(
            "sampler_file must be Megatron-Bridge's own file, .../megatron/bridge/data/samplers.py"
        )
    spec = importlib.util.spec_from_file_location(BRIDGE_SAMPLER_MODULE, bridge_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return Samplers(
            sequential=module.MegatronPretrainingSampler,
            cyclic=module.MegatronPretrainingRandomSampler,
            bridge_file=bridge_file,
        )
    except AttributeError as e:
        raise ConfigError(f"sampler_file does not define the two sampler classes: {e}") from e


def make_sampler(
    samplers: Samplers,
    segment: Segment,
    launch: Launch,
    rank: int,
    built: Built,
    *,
    consumed_samples: int | None = None,
):
    """One sampler for one rank of one launch, with the arguments Megatron's
    ``build_pretraining_data_loader`` passes. That factory is not called: Megatron-
    LM's reads global arguments and the process groups of a distributed launch, so
    the rank and the data-parallel size are given here instead (plan F5).

    ``consumed_samples`` replaces the launch's consumed count. Only the order
    digests use it, to drive the class from the start of an epoch."""
    common = dict(
        total_samples=built.samples,
        consumed_samples=launch.consumed_at_start if consumed_samples is None else consumed_samples,
        micro_batch_size=segment.micro_batch_size,
        data_parallel_rank=rank,
        data_parallel_size=launch.data_parallel_size,
    )
    try:
        if segment.dataloader_type == DATALOADER_SINGLE:
            return samplers.sequential(**common)
        return samplers.cyclic(built.top, data_sharding=segment.data_sharding, **common)
    except AssertionError as e:
        raise SamplerRefused(
            f"segment {segment.segment!r}, launch {launch.position}, rank {rank}: Megatron's "
            f"sampler refused its arguments ({common}): {e}"
        ) from e


class RankStream:
    """The micro batches one rank draws in one launch.

    The trainer wraps its loader so that it starts a new pass when one ends
    (``cyclic_iter`` in Megatron-LM's ``megatron/training/training.py`` and in
    Megatron-Bridge's ``data/loaders.py``). This class does the same: when the
    iterator ends, it asks the iterable for a new one. What a new pass yields is
    the sampler's own doing: the cyclic sampler moves to its next epoch, and the
    sequential sampler starts again from the consumed count it was built with.

    ``restart_allowed`` false makes the end of a pass an error instead. It is how a
    sequential launch that outruns its dataset is stopped even if the count made
    before the walk (``stepping.check_launches_fit``) missed it."""

    def __init__(self, sampler, iterable, where: str, *, restart_allowed: bool = True) -> None:
        self.sampler = sampler
        self._iterable = iterable
        self._iterator = None
        self._where = where
        self._batches_in_pass = 0
        self._restart_allowed = restart_allowed
        self.passes = 0

    def next(self):
        while True:
            if self._iterator is None:
                self._iterator = iter(self._iterable)
                self._batches_in_pass = 0
                self.passes += 1
            try:
                batch = next(self._iterator)
            except StopIteration:
                if self._batches_in_pass == 0:
                    raise SamplerRefused(
                        f"{self._where}: the sampler gave no batch in a whole pass"
                    ) from None
                if not self._restart_allowed:
                    raise LaunchOutrunsDataset(
                        f"{self._where}: the sequential sampler ended after "
                        f"{self._batches_in_pass} micro batches and the launch asks for more; "
                        "set allow_sequential_restart to start the pass again"
                    ) from None
                self._iterator = None
                continue
            except AssertionError as e:
                raise SamplerRefused(f"{self._where}: Megatron's sampler stopped: {e!r}") from e
            self._batches_in_pass += 1
            return batch

    @property
    def epoch(self) -> int | None:
        """The epoch the cyclic sampler says its current pass is in. The sequential
        sampler has none."""
        epoch = getattr(self.sampler, "epoch", None)
        return None if epoch is None else int(epoch)


class TappedDataset(torch.utils.data.Dataset):
    """The dataset handed to torch's DataLoader for one rank: it passes each index
    to Megatron's item getter through the tap and keeps what was read."""

    def __init__(self, built: Built, tap: ReadTap) -> None:
        self._built = built
        self._tap = tap
        self._fetched: list[tuple[int, list[Read]]] = []

    def __len__(self) -> int:
        return self._built.samples

    def __getitem__(self, index):
        item, reads = self._tap.fetch(int(index))
        self._fetched.append((int(index), reads))
        return item

    def take(self) -> list[tuple[int, list[Read]]]:
        fetched, self._fetched = self._fetched, []
        return fetched


def make_loader(dataset: TappedDataset, sampler):
    """A DataLoader as Megatron's factories build theirs: the sampler is the batch
    sampler and there are no worker processes, so every read happens here.
    ``pin_memory`` is left off; it moves no data on a machine with no accelerator."""
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
