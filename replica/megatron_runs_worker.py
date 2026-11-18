"""The run wrapper's work inside the Megatron environment. A script for that environment.

``replica.runs`` runs this file with the interpreter of the test-time environment.
The replica itself never imports it, because it imports Megatron and torch.

    python -I megatron_runs_worker.py PLAN.json RESULT.json

For every segment of the plan it does what a training launch does before its first
step (plan 3.3 "Training", 5.4):

- builds the training datasets with Megatron's own builder, ``BlendedMegatronDatasetBuilder``
  over ``GPTDataset`` and a ``GPTDatasetConfig`` made the way ``pretrain_gpt.py``
  makes one, with ``path_to_cache`` set, so that Megatron itself writes the
  data-order cache folder (docs/interfaces.md 8.4) into a new, empty folder;
- reads the kept ``.npy`` files back and takes the digest of each index array
  (``waterwhite.ids.array_digest``, plan 3.5), tying a file to its dataset through
  the description hash Megatron puts at the start of the file's name;
- for a cyclic run, computes the shuffle digest of every epoch each launch touches
  and every rank, from the dataset length Megatron reported (the launch's real
  size) and the launch's sizes. The draw is the replica's own call to torch, as
  Megatron's cyclic sampler makes it (``torch.Generator`` seeded with the epoch
  number, ``torch.randperm`` over the rank's bucket); Megatron's sampler class is
  not run and nothing of the tool's join is called. The digest is
  ``waterwhite.ids.order_digest`` of the whole epoch's order for the rank,
  whatever part of the epoch the launch read (docs/interfaces.md 8.1).

It imports nothing of the oracle. It writes one JSON file; what a library prints
goes to standard error.
"""

import argparse
import json
import logging
import os
import sys
from importlib import metadata

SHARD_KINDS = ("document_index", "sample_index", "shuffle_index")
BLEND_KINDS = ("dataset_index", "dataset_sample_index")


def _tokenizer(spec):
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
    from megatron.training.arguments import _add_tokenizer_args

    parser = argparse.ArgumentParser(add_help=False)
    _add_tokenizer_args(parser)
    args = parser.parse_args(["--tokenizer-type", spec["type"], "--vocab-file", spec["vocab_file"],
                              "--merge-file", spec["merges_file"]])
    # The four values tools/preprocess_data.py sets before it builds its tokenizer.
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    return build_tokenizer(args)


def _megatron_default_data_sharding():
    """The value ``--data-sharding`` has when a launch does not set it: read from
    the argument table Megatron's training code parses."""
    from megatron.training.arguments import _add_vision_args  # the group that holds --no-data-sharding

    parser = argparse.ArgumentParser(add_help=False)
    _add_vision_args(parser)
    return bool(parser.parse_args([]).data_sharding)


def _build(segment, tokenizer):
    """The builder call of ``train_valid_test_datasets_provider`` in pretrain_gpt.py,
    with the config fields ``core_gpt_dataset_config_from_args`` passes (the values
    Megatron's argument table defaults to where the launch states none)."""
    from megatron.core.datasets.blended_dataset import BlendedDataset
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

    prefixes = list(segment["shard_prefixes"])
    first = segment["launches"][0]
    config = GPTDatasetConfig(
        random_seed=segment["random_seed"],
        sequence_length=segment["sequence_length"],
        blend=(prefixes, None),
        blend_per_split=None,
        split=segment["split"],
        num_dataset_builder_threads=1,
        path_to_cache=segment["cache_dir"],
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        mid_level_dataset_surplus=0.005,
        allow_ambiguous_pad_tokens=False,
        add_extra_token_to_sequence=segment["add_extra_token_to_sequence"],
        data_parallel_size=first["data_parallel_size"],
    )
    matrix = config.split_matrix
    if matrix[0] is None or matrix[1] is not None or matrix[2] is not None:
        raise SystemExit(f"split {segment['split']!r} gives documents to validation or test")
    requested = segment["train_iters"] * segment["global_batch_size"]
    datasets = BlendedMegatronDatasetBuilder(GPTDataset, [requested, 0, 0], lambda: True, config).build()
    top = datasets[0]
    if len(prefixes) == 1:
        if not isinstance(top, GPTDataset):
            raise SystemExit(f"one shard should give a GPTDataset, got {type(top).__name__}")
        shards, blend = [top], None
    else:
        if not isinstance(top, BlendedDataset):
            raise SystemExit(f"several shards should give a BlendedDataset, got {type(top).__name__}")
        shards, blend = list(top.datasets), top
    if len(shards) != len(prefixes):
        raise SystemExit(f"{len(prefixes)} shards were named, Megatron built {len(shards)} datasets")
    for position, (dataset, prefix) in enumerate(zip(shards, prefixes)):
        if os.path.realpath(dataset.dataset.path_prefix) != os.path.realpath(prefix):
            raise SystemExit(f"the dataset at position {position} was not built from shard {position}")
    return top, shards, blend


def _one_file(names, owner_hash, ending, what):
    found = [n for n in names if n.startswith(owner_hash + "-") and n.endswith("-" + ending)]
    if len(found) != 1:
        raise SystemExit(f"expected one cache file for {what}, found {len(found)}")
    return found[0]


def _cache_digests(cache_dir, shards, blend):
    """Digest the index arrays Megatron wrote: three kinds per shard position, then
    the two of the blend (shard null). The arrays are read from the kept files."""
    import numpy

    from waterwhite import ids

    names = sorted(os.listdir(cache_dir))
    used, rows = set(), []

    def take(owner, kind, shard, what):
        name = _one_file(names, owner.unique_description_hash, kind + ".npy", what)
        array = numpy.load(os.path.join(cache_dir, name), allow_pickle=False)
        used.add(name)
        rows.append({"kind": kind, "shard": shard,
                     "digest": ids.array_digest(array.dtype.str, array.shape,
                                                numpy.ascontiguousarray(array).tobytes())})

    for position, dataset in enumerate(shards):
        for kind in SHARD_KINDS:
            take(dataset, kind, position, f"{kind} of shard {position}")
        used.add(_one_file(names, dataset.unique_description_hash, "description.txt",
                           f"the description of shard {position}"))
    if blend is not None:
        for kind in BLEND_KINDS:
            take(blend, kind, None, f"{kind} of the blend")
        used.add(_one_file(names, blend.unique_description_hash, "description.txt",
                           "the description of the blend"))
    extra = sorted(set(names) - used)
    if extra:
        raise SystemExit(f"the cache folder holds files the build should not write: {extra}")
    return rows


def epoch_order(total, micro, dp_size, rank, epoch, sharding):
    """The whole order of one epoch for one rank, drawn as Megatron's cyclic sampler
    draws it: a generator seeded with the epoch number, ``randperm`` over the
    bucket. Only whole micro batches are yielded, so a partial last one is dropped."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(epoch)
    if sharding:
        bucket = (total // (micro * dp_size)) * micro
        start = rank * bucket
        return [start + int(x) for x in torch.randperm(bucket, generator=generator).tolist()]
    full = (total // micro) * micro
    order = [int(x) for x in torch.randperm(full, generator=generator).tolist()][rank::dp_size]
    return order[: (len(order) // micro) * micro]


def _shuffle_digests(segment, total, sharding):
    from waterwhite import ids

    micro, gbs = segment["micro_batch_size"], segment["global_batch_size"]
    rows = []
    for launch in segment["launches"]:
        dp_size = launch["data_parallel_size"]
        per_round = micro * dp_size
        active = total - total % per_round  # the consumed count one epoch adds
        if active <= 0 or gbs % per_round or launch["consumed_at_start"] % per_round:
            raise SystemExit(f"segment {segment['name']!r}, launch {launch['position']}: sizes the "
                             "cyclic sampler cannot take")
        steps = launch["last_step"] - launch["step_at_start"]
        first_epoch = launch["consumed_at_start"] // active
        last_epoch = (launch["consumed_at_start"] + steps * gbs - 1) // active
        for epoch in range(first_epoch, last_epoch + 1):
            for rank in range(dp_size):
                order = epoch_order(total, micro, dp_size, rank, epoch, sharding)
                rows.append({"launch": launch["position"], "epoch": epoch, "rank": rank,
                             "digest": ids.order_digest(order), "count": len(order)})
    return rows


def _sampler_class(dataloader_type):
    from megatron.training.datasets import data_samplers

    cls = (data_samplers.MegatronPretrainingSampler if dataloader_type == "single"
           else data_samplers.MegatronPretrainingRandomSampler)
    return f"{cls.__module__}.{cls.__qualname__}"


def main():
    plan_path, result_path = sys.argv[1:3]
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    sys.path.insert(0, plan["src"])
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    import torch

    tokenizer = _tokenizer(plan["tokenizer"])
    default_sharding = _megatron_default_data_sharding()
    out = {"versions": {"megatron_core": metadata.version("megatron-core"), "torch": torch.__version__},
           "segments": []}
    for segment in plan["segments"]:
        cache_dir = segment["cache_dir"]
        if not os.path.isdir(cache_dir) or os.listdir(cache_dir):
            raise SystemExit(f"the cache folder must be new and empty: {cache_dir}")
        top, shards, blend = _build(segment, tokenizer)
        sharding = segment["data_sharding"]
        if sharding is None:
            sharding = default_sharding
        total = len(top)
        out["segments"].append({
            "name": segment["name"],
            "dataset_samples": total,
            "data_sharding": sharding,
            "sampler_class": _sampler_class(segment["dataloader_type"]),
            "cache_digests": _cache_digests(cache_dir, shards, blend),
            "shuffle_digests": (_shuffle_digests(segment, total, sharding)
                                if segment["dataloader_type"] == "cyclic" else []),
        })
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)


if __name__ == "__main__":
    main()
