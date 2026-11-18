"""The oracle run against Megatron's own code, in this process.

Every test here builds Megatron's datasets over a small throwaway corpus (see
conftest.py). What is compared with what is said in each test: the oracle's log
against the raw shard bytes and against Megatron's uninstrumented item getter, its
orders against Megatron's sampler classes driven directly, and its digests
against the tool's definition of them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oracle import config as oracle_config
from oracle.errors import (
    CacheFolderNotEmpty,
    ConfigError,
    LaunchOutrunsDataset,
    ReadMismatch,
    SamplerRefused,
)

pytestmark = pytest.mark.megatron

PER_SHARD = ("document_index", "sample_index", "shuffle_index")
BLEND = ("dataset_index", "dataset_sample_index")


def parse(make, **kw) -> oracle_config.Segment:
    return oracle_config.parse_segment(make(**kw), make.base)


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def consume(segment):
    """Build a segment's index and run the consumer. Returns (rows, summary, built)."""
    from oracle import pipeline

    built, _document = pipeline.build_index(segment.data)
    summary = pipeline.consume_segment(segment, built)
    return read_log(segment.output("log")), summary, built


def samples_of(rows: list[dict]) -> list[tuple[tuple, list[dict]]]:
    """Group log rows into samples, keeping the order of the log."""
    out: list[tuple[tuple, list[dict]]] = []
    for row in rows:
        key = (row["step"], row["sample_pos"], row["global_index"], row["rank"])
        if not out or out[-1][0] != key:
            out.append((key, []))
        out[-1][1].append(row)
    return out


def by_kind(rows: list[dict]) -> dict:
    return {(r["kind"], r["shard"]): r["digest"] for r in rows}


def lm_orders(kind: str, *, total, consumed, micro_batch_size, rank, dp, sharding=True, passes=1):
    """Drive Megatron-LM's sampler class directly, with no oracle code between:
    the index lists of ``passes`` passes of one rank."""
    from megatron.training.datasets import data_samplers

    common = dict(total_samples=total, consumed_samples=consumed, micro_batch_size=micro_batch_size,
                  data_parallel_rank=rank, data_parallel_size=dp)
    if kind == "single":
        sampler = data_samplers.MegatronPretrainingSampler(**common)
    else:
        sampler = data_samplers.MegatronPretrainingRandomSampler(None, data_sharding=sharding, **common)
    return [[int(i) for batch in sampler for i in batch] for _ in range(passes)]


# --- Environment ------------------------------------------------------------------------------


def test_megatron_helpers_and_tokenizer_import(corpus):
    """Driven: megatron.core.datasets.helpers_cpp (import), build_tokenizer."""
    import megatron.core.datasets.helpers_cpp  # noqa: F401

    from oracle import environment, megatron_data

    tokenizer = megatron_data.build_megatron_tokenizer(oracle_config.parse_tokenizer(corpus.tokenizer, Path(".")))
    assert int(tokenizer.eod) == 50256
    assert environment.verify_installed_matches_checkout() == list(environment.RULE_FILES)
    assert environment.versions(None)["megatron_bridge"] is None


# --- Index build (plan F3, Check 2) ---------------------------------------------------------------


def test_index_build_repeats_and_follows_the_seed(segment_config):
    """Driven: BlendedMegatronDatasetBuilder.build with GPTDataset; BlendedDataset.

    Two builds into two new folders give equal digests for all five kinds. Another
    seed changes the three kinds of each shard and neither of the blend's two."""
    import numpy

    from oracle import pipeline
    from waterwhite import ids

    first = parse(segment_config)
    built, one = pipeline.build_index(first.data)
    _, two = pipeline.build_index(parse(segment_config).data)
    _, other_seed = pipeline.build_index(parse(segment_config, random_seed=99).data)

    a, b, c = (by_kind(d["cache_digests"]) for d in (one, two, other_seed))
    assert set(a) == {(k, s) for s in (0, 1) for k in PER_SHARD} | {(k, None) for k in BLEND}
    assert a == b
    for shard in (0, 1):
        for kind in PER_SHARD:
            assert a[(kind, shard)] != c[(kind, shard)], (kind, shard)
    for kind in BLEND:
        assert a[(kind, None)] == c[(kind, None)], kind

    # The oracle's digests are the tool's definition applied to the files on disk.
    cache = first.data.cache_dir
    from_files = set()
    for path in cache.glob("*.npy"):
        array = numpy.load(path)
        from_files.add(ids.array_digest(array.dtype.str, array.shape, numpy.ascontiguousarray(array).tobytes()))
    assert from_files == set(a.values())
    # The folder is kept as Megatron wrote it: five kinds of array and a
    # description per dataset; the blend's description names the shards in order.
    assert len(list(cache.iterdir())) == 4 + 4 + 3
    blend_description = next(cache.glob("*BlendedDataset*description.txt")).read_text()
    assert blend_description.index("alpha_text_document") < blend_description.index("beta_text_document")
    assert one["dataset_samples"] == built.samples == 60
    assert [s["samples"] for s in one["shards"]] == [36, 24]


def test_cache_folder_must_be_new_or_empty(segment_config):
    from oracle import pipeline

    segment = parse(segment_config)
    pipeline.build_index(segment.data)
    with pytest.raises(CacheFolderNotEmpty):
        pipeline.build_index(segment.data)


def test_one_shard_builds_no_blend(segment_config):
    """Driven: the builder's single-prefix path, which returns a GPTDataset."""
    segment = parse(segment_config, shards=("gamma",), train_iters=10)
    rows, summary, built = consume(segment)
    assert built.blend is None
    from oracle.megatron_data import cache_digests

    assert [(r["kind"], r["shard"]) for r in cache_digests(built)] == [(k, 0) for k in PER_SHARD]
    assert {r["shard_position"] for r in rows} == {0}
    assert summary["samples_read"] == 40 and built.samples >= 40


def test_a_split_with_validation_documents_is_refused(segment_config):
    from oracle import pipeline

    with pytest.raises(ConfigError, match="training split"):
        pipeline.build_index(parse(segment_config, split="9,1,0").data)


# --- The sequential sampler ------------------------------------------------------------------------


def test_sequential_log_equals_the_bin_tokens(corpus, segment_config):
    """Driven: MegatronPretrainingSampler, GPTDataset.__getitem__, IndexedDataset.get.

    The log's (shard, document, token range) rows are applied to the raw bytes of
    the .bin files, with document boundaries taken from an independent
    tokenization of the input lines. Each sample must then be 33 tokens and equal
    what Megatron's item getter returns from a second, uninstrumented build."""
    import numpy
    import torch
    from transformers import AutoTokenizer

    from oracle import megatron_data

    names = ("alpha", "beta")
    hf = AutoTokenizer.from_pretrained("gpt2")
    flat, starts = [], []
    for name in names:
        texts = [json.loads(line)["text"] for line in corpus.inputs[name].read_text().splitlines()]
        lengths = [len(hf(text)["input_ids"]) + 1 for text in texts]  # plus the end-of-document token
        tokens = numpy.fromfile(str(corpus.prefixes[name]) + ".bin", dtype=numpy.uint16)
        assert len(tokens) == sum(lengths)
        flat.append(tokens)
        starts.append(numpy.concatenate([[0], numpy.cumsum(lengths)]))

    rows, summary, _built = consume(parse(segment_config))
    plain = megatron_data.build_train_dataset(parse(segment_config).data)  # no tap on this one
    assert plain.tap is None

    samples = samples_of(rows)
    assert len(samples) == summary["samples_read"] == 60
    for n, ((step, sample_pos, global_index, rank), sample_rows) in enumerate(samples):
        # a step reads the next four samples in a row, counted from the consumed count
        assert (step, sample_pos, global_index, rank) == (n // 4 + 1, n % 4, n, 0)
        parts = []
        for row in sample_rows:
            begin = starts[row["shard_position"]][row["ordinal"]]
            parts.append(flat[row["shard_position"]][begin + row["token_start"] : begin + row["token_end"]])
        from_bin = numpy.concatenate(parts).astype(numpy.int64)
        assert len(from_bin) == 33
        item = plain.top[global_index]
        returned = torch.cat([item["tokens"], item["labels"][-1:]]).numpy()
        assert numpy.array_equal(from_bin, returned), (step, sample_pos)
        assert {r["shard_position"] for r in sample_rows} == {int(item["dataset_id"])}
    assert summary["sample_tokens"] == 33
    assert summary["sampler_class"] == "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
    assert summary["a_pass_ended_inside_a_launch"] is False


def test_a_resume_reads_the_tail_of_the_full_stream(segment_config):
    """A launch that starts at consumed count 24 reads what the full run reads
    from its seventh step on."""
    full, _, _ = consume(parse(segment_config))
    resumed, summary, _ = consume(
        parse(
            segment_config,
            launches=[{"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24, "step_at_start": 6, "steps": 9}],
        )
    )
    assert resumed == [row for row in full if row["step"] >= 7]
    assert resumed[0]["global_index"] == 24 and resumed[0]["step"] == 7
    assert summary["samples_read"] == 36


def test_sequential_ranks_and_micro_batches(segment_config):
    """Data-parallel size 2, micro batch 2, global batch 8: two micro batches per
    rank per step. The step still reads eight consecutive samples, and each
    sample's rank is the one Megatron's sampler gives that slice."""
    launches = [{"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0, "step_at_start": 0, "steps": 6}]
    rows, summary, built = consume(
        parse(segment_config, micro_batch_size=2, global_batch_size=8, train_iters=6, launches=launches)
    )
    assert built.samples == 48
    assert summary["launches"][0]["micro_batches_per_step"] == 2
    keys = [key for key, _ in samples_of(rows)]
    assert [(step, pos, index) for step, pos, index, _ in keys] == [(n // 8 + 1, n % 8, n) for n in range(48)]
    assert [rank for _, _, _, rank in keys[:8]] == [0, 0, 1, 1, 0, 0, 1, 1]
    for rank in (0, 1):
        direct = lm_orders("single", total=48, consumed=0, micro_batch_size=2, rank=rank, dp=2)[0]
        assert [index for _, _, index, r in keys if r == rank] == direct


def test_a_sequential_launch_that_outruns_its_dataset_is_refused(segment_config):
    """The pool holds 60 samples; from consumed count 40 that is five steps of 4,
    and the launch asks for seven. Every tool that walks the launch refuses it:
    the consumer before it writes a log, and the drive for the run record. A
    launch that ends exactly at the dataset's end is read."""
    from oracle import pipeline

    def launch(steps):
        return [{"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 40, "step_at_start": 10, "steps": steps}]

    segment = parse(segment_config, train_iters=20, launches=launch(7))
    built, document = pipeline.build_index(segment.data)
    assert built.samples == 60  # the smaller of the 80 requested and the pool
    with pytest.raises(LaunchOutrunsDataset, match="need 7 micro batches per rank.*gives 5 from consumed count 40"):
        pipeline.consume_segment(segment, built)
    assert not segment.output("log").exists()
    with pytest.raises(LaunchOutrunsDataset, match="allow_sequential_restart"):
        pipeline.segment_records(segment, built, document["cache_digests"])

    rows, summary, _ = consume(parse(segment_config, train_iters=20, launches=launch(5)))
    assert [key[2] for key, _ in samples_of(rows)] == list(range(40, 60))
    assert summary["a_pass_ended_inside_a_launch"] is False


def test_a_sequential_launch_may_start_its_pass_again_when_the_config_allows_it(segment_config):
    """With ``allow_sequential_restart`` the same launch is read. Megatron's
    sequential sampler ends, the pass is started again as the trainer's loop does,
    and the sampler begins again at the consumed count it was built with: 40
    here, not 0. The summary says that a pass ended."""
    launch = [{"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 40, "step_at_start": 10, "steps": 7}]
    rows, summary, built = consume(
        parse(segment_config, train_iters=20, launches=launch, allow_sequential_restart=True)
    )
    assert built.samples == 60
    keys = [key for key, _ in samples_of(rows)]
    assert [index for _, _, index, _ in keys] == list(range(40, 60)) + list(range(40, 48))
    assert [step for step, _, _, _ in keys[20:24]] == [16, 16, 16, 16]
    assert summary["a_pass_ended_inside_a_launch"] is True
    assert summary["launches"][0]["passes_per_rank"] == [2]
    assert lm_orders("single", total=60, consumed=40, micro_batch_size=4, rank=0, dp=1, passes=2) == [
        list(range(40, 60)),
        list(range(40, 60)),
    ]


def test_megatrons_sampler_refusals_are_named(segment_config):
    past_the_end = [{"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 60, "step_at_start": 15, "steps": 1}]
    with pytest.raises(SamplerRefused, match="refused its arguments"):
        consume(parse(segment_config, launches=past_the_end))
    off_a_batch = [{"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 6, "step_at_start": 1, "steps": 1}]
    with pytest.raises(SamplerRefused):
        consume(parse(segment_config, dataloader_type="cyclic", micro_batch_size=2, launches=off_a_batch))


# --- The cyclic sampler ---------------------------------------------------------------------------


def cyclic(make, launches, **kw):
    settings = dict(dataloader_type="cyclic", micro_batch_size=2, global_batch_size=4, data_sharding=True)
    settings.update(kw)
    return parse(make, launches=launches, **settings)


def test_cyclic_reads_agree_with_the_sampler_class(segment_config):
    """Driven: MegatronPretrainingRandomSampler, one per rank at data-parallel size
    2, data sharding on, over two epochs (the dataset holds 60 samples, so an
    epoch is 30 samples per rank, 15 steps).

    The indices in the log are the ones the sampler class yields when driven
    directly; another epoch gives another order; the recorded digests are the
    tool's definition applied to those lists."""
    from oracle import pipeline
    from waterwhite import ids

    launches = [{"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0, "step_at_start": 0, "steps": 30}]
    segment = cyclic(segment_config, launches, train_iters=30)
    built, document = pipeline.build_index(segment.data)
    record, _orders = pipeline.segment_records(segment, built, document["cache_digests"])
    summary = pipeline.consume_segment(segment, built)
    keys = [key for key, _ in samples_of(read_log(segment.output("log")))]

    assert built.samples == 60
    digests = {(d["launch"], d["epoch"], d["rank"]): d for d in record["shuffle_digests"]}
    assert set(digests) == {(0, epoch, rank) for epoch in (0, 1) for rank in (0, 1)}
    for rank in (0, 1):
        epochs = lm_orders("cyclic", total=60, consumed=0, micro_batch_size=2, rank=rank, dp=2, passes=2)
        assert [index for _, _, index, r in keys if r == rank] == epochs[0] + epochs[1]
        for epoch in (0, 1):
            # each rank reads its own bucket of 30 once per epoch
            assert sorted(epochs[epoch]) == list(range(rank * 30, rank * 30 + 30))
            assert digests[(0, epoch, rank)]["count"] == 30
            assert digests[(0, epoch, rank)]["digest"] == ids.order_digest(epochs[epoch])
        assert epochs[0] != epochs[1]
        assert digests[(0, 0, rank)]["digest"] != digests[(0, 1, rank)]["digest"]
    # inside a step: rank 0's micro batch, then rank 1's
    assert [(pos, rank) for step, pos, _, rank in keys if step == 1] == [(0, 0), (1, 0), (2, 1), (3, 1)]
    assert summary["a_pass_ended_inside_a_launch"] is True
    assert record["settings"]["sampler_class"].endswith("data_samplers.MegatronPretrainingRandomSampler")


def test_cyclic_resume_inside_an_epoch_and_two_launches(segment_config):
    """One segment, two launches: steps 1 to 8 from consumed count 0, then a resume
    at consumed count 32 that crosses the epoch boundary after step 15. The log
    equals that of one launch over all 20 steps.

    The order digests (DECISIONS.md D-16): each launch records, for every epoch
    it touches, the whole epoch's order of each rank, 30 indices, whatever part it
    read. Launch 0 read 16 indices of epoch 0, launch 1 the other 14 and 10 of
    epoch 1; the two launches record the same digest for epoch 0. What was read
    is in the summary's ``read_orders``."""
    from oracle import pipeline
    from waterwhite import ids

    def launch(consumed, step, steps):
        return {"world_size": 2, "data_parallel_size": 2, "consumed_at_start": consumed, "step_at_start": step, "steps": steps}

    whole, _, _ = consume(cyclic(segment_config, [launch(0, 0, 20)], train_iters=20))
    two = cyclic(segment_config, [launch(0, 0, 8), launch(32, 8, 12)], train_iters=20)
    built, document = pipeline.build_index(two.data)
    record, _ = pipeline.segment_records(two, built, document["cache_digests"])
    summary = pipeline.consume_segment(two, built)
    assert read_log(two.output("log")) == whole
    assert [l["continues_the_launch_before"] for l in summary["launches"]] == [None, True]

    got = {(d["launch"], d["epoch"], d["rank"]): (d["count"], d["digest"]) for d in record["shuffle_digests"]}
    for rank in (0, 1):
        epochs = lm_orders("cyclic", total=60, consumed=0, micro_batch_size=2, rank=rank, dp=2, passes=2)
        assert got[(0, 0, rank)] == (30, ids.order_digest(epochs[0]))
        assert got[(1, 0, rank)] == got[(0, 0, rank)]
        assert got[(1, 1, rank)] == (30, ids.order_digest(epochs[1]))
        assert got[(1, 1, rank)] != got[(1, 0, rank)]
        # a sampler built at the resume's consumed count gives the same tail
        assert lm_orders("cyclic", total=60, consumed=32, micro_batch_size=2, rank=rank, dp=2)[0] == epochs[0][16:]
        # a sampler built at the start of epoch 1 gives the whole of epoch 1
        assert lm_orders("cyclic", total=60, consumed=60, micro_batch_size=2, rank=rank, dp=2)[0] == epochs[1]
    assert len(got) == 6
    read = {(o["launch"], o["epoch"], o["rank"]): o for o in summary["read_orders"]}
    assert {key: o["count"] for key, o in read.items()} == {
        (0, 0, 0): 16, (0, 0, 1): 16, (1, 0, 0): 14, (1, 0, 1): 14, (1, 1, 0): 10, (1, 1, 1): 10,
    }
    assert [(read[key]["first_step"], read[key]["last_step"]) for key in ((0, 0, 0), (1, 0, 0), (1, 1, 0))] == [
        (1, 8), (9, 15), (16, 20),
    ]
    assert [(l["first_step"], l["last_step"]) for l in record["launches"]] == [(1, 8), (9, 20)]


def test_cyclic_with_sharding_off_and_megatron_bridges_classes(corpus, segment_config):
    """Driven: Megatron-Bridge's samplers.py loaded by path, beside Megatron-LM's.

    With sharding off, both read one permutation dealt over the ranks, and they
    agree while the micro batch size divides the dataset. With a micro batch size
    of 7 over 60 samples they differ, which is why the run record names the class."""
    from oracle import environment, pipeline

    bridge = str(corpus.env / "src" / "Megatron-Bridge" / "src" / "megatron" / "bridge" / "data" / "samplers.py")
    launches = [{"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0, "step_at_start": 0, "steps": 15}]
    lm_rows, _, _ = consume(cyclic(segment_config, launches, data_sharding=False))
    keys = [key for key, _ in samples_of(lm_rows)]
    for rank in (0, 1):
        direct = lm_orders("cyclic", total=60, consumed=0, micro_batch_size=2, rank=rank, dp=2, sharding=False)[0]
        assert [index for _, _, index, r in keys if r == rank] == direct

    with_bridge = cyclic(segment_config, launches, data_sharding=False, sampler_file=bridge)
    built, document = pipeline.build_index(with_bridge.data)
    record, _ = pipeline.segment_records(with_bridge, built, document["cache_digests"])
    pipeline.consume_segment(with_bridge, built)
    assert read_log(with_bridge.output("log")) == lm_rows
    assert record["settings"]["sampler_class"] == "megatron.bridge.data.samplers.MegatronPretrainingRandomSampler"
    assert record["versions"]["megatron_bridge"] == environment.ref_name(corpus.env / "src" / "Megatron-Bridge")
    assert record["versions"]["megatron_bridge"] is not None
    assert record["versions"]["release_tag"] == environment.ref_name(corpus.env / "src" / "Megatron-LM")

    one_rank = [{"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0, "step_at_start": 0, "steps": 8}]
    sevens = dict(data_sharding=False, micro_batch_size=7, global_batch_size=7, train_iters=10)
    lm_seven, _, _ = consume(cyclic(segment_config, one_rank, **sevens))
    bridge_seven, _, _ = consume(cyclic(segment_config, one_rank, sampler_file=bridge, **sevens))
    assert [r["global_index"] for r in lm_seven] != [r["global_index"] for r in bridge_seven]

    # the sequential classes of the two code bases read the same
    single_lm, _, _ = consume(parse(segment_config))
    single_bridge, _, _ = consume(parse(segment_config, sampler_file=bridge))
    assert single_lm == single_bridge


# --- The extra token and the read check -----------------------------------------------------------------


def test_samples_share_their_extra_token_in_sample_index_order(segment_config):
    """In a shard's own sample order, the last token a sample reads is the first
    token the next sample reads (plan 3.5): seen here in the log rows."""
    rows, _, built = consume(parse(segment_config))
    blend, checked = built.blend, 0
    by_index = {key[2]: sample_rows for key, sample_rows in samples_of(rows)}
    place = {}  # (shard, the shard's sample number before its shuffle) -> global index
    for index in range(built.samples):
        shard = int(blend.dataset_index[index])
        row = int(built.shards[shard].shuffle_index[int(blend.dataset_sample_index[index])])
        place[(shard, row)] = index
    for (shard, row), index in place.items():
        if (shard, row + 1) not in place:
            continue
        last, first = by_index[index][-1], by_index[place[(shard, row + 1)]][0]
        assert (last["ordinal"], last["token_end"] - 1) == (first["ordinal"], first["token_start"])
        checked += 1
    assert checked == 60 - 2  # every sample but the last of each shard


def test_a_read_the_log_does_not_explain_stops_the_run(segment_config):
    """Known-bad input: the indexed dataset hands back other tokens than the tap
    logged, as an unlogged read or padding would. The consumer must stop."""
    from oracle import pipeline
    from oracle.megatron_data import ReadTap

    segment = parse(segment_config)
    built, _ = pipeline.build_index(segment.data)
    tap = ReadTap.attach(built)
    assert ReadTap.attach(built) is tap
    index = next(i for i in range(built.samples) if int(built.blend.dataset_index[i]) == 0)
    tap.fetch(index)  # sound before the change

    low = built.shards[0].dataset
    logged_get = low.get

    def altered(idx, offset=0, length=None):
        out = logged_get(idx, offset=offset, length=length).copy()
        out[0] ^= 1
        return out

    low.get = altered
    with pytest.raises(ReadMismatch, match="not the concatenation"):
        tap.fetch(index)
    with pytest.raises(ReadMismatch):
        pipeline.consume_segment(segment, built)
    assert not segment.output("log").exists()  # no file that looks like a log is left


# --- F1 probe ------------------------------------------------------------------------------------------


def test_f1_probe_on_the_throwaway_corpus(corpus):
    """Driven: IndexedDataset, and the tokenizer of build_tokenizer."""
    from oracle import f1_probe

    tokenizer = oracle_config.parse_tokenizer(corpus.tokenizer, Path("."))
    result = f1_probe.probe(corpus.inputs["alpha"], corpus.prefixes["alpha"], tokenizer)
    assert result["f1_holds"] is True
    assert result["one_document_per_line"] is True and result["input_order"] is True
    assert (result["lines"], result["sequences"], result["documents"]) == (44, 44, 44)
    assert result["eod"] == {"token_id": 50256, "in_span": True}
    assert result["documents_equal_to_line_plus_eod"] == 44 and result["documents_equal_to_line_alone"] == 0
    assert result["zero_token_lines"] == []
    # Known-bad input: another file's lines against this shard.
    wrong = f1_probe.probe(corpus.inputs["beta"], corpus.prefixes["alpha"], tokenizer)
    assert wrong["f1_holds"] is False and wrong["eod"]["in_span"] is None
    assert wrong["first_line_that_matches_neither"] == 1


# --- The training stand-in and the driver ---------------------------------------------------------------


def test_run_segments_writes_records_logs_and_kept_caches(corpus, segment_config):
    """A fresh segment, a resume on a new mixture and a cyclic warm start from one
    call, each with its own new cache folder."""
    from oracle import pipeline
    from oracle.digests import compare_digests

    def one(dp, consumed, step, steps):
        return {"world_size": dp, "data_parallel_size": dp, "consumed_at_start": consumed, "step_at_start": step, "steps": steps}

    raw = {
        "segments": [
            segment_config(segment="E/seg-1", train_iters=6, launches=[one(1, 0, 0, 6)],
                           step_record={"checkpoints": [{"name": "E@6", "step": 6}]}),
            segment_config(segment="E/seg-2", shards=("alpha", "beta", "gamma"), train_iters=12,
                           start_kind="resume", parent_checkpoint="E@6", launches=[one(1, 24, 6, 6)],
                           step_record={"lowered": [{"step": 8, "last_sample_tokens_read": 16}],
                                        "checkpoints": [{"name": "E@9", "step": 9}]}),
            segment_config(segment="K/seg-1", dataloader_type="cyclic", micro_batch_size=2, train_iters=20,
                           start_kind="warm-start", parent_checkpoint="E@6",
                           launches=[one(2, 0, 0, 8), one(2, 32, 8, 12)]),
        ]
    }
    segments = oracle_config.parse_segments(raw, segment_config.base)
    result = pipeline.run_segments(segments)
    assert [s["segment"] for s in result["segments"]] == ["E/seg-1", "E/seg-2", "K/seg-1"]
    assert "consumed count 24" in result["segments"][1]["start"]
    assert [s["dataset_samples"] for s in result["segments"]] == [24, 48, 60]

    seg1, seg2, k = segments
    record = json.loads(seg2.output("run_record").read_text())
    assert record["format"] == "waterwhite.run-record/1"
    assert (record["run"], record["segment"], record["start_kind"], record["parent_checkpoint"]) == (
        "E", "E/seg-2", "resume", "E@6")
    assert record["settings"]["sampler_class"] == "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
    assert record["settings"]["add_extra_token_to_sequence"] is True and record["settings"]["train_iters"] == 12
    assert record["shuffle_digests"] == []
    for entry, name in zip(record["shards"], ("alpha", "beta", "gamma")):
        prefix = (seg2.output("run_record").parent / entry["prefix"]).resolve()
        assert prefix == corpus.prefixes[name].resolve()
        assert entry["bin_sha256"] == hashlib.sha256(Path(str(prefix) + ".bin").read_bytes()).hexdigest()
        assert entry["idx_sha256"] == hashlib.sha256(Path(str(prefix) + ".idx").read_bytes()).hexdigest()
    assert len(record["cache_digests"]) == 3 * 3 + 2
    assert record["launches"] == [{"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24,
                                   "step_at_start": 6, "first_step": 7, "last_step": 12}]
    import torch
    from importlib.metadata import version

    assert record["versions"]["torch"] == torch.__version__
    assert record["versions"]["megatron_core"] == version("megatron-core")
    assert record["versions"]["megatron_bridge"] is None

    steps = json.loads(seg2.output("step_record").read_text())
    assert steps["steps"][1] == {"step": 8, "batch_size": 4, "last_sample_tokens_read": 16}
    assert [s["step"] for s in steps["steps"]] == [7, 8, 9, 10, 11, 12]
    assert steps["checkpoints"] == [{"name": "E@9", "step": 9}]

    rows = read_log(seg2.output("log"))
    assert {r["segment"] for r in rows} == {"E/seg-2"}
    assert sorted({r["step"] for r in rows}) == [7, 8, 9, 10, 11, 12]
    assert [key[2] for key, _ in samples_of(rows)] == list(range(24, 48))
    assert {r["shard_position"] for r in rows} == {0, 1, 2}
    assert set(rows[0]) == {"segment", "step", "sample_pos", "global_index", "rank", "shard_position",
                            "ordinal", "token_start", "token_end"}
    assert len(json.loads(k.output("run_record").read_text())["shuffle_digests"]) == 6
    assert len(list(seg2.data.cache_dir.glob("*.npy"))) == 11

    # Replay (Check 2): a rebuild from the settings and the seed into another new
    # folder gives the recorded digests; with another seed it does not, and the
    # arrays that differ are the three of each shard.
    replay = pipeline.build_index(parse(segment_config, shards=("alpha", "beta", "gamma"), train_iters=12).data)[1]
    assert compare_digests(replay["cache_digests"], record["cache_digests"])["all_equal"] is True
    other = pipeline.build_index(
        parse(segment_config, shards=("alpha", "beta", "gamma"), train_iters=12, random_seed=5).data
    )[1]
    comparison = compare_digests(other["cache_digests"], record["cache_digests"])
    assert comparison["all_equal"] is False
    assert sorted((r["kind"], r["shard"]) for r in comparison["rows"] if not r["equal"]) == sorted(
        (kind, shard) for shard in (0, 1, 2) for kind in PER_SHARD
    )

    # A second run over the same folders is refused before anything is built.
    with pytest.raises(ConfigError, match="exists already"):
        pipeline.run_segments(segments)


def test_positions_report_on_a_real_log(corpus, segment_config):
    from oracle import positions

    segment = parse(segment_config, segment="T/seg-1")
    rows, _, _ = consume(segment)
    labels = ["alpha.jsonl", "beta.jsonl"]
    ids = {
        label: positions.read_ids(corpus.inputs[label.split(".")[0]], "id") for label in labels
    }
    report = positions.build_report(rows, {"T/seg-1": labels}, ids)
    assert len(report["steps"]) == 15
    for step in report["steps"]:
        assert [s["tokens"] for s in step["samples"]] == [33, 33, 33, 33]
    first_row = report["steps"][0]["samples"][0]["rows"][0]
    assert first_row["record"] == ids[first_row["shard"]][first_row["ordinal"]]
    flags = report["flags"]
    # samples of 33 tokens over documents of other lengths: some documents are split
    assert flags["split_across_samples"]
    assert all(item["reads"] for item in flags["split_across_samples"])
    # one pass over a blend reads no token twice
    assert not any(item["same_tokens_read_again"] for item in flags["read_in_more_than_one_step"])
    assert flags["same_ordinal_in_two_shards"]
    text = positions.render(report)
    assert "T/seg-1 step 15" in text and "Flags" in text
