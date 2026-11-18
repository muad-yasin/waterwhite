"""A segment's trace route follows from its run record, whoever calls the store
(plan 3.5; docs/interfaces.md section 2; DECISIONS.md D-25, D-36 and D-41).

The store's own call, ``Store.register_run_segment``, is put every case directly:

- a segment with no trace route can carry only one of the five trace causes of the
  data-order trace; any other named cause is refused as "invalid record" and nothing
  is written;
- a route "cache files" or "replay" needs a run record that holds a digest of every
  index array the segment needs (three kinds per shard and, with more than one
  shard, two for the blend): a replay counts only against digests recorded at
  training time (plan 3.5), and kept cache files count only against the same digests
  (a decision of the project's owner, stricter than plan 3.5 and 4.6: DECISIONS.md
  D-41); and for a segment read with the cyclic sampler also a shuffle digest.

A case that passes before the rule is in force is a control and says so in its name.
Every record below states its shard list, its launch and both digest lists itself, so
that the only thing that differs between a refused and an accepted case is the thing
the test is about. The last test goes through the command line.

Nothing here imports Megatron or torch.
"""

from __future__ import annotations

import io
import json

import pytest

from waterwhite import cli, errors
from waterwhite import vocabulary as v
from waterwhite.store import Store

CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
JOIN = ("core_v0.20.0", "single", CLASS)

PER_SHARD = ("document_index", "sample_index", "shuffle_index")
PER_BLEND = ("dataset_index", "dataset_sample_index")

# The five causes a segment with no trace route may carry (docs/interfaces.md section 2).
THE_FIVE = (
    "data-order trace missing",
    "data-order trace unverified for this version",
    "data-order trace unverified for this sampler type",
    "cyclic order not recorded",
    "cyclic order digest mismatch",
)

# Named causes of the answers that are none of the five.
OUTSIDE_THE_FIVE = (
    "step untraced",
    "mixture snapshot missing",
    "missing origin",
    "upstream record missing",
    "upper bound not established",
    "in scope only",
    "evidence level cannot decide the request",
    "label stronger than derivation",
    "candidate parents",
    "ancestor may contain",
    "unresolved record",
    "generating-model entry missing",
    "generating model exposed",
    "generating model not registered",
)


def digest_list(shards: int, *, kinds_missing: tuple = (), blend: bool | None = None) -> list[dict]:
    """The cache digests of a run of ``shards`` shards: three kinds for each and, with
    more than one shard, the two of the blend, except those named in ``kinds_missing``
    as (kind, shard position or None)."""
    wanted = [(kind, n) for n in range(shards) for kind in PER_SHARD]
    if (shards > 1) if blend is None else blend:
        wanted += [(kind, None) for kind in PER_BLEND]
    return [{"kind": kind, "shard": at, "digest": f"{n:02x}" * 32}
            for n, (kind, at) in enumerate(wanted) if (kind, at) not in kinds_missing]


def record(*, shards: int = 1, digests: list[dict] | None = None, sampler: str = "single",
           shuffle: bool = False) -> dict:
    """A run record of one launch with its shard list, its digests and, when asked, one
    shuffle digest. ``digests`` None gives the complete set for the shard list."""
    return {
        "format": "waterwhite.run-record/1", "run": "H", "segment": "H/seg-1", "start_kind": "fresh",
        "parent_checkpoint": None,
        "settings": {"dataloader_type": sampler, "sampler_class": CLASS, "data_sharding": True,
                     "micro_batch_size": 1, "global_batch_size": 1, "sequence_length": 4,
                     "add_extra_token_to_sequence": True, "random_seed": 1234, "split": "1,0,0",
                     "train_iters": 2, "train_samples": None, "batch_schedule": None},
        "versions": {"release_tag": JOIN[0], "megatron_core": "0.20.0", "megatron_bridge": None,
                     "torch": "2.0"},
        "shards": [{"position": n, "prefix": f"shards/shard-{n}", "bin_sha256": "b" * 64,
                    "idx_sha256": "c" * 64} for n in range(shards)],
        "launches": [{"position": 0, "world_size": 1, "data_parallel_size": 1,
                      "consumed_at_start": 0, "step_at_start": 0, "first_step": 1, "last_step": 2}],
        "cache_digests": digest_list(shards) if digests is None else digests,
        "shuffle_digests": ([{"launch": 0, "epoch": 0, "rank": 0, "digest": "0" * 64, "count": 1}]
                            if shuffle else []),
    }


def call(run_record: dict, route: str, cause: str | None = None, detail: str | None = None) -> dict:
    return dict(run="H", segment="H/seg-1", position=0, mixture_instance_id=None,
                parent_checkpoint=None, start_kind="fresh", run_record=run_record,
                steps=[(1, 1, None), (2, 1, None)], trace_route=route, trace_cause=cause,
                trace_cause_detail=detail, traces=[], checkpoints=[("H@2", 2)])


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser="control-chars/1", verified_joins=[JOIN]) as made:
        yield made


def counts(store: Store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def refused(store: Store, arguments: dict) -> None:
    """The call is refused as "invalid record" (exit 4) and the store is as it was."""
    before = counts(store)
    with pytest.raises(errors.InvalidRecord) as caught:
        store.register_run_segment(**arguments)
    assert (caught.value.name, caught.value.exit_code) == ("invalid record", 4)
    assert store.segment("H/seg-1") is None and counts(store) == before


# --- (a) A segment with no route carries one of the five causes --------------------------


@pytest.mark.parametrize("cause", OUTSIDE_THE_FIVE)
def test_a_cause_outside_the_five_is_refused_for_a_segment_with_no_route(store, cause):
    refused(store, call(record(), v.ROUTE_NONE, cause))


def test_control_a_word_that_is_no_cause_at_all_was_refused_already(store):
    refused(store, call(record(), v.ROUTE_NONE, "the cache is gone"))


@pytest.mark.parametrize("cause, run_record, detail", [
    (THE_FIVE[0], record(digests=[]), "no recorded digest"),
    (THE_FIVE[0], record(), "digest mismatch"),
    (THE_FIVE[1], record(), None),
    (THE_FIVE[2], record(), None),
    (THE_FIVE[3], record(sampler="cyclic"), None),
    (THE_FIVE[4], record(sampler="cyclic", shuffle=True), None),
])
def test_control_each_of_the_five_causes_is_kept_with_its_own_record(store, cause, run_record, detail):
    store.register_run_segment(**call(run_record, v.ROUTE_NONE, cause, detail))
    held = store.segment("H/seg-1")
    assert (held.trace_route, held.trace_cause, held.trace_cause_detail) == (v.ROUTE_NONE, cause, detail)


# --- (b) A route needs the digests of every index array the segment needs ------------------

ROUTES = (v.ROUTE_CACHE_FILES, v.ROUTE_REPLAY)

# Run records that hold no digest of some index array the segment needs. Each is
# complete in everything else: its shard list, its launch, its settings.
INCOMPLETE = {
    "no digest at all, one shard": record(digests=[]),
    "no document index, one shard": record(digests=digest_list(1, kinds_missing=(("document_index", 0),))),
    "no sample index, one shard": record(digests=digest_list(1, kinds_missing=(("sample_index", 0),))),
    "no shuffle index, one shard": record(digests=digest_list(1, kinds_missing=(("shuffle_index", 0),))),
    "no digest at all, two shards": record(shards=2, digests=[]),
    "both shards, no blend pair": record(shards=2, digests=digest_list(2, blend=False)),
    "only one of the blend pair": record(
        shards=2, digests=digest_list(2, kinds_missing=(("dataset_sample_index", None),))),
    "the second shard misses a kind": record(
        shards=2, digests=digest_list(2, kinds_missing=(("sample_index", 1),))),
    "the blend pair and the first shard only": record(
        shards=2, digests=digest_list(2, kinds_missing=tuple((k, 1) for k in PER_SHARD))),
}


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name", INCOMPLETE)
def test_a_route_the_run_record_does_not_support_is_refused_and_nothing_is_written(store, name, route):
    refused(store, call(INCOMPLETE[name], route))


@pytest.mark.parametrize("route", ROUTES)
def test_a_record_with_no_shard_list_and_no_digest_supports_no_route(store, route):
    bare = record(digests=[])
    bare["shards"] = []
    refused(store, call(bare, route))


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("shards", [1, 2])
def test_control_a_record_that_holds_every_digest_is_accepted(store, shards, route):
    store.register_run_segment(**call(record(shards=shards), route))
    held = store.segment("H/seg-1")
    assert held.trace_route == route
    assert len(held.cache_digests) == (3 if shards == 1 else 8)


@pytest.mark.parametrize("route", ROUTES)
def test_control_the_cyclic_record_without_a_shuffle_digest_stays_refused(store, route):
    # Every cache digest is held: the shuffle digest is the only thing missing.
    refused(store, call(record(sampler="cyclic", shuffle=False), route))


@pytest.mark.parametrize("route", ROUTES)
def test_control_the_cyclic_record_with_every_digest_is_accepted(store, route):
    store.register_run_segment(**call(record(sampler="cyclic", shuffle=True), route))
    assert store.segment("H/seg-1").trace_route == route


def test_a_cyclic_record_with_a_shuffle_digest_and_no_cache_digest_supports_no_route(store):
    # The shuffle digest alone does not make a route: the index arrays are still to be held
    # against recorded digests.
    refused(store, call(record(sampler="cyclic", shuffle=True, digests=[]), v.ROUTE_REPLAY))


# --- Through the command line -------------------------------------------------------------


def test_control_register_run_gives_route_none_to_a_record_with_no_digest(tmp_path):
    path = tmp_path / "s.db"
    Store.create(path, normaliser="control-chars/1", verified_joins=[JOIN]).close()
    run_file, steps_file = tmp_path / "run.json", tmp_path / "steps.json"
    run_file.write_text(json.dumps(record(digests=[])), encoding="utf-8")
    steps_file.write_text(json.dumps({
        "segment": "H/seg-1", "steps": [{"step": 1, "batch_size": 1}, {"step": 2, "batch_size": 1}],
        "checkpoints": [{"name": "H@2", "step": 2}]}), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["register-run", "--store", str(path), "--run-record", str(run_file),
                     "--steps", str(steps_file)], stdout=out, stderr=err)
    assert code == 0, err.getvalue()
    printed = json.loads(out.getvalue())
    assert (printed["route"], printed["cause"], printed["cause_detail"]) == (
        "none", "data-order trace missing", "no recorded digest")
    with Store.open(path) as store:
        held = store.segment("H/seg-1")
        assert (held.trace_route, held.trace_cause, held.trace_cause_detail) == (
            "none", "data-order trace missing", "no recorded digest")
