"""The stores that hold shards, built with Megatron's own preprocessing and
verified with Megatron's own tokenizer (plan 5.4, F1, Checks 3 and 6).

Every test here runs the real tokenizer, carries the markers ``megatron`` and
``slow`` and is skipped unless ``WATERWHITE_MEGATRON_ENV`` is set. One test takes
15 to 40 seconds; on a shared machine run them one at a time through the CPU lock.
``test_replica_maps.py`` and ``test_replica_set_maps.py`` ask the same questions
of toy shards in a few seconds.
"""

from __future__ import annotations

import os
import shutil

import pytest
from waterwhite import vocabulary as v
from waterwhite.records import OccurrenceRef
from waterwhite.store.store import Store

from replica import setups
from replica.fixture import add_recovery_maps, build_table_store, recover_uncaptured
from replica.megatron_tokenizer import ENV_VARIABLE, EOD_TOKEN_ID
from replica.sets import build_captured_set, build_uncaptured_set
from replica.shard_index import read_shard_index

# These tests ask what a store holds, not what a bound gives: their generation
# jobs run in process, so they need no sandbox (the table's stores run confined by
# default: test_replica_harness_cli.py).
IN_PROCESS = "in-process"

pytestmark = [pytest.mark.megatron, pytest.mark.slow]

DOCUMENTS = {"E/shard-1": 18, "E/shard-2": 19, "E/shard-3": 27}
SKIP_DROP = ("E/shard-2.sanitizer-input", "L6", v.DROP_SANITIZER_SKIP, "sanitize",
             "E/shard-2.input")


@pytest.fixture(autouse=True)
def needs_megatron():
    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")


def all_drops(store, built):
    name_of = {instance: name for name, instance in built.instances.items()}
    return sorted(
        (name_of[d.input.instance_id], d.input.locator, d.reason, d.step_name, name)
        for name, instance in built.instances.items()
        for d in store.drops_by_output(instance) if d.reason != v.DROP_DEDUP_LOSER)


def test_the_main_store_has_verified_capture_maps_and_five_snapshots(tmp_path):
    built = build_table_store(tmp_path / "main", "main", tokenize=True, parallel=3,
                              generation_harness=IN_PROCESS)
    assert built.summary["tokenizer"]["eod_in_span"] is True
    with Store.open(built.store, readonly=True) as store:
        for shard, documents in DOCUMENTS.items():
            (binding,) = store.bindings(built.instances[shard])
            assert (binding.namespace, binding.derivation, binding.verification,
                    binding.label, binding.append_eod, binding.eod_in_span) == (
                v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED,
                True, True)
            rows = list(store.map_rows(built.instances[shard]))
            assert [r.label for r in rows] == [v.RECOVERED] * documents
            index = read_shard_index(f"{built.file(shard)}.idx")
            assert [r.token_end - r.token_start for r in rows] == list(index.document_tokens)
        assert all_drops(store, built) == [SKIP_DROP]
        assert len(list(store.artifacts(v.KIND_MIXTURE))) == 5
        assert [s.shard_instance_id for s in store.mixture_shards(built.instances["K/mix-1"])] == [
            built.instances["E/shard-3"], built.instances["E/shard-2"]]
        store.check_integrity()


def test_the_no_capture_store_is_recovered_in_position_mode(tmp_path):
    built = build_table_store(tmp_path / "no-capture", "no-capture", tokenize=True,
                                parallel=3)
    with Store.open(built.store, readonly=True) as store:
        # Table rows c06.fixture.documents-mapped.* and c06.fixture.drops.
        for shard, documents in DOCUMENTS.items():
            (binding,) = store.bindings(built.instances[shard])
            assert (binding.namespace, binding.derivation, binding.verification) == (
                v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)
            assert [r.label for r in store.map_rows(built.instances[shard])] == (
                [v.RECOVERED] * documents)
        assert all_drops(store, built) == [SKIP_DROP]
        assert list(store.artifacts(v.KIND_MIXTURE)) == []
        store.check_integrity()


def test_scratch_notokens_as_megatron_writes_it(tmp_path):
    built = build_table_store(tmp_path / "s", "scratch-notokens", tokenize=True)
    name = "c03/shard-notokens"
    shard, input_instance = built.instances[name], built.instances[f"{name}.input"]
    # Table row c03.drops.no-tokens.planted-line. Megatron's preprocessing wrote
    # four sequences and kept a fifth, empty entry in the index's document list.
    assert built.summary["line_tokens"][2] == 0 and built.summary["documents"] == 4
    assert built.summary["input_lines"] == 5
    with Store.open(built.store, readonly=True) as store:
        assert [(d.input, d.reason) for d in store.drops_by_output(shard)] == [
            (OccurrenceRef(input_instance, "L3"), v.DROP_NO_TOKENS)]
        maps = store.bindings(shard)
        assert [(b.namespace, b.derivation, b.verification, b.label, b.append_eod)
                for b in maps] == [
            (v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED, True),
            (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED, True)]
        for binding in maps:
            rows = list(store.map_rows(shard, binding.namespace, derivation=binding.derivation))
            assert [(r.ordinal, r.label, r.parent.locator) for r in rows] == [
                (0, v.RECOVERED, "L1"), (1, v.RECOVERED, "L2"),
                (2, v.RECOVERED, "L4"), (3, v.RECOVERED, "L5")]
            assert [store.edge_of(r.parent).inputs[0].parent for r in rows] == [
                OccurrenceRef(built.instances["src/Z"], f"L{n}") for n in (1, 2, 3, 4)]
        store.check_integrity()


def test_scratch_500_as_megatron_writes_it(tmp_path):
    built = build_table_store(tmp_path / "s", "scratch-500", tokenize=True)
    assert built.summary["documents"] == 500 and built.summary["input_lines"] == 500
    assert built.summary["tokenizer"]["eod_token_id"] == EOD_TOKEN_ID
    with Store.open(built.store, readonly=True) as store:
        shard = store.artifact("c04/shard-500")
        assert shard.kind == v.KIND_SHARD and store.bindings(shard.instance_id) == ()
        store.check_integrity()


@pytest.fixture(scope="module")
def four_chunk_set(plant_generate, set_parameters, tmp_path_factory):
    """The small set with its dedup off and four chunks, so that the test runs the
    tokenizer four times and not twenty."""
    folder = tmp_path_factory.mktemp("four-chunks") / "set"
    plant_generate.generate(folder, dedup=False, **{**set_parameters, "chunks": 4})
    return folder


def test_a_set_gets_its_capture_maps_and_both_recovery_maps(four_chunk_set, tmp_path):
    built = build_captured_set(tmp_path / "captured", four_chunk_set, parallel=4)
    add_recovery_maps(built)
    twins = 0
    with Store.open(built.store, readonly=True) as store:
        for n in (1, 2, 3, 4):
            shard = built.instances[f"set-rr/shard-{n}"]
            assert [(b.namespace, b.derivation, b.verification) for b in store.bindings(shard)] == [
                (v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED),
                (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED),
                (v.NAMESPACE_RECOVERY, v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED)]
            capture = list(store.map_rows(shard, v.NAMESPACE_CAPTURE))
            matched = list(store.map_rows(shard, v.NAMESPACE_RECOVERY,
                                          derivation=v.DERIVED_TOKEN_MATCHING))
            assert [(r.label, r.parent, set(r.candidates)) for r in matched] == [
                (r.label, r.parent, set(r.candidates)) for r in capture]
            twins += sum(r.label == v.CANDIDATES for r in capture)
        store.check_integrity()
    # With four chunks, some of the 64 identical pairs lie in one shard.
    assert twins >= 2 and twins % 2 == 0


def test_a_run_made_without_capture_is_recovered_and_a_changed_line_fails(four_chunk_set,
                                                                         tmp_path):
    run = build_uncaptured_set(tmp_path / "plain", four_chunk_set, parallel=4)
    shutil.copytree(run.out, tmp_path / "changed" / "build")
    recovered = recover_uncaptured(tmp_path / "plain", run, also_token_matching=True)
    with Store.open(recovered.store, readonly=True) as store:
        for n in (1, 2, 3, 4):
            assert {b.verification for b in store.bindings(
                recovered.instances[f"set-rr/shard-{n}"])} == {v.VERIFIED_PASSED}
    setups.change_first_line(tmp_path / "changed" / "build", "round-robin")
    failed = recover_uncaptured(tmp_path / "changed", run.moved_to(tmp_path / "changed" / "build"),
                                also_token_matching=True)
    with Store.open(failed.store, readonly=True) as store:
        assert sorted((b.derivation, b.verification, b.label) for b in store.bindings(
            failed.instances["set-rr/shard-1"])) == [
            (v.DERIVED_POSITION, v.VERIFIED_FAILED, v.UNKNOWN),
            (v.DERIVED_TOKEN_MATCHING, v.VERIFIED_FAILED, v.UNKNOWN)]
        assert {b.verification for b in store.bindings(
            failed.instances["set-rr/shard-2"])} == {v.VERIFIED_PASSED}
