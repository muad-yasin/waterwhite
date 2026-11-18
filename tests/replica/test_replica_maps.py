"""The capture points of tokenization and mixtures, the run made without capture
and its recovery, and the scratch stores, against the real store, wrappers,
recovery and tokenization maps (plan 5.4, Checks 3 and 6).

Nothing here runs Megatron: the stand-in of conftest.py writes each chunk as a toy
shard in the real layout, one token per byte of a text. The same questions are
asked of shards Megatron wrote in ``test_replica_megatron_stores.py``.

What is asked: which artifacts each store of the hand-written table holds
(fixture/table/README.md section 4), that each shard has the map its store calls
for, that a dropped line is recorded exactly once under the first registered
artifact that lacks it, and that a mixture's snapshot keeps the trainer's order.
"""

from __future__ import annotations

import pytest
from waterwhite import vocabulary as v
from waterwhite.records import OccurrenceRef
from waterwhite.store.store import Store

from replica.build import BuildOptions
from replica.fixture import (
    build_captured_fixture,
    build_table_store,
    build_uncaptured,
    recover_uncaptured,
    register_again,
)
from replica.plan import load_fixture_spec
from replica.shard_index import read_shard_index

# These tests ask what a store holds, not what a bound gives: their generation
# jobs run in process, so they need no sandbox (the table's stores run confined by
# default: test_replica_harness_cli.py).
IN_PROCESS = "in-process"

SOURCES = ["src/A", "src/B", "src/C", "src/Z"]
GENERATED = ["gen/P", "gen/T", "gen/S"]
MIXTURES = ["E/mix-1", "E/mix-2", "D/mix-1", "Y/mix-1", "K/mix-1"]
DOCUMENTS = {"E/shard-1": 18, "E/shard-2": 19, "E/shard-3": 27}


def shard_names(prefix: str) -> list[str]:
    return [name for n in (1, 2, 3)
            for name in (f"{prefix}/shard-{n}.sanitizer-input", f"{prefix}/shard-{n}.input",
                         f"{prefix}/shard-{n}")]


def all_drops(store, built) -> list[tuple[str, str, str, str, str]]:
    """Every drop of the store as (input name, locator, reason, step, output name)."""
    name_of = {instance: name for name, instance in built.instances.items()}
    return sorted(
        (name_of[d.input.instance_id], d.input.locator, d.reason, d.step_name, name)
        for name, instance in built.instances.items()
        for d in store.drops_by_output(instance))


def pipeline_drops(store, built):
    return [d for d in all_drops(store, built) if d[2] != v.DROP_DEDUP_LOSER]


@pytest.fixture(scope="module")
def main(tmp_path_factory, megatron_stand_in):
    """The table's main store, built with the stand-in, opened read-only."""
    base = tmp_path_factory.mktemp("maps-main")
    with megatron_stand_in(base / "env") as env:
        built = build_table_store(base / "main", "main", tokenize=True, megatron_env=env,
                                  generation_harness=IN_PROCESS)
    store = Store.open(built.store, readonly=True)
    yield built, store
    store.close()


@pytest.fixture(scope="module")
def uncaptured(tmp_path_factory, megatron_stand_in):
    """The table's no-capture store, built with the stand-in, opened read-only."""
    base = tmp_path_factory.mktemp("maps-no-capture")
    with megatron_stand_in(base / "env") as env:
        built = build_table_store(base / "no-capture", "no-capture", tokenize=True,
                                    megatron_env=env)
    store = Store.open(built.store, readonly=True)
    yield built, store
    store.close()


# --- The main store: capture maps and mixture snapshots ---------------------------------


def test_the_main_store_holds_the_three_shards_their_inputs_and_the_five_mixtures(main):
    built, store = main
    assert list(built.instances) == SOURCES + GENERATED + ["R1", "R2"] + shard_names("E") + MIXTURES
    kinds = {name: store.artifact(name).kind for name in built.instances}
    assert [n for n, k in kinds.items() if k == v.KIND_SHARD] == list(DOCUMENTS)
    assert [n for n, k in kinds.items() if k == v.KIND_MIXTURE] == MIXTURES
    assert built.files["E/shard-2"] == "pipeline/partitioned/shards/shard-2"
    assert built.files["E/shard-2.input"] == "pipeline/partitioned/tokenizer-input/shard-2.jsonl"
    assert built.files["K/mix-1"] == "mixtures/K/mix-1"
    # Runs, segments, step traces and checkpoints are registered after the build.
    assert store.runs() == () and store.artifact("E@6") is None
    store.check_integrity()


def test_each_shard_has_one_verified_capture_map_over_its_tokenizer_input(main):
    built, store = main
    for shard, documents in DOCUMENTS.items():
        instance = built.instances[shard]
        (binding,) = store.bindings(instance)
        assert (binding.namespace, binding.derivation, binding.verification, binding.label) == (
            v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED)
        # Every map records the --append-eod setting and the convention (plan F1).
        assert (binding.append_eod, binding.eod_in_span) == (True, True)
        rows = list(store.map_rows(instance))
        assert len(rows) == binding.declared_count == documents
        # Line i of the tokenizer input is document i - 1, and the tool was told so
        # by nothing but the files: every row names its line, labelled "recovered".
        input_instance = built.instances[f"{shard}.input"]
        assert [(r.label, r.parent) for r in rows] == [
            (v.RECOVERED, OccurrenceRef(input_instance, f"L{n + 1}")) for n in range(documents)]
        index = read_shard_index(f"{built.file(shard)}.idx")
        assert [r.token_end - r.token_start for r in rows] == list(index.document_tokens)
        assert rows[-1].token_end == index.tokens
        assert store.artifact(shard).inputs == (input_instance,)
        assert built.summary["layouts"]["partitioned"]["shards"][
            int(shard[-1]) - 1]["capture"]["maps"][0]["recovered"] == documents


def test_the_skipped_line_is_one_drop_whose_output_is_the_tokenizer_input(main):
    built, store = main
    # The one line the sanitizer skipped is line 6 of the chunk of shard 2. It is
    # recorded once, under the first registered artifact that lacks it: the
    # tokenizer input. The capture map, written over that input, sees no drop.
    assert pipeline_drops(store, built) == [
        ("E/shard-2.sanitizer-input", "L6", v.DROP_SANITIZER_SKIP, "sanitize", "E/shard-2.input")]
    for shard in DOCUMENTS:
        assert list(store.drops_by_output(built.instances[shard])) == []
    capture = built.summary["layouts"]["partitioned"]["shards"][1]["capture"]
    assert capture["tokenizer_input"]["drops"] == 1 and capture["drops"] == []
    assert capture["tokenizer_input"]["input_lines_unaccounted"] == 0


def test_a_document_is_linked_back_to_its_record_through_the_recovery_points(main):
    built, store = main
    # T is a document of shard 3. Its chain to gen/T passes the two files of the
    # recovery points: the tokenizer input (sanitizer, re-run) and the chunk as the
    # sanitizer read it (copy steps, by fingerprint).
    row = list(store.map_rows(built.instances["E/shard-3"]))[built.ordinal_of("E/shard-3", "T")]
    assert row.parent == OccurrenceRef(built.instances["E/shard-3.input"],
                                       f"L{built.line_of('E/shard-3.input', 'T')}")
    sanitized = store.edge_of(row.parent)
    assert (sanitized.derivation, sanitized.label, sanitized.transform.name) == (
        v.DERIVED_FUNCTION, v.RECOVERED, "sanitize")
    (chunk_line,) = [i.parent for i in sanitized.inputs]
    assert chunk_line.instance_id == built.instances["E/shard-3.sanitizer-input"]
    copied = store.edge_of(chunk_line)
    assert (copied.derivation, copied.label, copied.transform.name) == (
        v.DERIVED_FINGERPRINT, v.RECOVERED, "replica.shuffle-chunk-merge")
    assert [i.parent for i in copied.inputs] == [OccurrenceRef(built.instances["gen/T"], "L1")]


def test_b_copy_of_the_shared_passage_has_one_parent_in_its_shard(main):
    built, store = main
    # X@B lies in shard 1 or 2. Those shards list the files of B and Z only, so
    # the copy of A is no candidate and the line has one parent (plan L12).
    (shard,) = [s for s in ("E/shard-1", "E/shard-2")
                if any(r["handle"] == "X@B" for r in built.manifest_rows(f"{s}.input"))]
    chunk = f"{shard}.sanitizer-input"
    edge = store.edge_of(OccurrenceRef(built.instances[chunk], f"L{built.line_of(chunk, 'X@B')}"))
    assert edge.label == v.RECOVERED
    assert [i.parent for i in edge.inputs] == [
        OccurrenceRef(built.instances["src/B"], f"L{built.line_of('src/B', 'X@B')}")]


def test_a_mixture_snapshot_keeps_the_trainers_order_and_each_files_digest(main):
    built, store = main
    # Run K is handed shard 3 first (plan Decision 27): the snapshot keeps that
    # order, which a listing of the folder would not give.
    snapshot = store.mixture_shards(built.instances["K/mix-1"])
    assert [(s.position, s.shard_instance_id) for s in snapshot] == [
        (0, built.instances["E/shard-3"]), (1, built.instances["E/shard-2"])]
    assert [s.path for s in snapshot] == ["build/mixtures/K/mix-1/shard-3",
                                          "build/mixtures/K/mix-1/shard-2"]
    for entry in snapshot:
        assert entry.content_digest == store.artifact(entry.shard_instance_id).content_digest
    assert store.mixture_settings(built.instances["K/mix-1"]) == {
        "selection": "directory", "directory": "mixtures/K/mix-1"}
    assert [len(store.mixture_shards(built.instances[m])) for m in MIXTURES] == [2, 3, 2, 3, 2]
    # Two mixtures that select the same shards from two folders are two mixtures.
    assert len({built.instances[m] for m in MIXTURES}) == 5
    assert set(store.mixtures_holding(built.instances["E/shard-3"])) == {
        built.instances[m] for m in ("E/mix-2", "Y/mix-1", "K/mix-1")}


def test_the_same_build_registered_again_keeps_shards_maps_and_mixtures(tmp_path, no_megatron):
    first = build_captured_fixture(tmp_path / "f", tokenize=True, layouts=("partitioned",),
                                   releases=("R1",), imports=(), megatron_env=no_megatron)
    again = register_again(first, megatron_env=no_megatron)
    assert again.instances == first.instances
    with Store.open(first.store, readonly=True) as store:
        for shard in DOCUMENTS:
            assert len(store.bindings(first.instances[shard])) == 1
        assert len(list(store.artifacts(v.KIND_MIXTURE))) == 5
        store.check_integrity()


def test_the_round_robin_store_holds_its_shards_and_mixtures(tmp_path, no_megatron):
    built = build_table_store(tmp_path / "rr", "rr", tokenize=True, megatron_env=no_megatron,
                              generation_harness=IN_PROCESS)
    assert list(built.instances) == SOURCES + GENERATED + shard_names("E-rr") + [
        "E-rr/mix-1", "E-rr/mix-2"]
    shards = built.summary["layouts"]["round-robin"]["shards"]
    # Every listed file can reach every chunk.
    assert all(s["listed"] == ["sources/B.jsonl", "sources/Z.jsonl", "gen/T.jsonl"]
               for s in shards)
    assert sum(s["capture"]["documents"] for s in shards) == 64
    with Store.open(built.store, readonly=True) as store:
        for n in (1, 2, 3):
            (binding,) = store.bindings(built.instances[f"E-rr/shard-{n}"])
            assert (binding.namespace, binding.verification) == (
                v.NAMESPACE_CAPTURE, v.VERIFIED_PASSED)
        assert pipeline_drops(store, built) == [
            ("E-rr/shard-2.sanitizer-input", "L6", v.DROP_SANITIZER_SKIP, "sanitize",
             "E-rr/shard-2.input")]
        assert [s.shard_instance_id for s in
                store.mixture_shards(built.instances["E-rr/mix-2"])] == [
            built.instances[f"E-rr/shard-{n}"] for n in (1, 2, 3)]
        store.check_integrity()


def test_a_capture_map_needs_the_recorded_end_of_document_convention(tmp_path, fixture_plan,
                                                                    no_megatron):
    import dataclasses

    from replica.build import check_build
    from replica.errors import SpecError

    # A map records the convention (plan 3.3, F1). A tokenizer table that does not
    # hold it is refused before anything is written, where a map would be written.
    plan = dataclasses.replace(fixture_plan, tokenizer=dataclasses.replace(
        fixture_plan.tokenizer, eod_in_span=None))
    with pytest.raises(SpecError, match="eod_in_span"):
        check_build(plan, tmp_path / "out", BuildOptions(), capturing=True,
                    megatron_env=no_megatron)
    check_build(plan, tmp_path / "out", BuildOptions(), megatron_env=no_megatron)
    assert not (tmp_path / "out").exists()


# --- The run made without capture, recovered -------------------------------------------


def test_the_no_capture_store_holds_the_same_files_recovered_and_no_mixture(uncaptured):
    built, store = uncaptured
    assert built.summary["capture"] is False and built.summary["seeded"] is False
    assert list(built.instances) == SOURCES + GENERATED + ["R1", "R2"] + shard_names("E")
    # The mixture folders exist; a run made without capture has no snapshot (L6).
    assert (built.out / "mixtures" / "E" / "mix-1").is_dir()
    assert list(store.artifacts(v.KIND_MIXTURE)) == []
    assert built.summary["recovery"]["mixtures_not_registered"] == sorted(MIXTURES)
    roles = {name: store.artifact(name).role for name in SOURCES + GENERATED + ["R1", "R2"]}
    # A release found afterwards has the role of a release, so that a listing
    # covers it; the other found files are intermediate ones.
    assert roles == {**{n: v.ROLE_SOURCE for n in SOURCES},
                     **{n: v.ROLE_INTERMEDIATE for n in GENERATED},
                     "R1": v.ROLE_RELEASE, "R2": v.ROLE_RELEASE}
    store.check_integrity()


def test_the_listing_of_the_no_capture_store_covers_the_found_releases(uncaptured):
    from waterwhite import listing, requests

    built, store = uncaptured
    made = listing.listing(store, requests.withdraw(store, "A")).to_json()
    # R1 and R2 are found files with the role of a release: the listing covers them
    # and answers for them. A generation output is a found intermediate file and is
    # no release.
    assert made["covers"]["releases"] == ["R1", "R2"]
    answered = {a["artifact"] for a in made["answers"]}
    assert {"R1", "R2"} <= answered


def test_a_generation_output_is_a_plain_file_with_no_edge_and_no_member(uncaptured):
    built, store = uncaptured
    for name in GENERATED:
        instance = built.instances[name]
        assert list(store.edges(instance)) == [] and store.file_source_links(instance) == ()
        assert store.artifact(name).step is None
    assert {m.name for m in store.members()} == {"A/r1", "B/r1", "C/r1", "Z/r1"}
    assert not (built.out / "gen" / "harness").exists()


def test_the_row_two_sources_supply_has_both_copies_as_candidates(uncaptured):
    built, store = uncaptured
    # The dedup selection was not observed: the row's text equals X@A and X@B, and
    # no rule can tell which one it came from (plan L3; table row
    # c06.fixture.candidates.X).
    line = built.line_of("R1", "X@A")
    edge = store.edge_of(OccurrenceRef(built.instances["R1"], f"L{line}"))
    assert (edge.label, edge.parent_mode, edge.derivation) == (
        v.CANDIDATES, v.MODE_ANY, v.DERIVED_FINGERPRINT)
    assert {i.parent for i in edge.inputs} == {
        OccurrenceRef(built.instances["src/A"], f"L{built.line_of('src/A', 'X@A')}"),
        OccurrenceRef(built.instances["src/B"], f"L{built.line_of('src/B', 'X@B')}")}
    assert not any(i.selected for i in edge.inputs)
    assert built.summary["recovery"]["releases"]["R1"]["candidates"] == 1
    # The rows that hold P, T and S are linked to the plain files, and stop there.
    t = store.edge_of(OccurrenceRef(built.instances["R1"], f"L{built.line_of('R1', 'T')}"))
    assert [i.parent for i in t.inputs] == [OccurrenceRef(built.instances["gen/T"], "L1")]


def test_every_shard_is_recovered_in_position_mode(uncaptured):
    built, store = uncaptured
    # Table rows c06.fixture.documents-mapped.*: 18, 19 and 27 documents, each
    # recovered, none a candidate, in the recovery namespace.
    for shard, documents in DOCUMENTS.items():
        (binding,) = store.bindings(built.instances[shard])
        assert (binding.namespace, binding.derivation, binding.verification) == (
            v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)
        labels = [r.label for r in store.map_rows(built.instances[shard])]
        assert labels == [v.RECOVERED] * documents
        report = built.summary["recovery"]["shards"][shard]
        assert (report["documents"], report["maps"][0]["recovered"],
                report["maps"][0]["candidates"]) == (documents, documents, 0)


def test_the_skipped_line_of_the_no_capture_run_is_one_drop(uncaptured):
    built, store = uncaptured
    # Table row c06.fixture.drops: line 6 of the input of shard 2, "sanitizer
    # skip"; no empty line and no line without tokens. With the tokenizer input
    # kept, the drop's output is that file and the map over it records none.
    assert all_drops(store, built) == [
        ("E/shard-2.sanitizer-input", "L6", v.DROP_SANITIZER_SKIP, "sanitize", "E/shard-2.input")]
    assert built.summary["recovery"]["shards"]["E/shard-2"]["drops"] == []


def test_where_only_the_sanitizer_input_was_kept_the_drop_lies_under_the_shard(
        tmp_path, spec_path, no_megatron):
    options = BuildOptions(layouts=("partitioned",), seeded=False, releases=(), imports=())
    run = build_uncaptured(tmp_path / "run", load_fixture_spec(spec_path), options,
                           megatron_env=no_megatron)
    built = recover_uncaptured(tmp_path / "run", run, kept="sanitizer-input",
                               megatron_env=no_megatron)
    assert "E/shard-2.input" not in built.instances
    with Store.open(built.store, readonly=True) as store:
        # The map is recovered over the chunk with the sanitizer's log: the skipped
        # line is a drop whose output is the shard, and it is the only record of it.
        assert all_drops(store, built) == [
            ("E/shard-2.sanitizer-input", "L6", v.DROP_SANITIZER_SKIP, "sanitize", "E/shard-2")]
        rows = list(store.map_rows(built.instances["E/shard-2"]))
        chunk = built.instances["E/shard-2.sanitizer-input"]
        # No ordinal shifts: document 4 is line 5 and document 5 is line 7.
        assert [r.parent for r in rows[4:6]] == [OccurrenceRef(chunk, "L5"),
                                                 OccurrenceRef(chunk, "L7")]
        assert len(rows) == 19
        assert store.artifact("E/shard-2").inputs == (chunk,)
        store.check_integrity()


def test_without_tokenization_the_no_capture_store_holds_the_files_and_no_shard(tmp_path):
    built = build_table_store(tmp_path / "n", "no-capture", tokenize=False)
    inputs = [name for n in (1, 2, 3)
              for name in (f"E/shard-{n}.sanitizer-input", f"E/shard-{n}.input")]
    assert list(built.instances) == SOURCES + GENERATED + ["R1", "R2"] + inputs
    with Store.open(built.store, readonly=True) as store:
        assert list(store.artifacts(v.KIND_SHARD)) == []
        # The files are linked and the skipped line is recorded all the same.
        assert all_drops(store, built) == [
            ("E/shard-2.sanitizer-input", "L6", v.DROP_SANITIZER_SKIP, "sanitize",
             "E/shard-2.input")]
        report = built.summary["recovery"]["shards"]["E/shard-3"]
        assert (report["sanitizer_input"]["recovered"], report["tokenizer_input"]["recovered"],
                report["tokenizer_input"]["input_lines_unaccounted"]) == (27, 27, 0)
        store.check_integrity()


def test_a_recovery_refuses_a_captured_build_and_an_existing_store(tmp_path, spec_path):
    from replica.errors import OutputError, SpecError
    from replica.recover import Uncaptured

    captured = build_captured_fixture(tmp_path / "c", layouts=(), releases=(), imports=())
    as_run = Uncaptured(plan=load_fixture_spec(spec_path), out=captured.out,
                        summary=captured.summary, options=captured.options)
    with pytest.raises(OutputError, match="a recovery writes a new store"):
        recover_uncaptured(tmp_path / "c", as_run)
    with pytest.raises(SpecError, match="made with capture on"):
        recover_uncaptured(tmp_path / "elsewhere", as_run)


# --- The scratch stores ------------------------------------------------------------------


def test_scratch_notokens_holds_a_line_without_tokens_as_one_drop(tmp_path, no_megatron):
    built = build_table_store(tmp_path / "s", "scratch-notokens", tokenize=True,
                                 megatron_env=no_megatron)
    shard, name = built.instances["c03/shard-notokens"], "c03/shard-notokens"
    assert list(built.instances) == ["src/Z", f"{name}.input", name]
    # Table row c03.drops.no-tokens.planted-line: five input lines, four documents.
    assert built.summary["input_lines"] == 5 and built.summary["documents"] == 4
    assert [built.ordinal_of(name, f"Z-00{n}") for n in (1, 2, 3, 4)] == [0, 1, 2, 3]
    with Store.open(built.store, readonly=True) as store:
        input_instance = built.instances[f"{name}.input"]
        # The line is recorded once, as a drop whose output is the shard.
        assert [(d.input, d.reason, d.step_name) for d in store.drops_by_output(shard)] == [
            (OccurrenceRef(input_instance, "L3"), v.DROP_NO_TOKENS, "tokenize")]
        assert list(store.drops_by_output(input_instance)) == []
        maps = {(b.namespace, b.derivation): b for b in store.bindings(shard)}
        assert set(maps) == {(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION),
                             (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION)}
        for (namespace, derivation), binding in maps.items():
            assert (binding.verification, binding.label, binding.append_eod) == (
                v.VERIFIED_PASSED, v.RECOVERED, True)
            rows = list(store.map_rows(shard, namespace, derivation=derivation))
            # No ordinal shifts: lines 1, 2, 4 and 5 are documents 0 to 3.
            assert [(r.ordinal, r.label, r.parent.locator) for r in rows] == [
                (0, v.RECOVERED, "L1"), (1, v.RECOVERED, "L2"),
                (2, v.RECOVERED, "L4"), (3, v.RECOVERED, "L5")]
            # Each line is the sanitized text of one record of Z.
            records = [store.edge_of(r.parent).inputs[0].parent for r in rows]
            assert records == [OccurrenceRef(built.instances["src/Z"], f"L{n}")
                               for n in (1, 2, 3, 4)]
        store.check_integrity()


def test_scratch_500_holds_a_registered_shard_of_500_documents_and_no_map(tmp_path,
                                                                         no_megatron):
    built = build_table_store(tmp_path / "s", "scratch-500", tokenize=True,
                              megatron_env=no_megatron)
    assert list(built.instances) == ["src/A", "c04/shard-500.input", "c04/shard-500"]
    assert built.summary["documents"] == 500 and built.summary["map"] is None
    with Store.open(built.store, readonly=True) as store:
        shard = store.artifact("c04/shard-500")
        # The rows of Check 4 write the map themselves.
        assert shard.kind == v.KIND_SHARD and store.bindings(shard.instance_id) == ()
        assert shard.inputs == (built.instances["c04/shard-500.input"],)
        # The 500-line file is ingested under a source of its own; A is registered.
        assert {m.name: m.source for m in store.members()} == {"A/r1": "A", "c04/r1": "c04"}
        assert store.occurrence_count(built.instances["c04/shard-500.input"]) == 500
        fingerprints = [o.fingerprint for o in
                        store.occurrences(built.instances["c04/shard-500.input"])]
        assert len(set(fingerprints)) == 500 and None not in fingerprints
        store.check_integrity()
