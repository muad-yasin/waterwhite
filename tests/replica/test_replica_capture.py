"""The capture-on path against the real store, wrappers and harness (plan 5.4).

Each test builds the hand-checked fixture without tokenization with capture on and
then reads the store. What is asked: are the capture points of the first build
there (ingest, the three generation jobs, the dedup selection, the import), does
the store hold nothing for the recovery points (copy steps and the sanitizer), and
do the handed records carry the identity questions of plan Check 7.
"""

from __future__ import annotations

import pytest
from waterwhite import ids
from waterwhite import vocabulary as v
from waterwhite.errors import IntegrityFault
from waterwhite.harness import read_log
from waterwhite.records import OccurrenceRef
from waterwhite.store.store import Store

from replica.build import BuildOptions, RecordChanges, build
from replica.capture_hooks import StoreCapture
from replica.errors import CaptureError, ManifestError, OutputError, SpecError, TokenizerError
from replica.fixture import (
    TABLE_STORES,
    TABLE_STORES_LACK,
    TABLE_STORES_NOT_BUILT,
    build_captured_fixture,
    build_table_store,
    register_again,
)

# These tests ask what a store holds, not what a bound gives: their generation
# jobs run in process, so they need no sandbox (the table's stores run confined by
# default: test_replica_harness_cli.py).
IN_PROCESS = "in-process"

NAMES = ["src/A", "src/B", "src/C", "src/Z", "gen/P", "gen/T", "gen/S",
         "R1", "R1-swap", "R2", "import/T"]


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """One captured build of the fixture, without tokenization, and its store
    opened read-only."""
    built = build_captured_fixture(tmp_path_factory.mktemp("captured") / "fixture")
    store = Store.open(built.store, readonly=True)
    yield built, store
    store.close()


def parents(edge):
    return [(i.parent.instance_id, i.parent.locator) for i in edge.inputs if i.used]


# --- The call other code makes ---------------------------------------------------------


def test_the_call_returns_the_store_the_build_folder_and_the_manifests(captured, read_rows):
    built, store = captured
    assert built.store.name == "store.sqlite" and built.store.is_file()
    assert built.out == built.store.parent / "build" and built.out.is_dir()
    assert built.summary["capture"] is True and built.summary["tokenized"] is False
    assert list(built.instances) == NAMES
    manifest = read_rows(built.manifest("releases/R1.jsonl"))
    assert manifest[0] == {"line": 1, "handle": "X@A", "origin": ["A-001"]}
    assert store.normaliser == "control-chars/1"
    # The store lies beside the build folder, so every path it holds is relative.
    assert store.artifact("R1").path == "build/releases/R1.jsonl"


def test_a_folder_that_is_not_empty_is_refused(tmp_path):
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    with pytest.raises(OutputError):
        build_captured_fixture(tmp_path)
    assert not (tmp_path / "store.sqlite").exists()


def test_a_store_can_be_built_without_the_swap_and_without_the_import(tmp_path):
    built = build_captured_fixture(tmp_path / "main", releases=("R1", "R2"), imports=(),
                                   layouts=("partitioned",))
    assert list(built.instances) == [n for n in NAMES if n not in ("R1-swap", "import/T")]


# --- The stores of the hand-written table (fixture/table/README.md section 4) -------------


def test_the_main_store_holds_no_swapped_release_and_no_import(tmp_path):
    # Without tokenization the store holds the files and no shard or mixture
    # (test_replica_maps.py asks about those).
    built = build_table_store(tmp_path / "main", "main", tokenize=False, generation_harness=IN_PROCESS)
    assert list(built.instances) == ["src/A", "src/B", "src/C", "src/Z", "gen/P", "gen/T", "gen/S",
                                     "R1", "R2"]
    assert built.files["R1"] == "releases/R1.jsonl" and built.files["src/Z"] == "sources/Z.jsonl"
    assert list(built.summary["layouts"]) == ["partitioned"]
    with Store.open(built.store, readonly=True) as store:
        assert [a.names for a in store.artifacts()] == [(name,) for name in built.instances]
        store.check_integrity()


def test_the_swap_store_holds_the_swapped_release_beside_r1(tmp_path):
    built = build_table_store(tmp_path / "swap", "swap", generation_harness=IN_PROCESS)
    assert list(built.instances) == ["src/A", "src/B", "src/C", "src/Z", "gen/P", "gen/T", "gen/S",
                                     "R1", "R1-swap"]
    assert built.instances["R1"] != built.instances["R1-swap"]
    assert built.file("R1").read_bytes() == built.file("R1-swap").read_bytes()


def test_the_import_store_holds_the_sources_and_the_import_and_no_generation_output(tmp_path):
    built = build_table_store(tmp_path / "import", "import")
    assert list(built.instances) == ["src/A", "src/B", "src/C", "src/Z", "import/T"]
    # The generation outputs are written, since the import copies T's text, and
    # nothing about them is registered: no job ran in the harness.
    assert (built.out / "gen" / "T.jsonl").is_file()
    assert not (built.out / "gen" / "harness").exists()
    with Store.open(built.store, readonly=True) as store:
        assert store.artifacts_by_path(built.out / "gen" / "T.jsonl") == ()
        assert len(list(store.artifacts())) == 5
        store.check_integrity()


def test_a_store_that_will_hold_runs_lists_the_joins_of_those_runs_as_verified(tmp_path):
    # A run registered in a store whose list of verified joins is empty has no trace
    # route: the join answers "data-order trace unverified for this version" for every
    # segment (docs/join.md). The stores of the table that hold runs name the joins
    # their runs use (the release the spec pins, each run's sampler type and class).
    tag = "core_v0.20.0"
    single = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
    cyclic = "megatron.training.datasets.data_samplers.MegatronPretrainingRandomSampler"
    expected = {"main": [(tag, "cyclic", cyclic), (tag, "single", single)],
                "rr": [(tag, "single", single)],
                "swap": []}
    for name, joins in expected.items():
        built = build_table_store(tmp_path / name, name, generation_harness=IN_PROCESS)
        with Store.open(built.store, readonly=True) as store:
            assert list(store.verified_joins()) == joins, name


def test_every_store_of_the_table_has_a_build_and_an_unknown_name_is_refused(tmp_path):
    # The stores of fixture/table/README.md section 4. None is refused as not
    # built; what main and rr still lack of the table's list is said by name.
    assert set(TABLE_STORES) == {"main", "rr", "swap", "no-capture", "import", "scratch-500",
                                 "scratch-notokens"}
    assert TABLE_STORES_NOT_BUILT == {}
    # The call registers no run; the runs come from the run wrapper and its registration
    # (docs/replica.md section 13), and what the call lacks says so.
    assert set(TABLE_STORES_LACK) == {"main", "rr"}
    for name in ("main", "rr"):
        assert "build_runs" in TABLE_STORES_LACK[name]
        assert "register_runs" in TABLE_STORES_LACK[name]
        assert "next milestone" not in TABLE_STORES_LACK[name]
    assert "checkpoints E@6, E@9, D@3, Y@6, K@3, K@8 and K@40" in TABLE_STORES_LACK["main"]
    assert "E-rr@6" in TABLE_STORES_LACK["rr"]
    # The stores that hold shards are tokenized when the caller passes tokenize=True.
    assert {name for name, variant in TABLE_STORES.items() if variant.tokenize} == {
        "main", "rr", "no-capture", "scratch-500", "scratch-notokens"}
    with pytest.raises(SpecError, match="defines no store"):
        build_table_store(tmp_path / "attic", "attic")
    # A scratch store is nothing but a tokenized shard.
    with pytest.raises(SpecError, match="without tokenization"):
        build_table_store(tmp_path / "scratch", "scratch-500", tokenize=False)
    assert not (tmp_path / "attic").exists() and not (tmp_path / "scratch").exists()


def test_a_store_that_needs_megatron_is_refused_before_anything_is_written(tmp_path,
                                                                           monkeypatch):
    monkeypatch.delenv("WATERWHITE_MEGATRON_ENV", raising=False)
    for name in ("main", "rr", "no-capture", "scratch-500", "scratch-notokens"):
        with pytest.raises(TokenizerError, match="WATERWHITE_MEGATRON_ENV"):
            build_table_store(tmp_path / name, name, tokenize=True)
        assert not (tmp_path / name / "store.sqlite").exists()


def test_a_store_is_built_without_tokenizing_unless_tokenize_is_passed(tmp_path, monkeypatch):
    # The call the acceptance world makes: build_table_store(folder, name), with no
    # Megatron environment. Every store of the table but the two scratch stores is
    # built, with no shard, no map and no mixture; the scratch stores are refused.
    monkeypatch.delenv("WATERWHITE_MEGATRON_ENV", raising=False)
    for name in ("main", "rr", "swap", "import", "no-capture"):
        built = build_table_store(tmp_path / name, name, generation_harness=IN_PROCESS)
        assert built.options is None or built.options.tokenize is False
        assert not [n for n in built.instances if n.endswith(("/shard-1", "/shard-2", "/mix-1"))]
        with Store.open(built.store, readonly=True) as store:
            assert list(store.artifacts(v.KIND_SHARD)) == []
            assert list(store.artifacts(v.KIND_MIXTURE)) == []
    for name in ("scratch-500", "scratch-notokens"):
        with pytest.raises(SpecError, match="without tokenization"):
            build_table_store(tmp_path / name, name)


def test_a_release_cannot_be_captured_over_generation_outputs_that_are_not(tmp_path):
    with pytest.raises(CaptureError, match="gen/P.jsonl was not registered"):
        build_captured_fixture(tmp_path / "half", releases=("R1",), imports=(), layouts=(),
                               capture_generation=False)


def test_a_handle_is_found_through_the_manifest_of_its_file(captured, read_rows):
    built, _store = captured
    assert built.manifest_rows("R1") == read_rows(built.manifest("releases/R1.jsonl"))
    assert built.line_of("R1", "X@A") == 1 and built.line_of("R1-swap", "X@B") == 1
    assert built.line_of("src/A", "SA") == 2 and built.line_of("R1", "S") == 4
    assert built.file("gen/T") == built.out / "gen" / "T.jsonl"
    with pytest.raises(ManifestError, match="on 0 lines"):
        built.line_of("R1", "X@B")          # R1 kept A's copy
    with pytest.raises(ManifestError, match="registered nothing"):
        built.line_of("E/shard-3", "T")     # shards are not registered yet


def test_the_same_build_registered_again_keeps_every_instance(tmp_path):
    # Without tokenization; test_replica_maps.py asks the same of shards, maps
    # and mixtures.
    first = build_table_store(tmp_path / "main", "main", tokenize=False, generation_harness=IN_PROCESS)
    again = register_again(first)
    assert again.store == first.store and again.out == first.store.parent / "build-again"
    assert again.instances == first.instances and again.files == first.files
    assert again.file("R1").read_bytes() == first.file("R1").read_bytes()
    with Store.open(first.store, readonly=True) as store:
        assert len(list(store.artifacts())) == len(first.instances)
        # The stored path is the one of the first registration.
        assert store.artifact("R1").path == "build/releases/R1.jsonl"
        store.check_integrity()
    with pytest.raises(OutputError):
        register_again(first)               # the folder of the second build exists now
    assert register_again(first, "third").out.name == "build-third"


# --- The capture points ---------------------------------------------------------------


def test_the_store_holds_the_named_artifacts_and_nothing_for_the_recovery_points(captured):
    built, store = captured
    artifacts = list(store.artifacts())
    assert [a.names for a in artifacts] == [(name,) for name in NAMES]
    assert {a.instance_id for a in artifacts} == set(built.instances.values())
    assert [a.kind for a in artifacts] == [v.KIND_JSONL] * len(NAMES)
    # Copy steps and the sanitizer are recovery points: their files exist in the
    # build folder and the store knows nothing about them.
    for step in ("sanitizer-input", "tokenizer-input"):
        path = built.out / "pipeline" / "partitioned" / step / "shard-3.jsonl"
        assert path.is_file() and store.artifacts_by_path(path) == ()
    assert [a.role for a in artifacts] == (
        [v.ROLE_SOURCE] * 4 + [v.ROLE_INTERMEDIATE] * 3 + [v.ROLE_RELEASE] * 3 + [v.ROLE_IMPORT])


def test_every_source_file_is_ingested_under_its_member(captured):
    built, store = captured
    assert store.sources() == ("A", "B", "C", "Z")
    for key, lines in (("A", 2), ("B", 13), ("C", 4), ("Z", 90)):
        member = store.member(f"{key}/r1")
        instance = built.instances[f"src/{key}"]
        assert (member.source, member.revision, member.parts) == (key, "r1", ((instance, ""),))
        assert store.occurrence_count(instance) == lines
        digest = ids.file_digest(built.out / "sources" / f"{key}.jsonl")
        assert instance == ids.ingest_instance_id(digest, f"{key}/r1")
    # Every record of Z carries a fingerprint: no source line is a skipped line.
    assert all(o.fingerprint is not None for o in store.occurrences(built.instances["src/Z"]))


def test_each_generation_job_ran_in_the_harness_with_its_hand_off(captured, read_rows):
    built, store = captured
    src_a, src_c, gen_p = (built.instances[n] for n in ("src/A", "src/C", "gen/P"))
    expected = {
        "paraphrase": ("P", [OccurrenceRef(src_a, "L1")]),            # A-001, from the source file
        "translation": ("T", [OccurrenceRef(gen_p, "L1")]),           # P
        "summary": ("S", [OccurrenceRef(src_a, "L2"), OccurrenceRef(src_c, "L1")]),
    }
    for job, (output, handed) in expected.items():
        log = store.harness_log(job)
        assert list(log.handed) == handed
        assert [r.occurrence for r in log.reads] == handed and not any(r.refused for r in log.reads)
        # The fixture builds in process here, so the log establishes no bound
        # (Decision 13); its reads lie within the hand-off.
        assert log.model == "none" and log.reads_within_hand_off() and not log.bound_established()
        # The log file of this build, at a path of its own, is the one the store holds.
        path = built.out / "gen" / "harness" / output / f"{output}.log.jsonl"
        assert read_log(path) == log
    # P is made from the text of A's record as the registered source file holds it.
    source_text = read_rows(built.out / "sources" / "A.jsonl")[0]["text"]
    assert source_text.startswith("Every Thursday before sunrise")


def test_a_generation_output_is_registered_with_the_edge_of_its_job(captured):
    built, store = captured
    src_a, src_c, gen_p = (built.instances[n] for n in ("src/A", "src/C", "gen/P"))
    expected = {"gen/P": ("paraphrase", [(src_a, "L1")], (src_a,)),
                "gen/T": ("translation", [(gen_p, "L1")], (gen_p,)),
                "gen/S": ("summary", [(src_a, "L2"), (src_c, "L1")], (src_a, src_c))}
    for name, (job, named, inputs) in expected.items():
        artifact = store.artifact(name)
        (edge,) = store.edges(artifact.instance_id)
        assert edge.child == OccurrenceRef(artifact.instance_id, "L1")
        assert edge.is_generation and edge.parent_mode == v.MODE_ALL
        assert edge.derivation == v.DERIVED_STATED and edge.label == v.OBSERVED
        assert parents(edge) == named and all(i.used for i in edge.inputs)
        assert edge.harness_log.job_name == job and edge.bound_field is False  # in process
        assert artifact.inputs == inputs and artifact.step.name == "replica.generation"


def test_the_dedup_selection_states_an_any_edge_with_the_kept_copy_selected_and_a_drop(captured):
    built, store = captured
    src = {k: built.instances[f"src/{k}"] for k in "ABZ"}
    r1 = store.artifact("R1")
    edges = list(store.edges(r1.instance_id))
    assert len(edges) == 24 == store.occurrence_count(r1.instance_id)
    first = edges[0]
    assert first.child.locator == "L1" and first.parent_mode == v.MODE_ANY
    assert [(i.parent.instance_id, i.parent.locator, i.selected) for i in first.inputs] == [
        (src["A"], "L1", True), (src["B"], "L1", False)]
    # A row with no equivalent copy has a single-parent edge.
    for edge in edges[1:]:
        assert edge.parent_mode == v.MODE_ALL and len(edge.inputs) == 1
        assert edge.derivation == v.DERIVED_STATED and not edge.is_generation
        assert not edge.inputs[0].selected
    assert parents(edges[1]) == [(built.instances["gen/P"], "L1")]
    assert parents(edges[4]) == [(src["Z"], "L1")]
    drops = list(store.drops_by_output(r1.instance_id))
    assert [(d.input, d.reason, d.step_name, d.group_key) for d in drops] == [
        (OccurrenceRef(src["B"], "L1"), v.DROP_DEDUP_LOSER, "replica.release", "L1")]
    assert r1.inputs == (src["A"], src["B"], built.instances["gen/P"], built.instances["gen/T"],
                         built.instances["gen/S"], src["Z"])
    # R2 has no equivalent copies: no "any" edge and no drop.
    r2 = store.artifact("R2")
    assert all(e.parent_mode == v.MODE_ALL for e in store.edges(r2.instance_id))
    assert list(store.drops_by_output(r2.instance_id)) == [] and r2.inputs == (src["B"], src["Z"])


def test_the_declared_inputs_of_every_step_cover_every_parent_it_states(captured):
    # DECISIONS.md D-15: a stated edge's parent lies in a declared input.
    _, store = captured
    checked = 0
    for artifact in store.artifacts():
        for edge in store.edges(artifact.instance_id):
            for item in edge.inputs:
                assert item.parent.instance_id in artifact.inputs, (artifact.names, edge.child)
                checked += 1
        for drop in store.drops_by_output(artifact.instance_id):
            assert drop.input.instance_id in artifact.inputs
    assert checked == 4 + 25 + 25 + 33  # three jobs, R1, R1-swap, R2


def test_swapping_the_winner_changes_the_instance_and_not_one_byte(captured):
    built, store = captured
    r1, swap = store.artifact("R1"), store.artifact("R1-swap")
    assert (built.out / "releases" / "R1.jsonl").read_bytes() == (
        built.out / "releases" / "R1-swap.jsonl").read_bytes()
    assert r1.content_digest == swap.content_digest          # the same bytes
    assert r1.step == swap.step and r1.inputs == swap.inputs
    assert r1.edge_digest != swap.edge_digest
    assert r1.instance_id != swap.instance_id                # two instances (Decision 29)
    # A query by the shared digest cannot pick one of them.
    assert {a.instance_id for a in store.artifacts_by_digest(r1.content_digest)} == {
        r1.instance_id, swap.instance_id}
    selected = [next(i.parent for i in store.edge_of(OccurrenceRef(a.instance_id, "L1")).inputs
                     if i.selected) for a in (r1, swap)]
    assert selected == [OccurrenceRef(built.instances["src/A"], "L1"),
                        OccurrenceRef(built.instances["src/B"], "L1")]
    # The two occurrences of X carry different keys and the same fingerprint.
    x_r1 = store.occurrence(OccurrenceRef(r1.instance_id, "L1"))
    x_swap = store.occurrence(OccurrenceRef(swap.instance_id, "L1"))
    assert x_r1.ref.key != x_swap.ref.key and x_r1.fingerprint == x_swap.fingerprint
    (drop,) = store.drops_by_output(swap.instance_id)
    assert drop.input == OccurrenceRef(built.instances["src/A"], "L1")


def test_the_text_only_import_carries_its_file_family_and_no_parent(captured):
    built, store = captured
    imported = store.artifact("import/T")
    assert (imported.role, imported.member_name, imported.step) == (v.ROLE_IMPORT, "import/T", None)
    assert imported.instance_id != built.instances["gen/T"]
    (link,) = store.file_source_links(imported.instance_id)
    assert (link.locator, link.member, link.source, link.label, link.evidence) == (
        None, "A/r1", "A", v.FILE_LEVEL, "file family: translations made from source A")
    assert list(store.edges(imported.instance_id)) == []
    assert list(store.record_source_links(imported.instance_id)) == []


def test_the_loud_checks_pass_on_the_captured_store_and_stop_on_a_changed_file(tmp_path):
    built = build_captured_fixture(tmp_path / "fixture", layouts=("partitioned",))
    with Store.open(built.store, readonly=True) as store:
        store.check_integrity()
    release = built.out / "releases" / "R2.jsonl"
    release.write_bytes(release.read_bytes().replace(b"Eggs", b"Figs"))
    with Store.open(built.store, readonly=True) as store:
        with pytest.raises(IntegrityFault):
            store.check_integrity()


# --- A second build into the same store ---------------------------------------------------


def test_the_same_history_built_again_gives_the_same_instances(fixture_plan, tmp_path):
    first = build_captured_fixture(tmp_path / "fixture", layouts=("partitioned",))
    capture = StoreCapture.open(first.store, normaliser=fixture_plan.normaliser)
    try:
        again = build(fixture_plan, tmp_path / "fixture" / "again",
                      BuildOptions(tokenize=False, layouts=("partitioned",)), capture=capture)
    finally:
        capture.close()
    assert again["instances"] == first.instances
    with Store.open(first.store, readonly=True) as store:
        assert len(list(store.artifacts())) == len(NAMES)
        store.check_integrity()


def test_a_repaired_build_registers_under_a_prefix_beside_the_first(fixture_plan, tmp_path):
    first = build_captured_fixture(tmp_path / "fixture", layouts=("partitioned",))
    capture = StoreCapture.open(first.store, normaliser=fixture_plan.normaliser)
    options = BuildOptions(tokenize=False, layouts=("partitioned",), store_name_prefix="repair-1/",
                           changes=RecordChanges(leave_out=frozenset({"A-001", "P", "T", "S"})))
    try:
        repaired = build(fixture_plan, tmp_path / "fixture" / "repaired", options, capture=capture)
    finally:
        capture.close()
    names = list(repaired["instances"])
    assert all(n.startswith("repair-1/") for n in names) and "repair-1/import/T" not in names
    with Store.open(first.store, readonly=True) as store:
        # The sources and the generation outputs are the same instances under a
        # second name; the repaired release is a new one that takes X from B.
        assert set(store.artifact("src/A").names) == {"src/A", "repair-1/src/A"}
        assert set(store.artifact("gen/P").names) == {"gen/P", "repair-1/gen/P"}
        repaired_r1 = store.artifact("repair-1/R1")
        assert repaired_r1.instance_id != first.instances["R1"]
        edges = list(store.edges(repaired_r1.instance_id))
        assert len(edges) == 21 and all(e.parent_mode == v.MODE_ALL for e in edges)
        assert parents(edges[0]) == [(first.instances["src/B"], "L1")]
        assert list(store.drops_by_output(repaired_r1.instance_id)) == []
        # R2 holds no affected record: the repaired build's R2 is the first one's.
        assert store.artifact("repair-1/R2").instance_id == first.instances["R2"]
        store.check_integrity()


def test_a_store_made_with_another_normaliser_is_refused(tmp_path):
    Store.create(tmp_path / "other.sqlite", normaliser="another-rule/1").close()
    with pytest.raises(CaptureError, match="another-rule/1"):
        StoreCapture.open(tmp_path / "other.sqlite", normaliser="control-chars/1")


# --- A generated set with capture on ------------------------------------------------------


def test_a_generated_set_is_captured_at_the_same_points(small_set, tmp_path):
    from replica.plan import load_set

    plan = load_set(small_set)
    capture = StoreCapture.open(tmp_path / "store.sqlite", normaliser=plan.normaliser)
    try:
        summary = build(plan, tmp_path / "build", BuildOptions(tokenize=False), capture=capture)
    finally:
        capture.close()
    assert list(summary["instances"]) == [
        "src/A", "src/B", "src/C", "src/Z", "gen/paraphrase", "gen/translation", "gen/summary",
        "R1", "R2"]
    with Store.open(tmp_path / "store.sqlite", readonly=True) as store:
        r1 = store.artifact("R1")
        edges = list(store.edges(r1.instance_id))
        # 401 rows; 64 dedup groups (40 X pairs, 24 planted pairs), each with one drop.
        assert len(edges) == 401 == store.occurrence_count(r1.instance_id)
        assert sum(e.parent_mode == v.MODE_ANY for e in edges) == 64
        assert len(list(store.drops_by_output(r1.instance_id))) == 64
        # The six lines the sanitizer will skip are rows of R1 with no fingerprint,
        # each with its single-parent edge to the source line.
        bare = [o.ref for o in store.occurrences(r1.instance_id) if o.fingerprint is None]
        assert len(bare) == 6
        assert all(len(store.edge_of(ref).inputs) == 1 for ref in bare)
        # One harness run per generated row, named job/row number.
        log = store.harness_log("paraphrase/1")
        assert log.handed == (OccurrenceRef(summary["instances"]["src/A"], "L1"),)
        assert store.harness_log("summary/8").model == "none"
        store.check_integrity()


def test_a_second_release_can_drop_the_record_the_first_one_dropped(fixture_plan, tmp_path):
    # The store holds one drop per dropped record, step name and output artifact.
    # A second build of R1 with one unrelated row left out has other content and
    # drops B's copy of X again: it is registered beside the first, with its drop.
    first = build_captured_fixture(tmp_path / "fixture", layouts=(), releases=("R1",),
                                   imports=())
    capture = StoreCapture.open(first.store, normaliser=fixture_plan.normaliser)
    options = BuildOptions(tokenize=False, layouts=(), releases=("R1",), imports=(),
                           store_name_prefix="second/",
                           changes=RecordChanges(leave_out=frozenset({"Z-020"})))
    try:
        second = build(fixture_plan, tmp_path / "fixture" / "second", options, capture=capture)
    finally:
        capture.close()
    assert second["instances"]["second/R1"] != first.instances["R1"]
    with Store.open(first.store, readonly=True) as store:
        loser = OccurrenceRef(first.instances["src/B"], "L1")
        for release in (first.instances["R1"], second["instances"]["second/R1"]):
            assert [(d.input, d.reason) for d in store.drops_by_output(release)] == [
                (loser, v.DROP_DEDUP_LOSER)]
        assert len(store.drops_of(loser)) == 2
        store.check_integrity()
