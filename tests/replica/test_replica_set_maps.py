"""What Checks 3 and 6 need of a generated set, on a small one (plan 5.5): the
capture maps with the two recovery maps beside them, the run made without capture
recovered in both modes, and the two set-ups that change a tokenizer input before
a recovery is run (fixture/table/README.md section 7).

The set has 800 records, 20 round-robin chunks and its dedup switched off, so that
both members of each planted pair reach a shard. Nothing here runs Megatron: the
stand-in of conftest.py writes toy shards. Which pairs lie in one shard comes from
the plant list and the chunk-file ids, counted by ``fixture/plant/count.py``, never
from the tool.
"""

from __future__ import annotations

import re
import shutil

import pytest
from waterwhite import vocabulary as v
from waterwhite.records import OccurrenceRef
from waterwhite.store.store import Store

from replica import setups
from replica.errors import SpecError
from replica.fixture import add_recovery_maps, recover_uncaptured
from replica.sets import build_captured_set, build_uncaptured_set
from replica.shard_index import read_shard_index

LAYOUT = "round-robin"
MAPS = [(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION), (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION),
        (v.NAMESPACE_RECOVERY, v.DERIVED_TOKEN_MATCHING)]
RECOVERY_MAPS = MAPS[1:]


def shard_entries(built) -> list[dict]:
    return [s for s in built.summary["layouts"][LAYOUT]["shards"] if not s["empty"]]


def pairs_by_placement(plant_count, set_dir, out, *,
                       require: str = "auto") -> dict[str, list[dict]]:
    """The planted pairs of the build at ``out`` by where its chunk files put them.
    ``require`` is count.py's ``require_pair_kinds``: a build with unseeded shuffles
    can place no pair in one shard, and is looked at with "no"."""
    counted = plant_count.count(set_dir, chunks=out / "pipeline" / LAYOUT / "tokenizer-input",
                                layout=LAYOUT, require_pair_kinds=require)
    placed: dict[str, list[dict]] = {}
    for pair in counted["check3"]["pairs"]["list"]:
        placed.setdefault(pair["placement"], []).append(pair)
    return placed


def rows_of(store, built, shard: str, which) -> list:
    namespace, derivation = which
    return list(store.map_rows(built.instances[shard], namespace, derivation=derivation))


@pytest.fixture(scope="module")
def pair_set(plant_generate, set_parameters, tmp_path_factory):
    """The small set with its dedup switched off, as Check 3 builds the 100k set."""
    folder = tmp_path_factory.mktemp("pair-set") / "set"
    plant_generate.generate(folder, dedup=False, **set_parameters)
    return folder


@pytest.fixture(scope="module")
def captured(pair_set, tmp_path_factory, megatron_stand_in):
    """The set built with capture on, with the two recovery maps added beside the
    capture map of every shard, in one store."""
    base = tmp_path_factory.mktemp("set-captured")
    with megatron_stand_in(base / "env") as env:
        built = build_captured_set(base / "captured", pair_set, megatron_env=env)
        added = add_recovery_maps(built, megatron_env=env)
    store = Store.open(built.store, readonly=True)
    yield built, store, added
    store.close()


DRAWS = 12


@pytest.fixture(scope="module")
def run(pair_set, plant_count, tmp_path_factory, megatron_stand_in):
    """The set built with no wrapper and unseeded shuffles, and the same run
    recovered in both modes. Gives the base folder, the run and its recovery.

    The shuffles are drawn from the operating system, so a draw can place no
    planted pair in one shard (about three in ten draws do not, with 24 pairs in
    20 chunks). The tests below that read same-shard pairs need one, so the run
    is drawn again, into a new folder, until a draw has one: at most ``DRAWS``
    times, and the fixture fails after that."""
    base = tmp_path_factory.mktemp("set-run")
    for draw in range(DRAWS):
        folder = base / f"draw-{draw}"
        with megatron_stand_in(base / f"env-{draw}") as env:
            made = build_uncaptured_set(folder, pair_set, megatron_env=env)
            placed = pairs_by_placement(plant_count, pair_set, made.out, require="no")
            if not placed.get("same-shard"):
                continue
            recovered = recover_uncaptured(folder, made, also_token_matching=True,
                                           megatron_env=env)
        return base, made, recovered
    pytest.fail(f"no planted pair was placed in one shard in {DRAWS} unseeded runs")


# --- The capture maps and the recovery maps beside them -----------------------------------


def test_every_shard_holds_its_three_maps_side_by_side(captured):
    built, store, added = captured
    entries = shard_entries(built)
    assert len(entries) == 20
    for entry in entries:
        shard = built.instances[entry["store_name"]]
        bindings = store.bindings(shard)
        # Table row c03.map-fields: three maps per shard, each with its namespace,
        # derivation mode, verification result and label.
        assert [(b.namespace, b.derivation) for b in bindings] == MAPS
        assert {(b.verification, b.label, b.append_eod) for b in bindings} == {
            (v.VERIFIED_PASSED, v.RECOVERED, True)}
        for which in MAPS:
            assert {r.label for r in rows_of(store, built, entry["store_name"], which)} <= {
                v.RECOVERED, v.CANDIDATES}
        assert [m["derivation"] for m in added[entry["store_name"]]["maps"]] == [
            v.DERIVED_POSITION, v.DERIVED_TOKEN_MATCHING]
    store.check_integrity()


def test_every_span_is_the_documents_size_in_the_shards_own_index(captured):
    built, store, _ = captured
    for entry in shard_entries(built):
        index = read_shard_index(f"{built.file(entry['store_name'])}.idx")
        rows = rows_of(store, built, entry["store_name"], MAPS[0])
        assert [r.token_end - r.token_start for r in rows] == list(index.document_tokens)
        assert sum(r.token_end - r.token_start for r in rows) == index.tokens
        assert len(rows) == entry["input_lines"]


def test_a_same_shard_pair_is_candidates_with_exactly_its_two_lines_in_all_three_maps(
        captured, pair_set, plant_count):
    built, store, _ = captured
    placed = pairs_by_placement(plant_count, pair_set, built.out)
    assert placed["same-shard"] and placed["split"]
    for pair in placed["same-shard"]:
        first, second = pair["members"]
        name = f"set-rr/{first['shard']}"
        input_instance = built.instances[f"{name}.input"]
        both = {OccurrenceRef(input_instance, f"L{m['line']}") for m in (first, second)}
        for which in MAPS:
            rows = rows_of(store, built, name, which)
            for member in (first, second):
                row = rows[member["ordinal"]]
                assert row.label == v.CANDIDATES and row.parent is None
                assert set(row.candidates) == both


def test_a_split_pair_is_recovered_with_the_line_of_its_own_shard(captured, pair_set,
                                                                 plant_count):
    built, store, _ = captured
    for pair in pairs_by_placement(plant_count, pair_set, built.out)["split"]:
        for member in pair["members"]:
            name = f"set-rr/{member['shard']}"
            for which in MAPS:
                row = rows_of(store, built, name, which)[member["ordinal"]]
                assert (row.label, row.parent) == (v.RECOVERED, OccurrenceRef(
                    built.instances[f"{name}.input"], f"L{member['line']}"))


def test_token_matching_assigns_the_same_record_to_every_other_document(captured):
    built, store, _ = captured
    compared = 0
    for entry in shard_entries(built):
        position = rows_of(store, built, entry["store_name"], MAPS[0])
        matched = rows_of(store, built, entry["store_name"], MAPS[2])
        for a, b in zip(position, matched, strict=True):
            assert a.label == b.label
            if a.label == v.RECOVERED:
                assert a.parent == b.parent and a.parent.locator == f"L{a.ordinal + 1}"
                compared += 1
    assert compared > 300


def test_every_line_the_sanitizer_logged_is_one_drop_and_none_gives_no_tokens(captured,
                                                                             read_rows):
    built, store, _ = captured
    log = read_rows(built.out / "pipeline" / LAYOUT / "sanitizer.log.jsonl")
    assert len(log) == 6 and {row["reason"] for row in log} == {
        v.DROP_SANITIZER_SKIP, v.DROP_EMPTY_LINE}
    expected = sorted((f"set-rr/{row['shard']}.sanitizer-input", f"L{row['line']}",
                       row["reason"], f"set-rr/{row['shard']}.input") for row in log)
    name_of = {instance: name for name, instance in built.instances.items()}
    found = sorted(
        (name_of[d.input.instance_id], d.input.locator, d.reason, name)
        for name, instance in built.instances.items()
        for d in store.drops_by_output(instance) if d.reason != v.DROP_DEDUP_LOSER)
    # Each once, under the tokenizer input; none under a shard, so none for
    # giving no tokens (table rows c03.drops.sanitizer and c03.drops.no-tokens).
    assert found == expected


ID_IN_BYTES = re.compile(rb"[A-Z][A-Z0-9]*-\d{6}")


def test_no_id_of_the_set_reaches_a_store_and_a_job_name_is_a_row_number(
        captured, run, pair_set, read_rows):
    built, store, _ = captured
    path = built.file("set-rr/shard-01.input")
    ids = [row["id"] for row in read_rows(path)]
    assert ids and all(re.fullmatch(r"[A-Z0-9]+-\d{6}", i) for i in ids)
    # The tool reads the text field of a registered file and stores a locator and
    # a fingerprint per line. Plan Check 3 says it is not given the ids of the
    # chunk files, so none of them is stored: not as a handle, not in a path and
    # not in the name of a harness run. A job with many rows is one run per row,
    # named "<job>/<row number in the job's output>".
    generated = read_rows(pair_set / "generation.jsonl")
    every_id = {row["id"].encode() for row in generated} | {i.encode() for i in ids}
    held = built.store.read_bytes()
    assert not [i for i in every_id if i in held]
    assert ID_IN_BYTES.search(held) is None
    per_job: dict[str, int] = {}
    for row in generated:
        per_job[row["job"]] = per_job.get(row["job"], 0) + 1
        name = f"{row['job']}/{per_job[row['job']]}"
        assert store.harness_log(name).job_name == name
    # In the run made without capture no job ran: the store holds no id at all.
    _, _, recovered = run
    assert ID_IN_BYTES.search(recovered.store.read_bytes()) is None


# --- The run made without capture -------------------------------------------------------


def test_the_run_made_without_capture_is_recovered_in_both_modes(run, pair_set, plant_count,
                                                               read_rows):
    _, made, recovered = run
    assert recovered.summary["capture"] is False and recovered.summary["seeded"] is False
    # The ids are kept up to the tokenizer input in this run too.
    assert "id" in read_rows(made.out / "pipeline" / LAYOUT / "tokenizer-input"
                             / "shard-01.jsonl")[0]
    placed = pairs_by_placement(plant_count, pair_set, made.out, require="no")
    assert placed["same-shard"]
    with Store.open(recovered.store, readonly=True) as store:
        for entry in shard_entries(recovered):
            bindings = store.bindings(recovered.instances[entry["store_name"]])
            assert [(b.namespace, b.derivation, b.verification) for b in bindings] == [
                (*which, v.VERIFIED_PASSED) for which in RECOVERY_MAPS]
        for pair in placed["same-shard"]:
            for member in pair["members"]:
                for which in RECOVERY_MAPS:
                    row = rows_of(store, recovered, f"set-rr/{member['shard']}",
                                  which)[member["ordinal"]]
                    assert row.label == v.CANDIDATES and len(row.candidates) == 2
        assert list(store.artifacts(v.KIND_MIXTURE)) == []
        store.check_integrity()
    # The release's copy step was not observed: the rows of the planted pairs have
    # both members as candidates (plan L3; table row c06.generated.candidates).
    release = recovered.summary["recovery"]["releases"]["R1"]
    assert release["candidates"] >= 2 * 24 and release["no_edge"] == 6


def test_swapping_the_two_lines_of_a_twin_pair_changes_no_label_and_no_candidate_set(
        run, pair_set, plant_count, megatron_stand_in, read_rows):
    base, made, recovered = run
    same_shard = pairs_by_placement(plant_count, pair_set, made.out, require="no")["same-shard"]
    assert same_shard
    # Set-up t.twin-swap: before the recovery is run, the pair's two lines change
    # places in the shard's tokenizer input; nothing else changes.
    shutil.copytree(made.out, base / "swapped" / "build")
    swaps = [(pair["members"][0]["shard"], pair["members"][0]["line"],
              pair["members"][1]["line"]) for pair in same_shard]
    done = setups.swap_twin_lines(base / "swapped" / "build", LAYOUT, swaps)
    assert len(done) == len(swaps) and all(d["ids"][0] != d["ids"][1] for d in done)
    with megatron_stand_in(base / "env-swap") as env:
        swapped = recover_uncaptured(base / "swapped", made.moved_to(base / "swapped" / "build"),
                                     also_token_matching=True, megatron_env=env)

    def locators(row):
        return (row.label, row.parent and row.parent.locator,
                sorted(c.locator for c in row.candidates))

    with Store.open(recovered.store, readonly=True) as plain, \
            Store.open(swapped.store, readonly=True) as changed:
        for shard in {swap[0] for swap in swaps}:
            name = f"set-rr/{shard}"
            (first, _) = changed.bindings(swapped.instances[name])
            assert (first.derivation, first.verification) == (v.DERIVED_POSITION,
                                                              v.VERIFIED_PASSED)
            before = [locators(r) for r in rows_of(plain, recovered, name, RECOVERY_MAPS[0])]
            after = [locators(r) for r in rows_of(changed, swapped, name, RECOVERY_MAPS[0])]
            assert after == before
    # The manifest follows the lines: each row still names the record on its line.
    shard, first_line, second_line = swaps[0]
    for folder, order in ((made.out, (first_line, second_line)),
                          (swapped.out, (second_line, first_line))):
        path = folder / "pipeline" / LAYOUT / "tokenizer-input" / f"{shard}.jsonl"
        ids = [row["id"] for row in read_rows(path)]
        assert [ids[n - 1] for n in order] == done[0]["ids"]


def test_a_set_store_is_the_run_made_without_capture_recovered_by_the_tables_rule(
        pair_set, spec_path, no_megatron, tmp_path):
    from replica.fixture import build_set_store

    built = build_set_store(tmp_path / "set-store", pair_set, "no-capture", layout=LAYOUT,
                            tokenizer_spec=spec_path, parallel=2, megatron_env=no_megatron)
    # The same type as every other build: the store, the build folder and the names.
    assert built.store == tmp_path / "set-store" / "store.sqlite"
    assert built.out == tmp_path / "set-store" / "build"
    assert built.summary["capture"] is False and built.summary["seeded"] is False
    names = list(built.instances)
    assert names[:4] == ["src/A", "src/B", "src/C", "src/Z"]
    assert {"gen/paraphrase", "gen/translation", "gen/summary", "R1", "R2"} <= set(names)
    entries = shard_entries(built)
    assert len(entries) == 20
    with Store.open(built.store, readonly=True) as store:
        for entry in entries:
            name = entry["store_name"]
            assert name.startswith("set-rr/")
            assert f"{name}.input" in built.instances and f"{name}.sanitizer-input" in names
            # The rule of the table's no-capture store: the tokenizer input kept and
            # the position-mode map in the recovery namespace, and nothing else.
            assert [(b.namespace, b.derivation, b.verification)
                    for b in store.bindings(built.instances[name])] == [
                (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)]
        assert list(store.artifacts(v.KIND_MIXTURE)) == []
        assert {store.artifact(n).role for n in ("R1", "R2")} == {v.ROLE_RELEASE}
        store.check_integrity()
    assert built.files[f"{entries[0]['store_name']}.input"].startswith(
        "pipeline/round-robin/tokenizer-input/")
    # Only the table's store kind is built, and a layout of the replica.
    with pytest.raises(SpecError, match="no-capture"):
        build_set_store(tmp_path / "other", pair_set, "captured", layout=LAYOUT,
                        tokenizer_spec=spec_path, megatron_env=no_megatron)
    with pytest.raises(SpecError, match="layout must be one of"):
        build_set_store(tmp_path / "other", pair_set, "no-capture", layout="sorted",
                        tokenizer_spec=spec_path, megatron_env=no_megatron)
    assert not (tmp_path / "other").exists()


def test_a_line_changed_after_tokenization_fails_both_recoveries_of_its_shard(
        run, megatron_stand_in):
    base, made, _ = run
    # Set-up t.verification-fails: line 1 of the first shard's tokenizer input is
    # replaced after tokenization; the line count stays; both recoveries then run.
    shutil.copytree(made.out, base / "changed" / "build")
    changed = setups.change_first_line(base / "changed" / "build", LAYOUT)
    assert changed["shard"] == "shard-01" and changed["line"] == 1
    with megatron_stand_in(base / "env-changed") as env:
        failed = recover_uncaptured(base / "changed", made.moved_to(base / "changed" / "build"),
                                    also_token_matching=True, megatron_env=env)
    with Store.open(failed.store, readonly=True) as store:
        bindings = store.bindings(failed.instances["set-rr/shard-01"])
        # Table row c03.verification-failed: failed, "unknown", in both modes.
        assert sorted((b.derivation, b.verification, b.label) for b in bindings) == [
            (v.DERIVED_POSITION, v.VERIFIED_FAILED, v.UNKNOWN),
            (v.DERIVED_TOKEN_MATCHING, v.VERIFIED_FAILED, v.UNKNOWN)]
        for which in RECOVERY_MAPS:
            rows = rows_of(store, failed, "set-rr/shard-01", which)
            assert {(r.label, r.parent, r.candidates) for r in rows} == {(v.UNKNOWN, None, ())}
        # The other shards are untouched and verify.
        other = store.bindings(failed.instances["set-rr/shard-02"])
        assert {b.verification for b in other} == {v.VERIFIED_PASSED}
    report = failed.summary["recovery"]["shards"]["set-rr/shard-01"]
    assert report["verified"] is False and report["maps"][0]["failures"] >= 1
    with pytest.raises(Exception, match="different texts"):
        setups.swap_twin_lines(base / "changed" / "build", LAYOUT, [("shard-01", 1, 2)])
