"""One build of the hand-checked fixture, without tokenization: the completed spec,
files, manifests, the skipped sanitizer line, the dedup winner, reproducibility,
leaving out and swapping."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from replica.build import BuildOptions, RecordChanges, build, tree_digests
from replica.errors import InputError, OutputError, ReplicaError, SpecError
from replica.plan import load_fixture_spec
from replica.steps import expand_ids

LAYOUTS = ("partitioned", "round-robin")
SHARDS = ("shard-1", "shard-2", "shard-3")
SKIPPED_LINE = '{"id": "skip-1", "text": "this line is cut off'


def ids_of(rows):
    return [r["id"] for r in rows]


def command_environment(repo) -> dict:
    """The environment of ``python -m replica`` run from the repository root: the
    tool's package must be importable, since the sanitizer takes its normaliser
    from it (DECISIONS.md D-17)."""
    return {**os.environ, "PYTHONPATH": str(repo / "src")}


def changed_spec(spec_path, tmp_path, old: str, new: str):
    """A copy of the spec with one piece of text replaced, and the base folder its
    file paths are relative to."""
    text = spec_path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path = tmp_path / "spec.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path, spec_path.parent.parent.parent


# --- The completed spec ---------------------------------------------------------------


def test_the_mixtures_are_read_from_the_spec_in_the_trainers_order(fixture_plan):
    mixtures = {m.name: (m.layout, m.shards) for m in fixture_plan.mixtures}
    assert mixtures == {
        "E/mix-1": ("partitioned", ("shard-1", "shard-2")),
        "E/mix-2": ("partitioned", ("shard-1", "shard-2", "shard-3")),
        "D/mix-1": ("partitioned", ("shard-1", "shard-2")),
        "Y/mix-1": ("partitioned", ("shard-1", "shard-2", "shard-3")),
        # Run K's list is not in name order on purpose: shard-3 is dataset 0.
        "K/mix-1": ("partitioned", ("shard-3", "shard-2")),
        # A name such as "E-rr/shard-1" is a shard of the round-robin layout.
        "E-rr/mix-1": ("round-robin", ("shard-1", "shard-2")),
        "E-rr/mix-2": ("round-robin", ("shard-1", "shard-2", "shard-3")),
    }


def test_the_segments_name_their_run_and_mixture(fixture_plan):
    segments = [(s.run, s.name, s.position, s.mixture) for s in fixture_plan.segments]
    assert segments == [
        ("E", "E/seg-1", 1, "E/mix-1"), ("E", "E/seg-2", 2, "E/mix-2"),
        ("E-rr", "E-rr/seg-1", 1, "E-rr/mix-1"), ("E-rr", "E-rr/seg-2", 2, "E-rr/mix-2"),
        ("D", "D/seg-1", 1, "D/mix-1"), ("Y", "Y/seg-1", 1, "Y/mix-1"),
        ("K", "K/seg-1", 1, "K/mix-1"),
    ]


def test_the_skipped_line_and_the_tokenizer_files_are_read_from_the_spec(fixture_plan):
    assert [(i.layout, i.shard, i.line, i.text) for i in fixture_plan.inserted_lines] == [
        ("partitioned", "shard-2", 6, SKIPPED_LINE), ("round-robin", "shard-2", 6, SKIPPED_LINE)]
    tokenizer = fixture_plan.tokenizer
    assert len(tokenizer.revision) == 40 and int(tokenizer.revision, 16) >= 0
    snapshot = f"hf_home/hub/models--gpt2/snapshots/{tokenizer.revision}"
    assert tokenizer.vocab_file == f"{snapshot}/vocab.json"
    assert tokenizer.merges_file == f"{snapshot}/merges.txt"
    assert (tokenizer.type, tokenizer.name, tokenizer.append_eod, tokenizer.partitions) == (
        "GPT2BPETokenizer", "gpt2", True, 1)


def test_tables_and_keys_the_replica_does_not_use_are_left_alone(spec_path, tmp_path,
                                                               fixture_plan):
    path, base = changed_spec(
        spec_path, tmp_path, "[where]",
        "[runs.E.segments.\"E/seg-1\".later]\nnote = 1\n\n[something_new]\nkey = [1, 2]\n\n[where]")
    assert load_fixture_spec(path, base) == fixture_plan


@pytest.mark.parametrize("old, new, message", [
    ('"K/mix-1" = ["shard-3", "shard-2"]', '"K/mix-1" = ["shard-3", "shard-9"]',
     "names a shard the layout"),
    ('"K/mix-1" = ["shard-3", "shard-2"]', '"K/mix-1" = ["shard-3", "E-rr/shard-2"]',
     "lie in one layout"),
    ('"K/mix-1" = ["shard-3", "shard-2"]', '"K/mix-1" = ["shard-3", "F/shard-2"]',
     "names no layout"),
    ('"K/mix-1" = ["shard-3", "shard-2"]', '"K/mix-1" = { layout = "partitioned" }',
     "must be a list of strings"),
    # A segment and its mixture must list the same shards in the same order.
    ('"K/mix-1" = ["shard-3", "shard-2"]', '"K/mix-1" = ["shard-2", "shard-3"]',
     "the segment lists the shards"),
    ('mixture = "Y/mix-1"', 'mixture = "Y/mix-7"', "is not in \\[mixtures\\]"),
    # The skipped line must be one the sanitizer skips as unparseable.
    ("text = '{\"id\": \"skip-1\", \"text\": \"this line is cut off'",
     "text = '{\"id\": \"skip-1\", \"text\": \"this line is whole\"}'", "does not parse"),
    ('shard = "shard-2"\nline = 6', 'shard = "shard-5"\nline = 6', "has no shard"),
    ('revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"\n', "", "revision is missing"),
    ('vocab_file = "hf_home/', 'vocab_file = "/hf_home/', "relative to the Megatron environment"),
])
def test_a_wrong_spec_is_refused(spec_path, tmp_path, old, new, message):
    path, base = changed_spec(spec_path, tmp_path, old, new)
    with pytest.raises(SpecError, match=message):
        load_fixture_spec(path, base)


def record_files(out):
    """Every record file of a build: a JSONL that is neither a manifest nor a log."""
    return sorted(
        p for p in out.rglob("*.jsonl")
        if not p.name.endswith((".manifest.jsonl", ".dedup.jsonl", "sanitizer.log.jsonl"))
    )


# --- Files and manifests ---------------------------------------------------------------


def test_the_build_writes_the_fixed_layout(fixture_build):
    out, _ = fixture_build
    expected = {f"sources/{k}.jsonl" for k in "ABCZ"}
    expected |= {f"gen/{k}.jsonl" for k in "PTS"}
    expected |= {f"releases/{k}.jsonl" for k in ("R1", "R1-swap", "R2")}
    expected |= {"imports/T.jsonl"}
    expected |= {f"pipeline/{layout}/{step}/{s}.jsonl" for layout in LAYOUTS for s in SHARDS
                 for step in ("sanitizer-input", "tokenizer-input")}
    assert {p.relative_to(out).as_posix() for p in record_files(out)} == expected
    for layout in LAYOUTS:
        assert (out / "pipeline" / layout / "sanitizer.log.jsonl").is_file()
    assert not (out / "mixtures").exists()  # no shards without tokenization


def test_every_record_file_has_a_manifest_with_one_row_per_line(fixture_build, read_rows):
    out, _ = fixture_build
    without_origin = []
    for path in record_files(out):
        lines = path.read_text(encoding="utf-8").split("\n")[:-1]
        manifest = read_rows(path.parent / f"{path.name}.manifest.jsonl")
        assert [m["line"] for m in manifest] == list(range(1, len(lines) + 1)), path
        assert all(set(m) == {"line", "handle", "origin"} for m in manifest), path
        assert all(isinstance(m["origin"], list) for m in manifest), path
        without_origin += [(path.relative_to(out).as_posix(), m["line"], m["handle"])
                           for m in manifest if not m["origin"]]
    # Every line is a copy of a record, except the line the spec writes into chunk 2.
    assert without_origin == [
        (f"pipeline/{layout}/sanitizer-input/shard-2.jsonl", 6, "sanitizer_skip")
        for layout in LAYOUTS]


def test_manifest_handles_and_origins(fixture_build, read_rows):
    out, _ = fixture_build
    source_a = read_rows(out / "sources" / "A.jsonl.manifest.jsonl")
    assert source_a == [{"line": 1, "handle": "X@A", "origin": ["A-001"]},
                        {"line": 2, "handle": "SA", "origin": ["A-002"]}]
    assert read_rows(out / "sources" / "B.jsonl.manifest.jsonl")[0]["handle"] == "X@B"
    assert read_rows(out / "sources" / "C.jsonl.manifest.jsonl")[0]["handle"] == "SC"
    # A record with no handle is named by its id.
    assert read_rows(out / "sources" / "Z.jsonl.manifest.jsonl")[40] == {
        "line": 41, "handle": "Z-041", "origin": ["Z-041"]}
    # A generation output's origin is what the output states it used.
    assert read_rows(out / "gen" / "P.jsonl.manifest.jsonl") == [
        {"line": 1, "handle": "P", "origin": ["A-001"]}]
    assert read_rows(out / "gen" / "T.jsonl.manifest.jsonl") == [
        {"line": 1, "handle": "T", "origin": ["P"]}]
    assert read_rows(out / "gen" / "S.jsonl.manifest.jsonl") == [
        {"line": 1, "handle": "S", "origin": ["A-002", "C-001"]}]
    assert read_rows(out / "imports" / "T.jsonl.manifest.jsonl") == [
        {"line": 1, "handle": "T", "origin": ["T"]}]


def test_the_manifest_of_a_tokenizer_input_names_the_record_on_each_line(fixture_build, read_rows):
    out, _ = fixture_build
    for layout in LAYOUTS:
        for shard in SHARDS:
            path = out / "pipeline" / layout / "tokenizer-input" / f"{shard}.jsonl"
            rows = read_rows(path)
            manifest = read_rows(path.parent / f"{path.name}.manifest.jsonl")
            assert all(set(r) == {"id", "text"} for r in rows)
            assert [m["origin"] for m in manifest] == [[r["id"]] for r in rows]
    partitioned_3 = read_rows(out / "pipeline" / "partitioned" / "tokenizer-input"
                              / "shard-3.jsonl.manifest.jsonl")
    assert [m["handle"] for m in partitioned_3].count("T") == 1


def test_sources_are_copied_byte_for_byte(fixture_build, repo):
    out, _ = fixture_build
    for key in "ABCZ":
        assert (out / "sources" / f"{key}.jsonl").read_bytes() == (
            repo / "fixture" / "sources" / f"{key}.jsonl").read_bytes()


# --- Releases and the dedup winner ---------------------------------------------------


def test_release_rows_hold_the_text_and_nothing_else(fixture_build, read_rows):
    out, _ = fixture_build
    for name in ("R1", "R1-swap", "R2"):
        assert all(set(r) == {"text"} for r in read_rows(out / "releases" / f"{name}.jsonl"))
    assert set(read_rows(out / "imports" / "T.jsonl")[0]) == {"text"}


def test_r1_keeps_the_copy_of_a_and_the_swap_keeps_the_copy_of_b_in_the_same_bytes(
        fixture_build, read_rows):
    out, summary = fixture_build
    r1 = read_rows(out / "releases" / "R1.jsonl.manifest.jsonl")
    swap = read_rows(out / "releases" / "R1-swap.jsonl.manifest.jsonl")
    expected_rest = ["P", "T", "S"] + expand_ids(["Z-001..Z-020"])
    assert [m["handle"] for m in r1] == ["X@A"] + expected_rest
    assert [m["handle"] for m in swap] == ["X@B"] + expected_rest
    assert r1[0]["origin"] == ["A-001"] and swap[0]["origin"] == ["B-001"]
    assert (out / "releases" / "R1.jsonl").read_bytes() == (
        out / "releases" / "R1-swap.jsonl").read_bytes()
    assert read_rows(out / "releases" / "R1.dedup.jsonl") == [
        {"line": 1, "kept": "A-001", "dropped": ["B-001"], "rule": "winner source"}]
    assert read_rows(out / "releases" / "R1-swap.dedup.jsonl") == [
        {"line": 1, "kept": "B-001", "dropped": ["A-001"], "rule": "winner source"}]
    assert summary["releases"]["R1"] == {"lines": 24, "dedup": True, "dropped": 1}


def test_r2_holds_b_and_unrelated_records_only(fixture_build, read_rows):
    out, _ = fixture_build
    manifest = read_rows(out / "releases" / "R2.jsonl.manifest.jsonl")
    assert [m["origin"][0] for m in manifest] == expand_ids(["B-001..B-013", "Z-021..Z-040"])
    assert manifest[0]["handle"] == "X@B"
    assert read_rows(out / "releases" / "R2.dedup.jsonl") == []


def test_generation_reads_the_source_revision(fixture_build, read_rows, repo):
    out, _ = fixture_build
    fixed = {r["id"]: r["text"] for r in read_rows(repo / "fixture" / "sources" / "generation.jsonl")}
    for name in "PTS":
        assert read_rows(out / "gen" / f"{name}.jsonl") == [{"id": name, "text": fixed[name]}]


# --- The pipeline: shards, the sanitizer's log ---------------------------------------------


def test_partitioned_shards_hold_the_records_the_spec_lists(fixture_build, fixture_plan, read_rows):
    out, summary = fixture_build
    groups = dict(fixture_plan.layouts["partitioned"].groups)
    for shard in SHARDS:
        rows = read_rows(out / "pipeline" / "partitioned" / "tokenizer-input" / f"{shard}.jsonl")
        assert sorted(ids_of(rows)) == sorted(groups[shard])
        assert ids_of(rows) != list(groups[shard])  # shuffled
    lines = {s["shard"]: (s["chunk_lines"], s["input_lines"], s["skipped"])
             for s in summary["layouts"]["partitioned"]["shards"]}
    # Shard-1 holds all 18 of its records now that Z-046 is an ordinary passage;
    # shard-2 reads 20 lines, skips one and still holds 19 documents.
    assert lines == {"shard-1": (18, 18, 0), "shard-2": (20, 19, 1), "shard-3": (27, 27, 0)}


def test_round_robin_holds_the_same_records_dealt_over_three_chunks(fixture_build, fixture_plan,
                                                                   read_rows):
    out, summary = fixture_build
    everything = [i for _, items in fixture_plan.layouts["partitioned"].groups for i in items]
    dealt = []
    for shard in SHARDS:
        dealt += ids_of(read_rows(out / "pipeline" / "round-robin" / "tokenizer-input"
                                  / f"{shard}.jsonl"))
    assert sorted(dealt) == sorted(everything)
    shards = summary["layouts"]["round-robin"]["shards"]
    assert sorted(s["input_lines"] for s in shards) == [21, 21, 22]  # 64 records, three chunks
    assert [s["skipped"] for s in shards] == [0, 1, 0]  # the spec's line, in chunk 2


def test_the_specs_skipped_line_is_line_6_of_chunk_2_and_shifts_no_ordinal(
        fixture_build, read_rows):
    out, _ = fixture_build
    for layout in LAYOUTS:
        base = out / "pipeline" / layout
        chunk = (base / "sanitizer-input" / "shard-2.jsonl").read_text(encoding="utf-8")
        lines = chunk.split("\n")[:-1]
        # The sanitizer reads the line as the spec wrote it, at line 6.
        assert lines[5] == SKIPPED_LINE and lines.count(SKIPPED_LINE) == 1
        # Exactly one skipped line in the whole layout, logged with the reason the
        # tool spells "sanitizer skip". It came from no file and has no id.
        assert read_rows(base / "sanitizer.log.jsonl") == [{
            "shard": "shard-2", "input": f"pipeline/{layout}/sanitizer-input/shard-2.jsonl",
            "line": 6, "id": None, "origin_file": None, "origin_line": None,
            "reason": "sanitizer skip", "detail": "not JSON"}]
        # A drop, not a shift: the surviving lines are the other lines, in order,
        # so the document after the skipped line has the ordinal the line had.
        survivors = read_rows(base / "tokenizer-input" / "shard-2.jsonl")
        chunk_ids = [json.loads(line)["id"] for line in lines if line != SKIPPED_LINE]
        assert ids_of(survivors) == chunk_ids and len(survivors) == len(lines) - 1
        assert "skip-1" not in ids_of(survivors)
        for shard in ("shard-1", "shard-3"):
            assert SKIPPED_LINE not in (base / "sanitizer-input" / f"{shard}.jsonl").read_text(
                encoding="utf-8")
    partitioned = read_rows(out / "pipeline" / "partitioned" / "tokenizer-input" / "shard-2.jsonl")
    assert len(partitioned) == 19  # the spec's search condition 2


def test_the_sanitizer_input_holds_the_source_lines_byte_for_byte(fixture_build, repo):
    out, _ = fixture_build
    raw = {}
    for key in "BZ":
        for line in (repo / "fixture" / "sources" / f"{key}.jsonl").read_text(
                encoding="utf-8").split("\n")[:-1]:
            raw[line] = key
    chunk = (out / "pipeline" / "partitioned" / "sanitizer-input" / "shard-1.jsonl").read_text(
        encoding="utf-8").split("\n")[:-1]
    assert len(chunk) == 18 and all(line in raw for line in chunk)


# --- Seeded and unseeded -------------------------------------------------------------


def test_a_seeded_build_is_the_same_bytes_in_a_second_folder(fixture_build, fixture_plan, tmp_path):
    out, _ = fixture_build
    build(fixture_plan, tmp_path / "again", BuildOptions(tokenize=False))
    assert tree_digests(tmp_path / "again") == tree_digests(out)
    # 23 record files, 23 manifests, 3 dedup logs, 2 sanitizer logs.
    assert len(tree_digests(out)) == 51


def test_another_seed_moves_the_lines_and_keeps_the_records(fixture_build, fixture_plan, tmp_path,
                                                          read_rows):
    out, _ = fixture_build
    summary = build(fixture_plan, tmp_path / "seed5",
                    BuildOptions(tokenize=False, layouts=("partitioned",),
                                 shuffle_seeds={"partitioned": 5}))
    assert summary["layouts"]["partitioned"]["shuffle_seed"] == 5
    for shard in SHARDS:
        relative = f"pipeline/partitioned/tokenizer-input/{shard}.jsonl"
        first, second = ids_of(read_rows(out / relative)), ids_of(read_rows(tmp_path / "seed5" / relative))
        assert sorted(first) == sorted(second) and first != second


def test_unseeded_builds_differ_and_hold_the_same_records(fixture_plan, tmp_path, read_rows):
    options = BuildOptions(tokenize=False, seeded=False)
    one = build(fixture_plan, tmp_path / "one", options)
    build(fixture_plan, tmp_path / "two", options)
    assert one["seeded"] is False and one["layouts"]["partitioned"]["shuffle_seed"] is None
    orders = []
    for folder in ("one", "two"):
        order = []
        for layout in LAYOUTS:
            for shard in SHARDS:
                order.append(ids_of(read_rows(
                    tmp_path / folder / "pipeline" / layout / "tokenizer-input" / f"{shard}.jsonl")))
        orders.append(order)
    assert orders[0] != orders[1]
    for a, b in zip(orders[0][:3], orders[1][:3]):  # the partitioned shards keep their records
        assert sorted(a) == sorted(b)
    # The releases do not pass a shuffle, so they stay the same bytes.
    for name in ("R1", "R2"):
        assert (tmp_path / "one" / "releases" / f"{name}.jsonl").read_bytes() == (
            tmp_path / "two" / "releases" / f"{name}.jsonl").read_bytes()


# --- Leaving out and swapping (plan 4.4) -------------------------------------------------


def test_leaving_out_the_affected_records_takes_x_from_b(fixture_plan, tmp_path, read_rows):
    changes = RecordChanges(leave_out=frozenset({"A-001", "P", "T", "S"}))
    out = tmp_path / "repaired"
    summary = build(fixture_plan, out, BuildOptions(tokenize=False, changes=changes))
    manifest = read_rows(out / "releases" / "R1.jsonl.manifest.jsonl")
    assert [m["handle"] for m in manifest] == ["X@B"] + expand_ids(["Z-001..Z-020"])
    assert read_rows(out / "releases" / "R1.dedup.jsonl") == []
    shard_3 = read_rows(out / "pipeline" / "partitioned" / "tokenizer-input" / "shard-3.jsonl")
    assert "T" not in ids_of(shard_3) and len(shard_3) == 26
    assert not (out / "imports" / "T.jsonl").exists()
    assert summary["left_out"] == ["A-001", "P", "S", "T"]
    # Sources and generation outputs are built as their inputs say.
    assert (out / "gen" / "T.jsonl").is_file()


def test_an_untouched_shard_is_the_same_bytes_after_a_record_elsewhere_is_left_out(
        fixture_build, fixture_plan, tmp_path):
    out, _ = fixture_build
    build(fixture_plan, tmp_path / "repaired",
          BuildOptions(tokenize=False, layouts=("partitioned",),
                       changes=RecordChanges(leave_out=frozenset({"T"}))))
    for shard, same in (("shard-1", True), ("shard-2", True), ("shard-3", False)):
        for step in ("sanitizer-input", "tokenizer-input"):
            relative = f"pipeline/partitioned/{step}/{shard}.jsonl"
            assert ((out / relative).read_bytes()
                    == (tmp_path / "repaired" / relative).read_bytes()) is same


def test_a_build_can_be_told_which_releases_and_imports_to_make(fixture_plan, tmp_path):
    out = tmp_path / "main"
    summary = build(fixture_plan, out, BuildOptions(tokenize=False, layouts=("partitioned",),
                                                    releases=("R1", "R2"), imports=()))
    assert sorted(summary["releases"]) == ["R1", "R2"]
    assert not (out / "releases" / "R1-swap.jsonl").exists() and not (out / "imports").exists()
    with pytest.raises(SpecError, match="no release named"):
        build(fixture_plan, tmp_path / "bad", BuildOptions(tokenize=False, releases=("R9",)))


def test_a_swap_puts_the_new_record_where_the_old_one_was(fixture_plan, tmp_path, read_rows):
    out = tmp_path / "swapped"
    build(fixture_plan, out, BuildOptions(tokenize=False, layouts=("partitioned",),
                                          changes=RecordChanges(swap={"Z-003": "Z-039"})))
    r1 = [m["origin"][0] for m in read_rows(out / "releases" / "R1.jsonl.manifest.jsonl")]
    assert r1[6] == "Z-039" and "Z-003" not in r1


def test_a_swap_onto_a_record_that_is_already_selected_is_refused(fixture_plan, tmp_path):
    with pytest.raises(ReplicaError, match="selected twice"):
        build(fixture_plan, tmp_path / "bad",
              BuildOptions(tokenize=False, changes=RecordChanges(swap={"Z-003": "Z-004"})))
    with pytest.raises(InputError, match="no record with id"):
        build(fixture_plan, tmp_path / "bad2",
              BuildOptions(tokenize=False, changes=RecordChanges(swap={"Z-003": "Z-999"})))
    with pytest.raises(SpecError):
        RecordChanges(leave_out=frozenset({"Z-003"}), swap={"Z-003": "Z-039"})


# --- Refusals ----------------------------------------------------------------------


def test_a_folder_that_is_not_empty_is_refused(fixture_plan, tmp_path):
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "note.txt").write_text("x", encoding="utf-8")
    with pytest.raises(OutputError):
        build(fixture_plan, tmp_path / "used", BuildOptions(tokenize=False))


def test_tokenization_without_the_environment_variable_is_refused(fixture_plan, tmp_path,
                                                                 monkeypatch):
    from replica.errors import TokenizerError

    monkeypatch.delenv("WATERWHITE_MEGATRON_ENV", raising=False)
    with pytest.raises(TokenizerError, match="WATERWHITE_MEGATRON_ENV"):
        build(fixture_plan, tmp_path / "out", BuildOptions(tokenize=True))
    assert not (tmp_path / "out").exists()


def test_an_unknown_layout_is_refused(fixture_plan, tmp_path):
    with pytest.raises(SpecError):
        build(fixture_plan, tmp_path / "out", BuildOptions(tokenize=False, layouts=("striped",)))


# --- The command line, and the wall with capture off -----------------------------------------


def test_with_capture_off_the_command_line_loads_the_tools_normaliser_and_nothing_else(
        repo, spec_path, tmp_path):
    # With capture off the replica calls one thing of the tool, the normaliser
    # (DECISIONS.md D-17). The store, the capture wrappers and the harness are not
    # even loaded.
    code = (
        "import sys, runpy\n"
        f"sys.argv = ['replica', 'build-fixture', '--spec', {str(spec_path)!r}, "
        f"'--out', {str(tmp_path / 'cli')!r}, '--no-tokenize', '--layout', 'partitioned']\n"
        "try:\n"
        "    runpy.run_module('replica', run_name='__main__')\n"
        "except SystemExit as stop:\n"
        "    assert stop.code == 0, stop.code\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'waterwhite')\n"
        "allowed = ['waterwhite', 'waterwhite.errors', 'waterwhite.ids', 'waterwhite.normalise']\n"
        "assert 'waterwhite.normalise' in loaded, loaded\n"
        "assert not set(loaded) - set(allowed), loaded\n"
        "assert 'replica.capture_hooks' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True,
                          env=command_environment(repo))
    assert done.returncode == 0, done.stderr
    assert '"capture": false' in done.stdout
    assert (tmp_path / "cli" / "releases" / "R1.jsonl").is_file()
    assert not (tmp_path / "cli" / "pipeline" / "round-robin").exists()


def test_without_the_tools_package_on_the_path_the_replica_says_what_is_missing(repo, tmp_path):
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-S", "-m", "replica", "digests", "--out", str(tmp_path)],
                          cwd=repo, capture_output=True, text=True, env=environment)
    assert done.returncode != 0
    assert "waterwhite.normalise" in done.stderr and "PYTHONPATH=src" in done.stderr


def test_the_command_line_reports_an_error_and_exits_with_1(repo, spec_path, tmp_path):
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "replica", "build-fixture", "--spec", str(spec_path),
         "--out", str(tmp_path / "used"), "--no-tokenize"],
        cwd=repo, capture_output=True, text=True, env=command_environment(repo))
    assert done.returncode == 1
    assert done.stdout == "" and "OutputError" in done.stderr


def test_the_digests_command_lists_every_file(fixture_build, repo):
    out, _ = fixture_build
    done = subprocess.run([sys.executable, "-m", "replica", "digests", "--out", str(out)],
                          cwd=repo, capture_output=True, text=True,
                          env=command_environment(repo))
    assert done.returncode == 0
    listed = dict(reversed(line.split("  ", 1)) for line in done.stdout.splitlines())
    assert listed == tree_digests(out)
