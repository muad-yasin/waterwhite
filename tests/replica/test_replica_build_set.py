"""A generated set, as ``fixture/plant/generate.py`` writes it (docs/plant.md): ids
that name lines, lines that do not parse carried up to the sanitizer, the dedup
priority, the two layouts, and the two sides of a repair.

The expected values come from the plant list and the counting script
(``fixture/plant/count.py``), which share no code with the replica.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from replica.build import BuildOptions, RecordChanges, build
from replica.errors import InputError, ReplicaError, SpecError
from replica.plan import load_set, load_tokenizer_settings
from replica.steps import expand_ids


NO_TOKENS = BuildOptions(tokenize=False)
LAYOUTS = ("partitioned", "round-robin")


def tokenizer_input(out, layout):
    return out / "pipeline" / layout / "tokenizer-input"


def plant_rows(set_dir, read_rows):
    return {row["id"]: row for row in read_rows(set_dir / "plant_list.jsonl")}


def copy_set(small_set, tmp_path, name="set"):
    """A copy of the small set that a test may damage."""
    target = tmp_path / name
    (target / "sources").mkdir(parents=True)
    for path in small_set.rglob("*"):
        if path.is_file():
            (target / path.relative_to(small_set)).write_bytes(path.read_bytes())
    return target


# --- Reading set.toml ------------------------------------------------------------------


def test_the_set_file_is_read_into_a_plan(small_set):
    plan = load_set(small_set)
    assert plan.mode == "set" and plan.normaliser == "control-chars/1"
    assert plan.tokenizer is None and plan.mixtures == () and plan.pipeline_release == "R1"
    assert [(s.key, s.source, s.member, s.revision, len(s.records), s.by_line)
            for s in plan.sources] == [
        ("A", "A", "A/r1", "r1", 48, True), ("B", "B", "B/r1", "r1", 80, True),
        ("C", "C", "C/r1", "r1", 40, True), ("Z", "Z", "Z/r1", "r1", 542, True)]
    # An id names a line, with six digits.
    assert plan.sources[3].records[0] == "Z-000001" and plan.sources[3].records[-1] == "Z-000542"
    assert [(g.name, len(g.rows)) for g in plan.generation] == [
        ("paraphrase", 42), ("translation", 40), ("summary", 8)]
    first = plan.generation[0].rows[0]
    # A job with many rows is one harness run per row, named by the row's number in
    # the job's output and never by its id (plan Check 3: the ids are not the
    # tool's).
    assert (first.id, first.job_name, first.handed) == (
        "P-000001", "paraphrase/1", ("A-000001",))
    assert [r.job_name for r in plan.generation[2].rows] == [f"summary/{n}" for n in range(1, 9)]
    r1, r2 = plan.releases
    assert (r1.name, r1.dedup, r1.priority) == ("R1", True, ("A", "B", "C", "Z"))
    assert len(r1.inputs) == 40 + 40 + 90 + 24 + 271 and len(r2.inputs) == 80 + 271


def test_the_set_gets_its_seed_and_its_two_layouts(small_set, set_parameters):
    plan = load_set(small_set)
    dealt = plan.layouts["round-robin"]
    # The listed files: the files R1's inputs come from, in the order it names them.
    assert [name for name, _ in dealt.groups] == [
        "A", "B", "gen/paraphrase", "gen/translation", "gen/summary", "C", "Z"]
    assert (dealt.chunks, dealt.shuffle_seed, dealt.store_prefix) == (
        20, set_parameters["seed"], "set-rr")
    by_source = plan.layouts["partitioned"]
    assert by_source.groups == (
        ("shard-A", ("A",)), ("shard-B", ("B",)), ("shard-C", ("C",)), ("shard-Z", ("Z",)),
        ("shard-gen", ("gen/paraphrase", "gen/translation", "gen/summary")))
    assert (by_source.chunks, by_source.shuffle_seed, by_source.store_prefix) == (
        None, set_parameters["seed"], "set")


def test_a_set_with_ten_sources_has_a_shard_for_each(plant_generate, plant_count, set_parameters,
                                                    tmp_path):
    folder = tmp_path / "ten"
    plant_generate.generate(folder, **{**set_parameters, "records": 1200, "sources": 10})
    plan = load_set(folder)
    assert [s.key for s in plan.sources] == ["A", "B", "C", "Z"] + [f"Z{n}" for n in range(2, 8)]
    summary = build(plan, tmp_path / "out", BuildOptions(tokenize=False, layouts=("partitioned",)))
    shards = {s["shard"]: s for s in summary["layouts"]["partitioned"]["shards"]}
    assert list(shards) == ["shard-A", "shard-B", "shard-C", "shard-Z"] + [
        f"shard-Z{n}" for n in range(2, 8)] + ["shard-gen"]
    counted = plant_count.count(folder, chunks=tokenizer_input(tmp_path / "out", "partitioned"),
                                layout="partitioned")
    assert sorted(counted["shards"]["keep"]) == ["shard-C", "shard-Z"] + [
        f"shard-Z{n}" for n in range(2, 8)]
    assert counted["shards"]["rewrite"] == ["shard-A", "shard-gen"]


# --- The releases ---------------------------------------------------------------------


def test_r1_is_built_from_the_listed_ids_and_matches_the_plant_list(small_set, small_build,
                                                                   read_rows):
    out, summary = small_build
    plant = plant_rows(small_set, read_rows)
    manifest = read_rows(out / "releases" / "R1.jsonl.manifest.jsonl")
    present = [i for i, row in plant.items()
               if row["releases"].get("R1", {}).get("row") == "present"]
    losers = [i for i, row in plant.items()
              if row["releases"].get("R1", {}).get("row") == "dedup-loser"]
    assert sorted(m["origin"][0] for m in manifest) == sorted(present)
    assert summary["releases"]["R1"] == {"lines": len(present), "dedup": True,
                                         "dropped": len(losers)}
    assert (len(present), len(losers)) == (401, 64)
    dropped = [i for row in read_rows(out / "releases" / "R1.dedup.jsonl") for i in row["dropped"]]
    assert sorted(dropped) == sorted(losers)
    # A's copy of X is kept; of a planted pair the copy of C is kept, since C comes
    # before Z in the dedup priority.
    kept = {row["kept"] for row in read_rows(out / "releases" / "R1.dedup.jsonl")}
    assert {k.split("-")[0] for k in kept} == {"A", "C"}


def test_a_line_with_no_text_is_in_no_dedup_group_and_stays_a_row_until_the_sanitizer(
        small_set, small_build, read_rows):
    # docs/plant.md 5.1 point 3, DECISIONS.md D-17: the plant list counts on it.
    out, _ = small_build
    plant = plant_rows(small_set, read_rows)
    skipped = sorted(i for i, row in plant.items() if row["sanitizer"] and row["sanitizer"]["skip"])
    assert len(skipped) == 6
    assert {plant[i]["sanitizer"]["case"] for i in skipped} == {
        "unparseable", "empty", "control-only", "control-and-space"}
    manifest = read_rows(out / "releases" / "R1.jsonl.manifest.jsonl")
    line_of = {m["origin"][0]: m["line"] for m in manifest}
    assert all(i in line_of for i in skipped)           # each is a row of the release
    grouped = {i for row in read_rows(out / "releases" / "R1.dedup.jsonl")
               for i in [row["kept"], *row["dropped"]]}
    assert not grouped & set(skipped)                   # and none is in a dedup group
    release = (out / "releases" / "R1.jsonl").read_text(encoding="utf-8").split("\n")
    sources = (small_set / "sources" / "Z.jsonl").read_text(encoding="utf-8").split("\n")
    for record in skipped:
        row = release[line_of[record] - 1]
        if plant[record]["sanitizer"]["case"] == "unparseable":
            # A line that does not parse is carried as it is.
            assert row == sources[plant[record]["line"] - 1]
        else:
            assert set(json.loads(row)) == {"text"}
    # None of them reaches a tokenizer input, in either layout.
    for layout in LAYOUTS:
        for path in tokenizer_input(out, layout).glob("shard-*.jsonl"):
            if not path.name.endswith(".manifest.jsonl"):
                assert not {r["id"] for r in read_rows(path)} & set(skipped)


# --- The sanitizer and the tokenizer inputs --------------------------------------------------


def test_the_sanitizer_logs_exactly_the_lines_the_plant_list_says_it_must(
        small_set, small_build, plant_count, read_rows):
    out, _ = small_build
    expected = plant_count.count(small_set)["check3"]["sanitizer_drops"]["R1"]
    assert len(expected) == 6
    for layout in LAYOUTS:
        log = read_rows(out / "pipeline" / layout / "sanitizer.log.jsonl")
        assert sorted((row["id"], row["reason"]) for row in log) == sorted(
            (row["id"], row["reason"]) for row in expected)
        raw = {row["id"]: row["raw_sha256"] for row in expected}
        for row in log:
            # The log names the line of the file the sanitizer read, and that line
            # is the source's line, byte for byte.
            lines = (out / row["input"]).read_text(encoding="utf-8").split("\n")
            assert hashlib.sha256(lines[row["line"] - 1].encode("utf-8")).hexdigest() == raw[row["id"]]
            assert row["origin_file"] == "releases/R1.jsonl"


def test_the_counting_script_accepts_the_tokenizer_inputs_of_both_layouts(
        small_set, small_build, plant_count):
    out, summary = small_build
    by_source = plant_count.count(small_set, chunks=tokenizer_input(out, "partitioned"),
                                  layout="partitioned")
    assert by_source["shards"]["keep"] == ["shard-C", "shard-Z"]
    assert by_source["shards"]["rewrite"] == ["shard-A", "shard-gen"]
    assert by_source["shards"]["all_rewrite"] is False and by_source["shards"]["lines"] == 395
    dealt = plant_count.count(small_set, chunks=tokenizer_input(out, "round-robin"),
                              layout="round-robin")
    # A's records reach every one of the 20 chunks (plan 4.5, 5.2).
    assert dealt["shards"]["all_rewrite"] is True and dealt["shards"]["keep"] == []
    assert len(dealt["shards"]["rewrite"]) == 20 and dealt["shards"]["lines"] == 395
    # With dedup on one member of every pair was dropped before chunking.
    assert dealt["check3"]["pairs"]["counts"]["pair-member"]["single"] == 24
    # B's rows of R1 all lost the dedup: its shard holds no line and is not written.
    shards = {s["shard"]: s for s in summary["layouts"]["partitioned"]["shards"]}
    assert shards["shard-B"] == {"shard": "shard-B", "store_name": "set/shard-B",
                                 "chunk_lines": 0, "input_lines": 0, "skipped": 0, "empty": True}
    assert not (tokenizer_input(out, "partitioned") / "shard-B.jsonl").exists()
    assert shards["shard-Z"]["skipped"] == 6 and shards["shard-Z"]["input_lines"] == 241


def test_only_the_rows_of_r1_are_chunked(small_set, small_build, read_rows):
    out, _ = small_build
    plant = plant_rows(small_set, read_rows)
    for layout in LAYOUTS:
        ids = [r["id"] for path in sorted(tokenizer_input(out, layout).glob("shard-*.jsonl"))
               if not path.name.endswith(".manifest.jsonl") for r in read_rows(path)]
        assert len(ids) == len(set(ids)) == 395
        assert all(plant[i]["releases"].get("R1", {}).get("row") == "present" for i in ids)
        assert all(len(i.rsplit("-", 1)[1]) == 6 for i in ids)   # six-digit ids, kept


def test_with_dedup_switched_off_both_twins_reach_a_shard(plant_generate, plant_count,
                                                          set_parameters, tmp_path):
    folder = tmp_path / "set"
    plant_generate.generate(folder, dedup=False, **set_parameters)
    summary = build(load_set(folder), tmp_path / "out",
                    BuildOptions(tokenize=False, layouts=("partitioned",)))
    assert summary["releases"]["R1"] == {"lines": 465, "dedup": False, "dropped": 0}
    counted = plant_count.count(folder, chunks=tokenizer_input(tmp_path / "out", "partitioned"),
                                layout="partitioned", require_pair_kinds="no")
    # In a layout partitioned by source the two members of a pair lie in two shards.
    assert counted["check3"]["pairs"]["counts"]["pair-member"]["split"] == 24
    assert counted["check3"]["pairs"]["counts"]["x-copy"]["split"] == 40
    assert "shard-B" in counted["shards"]["keep"]


# --- The two sides of a repair (plan 4.4) -------------------------------------------------


def test_a_repaired_build_equals_a_build_from_the_clean_set(small_set, small_build,
                                                          plant_generate, plant_count,
                                                          set_parameters, tmp_path):
    full_out, _ = small_build
    omitted = expand_ids(plant_count.count(small_set)["clean_omits"]["ids"])
    assert len(omitted) == 48 + 88   # A's lines and everything derived from them
    repaired = tmp_path / "repaired"
    build(load_set(small_set), repaired,
          BuildOptions(tokenize=False, changes=RecordChanges(leave_out=frozenset(omitted))))
    clean = tmp_path / "clean"
    plant_generate.generate(clean, clean=True, **set_parameters)
    reference = tmp_path / "reference"
    build(load_set(clean), reference, NO_TOKENS)

    def files(out):
        found = {}
        for folder in ("releases", "pipeline"):
            for path in sorted((out / folder).rglob("*.jsonl")):
                found[path.relative_to(out).as_posix()] = path.read_bytes()
        return found

    assert files(repaired) == files(reference)
    # The successor takes X from B, and a shard none of whose records left is the
    # same bytes as before the repair.
    for shard in ("shard-C", "shard-Z"):
        relative = f"pipeline/partitioned/tokenizer-input/{shard}.jsonl"
        assert (repaired / relative).read_bytes() == (full_out / relative).read_bytes()
    assert (tokenizer_input(repaired, "partitioned") / "shard-B.jsonl").is_file()
    assert not (tokenizer_input(repaired, "partitioned") / "shard-A.jsonl").exists()


# --- Refusals -------------------------------------------------------------------------


@pytest.mark.parametrize("old, new, error, message", [
    ('format = "waterwhite.plant-set/1"', 'format = "waterwhite.replica-set/1"', SpecError,
     "format must be"),
    ("lines = 48", "lines = 47", SpecError, "an id names a line"),
    ('paraphrase = ["P-000001..P-000040", ', 'paraphrase = ["P-000001..P-000039", ', SpecError,
     "jobs.paraphrase"),
    ("rows = 90", "rows = 89", SpecError, "rows is 89"),
    ('"B-000001..B-000040",', '"B-000001..B-000040",\n    "Q-000001",', SpecError,
     "name no line"),
    ('dedup_winner = "A"\ndedup_priority = [', 'dedup_winner = "B"\ndedup_priority = [',
     SpecError, "dedup_winner is 'B'"),
    ('name = "control-chars/1"', 'name = "lowercase/1"',
     SpecError, "unknown normaliser"),
])
def test_a_wrong_set_file_is_refused(small_set, tmp_path, old, new, error, message):
    folder = copy_set(small_set, tmp_path)
    text = (folder / "set.toml").read_text(encoding="utf-8")
    assert old in text, old
    (folder / "set.toml").write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(error, match=message):
        load_set(folder)


def test_a_row_that_carries_another_id_than_its_lines_is_refused(small_set, tmp_path):
    folder = copy_set(small_set, tmp_path)
    path = folder / "sources" / "C.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"C-000002"', '"C-000003"', 1),
                    encoding="utf-8")
    with pytest.raises(InputError, match="the id of this line is 'C-000002'"):
        build(load_set(folder), tmp_path / "out", NO_TOKENS)


def test_a_source_file_with_a_line_too_few_is_refused(small_set, tmp_path):
    folder = copy_set(small_set, tmp_path)
    path = folder / "sources" / "C.jsonl"
    path.write_text("\n".join(path.read_text(encoding="utf-8").split("\n")[:-2]) + "\n",
                    encoding="utf-8")
    with pytest.raises(InputError, match="39 lines, and 40 ids are listed"):
        build(load_set(folder), tmp_path / "out", NO_TOKENS)


def test_a_set_names_no_tokenizer_so_the_build_must_be_given_one(small_set, spec_path, tmp_path):
    with pytest.raises(SpecError, match="names no tokenizer"):
        build(load_set(small_set), tmp_path / "out", BuildOptions(tokenize=True))
    assert not (tmp_path / "out").exists()
    settings = load_tokenizer_settings(spec_path)
    assert load_set(small_set, settings).tokenizer == settings


def test_the_command_line_builds_a_set_and_asks_for_a_tokenizer(small_set, spec_path, repo,
                                                              tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(repo / "src")}
    command = [sys.executable, "-m", "replica", "build-set", "--set", str(small_set),
               "--layout", "partitioned"]
    done = subprocess.run([*command, "--out", str(tmp_path / "out"), "--no-tokenize"],
                          cwd=repo, capture_output=True, text=True, env=environment)
    assert done.returncode == 0, done.stderr
    summary = json.loads(done.stdout)
    assert summary["mode"] == "set" and summary["layouts"]["partitioned"]["shuffle_seed"] == 7
    done = subprocess.run([*command, "--out", str(tmp_path / "second")],
                          cwd=repo, capture_output=True, text=True, env=environment)
    assert done.returncode == 1 and "--tokenizer-spec" in done.stderr
    assert not (tmp_path / "second").exists()


def test_replica_error_is_the_base_of_every_refusal():
    from replica import errors

    for name in ("SpecError", "InputError", "OutputError", "TokenizerError", "CaptureError"):
        assert issubclass(getattr(errors, name), ReplicaError)
