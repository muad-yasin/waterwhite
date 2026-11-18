"""The ``recover`` command, through ``cli.main`` (docs/cli.md; plan 3.3, 5.4).

A run made without capture, in miniature: two source files, a shuffled chunk
with one line that does not parse, the sanitizer's output and log, and a shard
written by the toy tokenizer of ``tm_world.py``. The sources are ingested from
Python (the command line has no ingest); everything after that goes through the
command line.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
import tm_world

from waterwhite import capture, cli, errors
from waterwhite import vocabulary as v
from waterwhite.store import Store

A_TEXTS = ["shared passage", "a two", "a  thr\u0007ee"]  # a control character the sanitizer removes
B_TEXTS = ["shared passage", "b two"]


def run(*argv: str) -> tuple[int, dict | None, dict | None]:
    """Run one command; return its exit code, its JSON output and its JSON error."""
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(item) for item in argv], stdout=out, stderr=err)
    printed, failed = out.getvalue(), err.getvalue()
    assert not (printed and failed), "an error prints nothing on standard output"
    return (code, json.loads(printed) if printed else None, json.loads(failed) if failed else None)


class Pipeline:
    """The files of the miniature run, and the store beside them."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.store = folder / "evidence.db"
        assert run("init", "--store", self.store, "--normaliser", tm_world.NORMALISER)[0] == 0
        sources = {}
        for key, texts in (("A", A_TEXTS), ("B", B_TEXTS)):
            sources[key] = tm_world.write_rows(folder / f"sources/{key}.jsonl", texts, id_prefix=key)
        with Store.open(self.store) as store:
            for key, path in sources.items():
                capture.ingest_jsonl(store, path, source=key, member=f"{key}/r1", revision="r1",
                                     name=f"src/{key}")
        # The chunk as the sanitizer read it: shuffled, ids gone, line 3 unparseable.
        self.chunk = tm_world.write_raw_lines(folder / "pipeline/chunk.jsonl", [
            json.dumps({"text": "b two"}), json.dumps({"text": "shared passage"}),
            "this line is not JSON", json.dumps({"text": "a  thr\u0007ee"}),
            json.dumps({"text": "a two"})])
        # What the sanitizer wrote, and its log.
        self.clean_texts = ["b two", "shared passage", "a  three", "a two"]
        self.clean = tm_world.write_rows(folder / "pipeline/clean.jsonl", self.clean_texts)
        self.log = folder / "pipeline/sanitizer.log.jsonl"
        self.log.write_text(json.dumps({
            "shard": "shard-1", "input": "pipeline/chunk.jsonl", "line": 3,
            "reason": "sanitizer skip", "detail": "not JSON"}) + "\n", encoding="utf-8")
        self.prefix = folder / "shards/shard-1"
        tm_world.toy_preprocess(self.clean, self.prefix)
        # The team's tokenizer as a command and as a module.
        self.command = folder / "team_tokenizer.py"
        self.command.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    text = json.loads(line)['text']\n"
            "    print(json.dumps({'tokens': list(text.encode('utf-8'))}), flush=True)\n",
            encoding="utf-8")

    def go(self, *argv: str) -> tuple[int, dict | None, dict | None]:
        return run(*argv, "--store", self.store)

    def register_files(self) -> None:
        assert self.go("recover", "file", self.chunk, "--name", "chunk")[0] == 0
        assert self.go("recover", "file", self.clean, "--name", "clean")[0] == 0

    def register_shard(self) -> None:
        assert self.go("recover", "shard", self.prefix, "--name", "shard-1", "--input", "clean",
                       "--tokenizer-version", "toy/1")[0] == 0

    def map_flags(self) -> list[str]:
        return ["--append-eod", "yes", "--eod-in-span", "yes", "--eod-token-id", str(tm_world.EOD),
                "--tokenizer-command", f"{sys.executable} {self.command}"]


@pytest.fixture
def pipeline(tmp_path) -> Pipeline:
    return Pipeline(tmp_path)


def table_counts(path: Path) -> dict[str, int]:
    with Store.open(path, readonly=True) as store:
        names = [r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


# --- recover file, recover shard ---------------------------------------------------------------


def test_recover_file_registers_a_found_file_and_a_dry_run_registers_nothing(pipeline):
    before = table_counts(pipeline.store)
    code, out, _ = pipeline.go("recover", "file", pipeline.chunk, "--name", "chunk", "--dry-run")
    assert code == 0
    assert out == {"artifact": "chunk", "role": "intermediate", "already_registered": False,
                   "instance": None, "records": 5,
                   "content_digest": capture.file_digest(pipeline.chunk), "dry_run": True}
    assert table_counts(pipeline.store) == before

    code, out, _ = pipeline.go("recover", "file", pipeline.chunk, "--name", "chunk")
    assert (code, out["records"], out["role"], out["already_registered"], out["dry_run"]) == (
        0, 5, "intermediate", False, False)
    code, shown, _ = pipeline.go("show", "chunk")
    assert (shown["instance"], shown["role"], shown["step"]) == (out["instance"], "intermediate", None)
    # The same file again is the same instance; a release keeps its role.
    assert pipeline.go("recover", "file", pipeline.chunk, "--name", "chunk")[1][
        "already_registered"] is True
    code, out, _ = pipeline.go("recover", "file", pipeline.clean, "--name", "R-found",
                               "--role", "release")
    assert (code, out["role"]) == (0, "release")
    code, listing, _ = pipeline.go("query", "withdraw", "A")
    assert listing["covers"]["releases"] == ["R-found"]


def test_recover_shard_registers_the_pair_under_the_tokenizer_step(pipeline):
    pipeline.register_files()
    code, out, _ = pipeline.go("recover", "shard", pipeline.prefix, "--name", "shard-1",
                               "--input", "clean", "--tokenizer-version", "toy/1",
                               "--tokenizer-params", '{"append_eod": true}', "--dry-run")
    assert (code, out["documents"], out["instance"], out["dry_run"]) == (0, 4, None, True)
    assert pipeline.go("show", "shard-1")[1]["registered"] is False
    code, out, _ = pipeline.go("recover", "shard", pipeline.prefix, "--name", "shard-1",
                               "--input", "clean", "--tokenizer-version", "toy/1",
                               "--tokenizer-params", '{"append_eod": true}')
    assert (code, out["documents"], out["inputs"], out["has_map"]) == (0, 4, ["clean"], False)
    shown = pipeline.go("show", "shard-1")[1]
    assert (shown["kind"], shown["step"]["name"], shown["own_record"]) == ("shard", "tokenize", False)
    # A shard with no map is "not covered" until its map is recovered.
    answer = pipeline.go("query", "withdraw", "A", "--artifact", "shard-1")[1]
    assert (answer["answer"], answer["reason"]) == ("not covered", "record missing")

    code, _, error = pipeline.go("recover", "shard", pipeline.prefix, "--name", "other",
                                 "--input", "nowhere", "--tokenizer-version", "toy/1")
    assert (code, error["error"]) == (errors.EXIT_REFUSED, "request refused")
    code, _, error = pipeline.go("recover", "shard", pipeline.prefix, "--name", "other",
                                 "--tokenizer-version", "toy/1", "--tokenizer-params", "{not json")
    assert (code, error["error"]) == (errors.EXIT_USAGE, "usage")


# --- recover copy-step, recover sanitizer -----------------------------------------------------


def test_recover_copy_step_and_sanitizer_link_the_files_record_by_record(pipeline):
    pipeline.register_files()
    arguments = ["recover", "copy-step", "chunk", "--input", "src/A", "--input", "src/B",
                 "--step-name", "shuffle-chunk-merge"]
    before = table_counts(pipeline.store)
    code, dry, _ = pipeline.go(*arguments, "--dry-run")
    assert (code, dry["dry_run"], dry["edges_written"]) == (0, True, 0)
    assert table_counts(pipeline.store) == before
    code, out, _ = pipeline.go(*arguments)
    # Four lines with a text: three with one parent, the shared passage with two
    # candidates (plan L3); the unparseable line has no text and no edge.
    assert code == 0
    assert (out["kind"], out["records"], out["recovered"], out["candidates"], out["no_edge"],
            out["no_text"], out["surplus"], out["edges_written"], out["dry_run"]) == (
        "copy step", 5, 3, 1, 1, 1, 0, 4, False)
    assert out["step"] == {"name": "shuffle-chunk-merge", "version": "not recorded",
                           "param_digest": capture.param_digest(None)}
    assert pipeline.go(*arguments)[1]["wrote_nothing"] is True
    record = pipeline.go("show", "chunk", "--record", "L2")[1]
    assert (record["edge"]["label"], record["edge"]["candidate_sources"]) == ("candidates", ["A", "B"])

    arguments = ["recover", "sanitizer", "clean", "--input", "chunk", "--log", pipeline.log,
                 "--log-input", "pipeline/chunk.jsonl", "--step-version", "control-chars/1"]
    code, out, _ = pipeline.go(*arguments)
    assert code == 0
    assert (out["kind"], out["outcome"], out["recovered"], out["drops"], out["drops_written"],
            out["input_lines_unaccounted"], out["step"]["name"]) == (
        "sanitizer", "record-level", 4, 1, 1, 0, v.STEP_SANITIZE)
    # The skipped line is a drop whose output is the file that lacks it.
    dropped = pipeline.go("show", "chunk", "--record", "L3")[1]
    assert dropped["drops"] == [{"reason": "sanitizer skip", "step": "sanitize", "output": "clean"}]
    # The sanitized file is now traced to its sources through the chunk.
    answer = pipeline.go("query", "withdraw", "A", "--artifact", "clean")[1]
    assert answer["answer"] == "contains"
    assert [hop["hop"] for hop in answer["chain"]] == ["clean:L3", "chunk:L4", "src/A:L3"]

    code, _, error = pipeline.go("recover", "copy-step", "chunk", "--step-name", "shuffle")
    assert (code, error["error"]) == (errors.EXIT_USAGE, "usage")
    code, _, error = pipeline.go("recover", "sanitizer", "clean", "--input", "chunk",
                                 "--log-input", "pipeline/chunk.jsonl")
    assert (code, error["error"]) == (errors.EXIT_USAGE, "usage")
    code, _, error = pipeline.go("recover", "copy-step", "nowhere", "--input", "src/A",
                                 "--step-name", "shuffle")
    assert (code, error["error"]) == (errors.EXIT_WRITE_REFUSED, "invalid record")


# --- recover map --------------------------------------------------------------------------------


def test_recover_map_in_position_mode_token_matching_mode_and_both(pipeline):
    pipeline.register_files()
    pipeline.register_shard()
    base = ["recover", "map", "shard-1", "--input", "clean", *pipeline.map_flags()]

    # A dry run prints the result and writes nothing; the store is opened read-only.
    before = table_counts(pipeline.store)
    code, out, _ = pipeline.go(*base, "--mode", "both", "--dry-run")
    assert code == 0
    assert (out["artifact"], out["input_artifact"], out["mode"], out["dry_run"], out["verified"],
            out["documents"], out["surviving_lines"]) == (
        "shard-1", "clean", "both", True, True, 4, 4)
    assert [(m["derivation"], m["verification"], m["label"], m["recovered"], m["stored"])
            for m in out["maps"]] == [("position", "passed", "recovered", 4, False),
                                      ("token-matching", "passed", "recovered", 4, False)]
    assert out["notes"] == ["nothing was written: the call asked for the result only"]
    assert table_counts(pipeline.store) == before
    assert pipeline.go("show", "shard-1")[1]["own_record"] is False

    # Position mode alone (the default).
    code, out, _ = pipeline.go(*base)
    assert (code, [m["derivation"] for m in out["maps"]], out["maps"][0]["stored"]) == (
        0, ["position"], True)
    assert (out["maps"][0]["append_eod"], out["maps"][0]["eod_in_span"]) == (True, True)
    # Token matching alone: position mode is not run.
    code, out, _ = pipeline.go(*base, "--mode", "token-matching")
    assert (code, [m["derivation"] for m in out["maps"]]) == (0, ["token-matching"])
    assert out["notes"] == ["F1 is said not to hold: position mode was not run"]
    # Both again: both are held already, nothing changes.
    counts = table_counts(pipeline.store)
    code, out, _ = pipeline.go(*base, "--mode", "both")
    assert (code, [m["stored"] for m in out["maps"]]) == (0, [True, True])
    assert table_counts(pipeline.store) == counts

    shown = pipeline.go("show", "shard-1")[1]
    assert [(b["namespace"], b["derivation"], b["verification"]) for b in shown["bindings"]] == [
        ("recovery", "position", "passed"), ("recovery", "token-matching", "passed")]
    document = pipeline.go("show", "shard-1", "--record", "D2")[1]
    assert (document["label"], document["parent"]["artifact"], document["parent"]["record"]) == (
        "recovered", "clean", "L3")


def test_recover_map_takes_the_tokenizer_as_an_entry_point(pipeline, monkeypatch):
    pipeline.register_files()
    pipeline.register_shard()
    (pipeline.folder / "team_tok_module.py").write_text(
        "def tokenize(text):\n    return list(text.encode('utf-8'))\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(pipeline.folder))
    code, out, _ = pipeline.go(
        "recover", "map", "shard-1", "--input", "clean", "--append-eod", "yes", "--eod-in-span",
        "yes", "--eod-token-id", str(tm_world.EOD), "--tokenizer-entry", "team_tok_module:tokenize")
    assert (code, out["verified"], out["maps"][0]["documents_compared"]) == (0, True, 4)
    sys.modules.pop("team_tok_module", None)


def test_recover_map_without_a_tokenizer_compares_nothing(pipeline):
    pipeline.register_files()
    pipeline.register_shard()
    code, out, _ = pipeline.go("recover", "map", "shard-1", "--input", "clean", "--append-eod",
                               "yes", "--eod-in-span", "yes", "--eod-token-id", str(tm_world.EOD))
    # Exit code 0: a map that is not verified is a result, and the output says so.
    assert (code, out["verified"]) == (0, False)
    assert (out["maps"][0]["verification"], out["maps"][0]["label"], out["maps"][0]["unknown"]) == (
        "not run", "unknown", 4)
    # The corrected run, with the tokenizer, is written beside it and is what a reader gets.
    code, out, _ = pipeline.go("recover", "map", "shard-1", "--input", "clean",
                               *pipeline.map_flags())
    assert (code, out["verified"], out["maps"][0]["stored"]) == (0, True, True)
    shown = pipeline.go("show", "shard-1")[1]
    assert [b["verification"] for b in shown["bindings"]] == ["passed", "not run"]


def test_recover_map_through_the_sanitizer_log_records_each_skip_once(pipeline):
    """Only the sanitizer's input is registered: the map is recovered over it with
    the log, and the skip is a drop whose output is the shard."""
    assert pipeline.go("recover", "file", pipeline.chunk, "--name", "chunk")[0] == 0
    assert pipeline.go("recover", "shard", pipeline.prefix, "--name", "shard-1", "--input",
                       "chunk", "--tokenizer-version", "toy/1")[0] == 0
    logged = ["--sanitizer-log", pipeline.log, "--log-shard", "shard-1"]
    code, out, _ = pipeline.go("recover", "map", "shard-1", "--input", "chunk",
                               *pipeline.map_flags(), *logged)
    assert (code, out["verified"], out["documents"], out["surviving_lines"]) == (0, True, 4, 4)
    assert out["drops"] == [{"record": "L3", "reason": "sanitizer skip", "step": "sanitize"}]
    dropped = pipeline.go("show", "chunk", "--record", "L3")[1]
    assert dropped["drops"] == [{"reason": "sanitizer skip", "step": "sanitize", "output": "shard-1"}]

    # The tokenizer input turns up and the sanitizer is recovered with the same
    # log: the skip is already recorded, and the command refuses by name.
    assert pipeline.go("recover", "file", pipeline.clean, "--name", "clean")[0] == 0
    counts = table_counts(pipeline.store)
    code, out, error = pipeline.go("recover", "sanitizer", "clean", "--input", "chunk",
                                   "--log", pipeline.log, "--log-input", "pipeline/chunk.jsonl")
    assert (code, out, error["error"]) == (errors.EXIT_WRITE_REFUSED, None, "drop already recorded")
    assert table_counts(pipeline.store) == counts


def test_recover_map_refuses_a_log_for_the_sanitizers_own_output(pipeline):
    pipeline.register_files()
    pipeline.register_shard()
    assert pipeline.go("recover", "sanitizer", "clean", "--input", "chunk", "--log", pipeline.log,
                       "--log-input", "pipeline/chunk.jsonl")[0] == 0
    counts = table_counts(pipeline.store)
    for input_name in ("clean", "chunk"):
        code, out, error = pipeline.go(
            "recover", "map", "shard-1", "--input", input_name, *pipeline.map_flags(),
            "--sanitizer-log", pipeline.log, "--log-shard", "shard-1")
        assert (code, out, error["error"]) == (
            errors.EXIT_WRITE_REFUSED, None, "drop already recorded")
    assert table_counts(pipeline.store) == counts


# --- Exit codes ----------------------------------------------------------------------------------


def test_the_exit_codes_of_recover(pipeline, tmp_path):
    pipeline.register_files()
    pipeline.register_shard()
    settings = ["--append-eod", "yes", "--eod-in-span", "yes", "--eod-token-id", str(tm_world.EOD)]
    base = ["recover", "map", "shard-1", "--input", "clean", *settings]

    def refused(*argv: str) -> tuple[int, str]:
        code, out, error = pipeline.go(*argv)
        assert out is None
        return code, error["error"]

    # 1: a command line the tool cannot carry out, and a tokenizer that fails.
    assert refused("recover") == (errors.EXIT_USAGE, "usage")
    assert refused(*base, "--mode", "token-matching") == (errors.EXIT_USAGE, "usage")
    assert refused(*base, "--mode", "both") == (errors.EXIT_USAGE, "usage")
    assert refused(*base, "--tokenizer-command", "x", "--tokenizer-entry", "a:b") == (
        errors.EXIT_USAGE, "usage")
    assert refused(*base, "--log-shard", "shard-1") == (errors.EXIT_USAGE, "usage")
    assert refused("recover", "map", "shard-1", "--input", "clean", "--append-eod", "yes") == (
        errors.EXIT_USAGE, "usage")
    assert refused(*base, "--tokenizer-command", str(tmp_path / "no-such-tokenizer")) == (
        errors.EXIT_USAGE, "tokenizer failed")
    assert refused(*base, "--tokenizer-entry", "no_such_team_module:tokenize") == (
        errors.EXIT_USAGE, "tokenizer failed")
    # 2: no store at the path.
    code, _, error = run("recover", "file", pipeline.chunk, "--name", "x", "--store",
                         tmp_path / "nowhere.db")
    assert (code, error["error"]) == (errors.EXIT_REFUSED, "not a store")
    # 4: a write the store or the recovery refuses.
    assert refused("recover", "map", "shard-1", "--input", "clean", "--append-eod", "no",
                   "--eod-in-span", "yes", "--eod-token-id", "256") == (
        errors.EXIT_WRITE_REFUSED, "invalid record")
    assert refused("recover", "map", "nowhere", "--input", "clean", *settings) == (
        errors.EXIT_WRITE_REFUSED, "invalid record")
    assert refused("recover", "file", pipeline.clean, "--name", "chunk") == (
        errors.EXIT_WRITE_REFUSED, "invalid record")
    # 3: a registered file that changed on disk.
    with open(pipeline.clean, "ab") as f:
        f.write(b'{"text": "added later"}\n')
    assert refused(*base) == (errors.EXIT_INTEGRITY, "artifact digest mismatch")
    # Nothing of all this left a map.
    assert pipeline.go("show", "shard-1")[1]["own_record"] is False


def test_recover_prints_text_for_a_person(pipeline):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["recover", "file", str(pipeline.chunk), "--name", "chunk", "--store",
                     str(pipeline.store), "--format", "text"], stdout=out, stderr=err)
    assert code == 0 and err.getvalue() == ""
    assert "artifact: chunk" in out.getvalue() and "records: 5" in out.getvalue()
