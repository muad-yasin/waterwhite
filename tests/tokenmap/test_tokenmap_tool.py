"""Where the tokenization maps meet the rest of the tool: one definition of the
text a line holds, one reader of a shard, a map and its drops in one transaction,
a corrected run beside a map that did not verify, a sanitizer's skip recorded
once, and the team's tokenizer reached as a command or by name (plan 3.3, 3.4,
3.1 "Drop"; Checks 3 and 6 of plan 5.5).
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest
import tm_world
from tm_store import CAPTURE, MATCHING, POSITION, RECOVERY, SETTINGS, STEP

from waterwhite import capture, errors, recovery, tokenmap
from waterwhite import vocabulary as v
from waterwhite.store import artifact_files
from waterwhite.tokenmap import _inputs, _shard


def recover(world, shard: str, input_name: str, **changes) -> tokenmap.ShardMaps:
    arguments = dict(shard=shard, input=input_name, settings=SETTINGS,
                     tokenize=tm_world.toy_tokenize)
    arguments.update(changes)
    return tokenmap.recover_map(world.store, **arguments)


def table_counts(store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


# --- One definition, one reader ---------------------------------------------------------------


def test_the_package_holds_no_second_definition_of_a_lines_text_or_of_a_shard():
    assert _inputs.line_text is capture.line_text
    assert tokenmap.ShardTokens is artifact_files.ShardTokens is _shard.ShardTokens
    assert not hasattr(_shard, "_TOKEN_TYPES")
    # The sanitizer's words are the vocabulary's, and the recovery's.
    assert tokenmap.SANITIZER_REASONS is recovery.SANITIZER_REASONS is v.SANITIZER_DROP_REASONS
    assert tokenmap.STEP_SANITIZE == v.STEP_SANITIZE == recovery.sanitize_step("1").name
    assert tokenmap.SanitizerRun().step_name == v.STEP_SANITIZE
    assert tokenmap.STEP_TOKENIZE == v.STEP_TOKENIZE == tokenmap.tokenize_step("1").name


# --- A map and its drops are one transaction (plan 3.4) ---------------------------------------


class Stopped(Exception):
    """Stands for a writer that is killed."""


def test_a_writer_stopped_between_a_map_and_its_drops_leaves_neither(world):
    """A kill between the map and its drops cannot leave a map without its drops:
    they are written in one transaction. The writer is stopped when the drops
    are due; the shard then holds no map."""
    texts = ["first", "", "third"]
    world.input("in", texts)
    instance = world.bare_shard("shard", "in", tm_world.documents_of(texts))

    def stop_at_the_drops(table: str, written: int) -> None:
        if table == "drop_record":
            raise Stopped()

    world.store.set_row_hook(stop_at_the_drops)
    with pytest.raises(Stopped):
        recover(world, "shard", "in")
    world.store.set_row_hook(None)
    assert world.store.bindings(instance) == ()
    assert world.drops("shard") == []

    # The same call again, not stopped, writes the map and its drop together.
    result = recover(world, "shard", "in")
    assert [m.stored for m in result.maps] == [True]
    assert world.head("shard") == [(RECOVERY, POSITION, v.VERIFIED_PASSED, v.RECOVERED)]
    assert world.drops("shard") == [("L2", v.DROP_NO_TOKENS, "tokenize")]


# --- A corrected run beside a map that did not verify -----------------------------------------


def test_a_corrected_recovery_stands_beside_the_one_that_failed(world):
    texts = ["first", "second", "third"]
    world.input("true-input", texts)
    world.input("wrong", ["first", "something else", "third"])
    instance = world.bare_shard("shard", "true-input", tm_world.documents_of(texts))

    # A recovery against the wrong file: both modes fail and both maps are stored.
    wrong = recover(world, "shard", "wrong")
    assert not wrong.verified
    assert world.head("shard") == [(RECOVERY, POSITION, v.VERIFIED_FAILED, v.UNKNOWN),
                                   (RECOVERY, MATCHING, v.VERIFIED_FAILED, v.UNKNOWN)]
    assert world.store.map_row(instance, 0).parent is None

    # The corrected run is not shut out: its verified map is written beside the
    # failed one, and a reader takes it.
    right = recover(world, "shard", "true-input")
    assert right.verified and [m.stored for m in right.maps] == [True]
    assert world.head("shard") == [(RECOVERY, POSITION, v.VERIFIED_PASSED, v.RECOVERED),
                                   (RECOVERY, POSITION, v.VERIFIED_FAILED, v.UNKNOWN),
                                   (RECOVERY, MATCHING, v.VERIFIED_FAILED, v.UNKNOWN)]
    assert [r.parent for r in world.store.map_rows(instance)] == [
        world.ref("true-input", n) for n in (1, 2, 3)]
    # The failed maps are still there, whole.
    assert world.store.stored_row_count(instance, RECOVERY, derivation=POSITION,
                                        verification=v.VERIFIED_FAILED) == 3
    world.store.check_integrity()
    # Either run again changes nothing.
    counts = table_counts(world.store)
    recover(world, "shard", "wrong")
    recover(world, "shard", "true-input")
    assert table_counts(world.store) == counts


def test_a_capture_map_that_was_not_verified_gains_a_verified_one(world):
    texts = ["first", "second"]
    world.capture("shard", texts, tokenize=None)  # no tokenize function: "not run"
    assert world.head("shard") == [(CAPTURE, POSITION, v.VERIFIED_NOT_RUN, v.UNKNOWN)]
    again = tokenmap.map_tokenized_shard(
        world.store, input="shard.input", prefix=world.prefix("shard"), name="shard", step=STEP,
        settings=SETTINGS, tokenize=tm_world.toy_tokenize)
    assert again.verified
    assert world.head("shard") == [(CAPTURE, POSITION, v.VERIFIED_PASSED, v.RECOVERED),
                                   (CAPTURE, POSITION, v.VERIFIED_NOT_RUN, v.UNKNOWN)]


# --- A sanitizer's skip is recorded once ------------------------------------------------------

RAW = [
    '{"id": "Z-001", "text": "first"}',
    "this line is not JSON",
    '{"id": "Z-003", "text": "third"}',
]
SKIPS = (tokenmap.SanitizerSkip(2, v.DROP_SANITIZER_SKIP),)


def sanitizer_files(world):
    """The file a sanitizer read, registered, and the file it wrote, registered,
    with the sanitizer recovered between them from its log."""
    path = tm_world.write_raw_lines(world.folder / "sanitizer-input/shard.jsonl", RAW)
    world.register(path, "shard.sanitizer-input")
    world.input("shard.input", ["first", "third"])
    log = world.folder / "sanitizer.log.jsonl"
    log.write_text('{"line": 2, "reason": "sanitizer skip"}\n', encoding="utf-8")
    return log


def all_drops(world) -> list[tuple]:
    return [(row[0], row[1], row[2]) for row in world.store._conn.execute(
        "SELECT d.locator, d.reason, n.name FROM drop_record d "
        "JOIN artifact a ON a.instance_id = d.output_instance_id "
        "JOIN artifact_name n ON n.artifact_id = a.artifact_id ORDER BY d.drop_id")]


def test_a_skip_recorded_for_the_tokenizer_input_is_not_recorded_again_for_the_shard(world):
    """The double count: the tokenizer-input file is registered and the sanitizer
    was recovered onto it, so the skipped line has its drop there. A map recovery
    over the sanitizer's input, with the log, would record the same line again
    with the shard as its output. It is refused by name and writes nothing."""
    log = sanitizer_files(world)
    done = recovery.recover_sanitizer(
        world.store, "shard.input", "shard.sanitizer-input",
        step=recovery.sanitize_step("control-chars/1"), log_path=log)
    assert (done.recovered, done.drops_written) == (2, 1)
    assert all_drops(world) == [("L2", v.DROP_SANITIZER_SKIP, "shard.input")]
    instance = world.bare_shard("shard", "shard.input",
                                tm_world.documents_of(["first", "third"]))

    counts = table_counts(world.store)
    with pytest.raises(errors.DropAlreadyRecorded, match="already has its drop under shard.input"):
        recover(world, "shard", "shard.sanitizer-input", sanitizer=tokenmap.SanitizerRun(SKIPS))
    # Nothing was written: no map, and the line still has one drop.
    assert table_counts(world.store) == counts
    assert world.store.bindings(instance) == ()
    assert all_drops(world) == [("L2", v.DROP_SANITIZER_SKIP, "shard.input")]
    # A dry run is refused the same way: the result would be a plan to count twice.
    with pytest.raises(errors.DropAlreadyRecorded):
        recover(world, "shard", "shard.sanitizer-input",
                sanitizer=tokenmap.SanitizerRun(SKIPS), write=False)

    # The map belongs over the registered tokenizer input, with no log. The line
    # is then recorded exactly once.
    result = recover(world, "shard", "shard.input")
    assert result.verified and result.drops == ()
    assert all_drops(world) == [("L2", v.DROP_SANITIZER_SKIP, "shard.input")]


def test_a_log_is_refused_for_a_file_that_is_the_sanitizers_output(world):
    """The input named with a log already is the output of recorded sanitizer
    drops: it is the file the sanitizer wrote, and the log numbers another file."""
    log = sanitizer_files(world)
    recovery.recover_sanitizer(
        world.store, "shard.input", "shard.sanitizer-input",
        step=recovery.sanitize_step("control-chars/1"), log_path=log)
    instance = world.bare_shard("shard", "shard.input",
                                tm_world.documents_of(["first", "third"]))
    counts = table_counts(world.store)
    with pytest.raises(errors.DropAlreadyRecorded, match="is the output of recorded sanitizer drops"):
        recover(world, "shard", "shard.input", sanitizer=tokenmap.SanitizerRun(SKIPS))
    assert table_counts(world.store) == counts and world.store.bindings(instance) == ()
    assert errors.DropAlreadyRecorded.exit_code == errors.EXIT_WRITE_REFUSED
    assert errors.DropAlreadyRecorded.name == "drop already recorded"


def test_with_only_the_sanitizer_input_kept_the_skip_is_recorded_for_the_shard_once(world):
    """Only the sanitizer's input was kept: the shard is the first registered
    artifact that lacks the line, and the map recovery records the drop there.
    The recovery of the sanitizer then refuses the same log for a file found
    later, and so does a second step name for the same line and shard."""
    path = tm_world.write_raw_lines(world.folder / "sanitizer-input/shard.jsonl", RAW)
    world.register(path, "shard.sanitizer-input")
    world.bare_shard("shard", "shard.sanitizer-input", tm_world.documents_of(["first", "third"]))
    result = recover(world, "shard", "shard.sanitizer-input",
                     sanitizer=tokenmap.SanitizerRun(SKIPS))
    assert result.verified
    assert all_drops(world) == [("L2", v.DROP_SANITIZER_SKIP, "shard")]
    # The same call again: the same drop, once.
    recover(world, "shard", "shard.sanitizer-input", sanitizer=tokenmap.SanitizerRun(SKIPS))
    assert all_drops(world) == [("L2", v.DROP_SANITIZER_SKIP, "shard")]
    with pytest.raises(errors.DropAlreadyRecorded, match="under the step name 'sanitize'"):
        recover(world, "shard", "shard.sanitizer-input",
                sanitizer=tokenmap.SanitizerRun(SKIPS, step_name="clean"))

    # The tokenizer input turns up afterwards. Its edges can be recovered; the
    # log's drop cannot be recorded a second time.
    world.input("shard.input", ["first", "third"])
    log = world.folder / "sanitizer.log.jsonl"
    log.write_text('{"line": 2, "reason": "sanitizer skip"}\n', encoding="utf-8")
    counts = table_counts(world.store)
    with pytest.raises(errors.DropAlreadyRecorded, match="already has its drop under the shard shard"):
        recovery.recover_sanitizer(
            world.store, "shard.input", "shard.sanitizer-input",
            step=recovery.sanitize_step("control-chars/1"), log_path=log)
    assert table_counts(world.store) == counts
    edges_only = recovery.recover_sanitizer(
        world.store, "shard.input", "shard.sanitizer-input",
        step=recovery.sanitize_step("control-chars/1"))
    assert (edges_only.recovered, edges_only.drops) == (2, 0)
    assert all_drops(world) == [("L2", v.DROP_SANITIZER_SKIP, "shard")]


def test_two_tokenizer_runs_over_one_sanitizer_input_each_record_the_skip(world):
    path = tm_world.write_raw_lines(world.folder / "sanitizer-input/shard.jsonl", RAW)
    world.register(path, "chunk")
    for shard in ("run-1", "run-2"):
        world.bare_shard(shard, "chunk", tm_world.documents_of(["first", "third"]))
        # The two shards hold the same bytes; a second step makes a second instance.
    run = tokenmap.SanitizerRun(SKIPS)
    assert recover(world, "run-1", "chunk", sanitizer=run).verified
    # The same bytes under the same step and input are one instance with two names.
    assert world.store.artifact("run-1").instance_id == world.store.artifact("run-2").instance_id
    tm_world.write_shard(world.prefix("other"), tm_world.documents_of(["first", "third"]))
    other_step = tokenmap.tokenize_step("toy/2", {"tokenizer": "bytes", "append_eod": True})
    world.store.register_shard(world.prefix("other"), name="other", step=other_step,
                               inputs=[world.store.artifact("chunk").instance_id])
    assert recover(world, "other", "chunk", sanitizer=run).verified
    assert sorted(d[2] for d in all_drops(world)) == ["other", "run-1", "run-2"]
    assert len({(row[0]) for row in world.store._conn.execute(
        "SELECT output_instance_id FROM drop_record")}) == 2


# --- The result as JSON ------------------------------------------------------------------------


def test_a_result_gives_its_maps_without_their_rows(world):
    texts = ["first", "twin", "", "third", "twin"]
    world.input("in", texts)
    world.bare_shard("shard", "in", tm_world.documents_of(texts))
    out = recover(world, "shard", "in", also_token_matching=True, write=False).to_json()
    json.dumps(out)  # plain JSON values
    assert (out["documents"], out["surviving_lines"], out["verified"]) == (4, 5, True)
    assert [(m["namespace"], m["derivation"], m["verification"], m["label"], m["recovered"],
             m["candidates"], m["unknown"], m["stored"]) for m in out["maps"]] == [
        (RECOVERY, POSITION, v.VERIFIED_PASSED, v.RECOVERED, 2, 2, 0, False),
        (RECOVERY, MATCHING, v.VERIFIED_PASSED, v.RECOVERED, 2, 2, 0, False)]
    assert out["maps"][0]["append_eod"] is True and out["maps"][0]["documents_compared"] == 4
    assert out["drops"] == [{"record": "L3", "reason": v.DROP_NO_TOKENS, "step": "tokenize"}]
    assert out["notes"] == ["nothing was written: the call asked for the result only"]


# --- The team's tokenizer, as a command or by name ---------------------------------------------

TOKENIZER = textwrap.dedent('''
    import json, sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "good"
    for number, line in enumerate(sys.stdin, start=1):
        text = json.loads(line)["text"]
        if mode == "exit" and number == 2:
            sys.stderr.write("the vocabulary file is missing\\n")
            sys.exit(3)
        if mode == "garbage" and number == 2:
            print("not a reply", flush=True)
            continue
        if mode == "silent" and number == 2:
            import time
            time.sleep(30)
        print(json.dumps({"tokens": list(text.encode("utf-8"))}), flush=True)
    if mode == "late":
        sys.exit(5)
''')


@pytest.fixture
def tokenizer_script(tmp_path):
    path = tmp_path / "team_tokenizer.py"
    path.write_text(TOKENIZER, encoding="utf-8")
    return path


def test_a_tokenizer_command_answers_one_text_per_line(tokenizer_script, world):
    with tokenmap.command_tokenizer([sys.executable, str(tokenizer_script)]) as tokenize:
        assert tokenize("ab") == [97, 98]
        assert tokenize("") == []
        assert tokenize("line\nfeed é") == list("line\nfeed é".encode("utf-8"))
        # It is the tokenize function of a recovery: every document is compared.
        texts = ["first", "twin", "", "third", "twin"]
        world.input("in", texts)
        world.bare_shard("shard", "in", tm_world.documents_of(texts))
        result = recover(world, "shard", "in", tokenize=tokenize)
        assert result.verified and result.map(POSITION).verification.documents_compared == 4
    assert tokenize.exit_status == 0
    # One text of command and arguments is split as a shell splits words.
    with tokenmap.command_tokenizer(f"{sys.executable} {tokenizer_script} late") as late:
        assert late("a") == [97]
    assert late.exit_status == 5  # after its last reply: reported, not an error


@pytest.mark.parametrize(("mode", "said"), [
    ("exit", "ended before it answered text 2 (exit status 3); it said last: the vocabulary"),
    ("garbage", "answered text 2 with something that is not"),
])
def test_a_tokenizer_command_that_fails_is_a_named_error(tokenizer_script, mode, said):
    with pytest.raises(errors.TokenizerFailed) as raised:
        with tokenmap.command_tokenizer([sys.executable, str(tokenizer_script), mode]) as tokenize:
            assert tokenize("one") == [111, 110, 101]
            tokenize("two")
    assert said in raised.value.detail
    assert (errors.TokenizerFailed.name, errors.TokenizerFailed.exit_code) == (
        "tokenizer failed", errors.EXIT_USAGE)


def test_a_tokenizer_command_has_a_time_limit_and_must_exist(tokenizer_script, tmp_path):
    with pytest.raises(errors.TokenizerFailed, match="gave no reply for text 2 within 0.5 seconds"):
        with tokenmap.command_tokenizer([sys.executable, str(tokenizer_script), "silent"],
                                        timeout_seconds=0.5) as tokenize:
            tokenize("one")
            tokenize("two")
    with pytest.raises(errors.TokenizerFailed, match="could not be started"):
        tokenmap.command_tokenizer([str(tmp_path / "no-such-program")])
    with pytest.raises(errors.InvalidRecord, match="a program and its arguments"):
        tokenmap.command_tokenizer([])
    with pytest.raises(errors.InvalidRecord, match="above 0 seconds"):
        tokenmap.command_tokenizer([sys.executable], timeout_seconds=0)


def test_a_tokenizer_named_as_an_entry_point_is_imported_by_name(tmp_path, monkeypatch):
    (tmp_path / "team_tok.py").write_text(textwrap.dedent('''
        class Tokenizer:
            def encode(self, text):
                if text == "boom":
                    raise ValueError("no such piece")
                return list(text.encode("utf-8"))
        tokenizer = Tokenizer()
        def tokenize(text):
            return tokenizer.encode(text)
        table = {}
    '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert "team_tok" not in sys.modules  # nothing of it is imported until it is named
    assert tokenmap.entry_point_tokenizer("team_tok:tokenize")("ab") == [97, 98]
    dotted = tokenmap.entry_point_tokenizer("team_tok:tokenizer.encode")
    assert dotted("c") == [99]
    with pytest.raises(errors.TokenizerFailed, match="raised ValueError: no such piece"):
        dotted("boom")
    for spec, said in (("no_such_team_module:tokenize", "could not be imported"),
                       ("team_tok:missing", "which is not there"),
                       ("team_tok:table", "is not callable")):
        with pytest.raises(errors.TokenizerFailed, match=said):
            tokenmap.entry_point_tokenizer(spec)
    for spec in ("team_tok", "team_tok:", ":tokenize", "a:b:c"):
        with pytest.raises(errors.InvalidRecord, match="module:function"):
            tokenmap.entry_point_tokenizer(spec)
    sys.modules.pop("team_tok", None)


def test_the_tool_imports_no_tokenizer_library():
    """The tool reaches a tokenizer through the caller's function, a command or a
    name the caller gives; no module of the tool imports one."""
    import ast
    from pathlib import Path

    package = Path(tokenmap.__file__).resolve().parents[1]
    libraries = {"tokenizers", "transformers", "sentencepiece", "tiktoken", "megatron", "torch"}
    found = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            found += [(str(path.relative_to(package)), name)
                      for name in names if name.split(".")[0] in libraries]
    # One exception, for the join and not for tokenization: the shuffle of a cyclic
    # run is computed by torch, as the sampler draws it (plan 3.5), in one function
    # (tests/join/test_join_imports.py holds that file to that one place).
    assert found in ([], [("join/_shuffle.py", "torch")])
