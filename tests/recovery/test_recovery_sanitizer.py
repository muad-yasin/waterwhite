"""Recovering the sanitizer, a deterministic rewrite (plan 3.3, F8), and what
happens when its function is not available.

The test pipeline's sanitizer is in ``recovery_kit``: it applies the store's
normaliser, skips blank, emptied and unreadable lines, and writes a log in the
form docs/recovery.md gives.
"""

from __future__ import annotations

import json

import pytest
import recovery_kit as kit

from waterwhite import capture, errors, normalise, recovery
from waterwhite import vocabulary as v

COPY = capture.step_ref("shuffle-deal-merge", "not recorded")
STEP = capture.step_ref("sanitize", "1")
LOG_INPUT = "pipeline/partitioned/sanitizer-input/shard-1.jsonl"

WITH_CONTROLS = "a line\x00 with\x07 control characters\r\n"


def build(store, folder):
    """Sources A and Z, a chunk as the sanitizer reads it (with a line that does not
    parse, a blank line and a line whose text is empty after sanitizing), the
    sanitizer's output and its log."""
    a = kit.source(store, folder, "A", ["a plain line", WITH_CONTROLS, "a third line"])
    z = kit.source(store, folder, "Z", ["z one", "z two"])
    kit.source(store, folder, "C", ["c one"])
    unreadable = kit.Rec(("no file", 0), "{this line does not parse")
    blank = kit.Rec(("no file", 0), "")
    emptied = kit.Rec(("no file", 0), json.dumps({"text": " \x00\t "}))
    chunk = kit.found_file(
        store, folder, "sanitizer-input",
        [a[1], unreadable, z[0], blank, a[0], emptied, z[1], a[2]],
    )
    kept, log = kit.sanitize(chunk, LOG_INPUT)
    kit.found_file(store, folder, "tokenizer-input", kept)
    log_path = kit.write_log(folder / "pipeline/sanitizer.log.jsonl", log)
    return kept, log_path


def sanitizer(store, log_path, **more):
    more.setdefault("log_input", LOG_INPUT)
    return recovery.recover_sanitizer(
        store, "tokenizer-input", "sanitizer-input", step=STEP, log_path=log_path, **more
    )


def drops_of(store, name):
    instance = store.artifact(name).instance_id
    return sorted((d.input.locator, d.reason, d.step_name) for d in store.drops_by_output(instance))


# --- The sanitizer recovered -------------------------------------------------------------


def test_the_sanitizer_is_recovered_with_its_skips_as_drops(store, folder):
    kept, log_path = build(store, folder)
    copied = recovery.recover_copy_step(store, "sanitizer-input", ["src/A", "src/Z"], step=COPY)
    # Three lines of the sanitizer's input came from no source.
    assert (copied.recovered, copied.no_edge, copied.no_text) == (5, 3, 3)

    result = sanitizer(store, log_path)

    assert (result.kind, result.outcome, result.reason) == (
        recovery.KIND_SANITIZER, recovery.OUTCOME_RECORD_LEVEL, None)
    assert (result.records, result.recovered, result.candidates, result.no_edge) == (5, 5, 0, 0)
    assert (result.drops, result.drops_written, result.edges_written) == (3, 3, 5)
    assert result.input_lines_unaccounted == 0
    assert result.files_read == ("tokenizer-input", "sanitizer-input", str(log_path))

    # Every surviving line is linked to the line of the input it was written from.
    assert [record.origin for record in kept] == [
        ("sanitizer-input", n) for n in (1, 3, 5, 7, 8)]
    for line, record in enumerate(kept, 1):
        assert kit.parents(store, "tokenizer-input", line) == [record.origin]
        edge = store.edge_of(kit.ref(store, "tokenizer-input", line))
        assert (edge.label, edge.derivation, edge.verification, edge.transform) == (
            v.RECOVERED, v.DERIVED_FUNCTION, v.VERIFIED_PASSED, STEP)

    # The skipped line and the two empty lines are drops of the step, read from its log.
    assert drops_of(store, "tokenizer-input") == [
        ("L2", v.DROP_SANITIZER_SKIP, "sanitize"),
        ("L4", v.DROP_EMPTY_LINE, "sanitize"),
        ("L6", v.DROP_EMPTY_LINE, "sanitize"),
    ]
    store.check_integrity()


def test_a_record_with_control_characters_is_matched_after_normalisation(store, folder):
    kept, log_path = build(store, folder)
    sanitizer(store, log_path)

    before = json.loads(kit.file_path(folder, "sanitizer-input").read_text().splitlines()[0])
    after = json.loads(kit.file_path(folder, "tokenizer-input").read_text().splitlines()[0])
    assert before["text"] == WITH_CONTROLS
    assert after["text"] == "a line with control characters"
    assert before["text"] != after["text"]  # the raw texts differ
    assert kit.parents(store, "tokenizer-input", 1) == [("sanitizer-input", 1)]


def test_a_logged_line_is_a_drop_and_not_a_missing_origin(store, folder):
    _kept, log_path = build(store, folder)
    recovery.recover_copy_step(store, "sanitizer-input", ["src/A", "src/Z"], step=COPY)
    sanitizer(store, log_path)

    # The sanitizer's input holds a line that came from no source: for that file no
    # source can be ruled out.
    upstream = kit.ask(store, "C", "sanitizer-input")
    assert (upstream["answer"], upstream["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert upstream["failing"]["record"] == "L2"
    # The sanitizer dropped that line, so its output is traced whole: three hops
    # back to A, and a negative for a source it holds nothing of.
    held = kit.ask(store, "A", "tokenizer-input")
    assert (held["answer"], kit.hops(held)) == (
        "contains", ["tokenizer-input:L1", "sanitizer-input:L1", "src/A:L2"])
    assert [hop["link"] for hop in held["chain"]] == [v.RECOVERED, v.RECOVERED, "member"]
    clear = kit.ask(store, "C", "tokenizer-input")
    assert (clear["answer"], clear["scope"]["walked"]) == ("does not contain", 5)


def test_the_second_sanitizer_run_writes_nothing_and_says_so(store, folder):
    _kept, log_path = build(store, folder)
    first = sanitizer(store, log_path)
    held = kit.rows_held(store, "tokenizer-input")
    second = sanitizer(store, log_path)

    assert (first.edges_written, first.drops_written, first.wrote_nothing) == (5, 3, False)
    assert (second.recovered, second.drops, second.input_lines_unaccounted) == (5, 3, 0)
    assert (second.edges_written, second.drops_written, second.links_written) == (0, 0, 0)
    assert second.wrote_nothing is True
    assert kit.rows_held(store, "tokenizer-input") == held == (5, 3, 0)


def test_a_line_the_log_names_is_no_candidate_parent(store, folder):
    # Two lines of the input hold one text. The sanitizer skipped the first (in this
    # pipeline a row with no id is skipped) and wrote the second.
    same = "one text on two lines"
    path = kit.write_lines(folder / "pipeline/in.jsonl", [
        json.dumps({"text": same}), json.dumps({"id": "k-2", "text": same})])
    recovery.register_found_file(store, path, name="in")
    out = kit.write_lines(folder / "pipeline/out.jsonl", [json.dumps({"text": same})])
    recovery.register_found_file(store, out, name="out")
    log_path = kit.write_log(folder / "pipeline/in.log.jsonl", [
        {"line": 1, "reason": "sanitizer skip", "detail": "no id"}])

    result = recovery.recover_sanitizer(store, "out", "in", step=STEP, log_path=log_path)

    assert kit.parents(store, "out", 1) == [("in", 2)]
    assert (result.recovered, result.candidates, result.drops) == (1, 0, 1)
    assert result.input_lines_unaccounted == 0


def test_without_a_log_the_skipped_lines_are_unaccounted_for(store, folder):
    _kept, _log_path = build(store, folder)
    result = recovery.recover_sanitizer(store, "tokenizer-input", "sanitizer-input", step=STEP)

    assert (result.recovered, result.drops, result.drops_written) == (5, 0, 0)
    assert result.input_lines_unaccounted == 3
    assert result.files_read == ("tokenizer-input", "sanitizer-input")
    assert drops_of(store, "tokenizer-input") == []


# --- The log ------------------------------------------------------------------------------


def test_a_log_of_several_files_gives_only_the_rows_of_the_file_asked_for(folder):
    rows = [
        {"input": "in/shard-1.jsonl", "line": 4, "reason": "empty line"},
        {"input": "in/shard-2.jsonl", "line": 6, "reason": "sanitizer skip"},
        {"input": "in/shard-1.jsonl", "line": 2, "reason": "sanitizer skip"},
        {"input": "in/shard-1.jsonl", "line": 2, "reason": "sanitizer skip"},
    ]
    log_path = kit.write_log(folder / "log.jsonl", rows)

    assert recovery.read_sanitizer_log(log_path, input="in/shard-1.jsonl") == (
        recovery.LoggedSkip(2, "sanitizer skip"), recovery.LoggedSkip(4, "empty line"))
    assert recovery.read_sanitizer_log(log_path, input="in/shard-2.jsonl") == (
        recovery.LoggedSkip(6, "sanitizer skip"),)
    assert recovery.read_sanitizer_log(log_path, input="in/shard-3.jsonl") == ()
    # Which file a row is about is never guessed.
    with pytest.raises(errors.InvalidRecord, match="not assigned to a file by guess"):
        recovery.read_sanitizer_log(log_path)


@pytest.mark.parametrize("row, match", [
    ('{"line": 2, "reason": "too short"}', "the reason of a skipped line"),
    ('{"line": 0, "reason": "empty line"}', "a whole number from 1"),
    ('{"line": "2", "reason": "empty line"}', "a whole number from 1"),
    ('["line", 2]', "not a JSON object"),
    ("line 2 was skipped", "not JSON"),
    ('{"line": 2, "reason": "empty line"}\n{"line": 2, "reason": "sanitizer skip"}',
     "logged twice"),
])
def test_a_log_row_the_reader_cannot_use_is_refused(folder, row, match):
    log_path = kit.write_lines(folder / "log.jsonl", [row])
    with pytest.raises(errors.InvalidRecord, match=match):
        recovery.read_sanitizer_log(log_path)


def test_a_log_that_does_not_fit_the_file_is_refused_and_nothing_is_written(store, folder):
    _kept, log_path = build(store, folder)

    wrong = kit.write_log(folder / "pipeline/wrong.log.jsonl", [
        {"input": LOG_INPUT, "line": 9, "reason": "empty line"}])  # the input has 8 lines
    with pytest.raises(errors.InvalidRecord, match="not the log of this file"):
        sanitizer(store, wrong)
    with pytest.raises(errors.InvalidRecord):
        sanitizer(store, log_path, log_input=None)  # the rows name a file, the call none
    with pytest.raises(errors.InvalidRecord, match="no sanitizer log"):
        sanitizer(store, folder / "pipeline/no-such.log.jsonl")
    with pytest.raises(errors.InvalidRecord, match="no log is given"):
        sanitizer(store, None)
    assert kit.rows_held(store, "tokenizer-input") == (0, 0, 0)

    # A stored drop is not replaced by a log that gives the line another reason.
    sanitizer(store, log_path)
    other = kit.write_log(folder / "pipeline/other.log.jsonl", [
        {"input": LOG_INPUT, "line": 2, "reason": "empty line"}])
    with pytest.raises(errors.InvalidRecord, match="not replaced"):
        sanitizer(store, other)
    assert kit.rows_held(store, "tokenizer-input") == (5, 3, 0)


# --- The normaliser is not available (plan 3.3, F8) ---------------------------------------

TEAM = "team-sanitizer/1"


def team_sanitizer(text: str) -> str:
    return " ".join(text.lower().split())


@pytest.fixture
def team_store(folder, monkeypatch):
    """A store whose fingerprints are taken after a team's own sanitizer, which is
    registered for this test only."""
    monkeypatch.setitem(normalise._REGISTRY, TEAM, team_sanitizer)
    opened = kit.new_store(folder, TEAM)
    yield opened
    opened.close()


def sanitized(records):
    return [
        kit.Rec(record.origin, json.dumps({"text": team_sanitizer(record.text)}))
        for record in records
        if team_sanitizer(record.text)
    ]


def test_without_the_normaliser_a_file_tied_to_a_member_degrades_to_file_level(
    team_store, folder, monkeypatch
):
    store = team_store
    # The sanitizer read the source file itself.
    a = kit.source(store, folder, "A", ["  First  Line ", "   ", "Second Line"])
    kit.source(store, folder, "C", ["c one"])
    kit.found_file(store, folder, "sanitized", sanitized(a))
    log_path = kit.write_log(folder / "pipeline/sanitizer.log.jsonl", [
        {"line": 2, "reason": "empty line", "detail": "empty after sanitizing"}])

    monkeypatch.delitem(normalise._REGISTRY, TEAM)  # the function is gone
    result = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP, log_path=log_path)

    assert result.outcome == recovery.OUTCOME_FILE_LEVEL == v.FILE_LEVEL
    assert result.reason.startswith("unknown normaliser: 'team-sanitizer/1'")
    assert (result.members_linked, result.links_written) == (("A/r1",), 1)
    assert (result.records, result.recovered, result.candidates, result.no_edge) == (2, 0, 0, 2)
    assert (result.edges_written, result.drops, result.drops_written) == (0, 1, 1)
    assert result.input_lines_unaccounted is None
    assert result.files_read == (str(log_path),)  # no record file was opened

    # One file-level link, no record-level link, and the log's drop.
    assert kit.rows_held(store, "sanitized") == (0, 1, 1)
    (link,) = store.file_source_links(store.artifact("sanitized").instance_id)
    assert (link.member, link.label, link.derivation, link.locator) == (
        "A/r1", v.FILE_LEVEL, v.DERIVED_FILE, None)
    assert "team-sanitizer/1" in link.evidence

    # What a file-level link decides (plan 3.2): the whole source, and nothing else.
    whole = kit.ask(store, "A", "sanitized")
    assert (whole["answer"], whole["chain"][0]["link"]) == ("contains", "file-level link")
    narrowed = kit.ask(store, "A", "sanitized", records=["src/A:L1"])
    assert (narrowed["answer"], narrowed["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK)
    other = kit.ask(store, "C", "sanitized")
    assert (other["answer"], other["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)

    # The same degraded run again writes nothing.
    second = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP, log_path=log_path)
    assert (second.outcome, second.wrote_nothing) == (recovery.OUTCOME_FILE_LEVEL, True)
    assert kit.rows_held(store, "sanitized") == (0, 1, 1)


def test_without_the_normaliser_a_file_tied_to_no_member_degrades_to_unknown(
    team_store, folder, monkeypatch
):
    store = team_store
    a = kit.source(store, folder, "A", ["First Line", "Second Line"])
    # The sanitizer read a chunk two steps after the source, a file with no member.
    chunk = kit.found_file(store, folder, "sanitizer-input", [a[1], a[0]])
    recovery.recover_copy_step(store, "sanitizer-input", ["src/A"], step=COPY)
    kit.found_file(store, folder, "sanitized", sanitized(chunk))

    monkeypatch.delitem(normalise._REGISTRY, TEAM)
    result = recovery.recover_sanitizer(store, "sanitized", "sanitizer-input", step=STEP)

    assert result.outcome == recovery.OUTCOME_UNKNOWN == v.UNKNOWN
    assert result.reason.startswith("unknown normaliser")
    assert (result.members_linked, result.no_edge, result.wrote_nothing) == ((), 2, True)
    assert result.files_read == ()
    assert kit.rows_held(store, "sanitized") == (0, 0, 0)
    # Nothing is known about the records, and the answer says so.
    answer = kit.ask(store, "A", "sanitized")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)


def test_without_the_normaliser_a_copy_step_is_refused_by_name(team_store, folder, monkeypatch):
    store = team_store
    a = kit.source(store, folder, "A", ["First Line"])
    kit.found_file(store, folder, "chunk-1", [a[0]])

    monkeypatch.delitem(normalise._REGISTRY, TEAM)
    with pytest.raises(errors.UnknownNormaliser):
        recovery.recover_copy_step(store, "chunk-1", ["src/A"], step=COPY)
    assert kit.rows_held(store, "chunk-1") == (0, 0, 0)
