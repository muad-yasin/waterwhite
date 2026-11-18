"""What the recovery decides where the plan leaves it open, and where it meets
the rest of the tool: surplus copies, the degraded link, the role of a found
file, a run that writes nothing, and one definition of a line's text and of an
artifact's place on disk (plan 3.2, 3.3, F8, L3; Check 6 of plan 5.5).
"""

from __future__ import annotations

import json

import pytest
import recovery_kit as kit

from waterwhite import capture, errors, listing, normalise, recovery, requests
from waterwhite import vocabulary as v
from waterwhite.recovery import _files

COPY = capture.step_ref("shuffle-chunk-merge", "not recorded")
STEP = recovery.sanitize_step("team-sanitizer/1")


# --- One definition -----------------------------------------------------------------------


def test_the_recovery_reads_a_line_and_finds_a_file_as_the_rest_of_the_tool_does(store, folder):
    a = kit.source(store, folder, "A", ["a one"])
    artifact = store.artifact("src/A")
    assert _files.disk_path(store, artifact) == store.disk_path(artifact.instance_id)
    file = _files.RegisteredFile(store, artifact, function=normalise.get(kit.NORMALISER),
                                 text_field="text")
    assert file.record(a[0].raw.encode("utf-8"), 1)[0] == capture.line_text(
        a[0].raw.encode("utf-8"), "text") == "a one"
    assert not hasattr(capture, "_line_text")
    assert (recovery.OUTCOME_RECORD_LEVEL, recovery.OUTCOME_FILE_LEVEL, recovery.OUTCOME_UNKNOWN) == (
        v.OUTCOME_RECORD_LEVEL, v.OUTCOME_FILE_LEVEL, v.OUTCOME_UNKNOWN)
    assert recovery.OUTCOMES is v.RECOVERY_OUTCOMES


# --- A found file keeps its place in the pipeline -----------------------------------------


def test_a_found_file_is_an_intermediate_unless_told_otherwise(store, folder):
    lines = [json.dumps({"text": "one"}), json.dumps({"text": "two"})]
    chunk = kit.write_lines(folder / "pipeline/chunk.jsonl", lines)
    release = kit.write_lines(folder / "pipeline/release.jsonl", lines[:1])
    legacy = kit.write_lines(folder / "pipeline/legacy.jsonl", lines[1:])
    recovery.register_found_file(store, chunk, name="chunk")
    recovery.register_found_file(store, release, name="R1", role=capture.ROLE_RELEASE)
    recovery.register_found_file(store, legacy, name="legacy", role=capture.ROLE_IMPORT)
    assert [(store.artifact(n).role, store.artifact(n).step) for n in ("chunk", "R1", "legacy")] == [
        (v.ROLE_INTERMEDIATE, None), (v.ROLE_RELEASE, None), (v.ROLE_IMPORT, None)]
    with pytest.raises(errors.InvalidRecord, match="a source file is ingested"):
        recovery.register_found_file(store, chunk, name="again", role=capture.ROLE_SOURCE)
    # The same file under the same name with another role is refused, not changed.
    with pytest.raises(errors.InvalidRecord, match="the role"):
        recovery.register_found_file(store, chunk, name="chunk", role=capture.ROLE_RELEASE)

    # A listing names the found release among the releases it covers, and gives
    # every found file an answer.
    kit.source(store, folder, "A", ["one"])
    made = listing.listing(store, requests.withdraw(store, "A")).to_json()
    assert made["covers"]["releases"] == ["R1"]
    assert {a["artifact"] for a in made["answers"]} >= {"chunk", "R1", "legacy"}


# --- Surplus copies (plan 3.3 "Copy steps", L3) --------------------------------------------


def test_a_text_held_more_often_than_the_named_inputs_hold_it_gets_no_edge(store, folder):
    """The later file holds a text three times and the one named input holds it
    once. A copy step writes a record at most once for each record it read, so
    two of the three came through a file nobody named, and no rule tells which.
    None of the three is given a parent."""
    a = kit.source(store, folder, "A", ["shared passage", "only in a"])
    kit.source(store, folder, "C", ["c one"])
    extra = kit.Rec(("elsewhere", 1), json.dumps({"text": "shared passage"}))
    kit.found_file(store, folder, "merged", [a[0], a[1], extra, extra])

    result = recovery.recover_copy_step(store, "merged", ["src/A"], step=COPY)
    assert (result.records, result.recovered, result.candidates, result.no_edge) == (4, 1, 0, 3)
    assert (result.surplus, result.no_text, result.edges_written) == (3, 0, 1)
    assert result.to_json()["surplus"] == 3
    assert kit.parents(store, "merged", 2) == [("src/A", 2)]
    assert [kit.parents(store, "merged", line) for line in (1, 3, 4)] == [[], [], []]

    # The file holds a record of A, so it "contains" A through that record.
    assert kit.hops(kit.ask(store, "A", "merged")) == ["merged:L2", "src/A:L2"]
    # For any other source the copies have no recorded origin: never a negative.
    other = kit.ask(store, "C", "merged")
    assert (other["answer"], other["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert other["failing"]["record"] == "L1"
    narrowed = kit.ask(store, "A", "merged", records=["src/A:L1"])
    assert (narrowed["answer"], narrowed["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)


def test_surplus_copies_get_their_edges_once_every_input_is_named(store, folder):
    a = kit.source(store, folder, "A", ["shared passage", "only in a"])
    b = kit.source(store, folder, "B", ["shared passage"])
    kit.found_file(store, folder, "merged", [a[0], a[1], b[0]])

    short = recovery.recover_copy_step(store, "merged", ["src/A"], step=COPY)
    assert (short.recovered, short.surplus, short.edges_written) == (1, 2, 1)
    # Nothing was written for the two copies, so the run that names both inputs
    # is not shut out by an earlier edge: each copy gets its candidates.
    full = recovery.recover_copy_step(store, "merged", ["src/A", "src/B"], step=COPY)
    assert (full.recovered, full.candidates, full.surplus, full.edges_written) == (1, 2, 0, 2)
    assert kit.parents(store, "merged", 1) == kit.parents(store, "merged", 3) == [
        ("src/A", 1), ("src/B", 1)]
    label = store.edge_of(kit.ref(store, "merged", 1)).label
    assert label == v.CANDIDATES
    assert kit.ask(store, "B", "merged")["answer"] == "may contain"


def test_as_many_copies_as_the_inputs_hold_are_candidates_as_before(store, folder):
    a = kit.source(store, folder, "A", ["twin text", "twin text", "alone"])
    kit.found_file(store, folder, "shuffled", [a[1], a[2], a[0]])
    result = recovery.recover_copy_step(store, "shuffled", ["src/A"], step=COPY)
    assert (result.recovered, result.candidates, result.surplus) == (1, 2, 0)


# --- A run that writes nothing --------------------------------------------------------------


def test_with_write_false_a_recovery_returns_its_result_and_writes_nothing(store, folder):
    a = kit.source(store, folder, "A", ["a one", "", "a three"])
    chunk = kit.found_file(store, folder, "chunk", [a[2], a[1], a[0]])
    kept, rows = kit.sanitize(chunk, "chunk.jsonl")
    kit.found_file(store, folder, "clean", kept)
    log = kit.write_log(folder / "sanitizer.log.jsonl", rows)

    dry = recovery.recover_copy_step(store, "chunk", ["src/A"], step=COPY, write=False)
    assert (dry.recovered, dry.no_edge, dry.dry_run, dry.wrote_nothing) == (2, 1, True, True)
    assert kit.rows_held(store, "chunk") == (0, 0, 0)
    dry = recovery.recover_sanitizer(store, "clean", "chunk", step=STEP, log_path=log,
                                     log_input="chunk.jsonl", write=False)
    assert (dry.recovered, dry.drops, dry.drops_written, dry.dry_run) == (2, 1, 0, True)
    assert dry.to_json()["dry_run"] is True
    assert kit.rows_held(store, "clean") == (0, 0, 0)

    wet = recovery.recover_sanitizer(store, "clean", "chunk", step=STEP, log_path=log,
                                     log_input="chunk.jsonl")
    assert (wet.edges_written, wet.drops_written, wet.dry_run) == (2, 1, False)
    assert kit.rows_held(store, "clean") == (2, 1, 0)


# --- The degraded link (plan 3.3, F8) -------------------------------------------------------

TEAM = "team-sanitizer/1"


def team_sanitizer(text: str) -> str:
    return " ".join(text.lower().split())


@pytest.fixture
def team_store(folder, monkeypatch):
    monkeypatch.setitem(normalise._REGISTRY, TEAM, team_sanitizer)
    opened = kit.new_store(folder, TEAM)
    yield opened
    opened.close()


def sanitized(records):
    return [kit.Rec(record.origin, json.dumps({"text": team_sanitizer(record.text)}))
            for record in records if team_sanitizer(record.text)]


def test_without_the_normaliser_the_callers_word_alone_gives_no_file_level_link(
    team_store, folder, monkeypatch
):
    """No sanitizer's log is given. That the later file was written from the
    source file is then the caller's statement and nothing else, and the link
    does not degrade to file-level on it: the outcome is "unknown"."""
    store = team_store
    a = kit.source(store, folder, "A", ["  First  Line ", "Second Line"])
    kit.found_file(store, folder, "sanitized", sanitized(a))
    monkeypatch.delitem(normalise._REGISTRY, TEAM)

    result = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP)
    assert result.outcome == recovery.OUTCOME_UNKNOWN
    assert "the caller's statement alone" in result.reason
    assert (result.members_linked, result.links_written, result.wrote_nothing) == ((), 0, True)
    assert kit.rows_held(store, "sanitized") == (0, 0, 0)
    answer = kit.ask(store, "A", "sanitized")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)


def test_without_the_normaliser_a_log_that_does_not_fit_gives_no_file_level_link(
    team_store, folder, monkeypatch
):
    store = team_store
    a = kit.source(store, folder, "A", ["First Line", "   ", "Second Line", "Third Line"])
    kit.found_file(store, folder, "sanitized", sanitized(a)[:2])  # one line short
    log_path = kit.write_log(folder / "pipeline/sanitizer.log.jsonl", [
        {"line": 2, "reason": "empty line", "detail": "empty after sanitizing"}])
    monkeypatch.delitem(normalise._REGISTRY, TEAM)

    result = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP, log_path=log_path)
    assert result.outcome == recovery.OUTCOME_UNKNOWN
    assert "names 1 of the 4 lines of src/A and this file holds 2" in result.reason
    # The log's drop is the step's own statement and is recorded either way.
    assert (result.links_written, result.drops_written) == (0, 1)
    assert kit.rows_held(store, "sanitized") == (0, 1, 0)


def test_without_the_normaliser_the_step_s_own_log_supports_the_file_level_link(
    team_store, folder, monkeypatch
):
    store = team_store
    a = kit.source(store, folder, "A", ["First Line", "   ", "Second Line"])
    kit.found_file(store, folder, "sanitized", sanitized(a))
    log_path = kit.write_log(folder / "pipeline/sanitizer.log.jsonl", [
        {"line": 2, "reason": "empty line", "detail": "empty after sanitizing"}])
    monkeypatch.delitem(normalise._REGISTRY, TEAM)

    dry = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP, log_path=log_path,
                                     write=False)
    assert (dry.outcome, dry.links_written, dry.drops_written) == (recovery.OUTCOME_FILE_LEVEL, 0, 0)
    assert kit.rows_held(store, "sanitized") == (0, 0, 0)
    result = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP, log_path=log_path)
    assert (result.outcome, result.members_linked, result.links_written) == (
        recovery.OUTCOME_FILE_LEVEL, ("A/r1",), 1)
    (link,) = store.file_source_links(store.artifact("sanitized").instance_id)
    # The evidence names what the link rests on: the log's count.
    assert "by the step's own log: 1 of its 3 lines are logged as skipped" in link.evidence
    assert "this file holds 2" in link.evidence


def test_without_the_normaliser_a_file_family_of_the_later_file_is_the_link(
    team_store, folder, monkeypatch
):
    """The later file carries a file family of its own. That is the file-level
    link; the degraded run reports it and writes nothing."""
    store = team_store
    a = kit.source(store, folder, "A", ["First Line", "Second Line"])
    rows = [json.dumps({"text": team_sanitizer(r.text)}) for r in a]
    path = kit.write_lines(folder / "pipeline/sanitized.jsonl", rows)
    capture.import_jsonl(store, path, name="sanitized", member_name="sanitized",
                         file_family=("A/r1", "file family: sanitized lines of source A"))
    monkeypatch.delitem(normalise._REGISTRY, TEAM)

    result = recovery.recover_sanitizer(store, "sanitized", "src/A", step=STEP)
    assert (result.outcome, result.members_linked) == (recovery.OUTCOME_FILE_LEVEL, ("A/r1",))
    assert (result.links_written, result.wrote_nothing) == (0, True)
    assert kit.rows_held(store, "sanitized") == (0, 0, 1)
