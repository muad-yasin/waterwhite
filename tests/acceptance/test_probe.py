"""The probe of what ``show`` prints and the gate on the sandbox, tested on
documents and commands written by hand. The probe itself runs the command line on
a scratch store; here only its judgment is tried, so a change of what ``show``
prints cannot change these tests."""

from __future__ import annotations

import acceptance_probe as probe
import pytest
import table
from table import Outcome

SETTINGS = {"namespace": "capture", "declared_count": 2, "append_eod": True, "eod_in_span": True}
PRINTED = {"drops": [], "bindings": [SETTINGS, {**SETTINGS, "namespace": "recovery"}]}
USAGE = Outcome(1, "", '{"error": "usage", "detail": "unrecognized arguments"}')
REFUSED = Outcome(2, "", '{"error": "request refused", "detail": "it holds no map"}')


def test_what_show_prints_takes_its_need_out_of_the_waiting_ones():
    assert probe.classify_show(PRINTED, REFUSED) == {}
    assert set(probe.classify_show({"bindings": []}, USAGE)) == {
        table.NEED_SHOW_MAP_ROWS, table.NEED_SHOW_DROPS, table.NEED_SHOW_MAP_SETTINGS}


def test_each_output_is_told_alone():
    assert set(probe.classify_show(PRINTED, USAGE)) == {table.NEED_SHOW_MAP_ROWS}
    assert set(probe.classify_show({**PRINTED, "drops": None}, REFUSED)) == {
        table.NEED_SHOW_DROPS}
    one_short = {**PRINTED, "bindings": [SETTINGS, {"namespace": "recovery"}]}
    assert set(probe.classify_show(one_short, REFUSED)) == {table.NEED_SHOW_MAP_SETTINGS}
    half = {**PRINTED, "bindings": [{k: v for k, v in SETTINGS.items() if k != "eod_in_span"}]}
    assert set(probe.classify_show(half, REFUSED)) == {table.NEED_SHOW_MAP_SETTINGS}
    # A file with no binding cannot show that a binding carries the settings.
    assert set(probe.classify_show({**PRINTED, "bindings": []}, REFUSED)) == {
        table.NEED_SHOW_MAP_SETTINGS}


def test_the_probe_runs_the_command_line_and_agrees_with_what_it_prints():
    # The probe is not a list: it asks the tool. Whatever it finds is a subset of the
    # needs it names, each with the words that the skipped rows carry.
    found = probe.show_gaps()
    assert set(found) <= set(probe.REASONS)
    assert all(found[need] == probe.REASONS[need] for need in found)


def question(state="unmutated", artifact="gen/S", **expect):
    row = table.Row(id="c05.x", check=5, question="", plan="", core=True, extension=None,
                    given={"state": state, "artifact": artifact}, expect=expect, expect_under={},
                    source="", rule=None)
    return table.Question(row=row, kind=table.KIND_ANSWER, store="main", state=state,
                          request=table.REQUESTS["withdraw A"], artifacts=(artifact,),
                          record=None, expect=dict(expect))


def test_the_rows_that_rest_on_a_bound_are_the_ones_that_name_one():
    assert probe.rests_on_bound(question(answer="does not contain"))  # a generation output
    assert probe.rests_on_bound(question(artifact="R1", answer="does not contain"))
    assert probe.rests_on_bound(question(state="g.bound-field", artifact="E/shard-3",
                                         answer="may contain"))
    assert probe.rests_on_bound(question(artifact="R1", answer="may contain",
                                         failing={"condition": "g", "job": "summary"}))
    assert probe.rests_on_bound(question(artifact="R2", answer="does not contain",
                                         scope={"generation_edges": ["summary"]}))
    # A positive answer, and a negative for a file no job wrote, need no bound.
    assert not probe.rests_on_bound(question(answer="contains"))
    assert not probe.rests_on_bound(question(artifact="R2", answer="does not contain"))
    assert not probe.rests_on_bound(question(artifact="gen/P", answer="may contain",
                                             failing={"condition": "a"}))


@pytest.fixture
def no_bwrap(monkeypatch, tmp_path):
    """A command named bwrap that fails, first on PATH, as bubblewrap does where user
    namespaces are off."""
    fake = tmp_path / "bin" / "bwrap"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho 'bwrap: No permissions to create new namespace' >&2\n"
                    "exit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))
    probe.sandbox_missing.cache_clear()
    yield
    monkeypatch.undo()
    probe.sandbox_missing.cache_clear()


def test_where_the_sandbox_cannot_be_made_the_reason_is_the_tools_own(no_bwrap):
    reason = probe.sandbox_missing()
    assert reason is not None and "bwrap" in reason


def with_argument(folder, store, *, generation_harness="in-process"):
    return None


def without_argument(folder, store):
    return None


def test_a_store_is_built_confined_where_the_replica_can_and_the_sandbox_can():
    probe.sandbox_missing.cache_clear()
    assert probe.generation_arguments(without_argument) == {}
    expected = "confined" if probe.sandbox_missing() is None else "in-process"
    assert probe.generation_arguments(with_argument) == {"generation_harness": expected}


def test_no_store_is_built_confined_where_there_is_no_sandbox(no_bwrap):
    assert probe.generation_arguments(with_argument) == {"generation_harness": "in-process"}


# --- show SEGMENT --trace and replay ---------------------------------------------------------


def test_a_parser_that_does_not_know_the_trace_flag_leaves_the_form_waiting():
    assert set(probe.classify_trace(USAGE)) == {table.NEED_SHOW_TRACE}


def test_a_form_that_refuses_a_name_that_is_no_segment_is_there():
    assert probe.classify_trace(REFUSED) == {}


def test_any_other_ending_of_the_trace_probe_is_an_error_of_the_probe_and_not_a_skip():
    with pytest.raises(probe.ProbeError, match="exit code 0"):
        probe.classify_trace(Outcome(0, "{}", ""))
    with pytest.raises(probe.ProbeError, match="exit code 2"):
        probe.classify_trace(Outcome(2, "", '{"error": "invalid record"}'))
    with pytest.raises(probe.ProbeError, match="exit code 1"):
        probe.classify_trace(Outcome(1, "", "Traceback (most recent call last)"))


INVALID_CHOICE = Outcome(
    1, "", '{"error": "usage", "detail": "argument <command>: invalid choice: \'replay\' '
           '(choose from \'init\', \'query\')"}')


def test_a_command_the_parser_does_not_know_is_absent_and_one_that_prints_its_help_is_there():
    found = probe.classify_command(INVALID_CHOICE, "replay", table.NEED_REPLAYED_ARRAYS)
    assert set(found) == {table.NEED_REPLAYED_ARRAYS}
    assert probe.classify_command(Outcome(0, "usage: ...", ""), "replay",
                                  table.NEED_REPLAYED_ARRAYS) == {}


def test_a_command_that_is_broken_is_an_error_of_the_probe_and_another_word_is_no_answer():
    with pytest.raises(probe.ProbeError):
        probe.classify_command(Outcome(1, "", "ImportError: no module named numpy"), "replay",
                               table.NEED_REPLAYED_ARRAYS)
    with pytest.raises(probe.ProbeError):   # the parser refused another word
        probe.classify_command(INVALID_CHOICE, "register-run", table.NEED_REPLAYED_ARRAYS)


def test_the_two_forms_carry_the_words_the_skipped_rows_name():
    assert table.NEED_SHOW_TRACE == "show --trace"
    assert table.NEED_REPLAYED_ARRAYS == "waterwhite replay"
    assert {table.NEED_SHOW_TRACE, table.NEED_REPLAYED_ARRAYS} <= set(probe.REASONS)
