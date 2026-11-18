"""The set-ups of Check 4's scratch store (``acceptance_scratch``) on a small store written by
hand, and the words the rows compare against what ``show`` prints.

The table's store ``scratch-500`` is built by the replica and needs Megatron. The shard
here is one of 500 one-token documents with the same names (``mini_store.build_scratch``),
so the writer that is killed, the retry, the refused count and the edit of the stored
count are tried with no Megatron, in seconds, through the tool's own commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import acceptance_scratch as scratch
import mini_store
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")


def cli(store, *arguments) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "waterwhite.cli", *arguments, "--store", str(store)],
        capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONPATH": SRC})


def shown(store) -> dict:
    done = cli(store, "show", scratch.SHARD)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# --- The words of the rows, against what show prints ----------------------------------------

DIGEST = "d" * 64


def doc(bindings=(), rows=None) -> dict:
    return {"bindings": list(bindings), "stored_rows": rows, "content_digest": DIGEST}


def binding(count=500, digest=DIGEST) -> dict:
    return {"declared_count": count, "declared_digest": digest}


def test_a_shard_with_no_binding_and_no_rows_holds_no_map():
    assert scratch.held(doc()) == {scratch.NO_MAP}


def test_a_complete_map_is_the_complete_map_and_the_one_that_matches_its_shard():
    assert scratch.held(doc([binding()], 500)) == {scratch.COMPLETE, scratch.MATCHES}


def test_a_map_with_fewer_rows_than_the_shard_has_documents_is_partial():
    assert scratch.held(doc([binding()], 100)) == {scratch.PARTIAL, scratch.PARTIAL_MAP}


def test_rows_with_no_binding_are_named_and_a_binding_that_declares_another_count_mismatches():
    assert scratch.held(doc([], 100)) == {scratch.ROWS_WITHOUT_BINDING}
    assert scratch.held(doc([binding(count=499)], 500)) == {scratch.MISMATCHES}
    assert scratch.held(doc([binding(digest="e" * 64)], 500)) == {scratch.MISMATCHES}


def test_a_named_refusal_is_a_rejection_and_another_ending_is_not():
    assert scratch.held(doc(), "declared count mismatch") == {scratch.NO_MAP, scratch.REJECTED}
    assert scratch.held(doc(), "declared digest mismatch") == {scratch.NO_MAP, scratch.REJECTED}
    assert scratch.held(doc(), "invalid record") == {scratch.NO_MAP}


def test_the_sentences_the_world_checks_are_the_ones_of_the_table():
    import table

    used = set()
    for row in table.load_check(4):
        for key in ("one_of", "never"):
            used |= set(row.expect.get(key, ()))
        if "map_after" in row.expect:
            used.add(row.expect["map_after"])
    assert used <= scratch.SENTENCES, sorted(used - scratch.SENTENCES)


# --- The set-ups, on the small store ---------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return mini_store.build_scratch(tmp_path / "s").store


def test_the_writer_killed_after_100_rows_leaves_no_map(store):
    scratch.kill_writer(store, 100)
    assert scratch.held(shown(store)) == {scratch.NO_MAP}
    assert cli(store, "check").returncode == 0


def test_the_write_retried_after_the_kill_gives_the_complete_map(store):
    scratch.kill_writer(store, 100)
    done = scratch.write_complete_map(store)
    assert scratch.refusal(done) is None
    assert scratch.held(shown(store)) == {scratch.COMPLETE, scratch.MATCHES}


def test_a_writer_that_is_not_killed_is_an_error_of_the_set_up(store):
    with pytest.raises(scratch.ScratchError, match="was to be killed"):
        scratch.kill_writer(store, 100000)


def test_499_declared_documents_are_refused_by_name_and_leave_no_map(store):
    assert scratch.declare_wrong_count(store, 499) == "declared count mismatch"
    assert scratch.held(shown(store)) == {scratch.NO_MAP}
    answered = cli(store, "query", "withdraw", "A", "--artifact", scratch.SHARD)
    printed = json.loads(answered.stdout)
    assert (printed["answer"], printed["reason"]) == ("not covered", "record missing")


def test_a_write_that_is_accepted_is_an_error_of_the_set_up(store):
    with pytest.raises(scratch.ScratchError, match="accepted 500 declared"):
        scratch.declare_wrong_count(store, 500)


def test_the_edited_count_stops_the_query_as_a_loud_check(store):
    scratch.count_edit(store)
    assert shown(store)["bindings"][0]["declared_count"] == 499
    answered = cli(store, "query", "withdraw", "A", "--artifact", scratch.SHARD)
    assert answered.returncode == 3 and answered.stdout == ""
    assert json.loads(answered.stderr)["error"] == "map count mismatch"
