"""The rows of Check 2 that read what ``register-run`` printed and what ``replay`` wrote
(``World._registration``), put to the world with both replaced by documents written by hand.

``register-run`` prints the replay it tried, and ``replay`` writes the arrays it derived
from a run record, only in a tool that has those forms; this file was written against one
that has not. The documents have the shape the tool's contract gives them. Each row is put
once with what is right for it and once with the one thing it asks changed.
"""

from __future__ import annotations

import dataclasses
import json

import acceptance_runs as runs
import numpy as np
import pytest
import table
import test_trace as t

SPEC = runs.spec()
SEGMENTS = ["E/seg-1", "E/seg-2", "D/seg-1", "Y/seg-1"]


def question(name: str, check: int = 2) -> table.Question:
    row = next(r for r in table.load_check(check) if r.id == name)
    return table.interpret(row)


def printed(route="none", cause="data-order trace missing", detail="digest mismatch",
            replay="absent", attempted=True, accepted=False) -> dict:
    out = {"segment": "x", "route": route, "cause": cause, "cause_detail": detail,
           "steps": 6, "traced_steps": 0, "untraced_steps": [], "checkpoints": []}
    if replay != "absent":
        out["replay"] = {"attempted": attempted, "accepted": accepted, "reason": detail}
    return out


@pytest.fixture
def registered(world, monkeypatch):
    state = {"printed": {}}
    monkeypatch.setattr(world, "registered", lambda store, st="unmutated": state["printed"])
    return state


def test_a_replay_rejected_for_a_digest_that_differs_is_the_object_the_row_asks_for(
        world, registered, monkeypatch):
    q = question("c02.ii.replay-rejected")
    q = dataclasses.replace(q, expect={k: v for k, v in q.expect.items() if k != "digests"})
    registered["printed"] = {n: printed(replay="here") for n in SEGMENTS}
    assert world._registration(q) == []
    wrong = {n: printed(replay="here", attempted=False, accepted=None) for n in SEGMENTS}
    registered["printed"] = wrong
    found = world._registration(q)
    assert found and found[0].field.startswith("replay object")


def test_a_replay_not_attempted_for_want_of_a_recorded_digest(world, registered):
    q = question("c02.iii.no-replay")
    ok = printed(detail="no recorded digest", replay="here", attempted=False, accepted=None)
    registered["printed"] = {n: dict(ok) for n in SEGMENTS}
    assert world._registration(q) == []
    registered["printed"]["E/seg-2"] = printed(detail="no recorded digest", replay="here",
                                               attempted=True, accepted=False)
    found = world._registration(q)
    assert len(found) == 1
    found = found[0]
    assert "E/seg-2" in found.field


def test_a_tool_that_prints_no_replay_object_fails_the_row_and_a_wrong_route_too(
        world, registered):
    # The tool prints the replay object (docs/cli.md): its absence is a failure of the
    # row, never a skip.
    q = question("c02.iii.no-replay")
    registered["printed"] = {n: printed(detail="no recorded digest") for n in SEGMENTS}
    assert [m.field for m in world._registration(q)] == [
        "register-run's replay object (attempted, accepted, reason)"]
    registered["printed"]["D/seg-1"] = printed(route="cache files", cause=None, detail=None)
    found = world._registration(q)
    assert found and "D/seg-1" in found[0].field


# --- Check 2 (ii): the digests of the replayed arrays ---------------------------------------


def arrays(equal: list[bool]) -> dict:
    return {"segment": "E/seg-1", "arrays": [
        {"kind": "document_index", "shard": n, "file": f"document_index-{n}.npy",
         "digest": "a" if flag else "b", "recorded_digest": "a", "equal": flag}
        for n, flag in enumerate(equal)]}


def test_digests_that_differ_from_the_recorded_ones_are_found_and_equal_ones_are_not(
        world, registered, monkeypatch):
    q = question("c02.ii.replay-rejected")
    registered["printed"] = {n: printed(replay="here") for n in SEGMENTS}
    monkeypatch.setattr(world, "_replay", lambda store, state, name: (arrays([False, True]), None))
    assert world._registration(q) == []
    monkeypatch.setattr(world, "_replay", lambda store, state, name: (arrays([True, True]), None))
    found = world._registration(q)
    assert found and "every replayed array has its recorded digest" in str(found[0].got)
    monkeypatch.setattr(world, "_replay", lambda store, state, name: (
        {"arrays": [{**arrays([False])["arrays"][0], "equal": True}]}, None))
    assert any("equal agrees" in str(m.expected) for m in world._registration(q))


# --- Check 2 (i): the replayed arrays against the kept cache -------------------------------


@pytest.fixture
def kept_cache(world, monkeypatch, tmp_path):
    """The store's folder holds the replica's run record and cache for E/seg-1, as the run
    wrapper writes them; the world is told where."""
    runs_dir = tmp_path / "stores" / "main" / "runs" / "E"
    runs_dir.mkdir(parents=True)
    made = t.write_cache(runs_dir / "E-seg-1.cache")
    (runs_dir / "E-seg-1.run.json").write_text(json.dumps(t.RECORD))

    class Built:
        store = tmp_path / "stores" / "main" / "store.sqlite"

    monkeypatch.setattr(world, "store", lambda name: Built)
    return made, tmp_path


def test_the_arrays_replay_wrote_are_the_kept_ones(world, registered, monkeypatch, kept_cache):
    made, tmp_path = kept_cache
    out = tmp_path / "replayed"
    printed_doc = t.write_replay(out, made)
    monkeypatch.setattr(world, "_replay", lambda store, state, name: (printed_doc, out))
    q = question("c02.i.arrays.E/seg-1")
    registered["printed"] = {"E/seg-1": printed(route="none", replay="here")}
    assert world._registration(q) == []


def test_one_array_that_differs_from_the_kept_file_fails_the_row(world, registered, monkeypatch,
                                                                  kept_cache):
    made, tmp_path = kept_cache
    out = tmp_path / "replayed"
    changed = dict(made)
    changed["dataset_sample_index", None] = made["dataset_sample_index", None] + 1
    printed_doc = t.write_replay(out, changed)
    monkeypatch.setattr(world, "_replay", lambda store, state, name: (printed_doc, out))
    q = question("c02.i.arrays.E/seg-1")
    registered["printed"] = {"E/seg-1": printed(route="none", replay="here")}
    found = world._registration(q)
    assert len(found) == 1
    found = found[0]
    assert found.field == "replayed_arrays (E/seg-1)"
    assert "dataset_sample_index" in str(found.got)
    assert np.array_equal(made["dataset_sample_index", None], made["dataset_sample_index", None])
