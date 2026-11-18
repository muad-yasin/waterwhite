"""Each fault of ``acceptance_faults`` changes what fixture/table/README.md section
7 says it changes, and nothing beside it on the same record.

The faults on edges and logs are read back from the store the replica built, through
``show``. The faults on a tokenization map, a step trace, an edge parent and a file are
put on a small store written by hand (``mini_store.py``), with no Megatron and no run,
and read back from the SQLite file; the runs' own stores are graded by the rows.

A fault that hit nothing would let every row that expects the unmutated answer
pass, so the state of each copy is read back here, through ``show --record`` of the
command line. Three of these faults (the bound field, the read added to the log,
the log removed) are asked only of shard 3 by the table and so by no row yet; this
is the one place they are exercised until shards are registered.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing

import acceptance_faults
import acceptance_probe
import mini_store
import pytest
from table import RecordRef

P, T, S = RecordRef("gen/P", "P"), RecordRef("gen/T", "T"), RecordRef("gen/S", "S")
STORE = "main"


def need_a_bound():
    """The tests that read the bound of a generation edge are not run where there
    is no sandbox: only a log of the confined harness establishes a bound (Decision
    13), so on such a machine there is no bound to read, and what a fault leaves of
    one would be read from nothing. A skip is not a pass."""
    missing = acceptance_probe.sandbox_missing()
    if missing is not None:
        pytest.skip(f"not run: a bound is established only by the confined harness, and "
                    f"the sandbox cannot be made: {missing}")


def edge(world, state: str, ref: RecordRef) -> dict | None:
    copied = world.copy(STORE, state)
    return world.shown(copied.store, ref.artifact, "--record", world.locator(STORE, ref))["edge"]


def named(parents: list[dict]) -> list[str]:
    return [f"{p['artifact']}:{p['record']}" for p in parents]


def at(world, ref: RecordRef) -> str:
    return f"{ref.artifact}:{world.locator(STORE, ref)}"


def test_every_fault_written_here_is_a_state_of_the_table_and_is_tested_here():
    import table

    assert acceptance_faults.FAULTS <= set(table.STATES)
    tested = {"f.edge-deleted", "f.edge-file-level", "g.bound-field", "g.log-outside-read",
              "g.log-removed", "g.model-entry-removed", "s.in-scope-only", "s.file-level-link-B",
              "f.trace-deleted", "f.trace-half", "f.map-deleted", "f.digest", "f.label",
              "l.edge-parent", "l.trace-count", "l.artifact-digest", "w.declare-499",
              "l.count-499"}
    assert acceptance_faults.FAULTS == tested


def test_the_unmutated_edges_are_what_the_faults_start_from(world):
    need_a_bound()
    p = edge(world, "unmutated", P)
    assert (p["label"], p["derivation"], p["harness_log"], p["bound_field"], p["bound_from_log"],
            p["model"]) == ("observed", "stated", True, True, True, "none")
    assert named(p["parents"]) == [at(world, RecordRef("src/A", "X@A"))]
    s = edge(world, "unmutated", S)
    assert named(s["parents"]) == [at(world, RecordRef("src/A", "SA")),
                                          at(world, RecordRef("src/C", "SC"))]
    assert s["in_scope_only"] == []


def test_edge_deleted_removes_the_edge_of_p_and_no_other(world):
    assert edge(world, "f.edge-deleted", P) is None
    t = edge(world, "f.edge-deleted", T)
    assert named(t["parents"]) == [at(world, P)] and t["harness_log"] is True


def test_edge_file_level_changes_the_label_and_derivation_of_the_edge_of_t(world):
    t = edge(world, "f.edge-file-level", T)
    assert (t["label"], t["derivation"]) == ("file-level", "file")
    assert named(t["parents"]) == [at(world, P)] and t["harness_log"] is True
    assert edge(world, "f.edge-file-level", P)["label"] == "observed"


def test_bound_field_clears_the_field_and_leaves_the_log(world):
    need_a_bound()
    p = edge(world, "g.bound-field", P)
    assert (p["bound_field"], p["harness_log"], p["bound_from_log"]) == (False, True, True)


def test_log_outside_read_breaks_the_bound_the_log_gives_and_leaves_the_field(world):
    need_a_bound()
    p = edge(world, "g.log-outside-read", P)
    assert (p["bound_field"], p["harness_log"], p["bound_from_log"]) == (True, True, False)


def test_log_removed_leaves_the_field_established_with_no_log(world):
    need_a_bound()
    p = edge(world, "g.log-removed", P)
    assert (p["bound_field"], p["harness_log"]) == (True, False)
    assert edge(world, "g.log-removed", T)["harness_log"] is True


def test_model_entry_removed_leaves_a_log_with_no_model_entry(world):
    need_a_bound()
    p = edge(world, "g.model-entry-removed", P)
    assert (p["model"], p["harness_log"], p["bound_from_log"]) == (None, True, True)


def test_in_scope_only_moves_the_passage_of_c_out_of_the_parents_of_s(world):
    s = edge(world, "s.in-scope-only", S)
    assert named(s["parents"]) == [at(world, RecordRef("src/A", "SA"))]
    assert named(s["in_scope_only"]) == [at(world, RecordRef("src/C", "SC"))]


def test_file_level_link_b_adds_a_link_and_keeps_the_observed_edge(world):
    copied = world.copy(STORE, "s.file-level-link-B")
    record = world.shown(copied.store, "gen/P", "--record", world.locator(STORE, P))
    links = [(link["member"], link["label"], link["evidence"]) for link in record["source_links"]]
    assert links == [("B/r1", "file-level", acceptance_faults.FILE_FAMILY_B)]
    assert record["edge"]["label"] == "observed"
    assert named(record["edge"]["parents"]) == [at(world, RecordRef("src/A", "X@A"))]


def test_a_fault_that_finds_no_target_is_an_error(world):
    copied = world.copy(STORE, "f.edge-deleted")
    with pytest.raises(acceptance_faults.FaultError, match="the edge of gen/P:P"):
        # The edge of P is gone from this copy, so the fault has nothing to put itself on.
        acceptance_faults.apply("g.bound-field", copied.store,
                                lambda ref: world.locate(STORE, ref))
    with pytest.raises(acceptance_faults.FaultError, match="no fault"):
        acceptance_faults.apply("f.unheard-of", copied.store,
                                lambda ref: world.locate(STORE, ref))
