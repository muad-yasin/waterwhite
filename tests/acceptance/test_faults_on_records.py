"""The faults on a tokenization map, a step trace, an edge parent and a file
(fixture/table/README.md section 7), put on a small store written by hand
(``mini_store.py``) and read back from the SQLite file.

The store the replica builds has these records only where Megatron tokenizes the
shards and the run wrapper has written the runs; the rows that ask them are graded
there. Here each fault is shown to change exactly what the catalogue names, in
seconds and with no Megatron, and to refuse a target the store does not hold.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing

import acceptance_faults
import acceptance_scratch
import mini_store
import pytest
from table import RecordRef

P = RecordRef("gen/P", "P")
X_AT_A = RecordRef("src/A", "X@A")


@pytest.fixture
def mini(tmp_path):
    return mini_store.build(tmp_path / "original")


def faulted(mini, tmp_path, state: str):
    """The mini store's folder copied and the fault put on the copy: (the copy's folder,
    its store file). The original is read to see that it was left alone."""
    folder = tmp_path / f"copy-{state}"
    shutil.copytree(mini.folder, folder)
    where = {P: (mini.ids["gen/P"], "L1"), X_AT_A: (mini.ids["src/A"], "L1")}
    acceptance_faults.apply(state, folder / "store.sqlite", lambda ref: where[ref])
    return folder, folder / "store.sqlite"


def rows(path, sql: str, *args):
    with closing(sqlite3.connect(path)) as db:
        return db.execute(sql, args).fetchall()


def binding(path, shard: str):
    return rows(path, "SELECT b.binding_id, b.declared_count, b.declared_digest, b.label, "
                      "b.derivation, b.verification FROM binding b JOIN artifact_name n "
                      "USING (artifact_id) WHERE n.name = ?", shard)


def artifact_digest(path, name: str):
    return rows(path, "SELECT content_digest FROM artifact JOIN artifact_name USING "
                      "(artifact_id) WHERE name = ?", name)


def test_the_mini_store_holds_what_the_faults_start_from(mini):
    for shard in ("E/shard-1", "E/shard-2", "E/shard-3"):
        ((_, count, _, label, derivation, verification),) = binding(mini.store, shard)
        assert (count, label, derivation, verification) == (3, "recovered", "position", "passed")
    assert rows(mini.store, "SELECT COUNT(*), MIN(consumed_samples) FROM step_trace") == [(6, 4)]
    assert rows(mini.store, "SELECT COUNT(*) FROM trace_sample WHERE covered_tokens < 33") == [(0,)]


def test_map_deleted_removes_the_capture_map_of_shard_1_and_nothing_else(mini, tmp_path):
    _, store = faulted(mini, tmp_path, "f.map-deleted")
    assert binding(store, "E/shard-1") == []
    assert rows(store, "SELECT COUNT(*) FROM map_row WHERE binding_id = ?",
                binding(mini.store, "E/shard-1")[0][0]) == [(0,)]
    for shard in ("E/shard-2", "E/shard-3"):
        assert binding(store, shard) == binding(mini.store, shard)
    # The shard's artifact row and the snapshot that names it stay.
    assert rows(store, "SELECT COUNT(*) FROM artifact_name WHERE name = 'E/shard-1'") == [(1,)]
    assert rows(store, "SELECT COUNT(*) FROM mixture_shard") == [(2,)]
    assert rows(store, "SELECT COUNT(*) FROM map_row") == [(6,)]


def test_digest_replaces_the_first_hex_digit_of_shard_3s_map_digest_only(mini, tmp_path):
    _, store = faulted(mini, tmp_path, "f.digest")
    ((_, _, before, *_),) = binding(mini.store, "E/shard-3")
    ((_, _, after, label, derivation, verification),) = binding(store, "E/shard-3")
    assert after != before and after[1:] == before[1:] and after[0] != before[0]
    assert (label, derivation, verification) == ("recovered", "position", "passed")
    assert binding(store, "E/shard-1") == binding(mini.store, "E/shard-1")
    assert binding(store, "E/shard-2") == binding(mini.store, "E/shard-2")
    # The artifact's own digest is what the check compares the map's with, and it stays.
    assert artifact_digest(store, "E/shard-3") == artifact_digest(mini.store, "E/shard-3")


def test_label_sets_the_label_of_shard_2s_map_and_leaves_derivation_verification_and_rows(
        mini, tmp_path):
    _, store = faulted(mini, tmp_path, "f.label")
    ((_, _, digest, label, derivation, verification),) = binding(store, "E/shard-2")
    assert (label, derivation, verification) == ("observed", "position", "passed")
    assert digest == binding(mini.store, "E/shard-2")[0][2]
    assert rows(store, "SELECT DISTINCT label FROM map_row") == [("recovered",)]
    assert binding(store, "E/shard-1")[0][3] == binding(store, "E/shard-3")[0][3] == "recovered"


def test_trace_deleted_removes_the_trace_of_step_3_and_keeps_the_registered_step(mini, tmp_path):
    _, store = faulted(mini, tmp_path, "f.trace-deleted")
    assert rows(store, "SELECT step FROM step_trace ORDER BY step") == [(n,) for n in (1, 2, 4, 5, 6)]
    assert rows(store, "SELECT COUNT(*) FROM trace_sample") == [(20,)]
    assert rows(store, "SELECT COUNT(*) FROM trace_row") == [(20,)]
    assert rows(store, "SELECT COUNT(*) FROM segment_step WHERE step = 3") == [(1,)]


def test_trace_half_lowers_the_last_sample_of_step_4_and_keeps_the_count(mini, tmp_path):
    _, store = faulted(mini, tmp_path, "f.trace-half")
    lowered = rows(store, "SELECT t.step, s.sample_pos, s.covered_tokens, s.sample_tokens "
                          "FROM trace_sample s JOIN step_trace t USING (trace_id) "
                          "WHERE s.covered_tokens < s.sample_tokens")
    assert lowered == [(4, 3, 16, 33)]
    assert rows(store, "SELECT consumed_samples FROM step_trace WHERE step = 4") == [(4,)]
    assert rows(store, "SELECT COUNT(*) FROM trace_sample") == [(24,)]


def test_edge_parent_points_the_edge_of_p_at_no_occurrence_and_deletes_no_occurrence(
        mini, tmp_path):
    _, store = faulted(mini, tmp_path, "l.edge-parent")
    named = rows(store, "SELECT occurrence_id FROM edge_input WHERE edge_id = "
                        "(SELECT edge_id FROM edge WHERE is_generation = 1)")
    assert len(named) == 1
    assert rows(store, "SELECT COUNT(*) FROM occurrence WHERE occurrence_id = ?",
                named[0][0]) == [(0,)]
    assert rows(store, "SELECT COUNT(*) FROM occurrence") == rows(
        mini.store, "SELECT COUNT(*) FROM occurrence")


def test_trace_count_deletes_the_last_sample_of_step_5_and_sets_the_count_to_3(mini, tmp_path):
    _, store = faulted(mini, tmp_path, "l.trace-count")
    assert rows(store, "SELECT consumed_samples FROM step_trace WHERE step = 5") == [(3,)]
    assert rows(store, "SELECT COUNT(*) FROM trace_sample s JOIN step_trace t USING (trace_id) "
                       "WHERE t.step = 5") == [(3,)]
    assert rows(store, "SELECT COUNT(*) FROM trace_row r JOIN step_trace t USING (trace_id) "
                       "WHERE t.step = 5") == [(3,)]
    assert rows(store, "SELECT COUNT(*) FROM trace_sample") == [(23,)]


def test_artifact_digest_changes_one_byte_of_the_copys_file_of_r2_and_nothing_else(mini, tmp_path):
    folder, store = faulted(mini, tmp_path, "l.artifact-digest")
    before = (mini.folder / "R2.jsonl").read_bytes()
    after = (folder / "R2.jsonl").read_bytes()
    assert len(after) == len(before) and after.count(b"\n") == before.count(b"\n")
    assert sum(a != b for a, b in zip(before, after)) == 1
    # The original is untouched, and the store file is not edited.
    assert (mini.folder / "R2.jsonl").read_bytes() == before
    assert artifact_digest(store, "R2") == artifact_digest(mini.store, "R2")


def test_a_file_fault_refuses_a_path_that_leaves_the_copy(mini, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(mini.folder, folder)
    with closing(sqlite3.connect(folder / "store.sqlite")) as db, db:
        db.execute("UPDATE artifact SET path = ? WHERE artifact_id = "
                   "(SELECT artifact_id FROM artifact_name WHERE name = 'R2')",
                   (str(mini.folder / "R2.jsonl"),))
    original = (mini.folder / "R2.jsonl").read_bytes()
    with pytest.raises(acceptance_faults.FaultError, match="outside the copy"):
        acceptance_faults.apply("l.artifact-digest", folder / "store.sqlite", lambda ref: ("", ""))
    assert (mini.folder / "R2.jsonl").read_bytes() == original


def test_a_fault_on_a_record_the_store_does_not_hold_is_an_error(mini, tmp_path):
    _, store = faulted(mini, tmp_path, "f.map-deleted")
    with pytest.raises(acceptance_faults.FaultError, match="the map of E/shard-1"):
        acceptance_faults.apply("f.map-deleted", store, lambda ref: ("", ""))
    _, store = faulted(mini, tmp_path, "f.trace-deleted")
    with pytest.raises(acceptance_faults.FaultError, match="the trace of step 3"):
        acceptance_faults.apply("f.trace-deleted", store, lambda ref: ("", ""))


def test_a_fault_that_must_leave_a_row_standing_refuses_a_store_where_it_is_not(mini, tmp_path):
    folder = tmp_path / "copy-step"
    shutil.copytree(mini.folder, folder)
    with closing(sqlite3.connect(folder / "store.sqlite")) as db, db:
        db.execute("DELETE FROM segment_step WHERE step = 3")
    with pytest.raises(acceptance_faults.FaultError, match="the registered step 3"):
        acceptance_faults.apply("f.trace-deleted", folder / "store.sqlite", lambda ref: ("", ""))
    # The fault ran in one transaction: it left the trace where it was.
    assert rows(folder / "store.sqlite", "SELECT COUNT(*) FROM step_trace WHERE step = 3") == [(1,)]

    folder = tmp_path / "copy-shard"
    shutil.copytree(mini.folder, folder)
    with closing(sqlite3.connect(folder / "store.sqlite")) as db, db:
        db.execute("DELETE FROM artifact WHERE artifact_id = "
                   "(SELECT artifact_id FROM artifact_name WHERE name = 'E/shard-1')")
    with pytest.raises(acceptance_faults.FaultError, match="the shard's artifact row"):
        acceptance_faults.apply("f.map-deleted", folder / "store.sqlite", lambda ref: ("", ""))
    assert binding(folder / "store.sqlite", "E/shard-1") != []


# --- The faults of the scratch store ----------------------------------------------------------


@pytest.fixture
def scratch(tmp_path):
    return mini_store.build_scratch(tmp_path / "scratch")


def scratch_copy(scratch, tmp_path, state: str):
    folder = tmp_path / f"copy-{state}"
    shutil.copytree(scratch.folder, folder)
    acceptance_faults.apply(state, folder / "store.sqlite", lambda ref: ("", ""))
    return folder / "store.sqlite"


def test_declare_499_is_refused_by_name_and_leaves_the_store_without_a_map(scratch, tmp_path):
    store = scratch_copy(scratch, tmp_path, "w.declare-499")
    assert binding(store, "c04/shard-500") == []
    assert rows(store, "SELECT COUNT(*) FROM map_row") == [(0,)]
    # The original store was left alone, and its shard has no map either.
    assert binding(scratch.store, "c04/shard-500") == []


def test_count_499_sets_the_stored_count_of_the_complete_map_and_changes_no_other_value(
        scratch, tmp_path):
    store = scratch_copy(scratch, tmp_path, "l.count-499")
    (held,) = binding(store, "c04/shard-500")
    assert held[1] == 499
    assert rows(store, "SELECT COUNT(*) FROM map_row") == [(500,)]
    digest = artifact_digest(store, "c04/shard-500")
    assert digest == artifact_digest(scratch.store, "c04/shard-500")


def test_a_scratch_fault_on_a_store_without_the_shard_is_an_error(mini, tmp_path):
    folder = tmp_path / "copy-no-shard"
    shutil.copytree(mini.folder, folder)
    with pytest.raises(acceptance_scratch.ScratchError, match="0 artifacts named c04/shard-500"):
        acceptance_faults.apply("w.declare-499", folder / "store.sqlite", lambda ref: ("", ""))
