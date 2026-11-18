"""The loud checks over the digests and numbers a store keeps of its own records
(docs/store.md "The loud checks", checks 6 to 9).

Each store is built by the write methods (the hand-built world of
``tests/coverage/ww_world.py`` and small stores of this file). An alteration is a direct
SQL edit of the file the write API wrote. A test compares one observed value with the
value a document or a named constant gives; none takes an expected value from the code
it tests.

The sentences the expectations rest on (docs/store.md):

- check 6: the digest registered with a run record is the digest of the text the store
  holds for it;
- check 7: a mixture's content digest, parameter digest and instance id follow from its
  snapshot entries (the shards' instance ids and digests, in order) and its settings; a
  mixture with no snapshot entry at all is a record that is gone ("not covered, record
  missing"), not a fault;
- check 8: the occurrence rows of a JSONL binding that declares n lines are L1 to Ln,
  the map rows of a shard's binding that declares n documents are ordinals 0 to n-1;
- check 9: one array of the blend has one recorded digest;
- the order of the table: these checks come after the five that were there, and a
  scope that holds no run segment, mixture or binding does not meet them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from waterwhite import errors
from waterwhite import vocabulary as v
from waterwhite.records import SnapshotShard
from waterwhite.store.store import Store

REPO = Path(__file__).resolve().parents[2]


def _load_world():
    held = sys.modules.get("ww_world")
    if held is not None:
        return held
    spec = importlib.util.spec_from_file_location("ww_world", REPO / "tests" / "coverage" / "ww_world.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ww_world"] = module
    spec.loader.exec_module(module)
    return module


ww = _load_world()

# --- The expected values ------------------------------------------------------------------

NAME_STORED_DIGEST = "stored digest mismatch"
NAME_DOMAIN = "locator domain mismatch"
NAME_CACHE = "cache digest conflict"
NAME_ARTIFACT_DIGEST = "artifact digest mismatch"
NAME_TRACE_COUNT = "trace sample count mismatch"
EXIT_LOUD_FAULT = 3

RECORD_RUN = "run record"
RECORD_MIXTURE = "mixture snapshot"
RECORD_MIXTURE_SETTINGS = "mixture settings"


@pytest.fixture
def world(tmp_path):
    built = ww.build(tmp_path)
    yield built
    built.store.close()


def fault_of(store: Store, instance_ids=None):
    """The loud fault the check raises, or None."""
    try:
        store.check_integrity(instance_ids)
    except errors.IntegrityFault as fault:
        return fault
    return None


def named(fault) -> tuple | None:
    return None if fault is None else (fault.name, fault.exit_code)


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_record_text(world, segment: str) -> str:
    return world.scalar("SELECT run_record FROM run_segment WHERE name = ?", (segment,))


def altered_run_record(text: str) -> str:
    record = json.loads(text)
    record["settings"]["random_seed"] = 4321
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


# --- Check 6: the run record and its digest -----------------------------------------------


def test_control_the_world_passes_every_check(world):
    assert fault_of(world.store) is None


def test_a_run_record_changed_without_its_digest_is_a_stored_digest_mismatch(world):
    held = world.scalar("SELECT run_record_digest FROM run_segment WHERE name = 'E/seg-1'")
    changed = altered_run_record(run_record_text(world, "E/seg-1"))
    world.sql(("UPDATE run_segment SET run_record = ? WHERE name = 'E/seg-1'", (changed,)))
    fault = fault_of(world.store)
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault
    assert (fault.context["record"], fault.context["segment"]) == (RECORD_RUN, "E/seg-1")
    assert (fault.context["registered"], fault.context["computed"]) == (held, digest_of(changed))


def test_a_run_record_removed_while_its_digest_stays_is_a_stored_digest_mismatch(world):
    world.sql("UPDATE run_segment SET run_record = NULL WHERE name = 'E/seg-1'")
    fault = fault_of(world.store)
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault
    assert fault.context["computed"] is None


def test_a_digest_removed_while_its_run_record_stays_is_a_stored_digest_mismatch(world):
    world.sql("UPDATE run_segment SET run_record_digest = NULL WHERE name = 'E/seg-1'")
    fault = fault_of(world.store)
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault
    assert fault.context["registered"] is None


def test_a_segment_registered_with_no_run_record_has_no_digest_to_hold_it_against(tmp_path):
    with Store.create(tmp_path / "none.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        store.register_run_segment(
            run="N", segment="N/seg-1", position=0, mixture_instance_id=None, parent_checkpoint=None,
            start_kind="fresh", run_record=None, steps=[(1, 1, None)], trace_route=v.ROUTE_NONE,
            trace_cause=v.NAMES_TRACE_MISSING, trace_cause_detail=v.TRACE_MISSING_NO_DIGEST)
        assert fault_of(store) is None


def test_the_run_record_check_reaches_the_segments_of_a_checkpoints_history_only(world):
    world.sql(("UPDATE run_segment SET run_record = ? WHERE name = 'E/seg-1'",
               (altered_run_record(run_record_text(world, "E/seg-1")),)))
    for holder in ("E@3", "E@6", "D@2", "Y@2"):  # every history runs through E/seg-1
        assert named(fault_of(world.store, [world[holder]])) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), holder
    for outside in ("src/A", "gen/P", "R1", "shard-1", "mix-1"):  # no run segment in the closure
        assert fault_of(world.store, [world[outside]]) is None, outside


# --- Check 7: the mixture's identity ------------------------------------------------------


def mixture_entry(world, mixture: str, position: int):
    return world.store._conn.execute(
        "SELECT artifact_id, position, shard_instance_id, path, content_digest FROM mixture_shard "
        "WHERE position = ? AND artifact_id = (SELECT artifact_id FROM artifact WHERE instance_id = ?)",
        (position, world[mixture])).fetchone()


def test_a_snapshot_entry_removed_is_a_stored_digest_mismatch(world):
    world.sql(("DELETE FROM mixture_shard WHERE position = 1 AND artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-2"],)))
    fault = fault_of(world.store)
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault
    assert (fault.context["record"], fault.context["instance"]) == (RECORD_MIXTURE, world["mix-2"])


def test_a_snapshot_entry_naming_another_shard_is_a_stored_digest_mismatch(world):
    world.sql(("UPDATE mixture_shard SET shard_instance_id = ? WHERE position = 0 AND artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", ("f" * 64, world["mix-2"])))
    fault = fault_of(world.store, [world["mix-2"]])
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault


def test_the_digest_of_a_shard_the_store_was_never_told_of_is_held_by_the_mixture_id(world):
    ghost = SnapshotShard(0, "e" * 64, "shards/ghost", "d" * 64)
    world.ids["mix-ghost"] = world.store.register_mixture(name="mix-ghost", shards=[ghost],
                                                          settings={"mix": "ghost"})
    assert fault_of(world.store, [world["mix-ghost"]]) is None
    world.sql(("UPDATE mixture_shard SET content_digest = ? WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", ("c" * 64, world["mix-ghost"])))
    fault = fault_of(world.store, [world["mix-ghost"]])
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault


def test_settings_changed_after_registration_are_a_stored_digest_mismatch(world):
    world.sql(("UPDATE mixture_settings SET settings = ? WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", ('{"mix":3}', world["mix-2"])))
    fault = fault_of(world.store, [world["mix-2"]])
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault
    assert fault.context["record"] == RECORD_MIXTURE_SETTINGS


def test_settings_that_are_no_json_are_a_stored_digest_mismatch(world):
    world.sql(("UPDATE mixture_settings SET settings = 'not json' WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-2"],)))
    fault = fault_of(world.store, [world["mix-2"]])
    assert named(fault) == (NAME_STORED_DIGEST, EXIT_LOUD_FAULT), fault
    assert (fault.context["record"], fault.context["computed"]) == (RECORD_MIXTURE_SETTINGS, None)


def test_settings_in_another_layout_of_the_same_document_are_not_held_against_the_digest(world):
    # The text is not the canonical text of what it parses to, so it is not compared.
    world.sql(("UPDATE mixture_settings SET settings = '{\"mix\": 2}' WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-2"],)))
    assert fault_of(world.store, [world["mix-2"]]) is None


def test_a_mixture_whose_whole_snapshot_is_gone_is_a_record_that_is_gone_and_no_fault(world):
    world.sql(("DELETE FROM mixture_shard WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-1"],)))
    assert fault_of(world.store) is None
    assert fault_of(world.store, [world["mix-1"]]) is None


# --- Check 8: the numbers of the records of a binding -------------------------------------


def test_a_locator_changed_with_the_count_kept_is_a_locator_domain_mismatch(world):
    world.sql(("UPDATE occurrence SET locator = 'L999' WHERE occurrence_id = ?",
               (world.occurrence_id("src/A", 1),)))
    fault = fault_of(world.store)
    assert named(fault) == (NAME_DOMAIN, EXIT_LOUD_FAULT), fault
    assert (fault.context["instance"], fault.context["locator"], fault.context["declared"]) == (
        world["src/A"], "L999", 2)
    # What a query runs: the check over the closure of the artifact it asks about.
    assert named(fault_of(world.store, [world["gen/P"]])) == (NAME_DOMAIN, EXIT_LOUD_FAULT)
    assert fault_of(world.store, [world["src/Z"]]) is None


@pytest.mark.parametrize("bad", ["X1", "L0", "L", "L01", "L-1", "L1.5", "l1"])
def test_a_locator_that_is_no_line_number_is_a_locator_domain_mismatch(world, bad):
    world.sql(("UPDATE occurrence SET locator = ? WHERE occurrence_id = ?",
               (bad, world.occurrence_id("src/A", 1))))
    fault = fault_of(world.store, [world["src/A"]])
    assert named(fault) == (NAME_DOMAIN, EXIT_LOUD_FAULT), fault
    assert fault.context["locator"] == bad


def test_an_ordinal_past_the_documents_a_map_declares_is_a_locator_domain_mismatch(world):
    binding = world.binding_id("shard-1")
    world.sql(("UPDATE map_row SET ordinal = 99 WHERE ordinal = 0 AND binding_id = ?", (binding,)))
    fault = fault_of(world.store, [world["shard-1"]])
    assert named(fault) == (NAME_DOMAIN, EXIT_LOUD_FAULT), fault
    assert (fault.context["instance"], fault.context["ordinal"], fault.context["declared"]) == (
        world["shard-1"], 99, 3)


def test_an_artifact_whose_file_is_away_is_not_looked_at_by_the_check_of_numbers(world):
    # A gone file is no fault (DECISIONS.md D-18). Three of R1's six occurrence rows are
    # removed and its binding declares three fewer; the rows L3, L5 and L6 stay (shard-3's
    # map names them). With the file present the count of the file names the fault
    # (check 5). With the file away nothing is checked, as before; the rows that stay lie
    # outside L1 to L3 and are not found.
    r1 = world.scalar("SELECT artifact_id FROM artifact WHERE instance_id = ?", (world["R1"],))
    world.sql(("DELETE FROM occurrence WHERE artifact_id = ? AND locator IN ('L1', 'L2', 'L4')", (r1,)),
              ("UPDATE binding SET declared_count = declared_count - 3 WHERE artifact_id = ?", (r1,)))
    assert named(fault_of(world.store)) == ("map count mismatch", EXIT_LOUD_FAULT)
    os.rename(world.folder / "R1.jsonl", world.folder / "R1.archived")
    assert fault_of(world.store) is None
    assert fault_of(world.store, [world["R1"]]) is None


def test_a_locator_changed_in_the_store_is_found_when_the_file_is_back(world):
    world.sql(("UPDATE occurrence SET locator = 'L999' WHERE occurrence_id = ?",
               (world.occurrence_id("src/A", 1),)))
    os.rename(world.folder / "src/A.jsonl", world.folder / "src/A.archived")
    assert fault_of(world.store) is None
    os.rename(world.folder / "src/A.archived", world.folder / "src/A.jsonl")
    assert named(fault_of(world.store)) == (NAME_DOMAIN, EXIT_LOUD_FAULT)


# --- Check 9: one digest for one blend array ----------------------------------------------


def register_blend_segment(store: Store) -> None:
    record = ww.run_record("G", "G/seg-1", "fresh", 1, 1)
    record["cache_digests"] = [{"kind": v.CACHE_DATASET_INDEX, "shard": None, "digest": "a" * 64}]
    store.register_run_segment(
        run="G", segment="G/seg-1", position=0, mixture_instance_id=None, parent_checkpoint=None,
        start_kind="fresh", run_record=record, steps=[(1, 1, None)], trace_route=v.ROUTE_NONE,
        trace_cause=v.NAMES_TRACE_MISSING, trace_cause_detail=v.TRACE_MISSING_NO_DIGEST)


def add_blend_row(path: Path, digest: str) -> bool:
    """Write a second row for the blend array of G/seg-1 into a file whose unique index
    over a blend array (schema version 3) was removed, which is the one kind of store in
    which the check can meet two rows: with the index in place the file refuses the row,
    as tests/store/test_store_schema_3.py shows. True when the row was written."""
    raw = sqlite3.connect(path)
    try:
        raw.execute("DROP INDEX segment_cache_digest_one_blend")
        raw.execute(
            "INSERT INTO segment_cache_digest (segment_id, kind, shard_position, digest) VALUES "
            "((SELECT segment_id FROM run_segment WHERE name = 'G/seg-1'), 'dataset_index', NULL, ?)",
            (digest,))
        raw.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        raw.close()


def expected_cache_fault(inserted: bool):
    """The check names the conflict when the file holds two rows; when the file refuses
    the second row there is nothing to name."""
    return (NAME_CACHE, EXIT_LOUD_FAULT) if inserted else None


def test_two_digests_for_one_blend_array_are_a_cache_digest_conflict(tmp_path):
    path = tmp_path / "blend.db"
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store)
        assert fault_of(store) is None
    inserted = add_blend_row(path, "b" * 64)
    with Store.open(path) as store:
        fault = fault_of(store)
        assert named(fault) == expected_cache_fault(inserted), fault
        if inserted:
            assert (fault.context["segment"], fault.context["kind"]) == ("G/seg-1", v.CACHE_DATASET_INDEX)


def test_the_same_digest_twice_for_one_blend_array_is_a_cache_digest_conflict_too(tmp_path):
    path = tmp_path / "twice.db"
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store)
    inserted = add_blend_row(path, "a" * 64)
    with Store.open(path) as store:
        assert named(fault_of(store)) == expected_cache_fault(inserted)


def test_the_cache_check_reaches_a_segment_through_the_checkpoint_of_its_history(tmp_path):
    path = tmp_path / "ckpt.db"
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store)
        checkpoint = store.register_checkpoint(name="G@1", segment="G/seg-1", step=1)
    inserted = add_blend_row(path, "b" * 64)
    with Store.open(path) as store:
        fault = fault_of(store, [checkpoint])
        assert named(fault) == expected_cache_fault(inserted), fault
        if inserted:
            assert fault.context["segment"] == "G/seg-1"


def test_a_scope_with_no_run_segment_does_not_meet_the_cache_check(tmp_path):
    path = tmp_path / "scope.db"
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store)
    inserted = add_blend_row(path, "b" * 64)
    with Store.open(path) as store:
        assert fault_of(store, ["0" * 64]) is None
        assert named(fault_of(store)) == expected_cache_fault(inserted)


# --- The order of the table ---------------------------------------------------------------


def test_the_new_checks_come_after_the_five_that_were_there(world):
    # A run record altered beside a trace whose count is wrong: check 4 is named first.
    world.sql(("UPDATE run_segment SET run_record = ? WHERE name = 'E/seg-1'",
               (altered_run_record(run_record_text(world, "E/seg-1")),)),
              "UPDATE step_trace SET consumed_samples = 7 WHERE step = 1 AND segment_id = "
              "(SELECT segment_id FROM run_segment WHERE name = 'E/seg-1')")
    assert named(fault_of(world.store)) == (NAME_TRACE_COUNT, EXIT_LOUD_FAULT)
    # The count repaired, and a snapshot entry now carries another digest than its
    # registered shard, beside the altered run record: check 5 is named first.
    world.sql("UPDATE step_trace SET consumed_samples = 1 WHERE step = 1 AND segment_id = "
              "(SELECT segment_id FROM run_segment WHERE name = 'E/seg-1')",
              ("UPDATE mixture_shard SET content_digest = ? WHERE position = 0 AND artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", ("0" * 64, world["mix-2"])))
    fault = fault_of(world.store)
    assert named(fault) == (NAME_ARTIFACT_DIGEST, EXIT_LOUD_FAULT), fault


def test_the_new_checks_are_found_in_the_order_run_record_mixture_numbers_cache(world):
    original = run_record_text(world, "E/seg-1")
    removed = mixture_entry(world, "mix-2", 1)
    a1 = world.occurrence_id("src/A", 1)
    segment_id = world.scalar("SELECT segment_id FROM run_segment WHERE name = 'E/seg-1'")
    world.sql(
        ("UPDATE run_segment SET run_record = ? WHERE name = 'E/seg-1'", (altered_run_record(original),)),
        ("DELETE FROM mixture_shard WHERE position = 1 AND artifact_id = ?", (removed["artifact_id"],)),
        ("UPDATE occurrence SET locator = 'L999' WHERE occurrence_id = ?", (a1,)),
    )
    # Two digests for one blend array of E/seg-1, when the file holds both rows.
    world.store.close()
    raw = sqlite3.connect(world.path)
    try:
        raw.execute("DROP INDEX segment_cache_digest_one_blend")
        for digest in ("1" * 64, "2" * 64):
            raw.execute("INSERT INTO segment_cache_digest (segment_id, kind, shard_position, digest) "
                        "VALUES (?, 'dataset_index', NULL, ?)", (segment_id, digest))
        raw.commit()
        conflict = True
    except sqlite3.IntegrityError:
        raw.rollback()
        conflict = False
    finally:
        raw.close()
    world.store = Store.open(world.path)
    repairs = [
        ("UPDATE run_segment SET run_record = ? WHERE name = 'E/seg-1'", (original,)),
        ("INSERT INTO mixture_shard (artifact_id, position, shard_instance_id, path, content_digest) "
         "VALUES (?, ?, ?, ?, ?)", tuple(removed)),
        ("UPDATE occurrence SET locator = 'L1' WHERE occurrence_id = ?", (a1,)),
    ]
    if conflict:
        repairs.append(
            ("DELETE FROM segment_cache_digest WHERE segment_id = ? AND kind = 'dataset_index'", (segment_id,)))
    found = []
    for repair in repairs:
        fault = fault_of(world.store)
        found.append(None if fault is None else (fault.name, fault.context.get("record")))
        world.sql(repair)
    assert found == [
        (NAME_STORED_DIGEST, RECORD_RUN),
        (NAME_STORED_DIGEST, RECORD_MIXTURE),
        (NAME_DOMAIN, None),
    ] + ([(NAME_CACHE, None)] if conflict else [])
    assert fault_of(world.store) is None
