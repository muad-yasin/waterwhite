"""Developer tests of what the store gained when the tokenization maps, the
recovery and the confined harness became part of the tool: an artifact's place on
disk, a map and its drops in one transaction, a map beside one that did not
verify, a found file's role, the harness a stored log names, and the one reader
of a shard's tokens (plan 3.3, 3.4, 2.2(d), 2.2(f), F10).

Tiny files in a temporary folder; nothing here imports Megatron or torch.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from waterwhite import errors, ids
from waterwhite import vocabulary as v
from waterwhite.records import (
    DropRecord,
    EdgeInput,
    EdgeRecord,
    HarnessLog,
    HarnessRead,
    MapRecord,
    MapRow,
    OccurrenceRecord,
    OccurrenceRef,
    SnapshotShard,
    StepRef,
)
from waterwhite.store import Store, artifact_files

NORMALISER = "control-chars/1"
TEXTS = ["one", "two", "three"]


def step(name: str) -> StepRef:
    return StepRef(name, "1", ids.param_digest(None))


def write_jsonl(path: Path, texts: list[str]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return len(texts), ids.file_digest(path)


def occurrences(texts: list[str]) -> list[OccurrenceRecord]:
    return [OccurrenceRecord(f"L{n}", ids.fingerprint(t)) for n, t in enumerate(texts, start=1)]


def write_shard(prefix: Path, documents: list[list[int]], code: int = 4, letter: str = "i"):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    size = struct.calcsize(letter)
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for document in documents:
        if document:
            pointers.append(size * len(tokens))
            lengths.append(len(document))
            tokens.extend(document)
        boundaries.append(len(lengths))
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}{letter}", *tokens))
    index = b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, code, len(lengths), len(boundaries))
    index += struct.pack(f"<{len(lengths)}i", *lengths)
    index += struct.pack(f"<{len(pointers)}q", *pointers)
    index += struct.pack(f"<{len(boundaries)}q", *boundaries)
    Path(f"{prefix}.idx").write_bytes(index)
    digest = ids.shard_digest(ids.file_digest(f"{prefix}.bin"), ids.file_digest(f"{prefix}.idx"))
    return len(lengths), digest


class World:
    """A store with one ingested file of three lines and one shard of three
    documents made from it, with no map."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.store = Store.create(folder / "evidence.db", normaliser=NORMALISER)
        count, digest = write_jsonl(folder / "in" / "input.jsonl", TEXTS)
        self.input = self.store.register_jsonl(
            folder / "in" / "input.jsonl", name="input", role=v.ROLE_SOURCE, step=None,
            member_name="Z/r1", inputs=(), occurrences=occurrences(TEXTS),
            declared_count=count, declared_digest=digest)
        self.count, self.digest = write_shard(folder / "shards" / "shard", [[1], [2], [3]])
        self.shard = self.store.register_shard(
            folder / "shards" / "shard", name="shard", step=step("tokenize"), inputs=[self.input])

    def line(self, number: int) -> OccurrenceRef:
        return OccurrenceRef(self.input, f"L{number}")

    def map(self, verification: str, parents=(1, 2, 3), namespace: str = v.NAMESPACE_RECOVERY,
            derivation: str = v.DERIVED_POSITION, eod: bool = True) -> MapRecord:
        passed = verification == v.VERIFIED_PASSED
        label = v.RECOVERED if passed else v.UNKNOWN
        rows = tuple(MapRow(n, n, n + 1, label, self.line(parent) if passed else None)
                     for n, parent in enumerate(parents))
        return MapRecord(namespace, derivation, verification, label, eod, eod, rows)

    def write(self, record: MapRecord, **more) -> None:
        self.store.write_map(self.shard, record, declared_count=self.count,
                             declared_digest=self.digest, **more)

    def heads(self) -> list[tuple]:
        return [(b.namespace, b.derivation, b.verification) for b in self.store.bindings(self.shard)]

    def drops(self) -> list[tuple]:
        return [(d.input.locator, d.reason, d.step_name)
                for d in self.store.drops_by_output(self.shard)]


@pytest.fixture
def world(tmp_path):
    built = World(tmp_path)
    yield built
    built.store.close()


# --- An artifact's place on disk ------------------------------------------------------------


def test_the_store_gives_an_artifacts_place_on_disk(world, tmp_path):
    s = world.store
    assert s.disk_path(world.input) == tmp_path / "in" / "input.jsonl"
    # A shard: the prefix, without .bin and .idx.
    assert s.disk_path(world.shard) == tmp_path / "shards" / "shard"
    assert artifact_files.shard_paths(s.disk_path(world.shard))[0].is_file()
    # A file outside the store's folder is stored, and given back, absolute.
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "far.jsonl"
    count, digest = write_jsonl(outside, ["far"])
    far = s.register_jsonl(outside, name="far", role=v.ROLE_SOURCE, step=None, member_name="F/r1",
                           inputs=(), occurrences=occurrences(["far"]), declared_count=count,
                           declared_digest=digest)
    assert s.artifact(far).path == str(outside.resolve()) and s.disk_path(far) == outside.resolve()
    # No row, and an artifact with no file, give None. The file need not be there.
    assert s.disk_path("0" * 64) is None
    shard = s.artifact(world.shard)
    mixture = s.register_mixture(
        name="mix", settings={},
        shards=[SnapshotShard(0, world.shard, shard.path, shard.content_digest)])
    assert s.disk_path(mixture) is None
    (tmp_path / "in" / "input.jsonl").rename(tmp_path / "in" / "moved.jsonl")
    assert s.disk_path(world.input) == tmp_path / "in" / "input.jsonl"
    assert s.file_state(world.input) == "absent"


# --- A map and its drops are one transaction (plan 3.4) ------------------------------------


class Stopped(Exception):
    """Stands for a writer that is killed."""


def test_a_writer_stopped_at_the_drops_leaves_no_map(world):
    """Plan 3.4: ``write_map`` is one transaction. The drops of a map are written
    in it, so a writer that stops when the drops are due leaves neither."""
    drops = [DropRecord(world.line(2), v.DROP_NO_TOKENS, "tokenize")]

    def stop_at_the_drops(table: str, written: int) -> None:
        if table == "drop_record":
            raise Stopped()

    world.store.set_row_hook(stop_at_the_drops)
    with pytest.raises(Stopped):
        world.write(world.map(v.VERIFIED_PASSED), drops=drops)
    world.store.set_row_hook(None)
    assert world.heads() == [] and world.drops() == []
    assert world.store.binding(world.shard) is None

    # The same call again, not stopped, writes both.
    world.write(world.map(v.VERIFIED_PASSED), drops=drops)
    assert world.heads() == [(v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)]
    assert world.drops() == [("L2", v.DROP_NO_TOKENS, "tokenize")]
    # A map that is held gains a drop that was not handed with it the first time,
    # and holds the same drop again unchanged.
    more = drops + [DropRecord(world.line(3), v.DROP_SANITIZER_SKIP, "sanitize")]
    world.write(world.map(v.VERIFIED_PASSED), drops=more)
    world.write(world.map(v.VERIFIED_PASSED), drops=more)
    assert sorted(world.drops()) == [("L2", v.DROP_NO_TOKENS, "tokenize"),
                                     ("L3", v.DROP_SANITIZER_SKIP, "sanitize")]
    assert len(world.heads()) == 1


def test_a_refused_drop_refuses_the_map_with_it(world):
    world.store.add_drops(world.shard, [DropRecord(world.line(2), v.DROP_FILTER, "tokenize")])
    with pytest.raises(errors.InvalidRecord, match="another reason"):
        world.write(world.map(v.VERIFIED_PASSED),
                    drops=[DropRecord(world.line(2), v.DROP_NO_TOKENS, "tokenize")])
    assert world.heads() == []
    with pytest.raises(errors.InvalidRecord, match="unknown reason"):
        world.write(world.map(v.VERIFIED_PASSED),
                    drops=[DropRecord(world.line(1), "lost", "tokenize")])
    assert world.heads() == []


def test_the_maps_of_one_run_are_written_together_or_not_at_all(world):
    s = world.store
    position = world.map(v.VERIFIED_FAILED)
    matching = world.map(v.VERIFIED_PASSED, derivation=v.DERIVED_TOKEN_MATCHING)
    # The second map names a record the store does not hold: neither is written.
    broken = dataclasses.replace(matching, rows=(
        MapRow(0, 0, 1, v.RECOVERED, OccurrenceRef(world.input, "L9")), *matching.rows[1:]))
    with pytest.raises(errors.InvalidRecord, match="not a record of the store"):
        s.write_maps(world.shard, [position, broken], declared_count=world.count,
                     declared_digest=world.digest)
    assert world.heads() == []
    s.write_maps(world.shard, [position, matching], declared_count=world.count,
                 declared_digest=world.digest,
                 drops=[DropRecord(world.line(1), v.DROP_EMPTY_LINE, "sanitize")])
    assert world.heads() == [
        (v.NAMESPACE_RECOVERY, v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED),
        (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_FAILED)]
    assert world.drops() == [("L1", v.DROP_EMPTY_LINE, "sanitize")]
    with pytest.raises(errors.InvalidRecord, match="two maps are given"):
        s.write_maps(world.shard, [position, position], declared_count=world.count,
                     declared_digest=world.digest)
    with pytest.raises(errors.InvalidRecord, match="at least one map"):
        s.write_maps(world.shard, [], declared_count=world.count, declared_digest=world.digest)


# --- A map beside one that did not verify (plan 2.2(d), 2.2(f), 3.3) -----------------------


def test_a_map_that_did_not_verify_gets_a_verified_one_beside_it(world):
    """A failed or not-run map does not take its namespace and derivation for
    good: a later map of the same namespace and derivation is written beside it,
    nothing is overwritten, and a reader takes the verified one."""
    s = world.store
    failed = world.map(v.VERIFIED_FAILED)
    world.write(failed)
    assert s.binding(world.shard).verification == v.VERIFIED_FAILED
    assert s.map_row(world.shard, 0).parent is None

    passed = world.map(v.VERIFIED_PASSED)
    world.write(passed)
    assert world.heads() == [
        (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED),
        (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_FAILED)]
    # The reader's map of the namespace and derivation is the verified one.
    assert s.binding(world.shard, v.NAMESPACE_RECOVERY,
                     derivation=v.DERIVED_POSITION).verification == v.VERIFIED_PASSED
    assert [r.parent for r in s.map_rows(world.shard)] == [world.line(n) for n in (1, 2, 3)]
    # The map that did not verify is still there, whole, and is read by its result.
    old = list(s.map_rows(world.shard, verification=v.VERIFIED_FAILED))
    assert [(r.label, r.parent) for r in old] == [(v.UNKNOWN, None)] * 3
    assert s.stored_row_count(world.shard, verification=v.VERIFIED_FAILED) == 3
    assert s.binding(world.shard, verification=v.VERIFIED_FAILED).label == v.UNKNOWN
    # Both are bound to the shard by count and digest (2.2(d)); the checks pass.
    s.check_integrity([world.shard])
    # Each of the two again changes nothing.
    world.write(failed)
    world.write(passed)
    assert len(world.heads()) == 2


def test_a_verified_map_is_final_for_its_namespace_and_derivation(world):
    world.write(world.map(v.VERIFIED_PASSED))
    before = list(world.store.map_rows(world.shard))
    for other in (world.map(v.VERIFIED_PASSED, parents=(3, 2, 1)),
                  world.map(v.VERIFIED_FAILED), world.map(v.VERIFIED_NOT_RUN)):
        with pytest.raises(errors.InvalidRecord, match="a verified map is final"):
            world.write(other)
    assert world.heads() == [(v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)]
    assert list(world.store.map_rows(world.shard)) == before
    # The schema holds the rule too: no second verified map of one namespace and
    # derivation, by whatever route.
    conn = sqlite3.connect(world.store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO binding (artifact_id, namespace, declared_count, declared_digest, "
                "derivation, verification, label, append_eod, eod_in_span) "
                "SELECT artifact_id, namespace, declared_count, declared_digest, derivation, "
                "verification, label, append_eod, eod_in_span FROM binding "
                "WHERE derivation = 'position'")
    finally:
        conn.close()


def test_maps_that_did_not_verify_stand_side_by_side(world):
    world.write(world.map(v.VERIFIED_NOT_RUN))
    world.write(world.map(v.VERIFIED_FAILED))
    world.write(world.map(v.VERIFIED_FAILED))  # the same again
    # A reader gets the first that was written; neither names a record.
    assert world.heads() == [
        (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_NOT_RUN),
        (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_FAILED)]
    assert {r.label for r in world.store.map_rows(world.shard)} == {v.UNKNOWN}
    world.store.check_integrity([world.shard])


# --- A file with no producing step keeps its role (the role of a found file) ---------------


@pytest.mark.parametrize("role", [v.ROLE_INTERMEDIATE, v.ROLE_RELEASE])
def test_a_file_with_no_producing_step_keeps_its_place_in_the_pipeline(world, tmp_path, role):
    count, digest = write_jsonl(tmp_path / "found.jsonl", ["a", "b"])
    found = world.store.register_jsonl(
        tmp_path / "found.jsonl", name="found", role=role, step=None, member_name="found",
        inputs=(), occurrences=occurrences(["a", "b"]), declared_count=count,
        declared_digest=digest)
    artifact = world.store.artifact(found)
    assert (artifact.role, artifact.step, artifact.member_name) == (role, None, "found")
    assert found == ids.ingest_instance_id(digest, "found")
    # A file a step produced is still an intermediate or a release, and has no member.
    with pytest.raises(errors.InvalidRecord, match="is for a file with no producing step"):
        world.store.register_jsonl(
            tmp_path / "found.jsonl", name="made", role=v.ROLE_IMPORT, step=step("merge"),
            member_name=None, inputs=(), occurrences=occurrences(["a", "b"]),
            declared_count=count, declared_digest=digest)


# --- A stored harness log names its harness (plan F10) -------------------------------------


def generation(world, harness: str | None) -> EdgeRecord:
    log = HarnessLog("job-1", "digest-1", (world.line(1),), (HarnessRead(world.line(1)),),
                     "none", harness)
    return EdgeRecord("L1", (EdgeInput(world.line(1)),), step("paraphrase"), is_generation=True,
                      harness_log=log, job_name="job-1")


@pytest.mark.parametrize("harness", [v.HARNESS_IN_PROCESS, v.HARNESS_CONFINED, None])
def test_a_stored_harness_log_says_which_harness_wrote_it(world, tmp_path, harness):
    count, digest = write_jsonl(tmp_path / "out.jsonl", ["made"])
    arguments = dict(name="out", role=v.ROLE_INTERMEDIATE, step=step("paraphrase"),
                     member_name=None, inputs=[world.input], occurrences=occurrences(["made"]),
                     declared_count=count, declared_digest=digest)
    out = world.store.register_jsonl(tmp_path / "out.jsonl", edges=[generation(world, harness)],
                                     **arguments)
    assert world.store.harness_log("job-1").harness == harness
    assert world.store.edge_of(OccurrenceRef(out, "L1")).harness_log.harness == harness
    # The same history again with the same log is the same instance. With a log
    # that names another harness it is refused: the id does not cover the word.
    assert world.store.register_jsonl(tmp_path / "out.jsonl", edges=[generation(world, harness)],
                                      **arguments) == out
    other = v.HARNESS_CONFINED if harness != v.HARNESS_CONFINED else v.HARNESS_IN_PROCESS
    with pytest.raises(errors.InvalidRecord, match="the harness log of job 'job-1'"):
        world.store.register_jsonl(tmp_path / "out.jsonl", edges=[generation(world, other)],
                                   **arguments)


def test_a_harness_word_outside_the_vocabulary_is_refused(world, tmp_path):
    count, digest = write_jsonl(tmp_path / "out.jsonl", ["made"])
    with pytest.raises(errors.InvalidRecord, match="names its harness"):
        world.store.register_jsonl(
            tmp_path / "out.jsonl", name="out", role=v.ROLE_INTERMEDIATE, step=step("paraphrase"),
            member_name=None, inputs=[world.input], occurrences=occurrences(["made"]),
            edges=[generation(world, "sandboxed")], declared_count=count, declared_digest=digest)
    assert world.store.artifact("out") is None


def _schema(path: Path) -> list[tuple]:
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    conn.close()
    return rows


def test_a_version_1_store_is_migrated_to_version_2_and_matches_a_new_store(tmp_path):
    """Plan 3.4: the schema is versioned and migrated stepwise. A store of version 1
    (made by step 1 alone, as the build before step 2 made every store) opens as
    version 2: its harness logs gain the column naming their harness, empty for
    every log written before, and a map that did not verify no longer holds its
    namespace and derivation. The migrated schema equals a new store's."""
    from waterwhite.store import _migrate

    old = tmp_path / "old.db"
    conn = sqlite3.connect(old, isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    _migrate.apply_step(conn, 1)
    conn.execute("INSERT INTO meta (key, value) VALUES ('normaliser', ?)", (NORMALISER,))
    conn.execute("INSERT INTO harness_log (job_name, log_digest, model) VALUES ('old-job', 'd', 'none')")
    conn.execute("COMMIT")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(harness_log)")}
    conn.close()
    assert "harness" not in columns

    with Store.open(old) as store:
        assert store.schema_version == 4
    conn = sqlite3.connect(old)
    rows = conn.execute("SELECT job_name, model, harness FROM harness_log").fetchall()
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    conn.close()
    assert rows == [("old-job", "none", None)]
    assert "binding_one_verified_per_mode" in indexes and "binding_one_per_mode" not in indexes

    Store.create(tmp_path / "new.db", normaliser=NORMALISER).close()
    assert _schema(old) == _schema(tmp_path / "new.db")


# --- One reader of a shard's tokens ---------------------------------------------------------


def test_the_token_type_of_a_dtype_code():
    assert [artifact_files.token_dtype(code).str for code in (1, 2, 3, 4, 5, 8)] == [
        "|u1", "|i1", "<i2", "<i4", "<i8", "<u2"]
    for code in (0, 6, 7, 9):
        with pytest.raises(errors.UnsupportedFormat, match="holds no token ids"):
            artifact_files.token_dtype(code)


def test_a_document_is_read_from_its_own_byte_range(tmp_path):
    documents = [[5, 6, 7], [], [8], [9, 10]]
    for name, code, letter in (("s32", 4, "i"), ("s16", 8, "H")):
        write_shard(tmp_path / name, documents, code, letter)
        index = artifact_files.read_shard_index(f"{tmp_path / name}.idx")
        assert index.token_starts().tolist() == [0, 3, 4, 6]
        assert index.token_starts().dtype == np.int64
        with artifact_files.ShardTokens(tmp_path / name) as shard:
            assert (shard.count, shard.token_count) == (3, 6)
            assert [shard.read(k).tolist() for k in range(3)] == [[5, 6, 7], [8], [9, 10]]
            assert shard.read(1).dtype == np.int64 and shard.span(2) == (4, 6)
            with pytest.raises(IndexError):
                shard.read(3)
    # A .bin file that is not the size its index gives is no shard of this layout.
    with open(f"{tmp_path / 's32'}.bin", "ab") as f:
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(errors.UnsupportedFormat, match="its index gives 24"):
        artifact_files.ShardTokens(tmp_path / "s32")


def test_a_shard_whose_stored_offsets_are_not_the_running_sum_of_its_lengths_is_refused(world, tmp_path):
    """The outside review of a011cd5, finding 3. Megatron (core_v0.20.0, indexed_dataset.py:
    ``__getitem__`` reads at ``sequence_pointers[idx]``) reads a document at the stored
    byte offset; Waterwhite takes the offset from the lengths. For the contiguous layout
    the two must agree, so a shard whose offsets do not is refused wherever the index is
    read: by the reader, by the token reader and by the registration."""
    documents = [[5, 6, 7], [8, 9]]
    for name, code, letter, size in (("s32", 4, "i", 4), ("s16", 8, "H", 2)):
        write_shard(tmp_path / name, documents, code, letter)
        idx = Path(f"{tmp_path / name}.idx")
        good = idx.read_bytes()
        assert artifact_files.read_shard_index(idx).sequence_lengths.tolist() == [3, 2]
        first = 34 + 4 * 2  # where the two offsets start
        assert good[first:first + 16] == struct.pack("<2q", 0, 3 * size)
        for label, offsets in (("swapped", (3 * size, 0)), ("shifted", (0, 3 * size + size))):
            bad = good[:first] + struct.pack("<2q", *offsets) + good[first + 16:]
            idx.write_bytes(bad)
            with pytest.raises(errors.UnsupportedFormat, match="offset"):
                artifact_files.read_shard_index(idx)
            with pytest.raises(errors.UnsupportedFormat, match="offset"):
                artifact_files.ShardTokens(tmp_path / name)
            with pytest.raises(errors.UnsupportedFormat, match="offset"):
                world.store.register_shard(
                    tmp_path / name, name=f"{name}-{label}", step=step("tokenize"),
                    inputs=[world.input])
        idx.write_bytes(good)


def test_the_command_line_migrates_a_version_1_store_that_every_reading_command_refuses(tmp_path):
    """A command that reads a store opens it read-only, so it cannot run a step;
    it refuses by name and names the command that can (plan 3.4)."""
    import io

    from waterwhite import cli
    from waterwhite.store import _migrate

    old = tmp_path / "old.db"
    conn = sqlite3.connect(old, isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    _migrate.apply_step(conn, 1)
    conn.execute("INSERT INTO meta (key, value) VALUES ('normaliser', ?)", (NORMALISER,))
    conn.execute("COMMIT")
    conn.close()

    def run(*argv):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main([*argv, "--store", str(old)], stdout=out, stderr=err)
        return code, json.loads(out.getvalue() or err.getvalue())

    code, printed = run("check")
    assert code == errors.EXIT_REFUSED and "waterwhite migrate" in printed["detail"]
    code, said = run("migrate")
    assert code == 0 and (said["schema_version_found"], said["schema_version"], said["steps_run"]) == (1, 4, 3)
    assert said["migrated"]["from"] == 1 and "compare-hand-off" in said["migrated"]["note"]
    assert run("migrate") == (0, {"schema_version_found": 4, "schema_version": 4, "steps_run": 0})
    code, printed = run("check")
    assert code == 0 and printed["check"] == "passed"
