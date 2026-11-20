"""Developer tests of the evidence store (plan 3.4).

They build tiny JSONL files and a tiny hand-made .bin/.idx pair in a temporary
folder, write through the store's write methods and read back through its read
methods. Nothing here imports Megatron or torch. The acceptance checks of plan 5.5
are written elsewhere, from the hand-written table.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from waterwhite import errors, ids
from waterwhite import vocabulary as v
from waterwhite.records import (
    DropRecord,
    EdgeInput,
    EdgeRecord,
    HarnessRead,
    MapRecord,
    MapRow,
    OccurrenceRecord,
    OccurrenceRef,
    SnapshotShard,
    StepRef,
    StepTrace,
    TraceRow,
    TraceSample,
)
from waterwhite.ids import checkpoint_instance_id
from waterwhite.records import OWN_ARTIFACT
from waterwhite.records import HarnessLog as _HarnessLog


def HarnessLog(*args, harness=v.HARNESS_CONFINED, **kwargs) -> _HarnessLog:  # noqa: N802
    """The logs of these tests are written as the confined harness writes them: only
    such a log establishes a bound (Decision 13), and these tests are about the
    store's records of a bound, not about which harness gave it."""
    return _HarnessLog(*args, harness=harness, **kwargs)
from waterwhite.store import Store, _migrate, artifact_files, views

SRC = str(Path(__file__).resolve().parents[2] / "src")
NORMALISER = "control-chars/1"
JOIN = ("core_v0.20.0", "single", "megatron.training.datasets.data_samplers.MegatronPretrainingSampler")


# --- Builders of tiny artifacts -----------------------------------------------------


def fp(text: str) -> bytes:
    return ids.fingerprint(text)


def write_jsonl(path: Path, texts: list[str]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return len(texts), ids.file_digest(path)


def occurrences(texts: list[str]) -> list[OccurrenceRecord]:
    return [OccurrenceRecord(f"L{n}", fp(t) if t else None) for n, t in enumerate(texts, start=1)]


def write_shard(prefix: Path, documents: list[list[int]]) -> tuple[int, str]:
    """A hand-made shard in the documented index layout. An empty token list is a
    line that gave no tokens: a document boundary with no sequence."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for document in documents:
        if document:
            pointers.append(4 * len(tokens))
            lengths.append(len(document))
            tokens.extend(document)
        boundaries.append(len(lengths))
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}i", *tokens))
    index = b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, len(lengths), len(boundaries))
    index += struct.pack(f"<{len(lengths)}i", *lengths)
    index += struct.pack(f"<{len(pointers)}q", *pointers)
    index += struct.pack(f"<{len(boundaries)}q", *boundaries)
    Path(f"{prefix}.idx").write_bytes(index)
    digest = ids.shard_digest(ids.file_digest(f"{prefix}.bin"), ids.file_digest(f"{prefix}.idx"))
    return len(lengths), digest


def step(name: str, params: object = None) -> StepRef:
    return StepRef(name, "1", ids.param_digest(params))


def copy_edge(line: int, parent: OccurrenceRef, name: str = "merge") -> EdgeRecord:
    return EdgeRecord(f"L{line}", (EdgeInput(parent),), step(name))


def ingest(store: Store, path: Path, texts: list[str], source: str, name: str) -> str:
    count, digest = write_jsonl(path, texts)
    member = f"{source}/r1"
    instance = store.register_jsonl(
        path, name=name, role=v.ROLE_SOURCE, step=None, member_name=member, inputs=(),
        occurrences=occurrences(texts), declared_count=count, declared_digest=digest,
    )
    store.add_source(source)
    store.add_member(source, member, "r1", v.PART_REVISION, [(instance, "")])
    return instance


def table_counts(store: Store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


CACHE_DIGESTS = [
    {"kind": "document_index", "shard": 0, "digest": "d0"},
    {"kind": "dataset_index", "shard": None, "digest": "d1"},
    {"kind": "sample_index", "shard": 0, "digest": "d2"},
    {"kind": "shuffle_index", "shard": 0, "digest": "d3"},
    {"kind": "document_index", "shard": 1, "digest": "d4"},
    {"kind": "sample_index", "shard": 1, "digest": "d5"},
    {"kind": "shuffle_index", "shard": 1, "digest": "d6"},
    {"kind": "dataset_sample_index", "shard": None, "digest": "d7"},
]


def run_record(segment: str, start_kind: str, launches: list[dict]) -> dict:
    return {
        "format": "waterwhite.run-record/1",
        "run": "E",
        "segment": segment,
        "start_kind": start_kind,
        "parent_checkpoint": None,
        "settings": {
            "dataloader_type": "single",
            "sampler_class": JOIN[2],
            "data_sharding": True,
            "micro_batch_size": 2,
            "global_batch_size": 2,
            "sequence_length": 4,
            "add_extra_token_to_sequence": True,
            "random_seed": 1234,
            "split": "1,0,0",
            "train_iters": 3,
            "train_samples": None,
            "batch_schedule": None,
        },
        "versions": {"release_tag": JOIN[0], "megatron_core": "0.20.0", "megatron_bridge": None,
                     "torch": "2.0"},
        "shards": [{"position": 0, "prefix": "shards/shard-1", "bin_sha256": "b", "idx_sha256": "i"},
                   {"position": 1, "prefix": "shards/shard-2", "bin_sha256": "c", "idx_sha256": "j"}],
        "launches": launches,
        # Two shards: three arrays each and the two of the blend, which a route of
        # "cache files" or "replay" needs the record to hold (plan 3.5).
        "cache_digests": CACHE_DIGESTS,
        "shuffle_digests": [{"launch": 0, "epoch": 0, "rank": 0, "digest": "s0", "count": 6}],
    }


def whole_sample(pos: int, index: int, ordinal: int) -> TraceSample:
    return TraceSample(pos, index, 0, 5, 5, (TraceRow(0, ordinal, 0, 5),))


@dataclasses.dataclass
class World:
    """A small store that holds one of everything."""

    folder: Path
    store: Store
    A: str
    B: str
    Z: str
    P: str
    R1: str
    r1_args: dict
    shard: str
    shard_count: int
    shard_digest: str
    mixture: str
    ghost_shard: str
    ckpt2: str
    ckpt5: str
    log: _HarnessLog

    def ref(self, instance: str, line: int) -> OccurrenceRef:
        return OccurrenceRef(instance, f"L{line}")

    def copy(self, name: str = "copy.db") -> Path:
        """Close the store and copy its file beside it, as a check does before it
        edits a store outside the write API."""
        self.store.close()
        target = self.folder / name
        shutil.copy(self.folder / "store.db", target)
        return target


def r1_edges(w_a: str, w_b: str, w_p: str, w_z: str, winner: str) -> list[EdgeRecord]:
    """R1's stated edges. Line 1 is the shared passage X, kept from ``winner``."""
    a, b = OccurrenceRef(w_a, "L1"), OccurrenceRef(w_b, "L1")
    dedup = EdgeRecord(
        "L1",
        (EdgeInput(a, selected=winner == "A"), EdgeInput(b, selected=winner == "B")),
        step("dedup"),
        parent_mode=v.MODE_ANY,
    )
    return [
        dedup,
        copy_edge(2, OccurrenceRef(w_p, "L1")),
        copy_edge(3, OccurrenceRef(w_z, "L1")),
        copy_edge(4, OccurrenceRef(w_z, "L2")),
    ]


@pytest.fixture
def world(tmp_path: Path) -> World:
    store = Store.create(tmp_path / "store.db", normaliser=NORMALISER, verified_joins=[JOIN])
    a = ingest(store, tmp_path / "src/A.jsonl", ["passage X", "passage SA"], "A", "src/A")
    b = ingest(store, tmp_path / "src/B.jsonl", ["passage X", "b two"], "B", "src/B")
    z = ingest(store, tmp_path / "src/Z.jsonl", ["twin", "twin", "z three"], "Z", "src/Z")

    # A generation output P made from A's copy of X, in the harness.
    a1 = OccurrenceRef(a, "L1")
    log = HarnessLog("paraphrase", "log-digest-p", (a1,), (HarnessRead(occurrence=a1),), "none")
    p_texts = ["paraphrase of X"]
    count, digest = write_jsonl(tmp_path / "gen/P.jsonl", p_texts)
    p = store.register_jsonl(
        tmp_path / "gen/P.jsonl", name="gen/P", role=v.ROLE_INTERMEDIATE, step=step("paraphrase"),
        member_name=None, inputs=[a], occurrences=occurrences(p_texts),
        edges=[EdgeRecord("L1", (EdgeInput(a1),), step("paraphrase"), is_generation=True,
                          harness_log=log, job_name=log.job_name,
                          handed_texts=((a1, "passage X"),))],  # the job's text, so its log is stored compared
        declared_count=count, declared_digest=digest,
    )

    # The release: X (dedup kept A's copy), P, and the two twins of Z.
    r1_texts = ["passage X", "paraphrase of X", "twin", "twin"]
    count, digest = write_jsonl(tmp_path / "R1.jsonl", r1_texts)
    r1_args = dict(
        role=v.ROLE_RELEASE, step=step("release", {"dedup": True}), member_name=None,
        inputs=[a, b, p, z], occurrences=occurrences(r1_texts),
        declared_count=count, declared_digest=digest,
    )
    r1 = store.register_jsonl(
        tmp_path / "R1.jsonl", name="R1", edges=r1_edges(a, b, p, z, "A"),
        drops=[DropRecord(OccurrenceRef(b, "L1"), v.DROP_DEDUP_LOSER, "dedup", "group-X"),
               DropRecord(OccurrenceRef(z, "L3"), v.DROP_FILTER, "filter")],
        **r1_args,
    )

    # A shard of R1 with its capture map: two documents with one parent, two twins.
    shard_count, shard_digest = write_shard(tmp_path / "shards/shard-1", [[1, 2, 3], [4, 5], [6], [6]])
    shard = store.register_shard(tmp_path / "shards/shard-1", name="E/shard-1",
                                 step=step("tokenize"), inputs=[r1])
    twins = (OccurrenceRef(r1, "L3"), OccurrenceRef(r1, "L4"))
    store.write_map(
        shard,
        MapRecord(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED, True, True, (
            MapRow(0, 0, 3, v.RECOVERED, OccurrenceRef(r1, "L1"), fingerprint=fp("passage X")),
            MapRow(1, 3, 5, v.RECOVERED, OccurrenceRef(r1, "L2")),
            MapRow(2, 5, 6, v.CANDIDATES, candidates=twins),
            MapRow(3, 6, 7, v.CANDIDATES, candidates=twins),
        )),
        declared_count=shard_count, declared_digest=shard_digest,
    )

    # A mixture of that shard and one the store was never told of.
    ghost = "f" * 64
    mixture = store.register_mixture(
        name="E/mix-1",
        shards=[SnapshotShard(0, shard, "shards/shard-1", shard_digest),
                SnapshotShard(1, ghost, "shards/shard-2", "e" * 64)],
        settings={"split": "1,0,0"},
    )

    # Run E: segment 1 with three steps (step 3 has no trace, step 2 a half-covered
    # sample), a checkpoint at step 2, and segment 2 resumed from it.
    store.register_run_segment(
        run="E", segment="E/seg-1", position=0, mixture_instance_id=mixture, parent_checkpoint=None,
        start_kind="fresh",
        run_record=run_record("E/seg-1", "fresh", [
            {"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0,
             "step_at_start": 0, "first_step": 1, "last_step": 3}]),
        steps=[(1, 2, None), (2, 2, 3), (3, 2, None)], trace_route="cache files",
    )
    store.write_step_traces("E/seg-1", [
        StepTrace(1, 2, (whole_sample(0, 0, 0), whole_sample(1, 1, 1))),
        StepTrace(2, 2, (whole_sample(0, 2, 2),
                         TraceSample(1, 3, 0, 5, 3, (TraceRow(0, 3, 0, 1), TraceRow(0, 0, 0, 2))))),
    ])
    ckpt2 = store.register_checkpoint(name="E@2", segment="E/seg-1", step=2)
    store.register_run_segment(
        run="E", segment="E/seg-2", position=1, mixture_instance_id=mixture,
        parent_checkpoint=ckpt2, start_kind="resume", run_record=None,
        steps=[(4, 2, None), (5, 2, None)], trace_route="none",
        trace_cause=v.NAMES_TRACE_MISSING, trace_cause_detail=v.TRACE_MISSING_NO_DIGEST,
    )
    ckpt5 = store.register_checkpoint(name="E@5", segment="E/seg-2", step=5)
    return World(tmp_path, store, a, b, z, p, r1, r1_args, shard, shard_count, shard_digest,
                 mixture, ghost, ckpt2, ckpt5, log)


# --- Opening, settings, migration ---------------------------------------------------


def test_create_writes_meta_and_opens_with_the_fixed_pragmas(tmp_path):
    store = Store.create(tmp_path / "s.db", normaliser=NORMALISER, verified_joins=[JOIN, JOIN])
    assert store.meta() == {"schema_version": "5", "normaliser": NORMALISER}
    assert store.schema_version == 5 and store.normaliser == NORMALISER
    assert store.verified_joins() == (JOIN,)
    pragma = lambda name: store._conn.execute(f"PRAGMA {name}").fetchone()[0]
    assert (pragma("journal_mode"), pragma("synchronous"), pragma("foreign_keys")) == ("wal", 2, 1)
    store.close()
    store.close()  # closing twice is harmless
    with Store.open(tmp_path / "s.db") as again:
        assert again.meta()["normaliser"] == NORMALISER
        pragma = lambda name: again._conn.execute(f"PRAGMA {name}").fetchone()[0]
        assert (pragma("journal_mode"), pragma("synchronous"), pragma("foreign_keys")) == ("wal", 2, 1)


def test_create_and_open_refusals(tmp_path):
    Store.create(tmp_path / "s.db", normaliser=NORMALISER).close()
    with pytest.raises(errors.StoreFileExists) as caught:
        Store.create(tmp_path / "s.db", normaliser=NORMALISER)
    assert caught.value.name == "a file already exists at the store path"
    assert caught.value.exit_code == errors.EXIT_WRITE_REFUSED
    (tmp_path / "notes.txt").write_text("any file at all\n")
    with pytest.raises(errors.StoreFileExists):
        Store.create(tmp_path / "notes.txt", normaliser=NORMALISER)
    assert (tmp_path / "notes.txt").read_text() == "any file at all\n"
    with pytest.raises(errors.InvalidRecord):
        Store.create(tmp_path / "t.db", normaliser="no-version")
    assert not (tmp_path / "t.db").exists()

    with pytest.raises(errors.NotAStore) as caught:
        Store.open(tmp_path / "missing.db")
    assert (caught.value.name, caught.value.exit_code) == ("not a store", errors.EXIT_REFUSED)
    (tmp_path / "text.db").write_text("not a database, only some text\n" * 40)
    with pytest.raises(errors.NotAStore):
        Store.open(tmp_path / "text.db")
    other = sqlite3.connect(tmp_path / "other.db")  # an SQLite file of something else
    other.execute("CREATE TABLE t (x INTEGER)")
    other.commit()
    other.close()
    with pytest.raises(errors.NotAStore):
        Store.open(tmp_path / "other.db")
    with pytest.raises(errors.NotAStore):
        Store.open(tmp_path / "other.db", readonly=True)

    raw = sqlite3.connect(tmp_path / "s.db")
    raw.execute("UPDATE meta SET value = '6' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()
    with pytest.raises(errors.NewerSchemaVersion, match="schema version 6") as caught:
        Store.open(tmp_path / "s.db")
    assert caught.value.name == "store of a newer schema version"
    assert caught.value.exit_code == errors.EXIT_REFUSED


def test_the_package_exports_store_and_the_store_names_its_normaliser(tmp_path):
    import waterwhite.store
    from waterwhite.store.store import Store as from_module

    assert waterwhite.store.Store is Store is from_module
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER, verified_joins=[JOIN]) as store:
        assert store.normaliser == NORMALISER  # a property, not a method
        assert store.verified_joins() == (JOIN,)
    with pytest.raises(errors.InvalidRecord):
        Store.create(tmp_path / "bad.db", normaliser=NORMALISER, verified_joins=[("tag", "single")])
    assert not (tmp_path / "bad.db").exists()


def test_readonly_store_reads_and_refuses_every_write(tmp_path):
    Store.create(tmp_path / "s.db", normaliser=NORMALISER).close()
    store = Store.open(tmp_path / "s.db", readonly=True)
    assert store.settings() == {}
    with pytest.raises(errors.WriteRefused):
        store.set_setting("extension.content-ban", "on")
    with pytest.raises(errors.WriteRefused):
        store.add_source("A")
    store.close()


def test_settings_round_trip(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER) as store:
        assert store.settings() == {}
        store.set_setting("extension.generating-model", "on")
        store.set_setting("extension.generating-model", "off")
        # A switch holds "on" or "off" and names an extension of this build: a
        # value the tool could not read would be taken for off.
        for key, value in (("extension.generating-model", "yes"), ("extension.generating-model", ""),
                           ("extension.no-such-extension", "on")):
            with pytest.raises(errors.InvalidRecord):
                store.set_setting(key, value)
        assert store.settings()["extension.generating-model"] == "off"
        store.set_setting("team.note", "anything")  # another key is the team's own
        store.set_setting("extension.content-ban", "on")
        assert store.settings() == {"extension.generating-model": "off", "extension.content-ban": "on",
                                    "team.note": "anything"}


def test_migration_runs_step_by_step_and_a_failed_step_changes_nothing(tmp_path, monkeypatch):
    Store.create(tmp_path / "s.db", normaliser=NORMALISER).close()

    def step_6(conn):
        conn.execute("CREATE TABLE added_by_step_6 (x INTEGER)")

    def step_7(conn):
        conn.execute("CREATE TABLE added_by_step_7 (x INTEGER)")
        raise RuntimeError("step 7 breaks")

    monkeypatch.setattr(_migrate, "STEPS", {**_migrate.STEPS, 6: step_6, 7: step_7})
    monkeypatch.setattr(_migrate, "SCHEMA_VERSION", 7)
    monkeypatch.setattr("waterwhite.store.store.SCHEMA_VERSION", 7)
    with pytest.raises(errors.Refused, match="migrated"):
        Store.open(tmp_path / "s.db", readonly=True)
    with pytest.raises(RuntimeError, match="step 7 breaks"):
        Store.open(tmp_path / "s.db")
    raw = sqlite3.connect(tmp_path / "s.db")
    tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    version = raw.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    raw.close()
    # Step 6 ran and is recorded; step 7 was rolled back whole.
    assert "added_by_step_6" in tables and "added_by_step_7" not in tables
    assert version == "6"


def test_schema_script_splits_into_its_statements():
    statements = _migrate.split_statements(_migrate.schema_sql())
    assert sum(s.count("CREATE TABLE") for s in statements) == 32
    assert sum(s.count("CREATE INDEX") + s.count("CREATE UNIQUE INDEX") for s in statements) == 9
    assert all(sqlite3.complete_statement(s) for s in statements)
    with pytest.raises(ValueError):
        _migrate.split_statements("CREATE TABLE t (x INTEGER);\nCREATE TABLE u (")


def test_the_schema_holds_the_indexes_the_walks_use(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER) as store:
        indexed = {
            (row[0], row[1])
            for row in store._conn.execute(
                "SELECT tbl_name, name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")
        }
        columns = {
            name: [r[2] for r in store._conn.execute(f"PRAGMA index_info({name})")]
            for _, name in indexed
        }
    assert columns["map_row_parent"] == ["parent_occurrence_id"]
    assert columns["map_candidate_occurrence"] == ["occurrence_id"]
    assert columns["drop_record_output"] == ["output_instance_id"]
    assert columns["drop_record_group"] == ["group_key"]
    assert columns["source_link_artifact"] == ["artifact_id"]
    # One verified map per namespace and derivation, and one binding of a JSONL
    # artifact: a partial index, which leaves out the maps that did not verify.
    assert ("binding", "binding_one_verified_per_mode") in indexed
    assert columns["binding_one_verified_per_mode"][:2] == ["artifact_id", "namespace"]


# --- JSONL artifacts ----------------------------------------------------------------


def test_register_jsonl_reads_back(world):
    s = world.store
    a = s.artifact("src/A")
    assert a == s.artifact(world.A)
    assert (a.kind, a.role, a.step, a.member_name, a.names, a.inputs) == (
        v.KIND_JSONL, v.ROLE_SOURCE, None, "A/r1", ("src/A",), ())
    assert a.path == "src/A.jsonl"  # relative to the store's folder
    assert a.instance_id == ids.ingest_instance_id(a.content_digest, "A/r1")
    assert a.content_digest == ids.file_digest(world.folder / "src/A.jsonl")
    assert s.artifact("never heard of") is None

    (binding,) = s.bindings(world.A)
    assert (binding.namespace, binding.declared_count, binding.declared_digest, binding.derivation) == (
        v.NAMESPACE_CAPTURE, 2, a.content_digest, None)
    assert s.stored_row_count(world.A) == 2 and s.occurrence_count(world.A) == 2

    rows = s.occurrences(world.A)
    assert next(rows) == views.Occurrence(world.ref(world.A, 1), fp("passage X"))  # read lazily
    assert [o.ref.locator for o in rows] == ["L2"]
    assert s.occurrence(world.ref(world.A, 2)).fingerprint == fp("passage SA")
    assert s.occurrence(world.ref(world.A, 3)) is None
    twins = [o.ref for o in s.occurrences_by_fingerprint(fp("passage X"))]
    assert twins == [world.ref(world.A, 1), world.ref(world.B, 1), world.ref(world.R1, 1)]

    r1 = s.artifact("R1")
    assert (r1.role, r1.step, r1.inputs) == (
        v.ROLE_RELEASE, step("release", {"dedup": True}), (world.A, world.B, world.P, world.Z))
    assert [x.instance_id for x in s.artifacts(v.KIND_JSONL)] == [
        world.A, world.B, world.Z, world.P, world.R1]


def test_a_line_with_no_text_and_a_last_line_with_no_newline(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"text": "one"}\n\n{"text": "three"}')
    found = artifact_files.read_jsonl(path)
    assert (found.line_count, found.digest) == (3, ids.file_digest(path))
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER) as store:
        instance = store.register_jsonl(
            path, name="f", role=v.ROLE_IMPORT, step=None, member_name="legacy", inputs=(),
            occurrences=[OccurrenceRecord("L3", fp("three")), OccurrenceRecord("L1", fp("one")),
                         OccurrenceRecord("L2", None)],
            declared_count=3, declared_digest=found.digest,
        )
        assert [(o.ref.locator, o.fingerprint) for o in store.occurrences(instance)] == [
            ("L1", fp("one")), ("L2", None), ("L3", fp("three"))]


def register_plain(store: Store, path: Path, texts: list[str], **changes):
    count, digest = write_jsonl(path, texts)
    args = dict(name="x", role=v.ROLE_SOURCE, step=None, member_name="X/r1", inputs=(),
                occurrences=occurrences(texts), declared_count=count, declared_digest=digest)
    args.update(changes)
    return store.register_jsonl(path, **args)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"declared_count": 1}, errors.DeclaredCountMismatch),
        ({"declared_digest": "0" * 64}, errors.DeclaredDigestMismatch),
        ({"occurrences": occurrences(["one"])}, errors.DeclaredCountMismatch),
        ({"occurrences": [OccurrenceRecord("L1", fp("one")), OccurrenceRecord("L1", fp("two"))]},
         errors.InvalidRecord),
        ({"occurrences": [OccurrenceRecord("L1", fp("one")), OccurrenceRecord("D1", fp("two"))]},
         errors.InvalidRecord),
        ({"occurrences": [OccurrenceRecord("L1", b"short"), OccurrenceRecord("L2", None)]},
         errors.InvalidRecord),
        ({"role": "archive"}, errors.InvalidRecord),
        ({"member_name": None}, errors.InvalidRecord),
        ({"inputs": ["abc"]}, errors.InvalidRecord),  # an ingest has no inputs
        ({"step": step("merge"), "role": v.ROLE_INTERMEDIATE}, errors.InvalidRecord),  # step and member
        ({"name": ""}, errors.InvalidRecord),
    ],
)
def test_register_jsonl_refusals_leave_nothing(tmp_path, changes, error):
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER) as store:
        before = table_counts(store)
        with pytest.raises(error) as caught:
            register_plain(store, tmp_path / "x.jsonl", ["one", "two"], **changes)
        assert caught.value.exit_code == errors.EXIT_WRITE_REFUSED
        assert table_counts(store) == before
        assert store.artifact("x") is None and store.bindings("x") == ()


def test_register_jsonl_refuses_a_missing_file(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER) as store:
        with pytest.raises(errors.InvalidRecord, match="no file"):
            store.register_jsonl(
                tmp_path / "gone.jsonl", name="x", role=v.ROLE_SOURCE, step=None,
                member_name="X/r1", inputs=(), occurrences=[], declared_count=0, declared_digest="")


def test_a_path_outside_the_stores_folder_is_stored_absolute(tmp_path):
    (tmp_path / "inner").mkdir()
    with Store.create(tmp_path / "inner/s.db", normaliser=NORMALISER) as store:
        register_plain(store, tmp_path / "outside.jsonl", ["one"])
        stored = store.artifact("x").path
        assert Path(stored).is_absolute() and stored == str((tmp_path / "outside.jsonl").resolve())
        assert store.artifact_by_path(tmp_path / "outside.jsonl").names == ("x",)
        store.check_integrity()


# --- Identity (plan 3.1, Decision 29) ------------------------------------------------


def test_the_same_history_again_is_the_same_instance_and_writes_nothing(world):
    s = world.store
    before = table_counts(s)
    again = s.register_jsonl(
        world.folder / "R1.jsonl", name="R1",
        edges=r1_edges(world.A, world.B, world.P, world.Z, "A"), **world.r1_args)
    assert again == world.R1
    assert table_counts(s) == before
    # The same history under a second name: one more name, nothing else.
    second = s.register_jsonl(
        world.folder / "R1.jsonl", name="R1-alias",
        edges=r1_edges(world.A, world.B, world.P, world.Z, "A"), **world.r1_args)
    assert second == world.R1
    after = table_counts(s)
    assert after.pop("artifact_name") == before.pop("artifact_name") + 1 and after == before
    assert s.artifact("R1-alias").names == ("R1", "R1-alias")
    assert s.artifact("R1-alias") == s.artifact("R1")


def test_the_same_history_with_other_fingerprints_is_refused(world):
    other = occurrences(["passage X", "paraphrase of X", "twin", "not the twin"])
    with pytest.raises(errors.InvalidRecord, match="another fingerprint"):
        world.store.register_jsonl(
            world.folder / "R1.jsonl", name="R1",
            edges=r1_edges(world.A, world.B, world.P, world.Z, "A"),
            **{**world.r1_args, "occurrences": other})


def summary_file(world: World, name: str = "gen/S", **changes) -> str:
    """A generation output made from A's two records, of which it states one."""
    a1, a2 = world.ref(world.A, 1), world.ref(world.A, 2)
    path = world.folder / "gen/S.jsonl"
    count, digest = write_jsonl(path, ["summary"])
    fields = dict(
        log=HarnessLog("summary", "log-digest-s", (a1, a2),
                       (HarnessRead(occurrence=a1), HarnessRead(occurrence=a2)), "none"),
        inputs=(EdgeInput(a1), EdgeInput(a2, used=False)),
        label=v.OBSERVED, role=v.ROLE_INTERMEDIATE, fingerprint=fp("summary"), drops=(),
    )
    fields.update(changes)
    edge = EdgeRecord("L1", fields["inputs"], step("summary"), is_generation=True,
                      harness_log=fields["log"], label=fields["label"], job_name="summary")
    return world.store.register_jsonl(
        path, name=name, role=fields["role"], step=step("summary"), member_name=None,
        inputs=[world.A], occurrences=[OccurrenceRecord("L1", fields["fingerprint"])],
        edges=[edge], drops=fields["drops"], declared_count=count, declared_digest=digest)


def test_a_second_registration_that_disagrees_with_the_stored_instance_is_refused(world):
    """DECISIONS.md D-14. The instance id covers the content, the step, the inputs,
    the parents and the selected parent. A registration that agrees on those and
    differs in a field the id does not cover is refused, not passed over."""
    s = world.store
    a1, a2 = world.ref(world.A, 1), world.ref(world.A, 2)
    first = summary_file(world)
    before = table_counts(s)
    assert summary_file(world) == first  # everything agrees: the stored instance
    assert table_counts(s) == before

    stored_log = s.harness_log("summary")
    other_log = dataclasses.replace(stored_log, log_digest="another-log-digest")
    # What differs, and the words the refusal names it with.
    refused = [
        # a2 was in scope only; now a third record is in scope only too
        (dict(log=dataclasses.replace(stored_log, handed=(a1, a2, world.ref(world.B, 1))),
              inputs=(EdgeInput(a1), EdgeInput(a2, used=False),
                      EdgeInput(world.ref(world.B, 1), used=False))), "the edge of L1"),
        (dict(log=other_log), "the harness log of job 'summary'"),
        (dict(log=dataclasses.replace(stored_log, reads=(HarnessRead(occurrence=a1),))),
         "the harness log of job 'summary'"),
        (dict(log=dataclasses.replace(stored_log, model="some-model")),
         "the harness log of job 'summary'"),
        (dict(log=None), "the edge of L1"),
        (dict(fingerprint=fp("another text")), "another fingerprint for L1"),
        (dict(role=v.ROLE_RELEASE), "the role"),
        (dict(drops=(DropRecord(world.ref(world.A, 2), v.DROP_FILTER, "summary"),)),
         "the drop of line L2"),
    ]
    for changes, words in refused:
        with pytest.raises(errors.InvalidRecord, match=words) as caught:
            summary_file(world, **changes)
        assert caught.value.exit_code == errors.EXIT_WRITE_REFUSED, words
        assert table_counts(s) == before, words
    # Under another name the refusal is the same, and the name is not added.
    with pytest.raises(errors.InvalidRecord, match="registered already and .* the harness log"):
        summary_file(world, name="gen/S-again", log=other_log)
    assert s.artifact("gen/S-again") is None and s.artifact("gen/S").names == ("gen/S",)
    assert s.harness_log("summary") == stored_log
    assert s.edge_of(OccurrenceRef(first, "L1")).in_scope_only == (a2,)


def test_a_second_registration_with_another_label_is_refused(world):
    """A label is not part of the id: the same recovered link handed again with a
    weaker label is a disagreement (D-14)."""
    s = world.store
    path = world.folder / "import-T.jsonl"
    count, digest = write_jsonl(path, ["passage X"])
    recovered = EdgeRecord(
        "L1", (EdgeInput(world.ref(world.A, 1)),), step("recover-by-fingerprint"),
        derivation=v.DERIVED_FINGERPRINT, verification=v.VERIFIED_PASSED, label=v.RECOVERED)

    def register(edge, name="import/T"):
        return s.register_jsonl(
            path, name=name, role=v.ROLE_IMPORT, step=None, member_name="legacy/T", inputs=(),
            occurrences=occurrences(["passage X"]), edges=[edge], declared_count=count,
            declared_digest=digest)

    first = register(recovered)
    assert register(recovered, name="import/T-alias") == first
    before = table_counts(s)
    weaker = dataclasses.replace(recovered, verification=v.VERIFIED_NOT_RUN, label=v.UNKNOWN)
    with pytest.raises(errors.InvalidRecord, match="the edge of L1"):
        register(weaker)
    other_parent = dataclasses.replace(recovered, inputs=(EdgeInput(world.ref(world.B, 1)),))
    with pytest.raises(errors.InvalidRecord, match="the edge of L1"):
        register(other_parent)
    assert table_counts(s) == before
    assert s.edge_of(OccurrenceRef(first, "L1")).label == v.RECOVERED


def test_a_stated_parent_lies_in_a_declared_input_or_in_the_artifact_itself(world):
    """DECISIONS.md D-15. The inputs are part of the instance id, so a stated parent
    outside them would be history the id does not cover."""
    s = world.store
    a1, b1 = world.ref(world.A, 1), world.ref(world.B, 1)
    path = world.folder / "m.jsonl"
    count, digest = write_jsonl(path, ["passage X", "second"])

    def register(edges, inputs, name="m"):
        return s.register_jsonl(
            path, name=name, role=v.ROLE_INTERMEDIATE, step=step("merge"), member_name=None,
            inputs=inputs, occurrences=occurrences(["passage X", "second"]), edges=edges,
            declared_count=count, declared_digest=digest)

    before = table_counts(s)
    with pytest.raises(errors.InvalidRecord, match="no declared input") as caught:
        register([copy_edge(1, a1), copy_edge(2, b1)], [world.A])  # B is registered, not declared
    assert caught.value.context["parent"] == b1.key
    with pytest.raises(errors.InvalidRecord, match="no declared input"):
        register([copy_edge(1, a1)], [])
    any_edge = EdgeRecord("L1", (EdgeInput(a1, selected=True), EdgeInput(b1)), step("dedup"),
                          parent_mode=v.MODE_ANY)
    with pytest.raises(errors.InvalidRecord, match="no declared input"):
        register([any_edge], [world.A])  # every copy of an "any" edge is a parent
    assert table_counts(s) == before

    # The artifact itself is named with the instance id "" and needs no declaration.
    own = register([copy_edge(1, a1), copy_edge(2, OccurrenceRef(OWN_ARTIFACT, "L1"), "expand")],
                   [world.A])
    assert OWN_ARTIFACT == ""
    assert s.edge_of(OccurrenceRef(own, "L2")).parents == (OccurrenceRef(own, "L1"),)

    # A record that is in scope only is not a parent and is not held to the rule.
    log = HarnessLog("scoped", "d", (a1, b1), (HarnessRead(occurrence=a1),), "none")
    scoped = generation_file(world, "scoped", log, [EdgeInput(a1), EdgeInput(b1, used=False)])
    assert s.artifact(scoped).inputs == (world.A, world.B)

    # A recovered edge, written after registration, may name any registered record.
    imported = s.register_jsonl(
        path, name="import/m", role=v.ROLE_IMPORT, step=None, member_name="legacy/m", inputs=(),
        occurrences=occurrences(["passage X", "second"]), declared_count=count, declared_digest=digest)
    s.add_recovered_edges(imported, [EdgeRecord(
        "L1", (EdgeInput(b1),), step("recover-by-fingerprint"), derivation=v.DERIVED_FINGERPRINT,
        verification=v.VERIFIED_PASSED, label=v.RECOVERED)])
    assert s.edge_of(OccurrenceRef(imported, "L1")).parents == (b1,)
    assert s.artifact(imported).inputs == ()


def test_a_sole_handed_record_is_a_parent(world):
    """DECISIONS.md D-21: the store refuses a generation edge that marks the one
    record its job was handed as in scope only, or does not name it."""
    a1, a2 = world.ref(world.A, 1), world.ref(world.A, 2)
    sole = HarnessLog("sole", "d", (a1,), (HarnessRead(occurrence=a1),), "none")
    before = table_counts(world.store)
    with pytest.raises(errors.InvalidRecord, match="sole handed record"):
        generation_file(world, "sole", sole, [EdgeInput(a1, used=False)])
    with pytest.raises(errors.InvalidRecord, match="sole handed record"):
        generation_file(world, "sole", sole, [])
    assert table_counts(world.store) == before
    made = generation_file(world, "sole", sole, [EdgeInput(a1)])
    assert world.store.edge_of(OccurrenceRef(made, "L1")).parents == (a1,)
    # With two handed records only the stated ones are parents, and none may be stated.
    two = HarnessLog("two", "d", (a1, a2), (), "none")
    made = generation_file(world, "two", two, [EdgeInput(a1, used=False), EdgeInput(a2, used=False)])
    edge = world.store.edge_of(OccurrenceRef(made, "L1"))
    assert edge.parents == () and edge.in_scope_only == (a1, a2)


def test_a_swapped_dedup_winner_is_a_new_instance_with_the_same_bytes(world):
    s = world.store
    shutil.copy(world.folder / "R1.jsonl", world.folder / "R1-swap.jsonl")
    swap = s.register_jsonl(
        world.folder / "R1-swap.jsonl", name="R1-swap",
        edges=r1_edges(world.A, world.B, world.P, world.Z, "B"),
        drops=[DropRecord(world.ref(world.A, 1), v.DROP_DEDUP_LOSER, "dedup", "group-X")],
        **world.r1_args)
    assert swap != world.R1
    one, two = s.artifact("R1"), s.artifact("R1-swap")
    assert one.content_digest == two.content_digest and one.edge_digest != two.edge_digest
    assert (one.step, one.inputs) == (two.step, two.inputs)  # only the kept parent differs
    assert world.ref(world.R1, 1).key != world.ref(swap, 1).key
    assert s.edge_of(world.ref(world.R1, 1)).selected == world.ref(world.A, 1)
    assert s.edge_of(world.ref(swap, 1)).selected == world.ref(world.B, 1)

    assert {x.instance_id for x in s.artifacts_by_digest(one.content_digest)} == {world.R1, swap}
    with pytest.raises(errors.AmbiguousArtifact) as caught:
        s.artifact_by_digest(one.content_digest)
    assert caught.value.exit_code == errors.EXIT_REFUSED
    with pytest.raises(errors.AmbiguousArtifact):
        s.resolve(one.content_digest)
    assert s.resolve("R1-swap").instance_id == swap and s.resolve(swap).instance_id == swap
    assert s.artifact_by_path(world.folder / "R1-swap.jsonl").instance_id == swap
    assert s.artifact_by_path("R1.jsonl").instance_id == world.R1
    assert s.artifact_by_digest("0" * 64) is None and s.resolve("nothing") is None


def test_other_inputs_make_another_instance_and_one_path_can_hold_two(world):
    s = world.store
    other = s.register_jsonl(
        world.folder / "R1.jsonl", name="R1-other-inputs",
        edges=r1_edges(world.A, world.B, world.P, world.Z, "A"),
        **{**world.r1_args, "inputs": [world.B, world.A, world.P, world.Z]})
    assert other != world.R1
    with pytest.raises(errors.AmbiguousArtifact):
        s.artifact_by_path(world.folder / "R1.jsonl")
    assert len(s.artifacts_by_path("R1.jsonl")) == 2


def test_a_name_names_one_instance_for_good(world):
    with pytest.raises(errors.InvalidRecord, match="already names another instance"):
        world.store.register_jsonl(
            world.folder / "R1.jsonl", name="R1",
            edges=r1_edges(world.A, world.B, world.P, world.Z, "B"), **world.r1_args)
    assert world.store.artifact("R1").instance_id == world.R1


def test_the_same_bytes_under_two_members_are_two_instances(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser=NORMALISER) as store:
        one = register_plain(store, tmp_path / "x.jsonl", ["one"], name="x", member_name="X/r1")
        two = register_plain(store, tmp_path / "x.jsonl", ["one"], name="y", member_name="Y/r1")
        assert one != two


# --- Edges, harness logs, drops ------------------------------------------------------


def test_edges_read_back_upward_and_downward(world):
    s = world.store
    dedup = s.edge_of(world.ref(world.R1, 1))
    assert dedup.child == world.ref(world.R1, 1)
    assert (dedup.parent_mode, dedup.derivation, dedup.label, dedup.transform) == (
        v.MODE_ANY, v.DERIVED_STATED, v.OBSERVED, step("dedup"))
    assert dedup.parents == (world.ref(world.A, 1), world.ref(world.B, 1))
    assert dedup.selected == world.ref(world.A, 1)
    assert (dedup.is_generation, dedup.harness_log, dedup.bound_field) == (False, None, False)

    generation = s.edge_of(world.ref(world.P, 1))
    assert generation.is_generation and generation.bound_field is True
    assert generation.harness_log == world.log == s.harness_log("paraphrase")
    assert generation.job_name == "paraphrase"  # the edge keeps the name itself (D-24)
    assert s.edge_of(world.ref(world.R1, 1)).job_name is None  # a copy edge has no job
    assert generation.harness_log.bound_established() and generation.harness_log.model == "none"

    assert s.edge_of(world.ref(world.A, 1)) is None  # a source record has no edge: not a fault
    assert [e.child.locator for e in s.edges(world.R1)] == ["L1", "L2", "L3", "L4"]
    below = {e.child for e in s.edges_naming(world.ref(world.A, 1))}
    assert below == {world.ref(world.P, 1), world.ref(world.R1, 1)}
    assert [e.child for e in s.edges_naming(world.ref(world.P, 1))] == [world.ref(world.R1, 2)]
    assert list(s.edges_naming(world.ref(world.R1, 1))) == []


def generation_file(world: World, name: str, log: HarnessLog | None, inputs, **edge_changes) -> str:
    """A file with one generated line. The edge names its job: the job of its
    harness log, or for a job that wrote its own list ``job-of-<name>``."""
    path = world.folder / f"gen/{name}.jsonl"
    count, digest = write_jsonl(path, [f"output of {name}"])
    edge = EdgeRecord("L1", tuple(inputs), step(name), is_generation=True, harness_log=log,
                      job_name=f"job-of-{name}" if log is None else log.job_name)
    edge = dataclasses.replace(edge, **edge_changes)
    return world.store.register_jsonl(
        path, name=f"gen/{name}", role=v.ROLE_INTERMEDIATE, step=step(name), member_name=None,
        inputs=[world.A, world.B], occurrences=occurrences([f"output of {name}"]), edges=[edge],
        declared_count=count, declared_digest=digest)


def test_a_generation_parent_outside_the_hand_off_is_refused(world):
    a1, b1 = world.ref(world.A, 1), world.ref(world.B, 1)
    log = HarnessLog("summary", "d", (a1,), (HarnessRead(occurrence=a1),), "none")
    before = table_counts(world.store)
    with pytest.raises(errors.InvalidRecord, match="outside the hand-off"):
        generation_file(world, "S", log, [EdgeInput(a1), EdgeInput(b1)])
    assert table_counts(world.store) == before


def test_a_generation_edge_that_leaves_out_a_handed_record_is_refused(world):
    """The outside review of a011cd5, finding 1: every record the job was handed is an input
    of its edge (used, or in scope only); an edge that omits one would let the proof
    reach "does not contain" for a source whose record the job was handed."""
    a1, a2 = world.ref(world.A, 1), world.ref(world.A, 2)
    log = HarnessLog("two-handed", "d", (a1, a2), (HarnessRead(occurrence=a1),), "none")
    before = table_counts(world.store)
    with pytest.raises(errors.InvalidRecord, match="handed"):
        generation_file(world, "omits", log, [EdgeInput(a1)])
    assert table_counts(world.store) == before
    # the same log with both records on the edge is registered
    generation_file(world, "keeps", log, [EdgeInput(a1), EdgeInput(a2, used=False)])


def test_in_scope_only_inputs_and_what_breaks_a_bound(world):
    s = world.store
    a1, a2, b1 = world.ref(world.A, 1), world.ref(world.A, 2), world.ref(world.B, 1)
    # Handed two records, states the use of one; a refused read outside the hand-off.
    kept = HarnessLog("job-kept", "d1", (a1, a2),
                      (HarnessRead(occurrence=a1), HarnessRead(outside_ref="src/B:L1", refused=True)),
                      None)
    kept_file = generation_file(world, "kept", kept, [EdgeInput(a1), EdgeInput(a2, used=False)])
    edge = s.edge_of(OccurrenceRef(kept_file, "L1"))
    assert edge.parents == (a1,) and edge.in_scope_only == (a2,)
    assert edge.bound_field is True and edge.harness_log == kept
    assert edge.harness_log.model is None  # no model entry: read as not established, never "none"

    # A read outside the hand-off that the harness let through: no bound.
    broken = HarnessLog("job-broken", "d2", (a1,), (HarnessRead(occurrence=a1), HarnessRead(occurrence=b1)),
                        "none")
    broken_file = generation_file(world, "broken", broken, [EdgeInput(a1)])
    edge = s.edge_of(OccurrenceRef(broken_file, "L1"))
    assert edge.bound_field is False and not edge.harness_log.bound_established()
    assert edge.harness_log.reads[1].occurrence == b1

    # A job that wrote its own list: observed parents, no log, no bound.
    own_list = generation_file(world, "own-list", None, [EdgeInput(a1)])
    edge = s.edge_of(OccurrenceRef(own_list, "L1"))
    assert (edge.label, edge.harness_log, edge.bound_field) == (v.OBSERVED, None, False)
    assert edge.job_name == "job-of-own-list"

    # In scope only belongs to generation edges.
    with pytest.raises(errors.InvalidRecord, match="in scope only"):
        generation_file(world, "bad", None, [EdgeInput(a1), EdgeInput(a2, used=False)],
                        is_generation=False, job_name=None)
    # One job name, one log.
    clash = HarnessLog("job-kept", "another-digest", (a1,), (), "none")
    with pytest.raises(errors.InvalidRecord, match="already has a stored log"):
        generation_file(world, "clash", clash, [EdgeInput(a1)])


def test_a_version_1_store_from_before_the_edge_kept_its_job_is_refused_by_name(tmp_path):
    """Schema version 1 was revised in place (D-24). A file made before that is
    refused when it is opened, with a named error, and not later by SQLite."""
    raw = sqlite3.connect(tmp_path / "old.db")
    raw.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "INSERT INTO meta VALUES ('schema_version', '1'), ('normaliser', 'control-chars/1');"
        "CREATE TABLE edge (edge_id INTEGER PRIMARY KEY, child_id INTEGER, log_id INTEGER);"
    )
    raw.commit()
    raw.close()
    for readonly in (True, False):
        with pytest.raises(errors.NotAStore, match="before an edge kept the name of its job"):
            Store.open(tmp_path / "old.db", readonly=readonly)


def test_an_edge_keeps_the_name_of_its_job(world):
    """DECISIONS.md D-24: a generation edge names its job in a field of its own,
    so the name outlives the job's harness log (plan Check 5). Every generation
    edge has one, no other edge does, and it is the job of the edge's log."""
    s = world.store
    a1 = world.ref(world.A, 1)
    log = HarnessLog("job-named", "d", (a1,), (HarnessRead(occurrence=a1),), "none")
    before = table_counts(s)
    for changes, words in (
        (dict(job_name=None), "a generation edge names its job"),
        (dict(job_name=""), "a generation edge names its job"),
        (dict(job_name="another-job"), "carries the harness log of the job 'job-named'"),
    ):
        with pytest.raises(errors.InvalidRecord, match=words):
            generation_file(world, "named", log, [EdgeInput(a1)], **changes)
    with pytest.raises(errors.InvalidRecord, match="a generation edge names its job"):
        generation_file(world, "own", None, [EdgeInput(a1)], job_name=None)
    with pytest.raises(errors.InvalidRecord, match="only a generation edge names a job"):
        generation_file(world, "copy", None, [EdgeInput(a1)], is_generation=False, job_name="j")
    assert table_counts(s) == before

    named = generation_file(world, "named", log, [EdgeInput(a1)])
    assert s.edge_of(OccurrenceRef(named, "L1")).job_name == "job-named"
    (downward,) = [e for e in s.edges_naming(a1) if e.child.instance_id == named]
    assert downward.job_name == "job-named"

    # With the log gone the edge still names its job, read from the edge's own row.
    copy = world.copy()
    raw = sqlite3.connect(copy)
    try:
        log_id = raw.execute("SELECT log_id FROM harness_log WHERE job_name = 'job-named'").fetchone()[0]
        raw.execute("UPDATE edge SET log_id = NULL WHERE log_id = ?", (log_id,))
        for table in ("harness_read", "harness_handed", "harness_log"):
            raw.execute(f"DELETE FROM {table} WHERE log_id = ?", (log_id,))
        raw.commit()
    finally:
        raw.close()
    with Store.open(copy) as reopened:
        edge = reopened.edge_of(OccurrenceRef(named, "L1"))
        assert (edge.harness_log, edge.job_name, edge.bound_field) == (None, "job-named", True)
        assert reopened.harness_log("job-named") is None

        # The schema itself holds no generation edge without a job name, and no
        # job name on another edge.
        for statement in (
            "UPDATE edge SET job_name = NULL WHERE is_generation = 1",
            "UPDATE edge SET job_name = 'j' WHERE is_generation = 0",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                reopened._conn.execute(statement)


def test_the_job_name_of_an_edge_is_compared_when_a_history_is_registered_again(world):
    """DECISIONS.md D-14 and D-24: the instance id does not cover the job name, so
    a second registration that names another job is refused and changes nothing."""
    s = world.store
    a1 = world.ref(world.A, 1)
    first = generation_file(world, "own-list", None, [EdgeInput(a1)])
    before = table_counts(s)
    assert generation_file(world, "own-list", None, [EdgeInput(a1)]) == first
    with pytest.raises(errors.InvalidRecord, match="registered already and .* the edge of L1"):
        generation_file(world, "own-list", None, [EdgeInput(a1)], job_name="another-job")
    assert table_counts(s) == before
    assert s.edge_of(OccurrenceRef(first, "L1")).job_name == "job-of-own-list"


def test_edge_shape_refusals(world):
    a1, b1 = world.ref(world.A, 1), world.ref(world.B, 1)
    path = world.folder / "m.jsonl"
    count, digest = write_jsonl(path, ["passage X", "second"])

    def register(edges):
        return world.store.register_jsonl(
            path, name="m", role=v.ROLE_INTERMEDIATE, step=step("merge"), member_name=None,
            inputs=[world.A, world.B], occurrences=occurrences(["passage X", "second"]),
            edges=edges, declared_count=count, declared_digest=digest)

    any_edge = EdgeRecord("L1", (EdgeInput(a1), EdgeInput(b1)), step("dedup"), parent_mode=v.MODE_ANY)
    bad = [
        [any_edge],  # a stated "any" edge that does not say which parent was kept
        [copy_edge(1, a1), copy_edge(1, b1)],  # two edges for one record
        [copy_edge(3, a1)],  # no such line
        [copy_edge(1, world.ref(world.A, 9))],  # no such parent record
        [copy_edge(1, OccurrenceRef("0" * 64, "L1"))],  # no such parent artifact
        [EdgeRecord("L1", (EdgeInput(a1), EdgeInput(a1)), step("merge"))],  # a parent twice
        [EdgeRecord("L1", (EdgeInput(a1, selected=True),), step("merge"))],  # "all" selects none
        [dataclasses.replace(copy_edge(1, a1), derivation=v.DERIVED_POSITION, label=v.UNKNOWN)],
        [dataclasses.replace(copy_edge(1, a1), parent_mode="some")],
        [dataclasses.replace(copy_edge(1, a1), harness_log=world.log)],  # a log on a copy edge
    ]
    before = table_counts(world.store)
    for edges in bad:
        with pytest.raises(errors.InvalidRecord):
            register(edges)
    assert table_counts(world.store) == before

    # A parent in the artifact being written is named with OWN_ARTIFACT.
    own = register([copy_edge(1, a1), copy_edge(2, OccurrenceRef(OWN_ARTIFACT, "L1"), "expand")])
    assert world.store.edge_of(OccurrenceRef(own, "L2")).parents == (OccurrenceRef(own, "L1"),)
    assert [e.child for e in world.store.edges_naming(OccurrenceRef(own, "L1"))] == [
        OccurrenceRef(own, "L2")]


@pytest.mark.parametrize(
    ("derivation", "verification", "label", "ok"),
    [
        (v.DERIVED_STATED, None, v.OBSERVED, True),
        (v.DERIVED_STATED, None, v.RECOVERED, False),
        (v.DERIVED_FUNCTION, v.VERIFIED_PASSED, v.RECOVERED, True),
        (v.DERIVED_FUNCTION, v.VERIFIED_PASSED, v.OBSERVED, False),
        (v.DERIVED_FUNCTION, v.VERIFIED_NOT_RUN, v.RECOVERED, False),
        (v.DERIVED_FINGERPRINT, v.VERIFIED_FAILED, v.CANDIDATES, False),
        (v.DERIVED_FINGERPRINT, v.VERIFIED_FAILED, v.UNKNOWN, True),
        (v.DERIVED_POINTER, v.VERIFIED_PASSED, v.MAPPED, True),
        (v.DERIVED_POINTER, v.VERIFIED_NOT_RUN, v.MAPPED, False),
        (v.DERIVED_FILE, None, v.OBSERVED, False),
        (v.DERIVED_NONE, None, v.FILE_LEVEL, False),
    ],
)
def test_an_edge_label_stronger_than_its_derivation_is_refused(world, derivation, verification, label, ok):
    path = world.folder / "m.jsonl"
    count, digest = write_jsonl(path, ["passage X"])
    edge = dataclasses.replace(copy_edge(1, world.ref(world.A, 1)), derivation=derivation,
                               verification=verification, label=label)
    args = dict(name="m", role=v.ROLE_INTERMEDIATE, step=step("merge"), member_name=None,
                inputs=[world.A], occurrences=occurrences(["passage X"]), edges=[edge],
                declared_count=count, declared_digest=digest)
    if ok:
        instance = world.store.register_jsonl(path, **args)
        stored = world.store.edge_of(OccurrenceRef(instance, "L1"))
        assert (stored.derivation, stored.verification, stored.label) == (derivation, verification, label)
    else:
        before = table_counts(world.store)
        with pytest.raises(errors.LabelStrongerThanDerivation):
            world.store.register_jsonl(path, **args)
        assert table_counts(world.store) == before


def test_drops_read_back_by_input_group_and_output(world):
    s = world.store
    loser = views.StoredDrop(world.ref(world.B, 1), v.DROP_DEDUP_LOSER, "dedup", world.R1, "group-X")
    filtered = views.StoredDrop(world.ref(world.Z, 3), v.DROP_FILTER, "filter", world.R1, None)
    assert list(s.drops_from(world.B)) == [loser]
    assert s.drops_in_group("group-X") == (loser,)
    assert s.drops_of(world.ref(world.Z, 3)) == (filtered,)
    assert list(s.drops_by_output(world.R1)) == [loser, filtered]
    assert list(s.drops_from(world.A)) == [] and s.drops_in_group("no group") == ()

    # Drops known only after registration: a tokenizer line that gave no tokens.
    no_tokens = DropRecord(world.ref(world.R1, 4), v.DROP_NO_TOKENS, "tokenize")
    s.add_drops(world.shard, [no_tokens])
    s.add_drops(world.shard, [no_tokens])  # again: nothing new
    assert [d.reason for d in s.drops_by_output(world.shard)] == [v.DROP_NO_TOKENS]
    with pytest.raises(errors.InvalidRecord):
        s.add_drops(world.shard, [DropRecord(world.ref(world.R1, 4), "lost", "tokenize")])
    with pytest.raises(errors.InvalidRecord):
        s.add_drops(world.shard, [DropRecord(world.ref(world.R1, 9), v.DROP_NO_TOKENS, "tokenize")])
    # The same step name can drop the same record for another output: the output
    # is part of what makes a drop one drop.
    s.add_drops(world.R1, [no_tokens])
    assert [d.output_instance_id for d in s.drops_of(world.ref(world.R1, 4))] == [world.shard, world.R1]
    with pytest.raises(errors.InvalidRecord, match="one drop per record, step name and output"):
        s.add_drops(world.shard, [DropRecord(world.ref(world.R1, 4), v.DROP_EMPTY_LINE, "tokenize")])
    with pytest.raises(errors.InvalidRecord, match="not registered"):
        s.add_drops("0" * 64, [no_tokens])


def test_recovered_edges_never_change_an_instance_and_never_replace_an_edge(world):
    s = world.store
    path = world.folder / "import-T.jsonl"
    count, digest = write_jsonl(path, ["passage X", "unmatched"])
    imported = s.register_jsonl(
        path, name="import/T", role=v.ROLE_IMPORT, step=None, member_name="legacy/T", inputs=(),
        occurrences=occurrences(["passage X", "unmatched"]), declared_count=count, declared_digest=digest)
    candidates = EdgeRecord(
        "L1", (EdgeInput(world.ref(world.A, 1)), EdgeInput(world.ref(world.B, 1))),
        step("recover-by-fingerprint"), parent_mode=v.MODE_ANY, derivation=v.DERIVED_FINGERPRINT,
        verification=v.VERIFIED_PASSED, label=v.CANDIDATES)
    s.add_recovered_edges(imported, [candidates])
    s.add_recovered_edges(imported, [candidates])  # again: nothing new
    stored = s.edge_of(OccurrenceRef(imported, "L1"))
    assert (stored.label, stored.derivation, stored.selected) == (v.CANDIDATES, v.DERIVED_FINGERPRINT, None)
    assert len(stored.parents) == 2
    assert s.artifact("import/T").instance_id == imported  # the id did not move

    with pytest.raises(errors.InvalidRecord, match="identity"):
        s.add_recovered_edges(imported, [copy_edge(2, world.ref(world.A, 2))])
    with pytest.raises(errors.InvalidRecord, match="not replaced"):
        s.add_recovered_edges(imported, [dataclasses.replace(candidates, label=v.RECOVERED)])
    with pytest.raises(errors.LabelStrongerThanDerivation):
        s.add_recovered_edges(imported, [dataclasses.replace(candidates, child_locator="L2",
                                                             label=v.OBSERVED)])
    with pytest.raises(errors.InvalidRecord, match="not registered"):
        s.add_recovered_edges("0" * 64, [candidates])
    assert s.edge_of(OccurrenceRef(imported, "L2")) is None


# --- Sources, members, source links --------------------------------------------------


def test_sources_members_and_what_they_cover(world):
    s = world.store
    assert s.sources() == ("A", "B", "Z")
    (member,) = s.members("A")
    assert member == views.Member("A/r1", "A", "r1", v.PART_REVISION, ((world.A, ""),))
    assert s.member("A/r1") == member and s.member("C/r1") is None
    assert s.members("C") == () and len(s.members()) == 3
    assert list(s.member_occurrences("A/r1")) == [world.ref(world.A, 1), world.ref(world.A, 2)]
    assert [m.name for m in s.members_covering(world.ref(world.A, 2))] == ["A/r1"]
    assert s.members_covering(world.ref(world.R1, 1)) == ()

    # The identical call again changes nothing.
    before = table_counts(s)
    s.add_member("A", "A/r1", "r1", v.PART_REVISION, [(world.A, "")])
    assert table_counts(s) == before and s.member("A/r1") == member

    # A member made of single records, and a member that gains a part.
    s.add_member("Z", "Z/twins", "r1", v.PART_RECORDS, [(world.Z, "L1")])
    s.add_member("Z", "Z/twins", "r1", v.PART_RECORDS, [(world.Z, "L2"), (world.Z, "L1")])
    assert list(s.member_occurrences("Z/twins")) == [world.ref(world.Z, 1), world.ref(world.Z, 2)]
    assert [m.name for m in s.members_covering(world.ref(world.Z, 1))] == ["Z/r1", "Z/twins"]

    # What a walk reads per artifact and per record, with no member's part list
    # loaded: the members of the whole file, whether any member names single
    # records, the members that name one record, and the sources of an artifact.
    assert s.file_members(world.Z) == (("Z/r1", "Z"),)
    assert s.file_members(world.R1) == () and s.file_members("0" * 64) == ()
    assert s.has_record_parts(world.Z) and not s.has_record_parts(world.A)
    assert s.record_members(world.ref(world.Z, 1)) == (("Z/twins", "Z"),)
    assert s.record_members(world.ref(world.Z, 3)) == ()
    assert s.record_members(world.ref(world.A, 1)) == ()  # held by a whole-file member only
    assert s.sources_of_artifact(world.Z) == ("Z",) and s.sources_of_artifact(world.A) == ("A",)
    assert s.sources_of_artifact(world.R1) == ()  # a release holds no member's part
    s.add_source_link(world.R1, None, "A/r1", v.FILE_LEVEL, "file family", v.DERIVED_FILE)
    s.add_source_link(world.R1, "L3", "Z/r1", v.UNKNOWN, "a pointer nobody verified", v.DERIVED_POINTER)
    assert s.sources_of_artifact(world.R1) == ("A", "Z")  # named by its source links
    s.add_source("Z")  # idempotent
    assert s.sources() == ("A", "B", "Z")

    for call in (
        lambda: s.add_member("nobody", "N/r1", "r1", v.PART_REVISION, []),
        lambda: s.add_member("Z", "Z/twins", "r2", v.PART_RECORDS, []),  # another revision
        lambda: s.add_member("Z", "Z/x", "r1", "slice", []),
        lambda: s.add_member("Z", "Z/x", "r1", v.PART_RECORDS, [(world.Z, "L9")]),
        lambda: s.add_member("Z", "Z/x", "r1", v.PART_FILE, [("0" * 64, "")]),
    ):
        with pytest.raises(errors.InvalidRecord):
            call()
    assert s.member("Z/x") is None


def test_source_links_keep_a_weak_link_to_one_source_beside_a_strong_one_to_another(world):
    s = world.store
    # R1's first record has an observed edge to A's copy. Its file is also known
    # to derive from B (file-level), and one record carries a pointer to Z.
    s.add_source_link(world.R1, None, "B/r1", v.FILE_LEVEL, "file family: built from B", v.DERIVED_FILE)
    before = table_counts(s)
    s.add_source_link(world.R1, None, "B/r1", v.FILE_LEVEL, "file family: built from B",
                      derivation=v.DERIVED_FILE, verification=None)  # once
    assert table_counts(s) == before
    s.add_source_link(world.R1, "L3", "Z/r1", v.MAPPED, "publisher's pointer, verified on the row",
                      v.DERIVED_POINTER, v.VERIFIED_PASSED)
    file_link = views.SourceLink(world.R1, None, "B/r1", "B", v.FILE_LEVEL, "file family: built from B",
                                 v.DERIVED_FILE, None)
    record_link = views.SourceLink(world.R1, "L3", "Z/r1", "Z", v.MAPPED,
                                   "publisher's pointer, verified on the row",
                                   v.DERIVED_POINTER, v.VERIFIED_PASSED)
    assert s.file_source_links(world.R1) == (file_link,)
    assert s.source_links(world.ref(world.R1, 1)) == (file_link,)
    assert s.source_links(world.ref(world.R1, 3)) == (record_link, file_link)
    assert list(s.record_source_links(world.R1)) == [record_link]
    assert s.edge_of(world.ref(world.R1, 1)).label == v.OBSERVED  # the strong link is untouched

    # The label rule (plan 3.2, 2.2(f)) holds for a source link as for an edge.
    before = table_counts(s)
    for label, derivation, verification in (
        (v.OBSERVED, v.DERIVED_FILE, None),  # no parent record behind it: never observed
        (v.MAPPED, v.DERIVED_FILE, None),  # "mapped" needs a pointer
        (v.MAPPED, v.DERIVED_POINTER, None),  # and its verification
        (v.MAPPED, v.DERIVED_POINTER, v.VERIFIED_NOT_RUN),
        (v.MAPPED, v.DERIVED_POINTER, v.VERIFIED_FAILED),
        (v.FILE_LEVEL, v.DERIVED_POINTER, v.VERIFIED_PASSED),
        (v.FILE_LEVEL, v.DERIVED_NONE, None),
        (v.RECOVERED, v.DERIVED_NONE, None),
    ):
        with pytest.raises(errors.LabelStrongerThanDerivation):
            s.add_source_link(world.R1, "L1", "B/r1", label, "claimed", derivation, verification)
    for call in (
        lambda: s.add_source_link(world.R1, "L9", "B/r1", v.FILE_LEVEL, "e", v.DERIVED_FILE),
        lambda: s.add_source_link(world.R1, None, "C/r1", v.FILE_LEVEL, "e", v.DERIVED_FILE),
        lambda: s.add_source_link("0" * 64, None, "B/r1", v.FILE_LEVEL, "e", v.DERIVED_FILE),
        lambda: s.add_source_link(world.R1, None, "B/r1", v.FILE_LEVEL, "", v.DERIVED_FILE),
        # a source link is never stated by a step and never derived by a rule
        lambda: s.add_source_link(world.R1, "L1", "B/r1", v.OBSERVED, "e", v.DERIVED_STATED),
        lambda: s.add_source_link(world.R1, "L1", "B/r1", v.RECOVERED, "e", v.DERIVED_FINGERPRINT,
                                  v.VERIFIED_PASSED),
        lambda: s.add_source_link(world.R1, None, "B/r1", v.FILE_LEVEL, "e", v.DERIVED_FILE, "checked"),
    ):
        with pytest.raises(errors.InvalidRecord):
            call()
    with pytest.raises(TypeError):  # the derivation is never guessed
        s.add_source_link(world.R1, None, "B/r1", v.FILE_LEVEL, "e")
    assert table_counts(s) == before
    # A pointer that was not verified, and no link at all, can only be "unknown".
    s.add_source_link(world.R1, "L4", "Z/r1", v.UNKNOWN, "pointer, not checked", v.DERIVED_POINTER,
                      v.VERIFIED_NOT_RUN)
    assert [(x.label, x.derivation, x.verification) for x in s.record_source_links(world.R1)] == [
        (v.MAPPED, v.DERIVED_POINTER, v.VERIFIED_PASSED),
        (v.UNKNOWN, v.DERIVED_POINTER, v.VERIFIED_NOT_RUN)]


# --- Shards and maps ------------------------------------------------------------------


def test_the_index_reader_counts_documents_that_hold_tokens(tmp_path):
    count, digest = write_shard(tmp_path / "s", [[1, 2], [], [3], [4, 5, 6], []])
    index = artifact_files.read_shard_index(tmp_path / "s.idx")
    assert (index.document_count, index.sequence_count, index.empty_documents, index.token_count) == (
        3, 3, 2, 6)
    assert index.sequence_lengths.tolist() == [2, 1, 3]
    assert index.document_index.tolist() == [0, 1, 1, 2, 3, 3]
    found = artifact_files.read_shard(tmp_path / "s")
    assert (found.document_count, found.digest) == (count, digest) == (3, digest)

    good = (tmp_path / "s.idx").read_bytes()
    broken = {
        "magic": b"NOTANIDX\x00" + good[9:],
        "version": good[:9] + struct.pack("<Q", 2) + good[17:],
        "short": good[:20],
        "cut": good[:-8],
        # a document of two sequences: 2 sequences, boundaries 0 and 2
        "two": b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, 2, 2) + struct.pack("<2i", 1, 1)
        + struct.pack("<2q", 0, 4) + struct.pack("<2q", 0, 2),
    }
    for name, data in broken.items():
        (tmp_path / f"{name}.idx").write_bytes(data)
        with pytest.raises(errors.UnsupportedFormat):
            artifact_files.read_shard_index(tmp_path / f"{name}.idx")


def test_one_read_of_a_shard_gives_its_digest_and_its_document_count(tmp_path):
    """The index file is digested and read as an index in the same read. The
    digest is the one two separate reads give, also for an index with bytes after
    its arrays and for one that cannot be read as an index."""
    count, digest = write_shard(tmp_path / "s", [[1, 2], [], [3], [4, 5, 6], []])
    found = artifact_files.read_shard_once(tmp_path / "s")
    assert (found.document_count, found.digest, found.unreadable) == (count, digest, None) == (3, digest, None)

    hasher = ids.ContentHasher()
    index = artifact_files.read_shard_index(tmp_path / "s.idx", hasher)
    assert index.sequence_lengths.tolist() == [2, 1, 3]
    assert hasher.hexdigest() == ids.file_digest(tmp_path / "s.idx")
    assert hasher.size == (tmp_path / "s.idx").stat().st_size

    good = (tmp_path / "s.idx").read_bytes()
    tokens = (tmp_path / "s.bin").read_bytes()
    variants = {
        "trailing": (good + b"\x00\x01\x00", 3),  # sequence modes after the arrays
        "magic": (b"NOTANIDX\x00" + good[9:], None),
        "short": (good[:20], None),
        "cut": (good[:-8], None),
        "empty": (b"", None),
    }
    for name, (data, expected) in variants.items():
        (tmp_path / f"{name}.idx").write_bytes(data)
        (tmp_path / f"{name}.bin").write_bytes(tokens)
        found = artifact_files.read_shard_once(tmp_path / name)
        assert found.digest == artifact_files.shard_digest_only(tmp_path / name), name
        assert found.digest == ids.shard_digest(ids.bytes_digest(tokens), ids.bytes_digest(data)), name
        assert found.document_count == expected, name
        assert (found.unreadable is None) == (expected is not None), name


def test_register_shard_and_map_read_back(world):
    s = world.store
    shard = s.artifact("E/shard-1")
    assert (shard.kind, shard.role, shard.path, shard.inputs) == (
        v.KIND_SHARD, None, "shards/shard-1", (world.R1,))
    assert shard.content_digest == world.shard_digest
    assert s.artifact_by_path(world.folder / "shards/shard-1.bin").instance_id == world.shard
    # Registered again: the same instance.
    assert s.register_shard(world.folder / "shards/shard-1", name="E/shard-1", step=step("tokenize"),
                            inputs=[world.R1]) == world.shard

    binding = s.binding(world.shard)
    assert binding == views.Binding(world.shard, v.NAMESPACE_CAPTURE, 4, world.shard_digest,
                                    v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED, True, True)
    assert s.stored_row_count(world.shard) == 4
    rows = s.map_rows(world.shard)
    first = next(rows)  # read lazily
    assert first == MapRow(0, 0, 3, v.RECOVERED, world.ref(world.R1, 1), (), fp("passage X"))
    rest = list(rows)
    assert [r.label for r in rest] == [v.RECOVERED, v.CANDIDATES, v.CANDIDATES]
    assert set(rest[1].candidates) == {world.ref(world.R1, 3), world.ref(world.R1, 4)}
    assert rest[1].parent is None and rest[2].candidates == rest[1].candidates
    assert s.map_row(world.shard, 1).parent == world.ref(world.R1, 2)
    assert s.map_row(world.shard, 9) is None
    assert [r.ordinal for r in s.map_rows(world.shard, first_ordinal=1, last_ordinal=2)] == [1, 2]

    # The walk downward from a JSONL record into shards.
    assert list(s.documents_naming(world.ref(world.R1, 2))) == [
        views.MapLink(world.shard, v.NAMESPACE_CAPTURE, 1, v.RECOVERED, "parent", v.DERIVED_POSITION)]
    assert [(d.ordinal, d.via) for d in s.documents_naming(world.ref(world.R1, 3))] == [
        (2, "candidate"), (3, "candidate")]
    assert list(s.documents_naming(world.ref(world.A, 1))) == []


def test_register_shard_refuses_a_missing_or_unreadable_pair(world):
    s = world.store
    before = table_counts(s)
    (world.folder / "shards/half.bin").write_bytes(b"")
    with pytest.raises(errors.InvalidRecord, match="no file"):
        s.register_shard(world.folder / "shards/half", name="half", step=step("tokenize"), inputs=[])
    (world.folder / "shards/half.idx").write_bytes(b"not an index at all, only bytes of some length")
    with pytest.raises(errors.UnsupportedFormat):
        s.register_shard(world.folder / "shards/half", name="half", step=step("tokenize"), inputs=[])
    with pytest.raises(errors.InvalidRecord):
        s.register_shard(world.folder / "shards/shard-1", name="R1", step=step("tokenize"),
                         inputs=[world.R1])  # the name names another instance
    assert table_counts(s) == before and s.artifact("half") is None


def unknown_map(namespace: str, count: int, **changes) -> MapRecord:
    record = MapRecord(namespace, v.DERIVED_POSITION, v.VERIFIED_NOT_RUN, v.UNKNOWN, True, True,
                       tuple(MapRow(n, n, n + 1, v.UNKNOWN) for n in range(count)))
    return dataclasses.replace(record, **changes)


def test_a_reader_gets_the_capture_map_when_a_shard_has_both(world):
    s = world.store
    recovery = unknown_map(v.NAMESPACE_RECOVERY, 4, derivation=v.DERIVED_TOKEN_MATCHING,
                           verification=v.VERIFIED_FAILED)
    s.write_map(world.shard, recovery, declared_count=4, declared_digest=world.shard_digest)
    assert [b.namespace for b in s.bindings(world.shard)] == [v.NAMESPACE_CAPTURE, v.NAMESPACE_RECOVERY]
    assert s.binding(world.shard).namespace == v.NAMESPACE_CAPTURE
    assert [r.label for r in s.map_rows(world.shard)][0] == v.RECOVERED
    assert {r.label for r in s.map_rows(world.shard, v.NAMESPACE_RECOVERY)} == {v.UNKNOWN}
    assert s.binding(world.shard, v.NAMESPACE_RECOVERY).derivation == v.DERIVED_TOKEN_MATCHING
    # The capture map was not touched by the recovery write.
    assert s.map_row(world.shard, 0).parent == world.ref(world.R1, 1)

    # A shard with a recovery map only: the reader gets that one.
    count, digest = write_shard(world.folder / "shards/shard-9", [[1], [2]])
    other = s.register_shard(world.folder / "shards/shard-9", name="E/shard-9", step=step("tokenize"),
                             inputs=[world.R1])
    assert s.binding(other) is None and list(s.map_rows(other)) == [] and s.bindings(other) == ()
    recovered = MapRecord(v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED,
                          False, False, (MapRow(0, 0, 1, v.RECOVERED, world.ref(world.R1, 2)),
                                         MapRow(1, 1, 2, v.RECOVERED, world.ref(world.R1, 1))))
    s.write_map(other, recovered, declared_count=count, declared_digest=digest)
    assert s.binding(other).namespace == v.NAMESPACE_RECOVERY
    assert s.map_row(other, 0).parent == world.ref(world.R1, 2)
    assert s.binding(other, v.NAMESPACE_CAPTURE) is None
    # Downward: the recovery map counts where it is the reader's map, and only there.
    assert {(d.shard_instance_id, d.namespace) for d in s.documents_naming(world.ref(world.R1, 2))} == {
        (world.shard, v.NAMESPACE_CAPTURE), (other, v.NAMESPACE_RECOVERY)}


def test_a_shard_holds_one_recovery_map_per_derivation_and_the_reader_takes_them_in_order(world):
    """DECISIONS.md D-20: a position-mode and a token-matching recovery map stand
    side by side; a reader gets the capture map, else the verified position-mode
    recovery map, else the verified token-matching one."""
    s = world.store
    count, digest = write_shard(world.folder / "shards/shard-7", [[1], [2]])
    shard = s.register_shard(world.folder / "shards/shard-7", name="E/shard-7", step=step("tokenize"),
                             inputs=[world.R1])
    r1 = lambda line: world.ref(world.R1, line)

    def recovery(derivation, verification, parents):
        label = v.RECOVERED if verification == v.VERIFIED_PASSED else v.UNKNOWN
        rows = tuple(
            MapRow(n, n, n + 1, label, r1(parent) if label == v.RECOVERED else None)
            for n, parent in enumerate(parents))
        return MapRecord(v.NAMESPACE_RECOVERY, derivation, verification, label, True, True, rows)

    def write(record):
        s.write_map(shard, record, declared_count=count, declared_digest=digest)

    def reader():
        b = s.binding(shard)
        return (b.namespace, b.derivation, b.verification)

    # Position mode failed its verification: its label can only be "unknown". It is
    # still the shard's own record, so the reader gets it.
    position = recovery(v.DERIVED_POSITION, v.VERIFIED_FAILED, (1, 2))
    write(position)
    assert reader() == (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_FAILED)
    assert s.map_row(shard, 0).parent is None

    # Token matching then passes and is stored beside it, not instead of it.
    matching = recovery(v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED, (2, 1))
    write(matching)
    write(matching)  # the same map again: nothing changes
    assert [(b.derivation, b.verification) for b in s.bindings(shard)] == [
        (v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED), (v.DERIVED_POSITION, v.VERIFIED_FAILED)]
    assert reader() == (v.NAMESPACE_RECOVERY, v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED)
    assert [r.parent for r in s.map_rows(shard)] == [r1(2), r1(1)]
    assert s.stored_row_count(shard) == 2
    # Each map is read by naming its derivation, and both can be compared.
    assert {r.label for r in s.map_rows(shard, derivation=v.DERIVED_POSITION)} == {v.UNKNOWN}
    assert s.binding(shard, v.NAMESPACE_RECOVERY, derivation=v.DERIVED_POSITION).verification == (
        v.VERIFIED_FAILED)
    assert s.map_row(shard, 1, derivation=v.DERIVED_TOKEN_MATCHING).parent == r1(1)
    assert s.stored_row_count(shard, derivation=v.DERIVED_POSITION) == 2
    assert s.binding(shard, v.NAMESPACE_CAPTURE) is None
    # Downward, only the reader's map of a shard counts.
    assert [(d.shard_instance_id, d.derivation, d.ordinal) for d in s.documents_naming(r1(2))
            if d.shard_instance_id == shard] == [(shard, v.DERIVED_TOKEN_MATCHING, 0)]

    # One verified map per namespace and derivation: another one beside a verified
    # map is refused.
    with pytest.raises(errors.InvalidRecord, match="never overwritten"):
        write(recovery(v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED, (1, 2)))
    # The position map did not verify, so a verified position map is written
    # beside it, and the reader's order puts it before the token-matching map.
    write(recovery(v.DERIVED_POSITION, v.VERIFIED_PASSED, (1, 2)))
    assert reader() == (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)
    assert len(s.bindings(shard)) == 3
    with pytest.raises(errors.InvalidRecord, match="never overwritten"):
        write(recovery(v.DERIVED_POSITION, v.VERIFIED_PASSED, (2, 1)))
    s.check_integrity([shard])

    # Among verified maps the capture map comes first.
    write(dataclasses.replace(recovery(v.DERIVED_POSITION, v.VERIFIED_PASSED, (1, 2)),
                              namespace=v.NAMESPACE_CAPTURE))
    assert reader() == (v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED)
    assert [r.parent for r in s.map_rows(shard)] == [r1(1), r1(2)]
    assert len(s.bindings(shard)) == 4
    s.check_integrity([shard])


def test_a_verified_recovery_map_comes_before_a_capture_map_that_is_not_verified(world):
    """DECISIONS.md D-24 (amending D-20): across namespaces a verified map comes
    before an unverified one. A shard whose capture map failed its verification
    and that has a verified recovery map is read through the recovery map, and
    not through "unknown" links."""
    s = world.store
    count, digest = write_shard(world.folder / "shards/shard-6", [[1], [2]])
    shard = s.register_shard(world.folder / "shards/shard-6", name="E/shard-6", step=step("tokenize"),
                             inputs=[world.R1])

    def reader():
        b = s.binding(shard)
        return (b.namespace, b.derivation, b.verification)

    def write(record):
        s.write_map(shard, record, declared_count=count, declared_digest=digest)

    def verified(derivation, parents):
        return MapRecord(
            v.NAMESPACE_RECOVERY, derivation, v.VERIFIED_PASSED, v.RECOVERED, True, True,
            tuple(MapRow(n, n, n + 1, v.RECOVERED, world.ref(world.R1, parent))
                  for n, parent in enumerate(parents)))

    # With no verified map the capture map comes first, and it is the shard's own
    # record although every link of it is "unknown".
    write(unknown_map(v.NAMESPACE_RECOVERY, count, derivation=v.DERIVED_TOKEN_MATCHING))
    write(unknown_map(v.NAMESPACE_CAPTURE, count, verification=v.VERIFIED_FAILED))
    assert reader() == (v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_FAILED)
    assert {r.label for r in s.map_rows(shard)} == {v.UNKNOWN}
    assert list(s.documents_naming(world.ref(world.R1, 2))) == [
        d for d in s.documents_naming(world.ref(world.R1, 2)) if d.shard_instance_id != shard]

    # A verified recovery map then comes before the capture map that failed.
    write(verified(v.DERIVED_POSITION, (2, 1)))
    assert reader() == (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED)
    assert [(b.namespace, b.derivation, b.verification) for b in s.bindings(shard)] == [
        (v.NAMESPACE_RECOVERY, v.DERIVED_POSITION, v.VERIFIED_PASSED),
        (v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_FAILED),
        (v.NAMESPACE_RECOVERY, v.DERIVED_TOKEN_MATCHING, v.VERIFIED_NOT_RUN),
    ]
    assert [r.parent for r in s.map_rows(shard)] == [world.ref(world.R1, 2), world.ref(world.R1, 1)]
    assert s.map_row(shard, 0).label == v.RECOVERED and s.stored_row_count(shard) == count
    assert [(d.namespace, d.ordinal) for d in s.documents_naming(world.ref(world.R1, 2))
            if d.shard_instance_id == shard] == [(v.NAMESPACE_RECOVERY, 0)]
    # The capture map is still there and is read by naming its namespace.
    assert s.binding(shard, v.NAMESPACE_CAPTURE).verification == v.VERIFIED_FAILED
    assert {r.label for r in s.map_rows(shard, v.NAMESPACE_CAPTURE)} == {v.UNKNOWN}
    s.check_integrity([shard])


def test_with_both_recovery_maps_verified_the_reader_gets_the_position_mode_map(world):
    s = world.store
    count, digest = write_shard(world.folder / "shards/shard-8", [[1], [2]])
    shard = s.register_shard(world.folder / "shards/shard-8", name="E/shard-8", step=step("tokenize"),
                             inputs=[world.R1])
    for derivation, parents in ((v.DERIVED_TOKEN_MATCHING, (2, 1)), (v.DERIVED_POSITION, (1, 2))):
        s.write_map(
            shard,
            MapRecord(v.NAMESPACE_RECOVERY, derivation, v.VERIFIED_PASSED, v.RECOVERED, True, True,
                      tuple(MapRow(n, n, n + 1, v.RECOVERED, world.ref(world.R1, parent))
                            for n, parent in enumerate(parents))),
            declared_count=count, declared_digest=digest)
    assert s.binding(shard).derivation == v.DERIVED_POSITION
    assert s.map_row(shard, 0).parent == world.ref(world.R1, 1)
    assert s.map_row(shard, 0, derivation=v.DERIVED_TOKEN_MATCHING).parent == world.ref(world.R1, 2)


def fresh_shard(world: World, documents=((1,), (2,), (3,))) -> tuple[str, int, str]:
    count, digest = write_shard(world.folder / "shards/new", [list(d) for d in documents])
    instance = world.store.register_shard(world.folder / "shards/new", name="new",
                                          step=step("tokenize"), inputs=[world.R1])
    return instance, count, digest


def test_write_map_refuses_a_declared_count_or_digest_that_does_not_match(world):
    s = world.store
    shard, count, digest = fresh_shard(world)
    before = table_counts(s)
    with pytest.raises(errors.DeclaredCountMismatch):
        s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count - 1),
                    declared_count=count - 1, declared_digest=digest)
    with pytest.raises(errors.DeclaredCountMismatch):  # the right count declared, a row short
        s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count - 1),
                    declared_count=count, declared_digest=digest)
    with pytest.raises(errors.DeclaredDigestMismatch):
        s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count),
                    declared_count=count, declared_digest=world.shard_digest)
    assert table_counts(s) == before
    assert s.binding(shard) is None and list(s.map_rows(shard)) == []  # no map: "not covered"
    s.check_integrity()  # a shard with no map is not a fault

    s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count), declared_count=count, declared_digest=digest)
    assert s.stored_row_count(shard) == count


def test_write_map_refuses_bad_labels_rows_and_overwrites(world):
    s = world.store
    shard, count, digest = fresh_shard(world)
    r1 = lambda line: world.ref(world.R1, line)
    passed = dict(verification=v.VERIFIED_PASSED, label=v.RECOVERED)
    good_rows = (MapRow(0, 0, 1, v.RECOVERED, r1(1)), MapRow(1, 1, 2, v.RECOVERED, r1(2)),
                 MapRow(2, 2, 3, v.CANDIDATES, candidates=(r1(3), r1(4))))

    def write(record):
        s.write_map(shard, record, declared_count=count, declared_digest=digest)

    stronger = [
        unknown_map(v.NAMESPACE_CAPTURE, count, label=v.RECOVERED),  # not verified, says recovered
        unknown_map(v.NAMESPACE_CAPTURE, count, label=v.OBSERVED, verification=v.VERIFIED_PASSED),
        unknown_map(v.NAMESPACE_CAPTURE, count, verification=v.VERIFIED_FAILED,
                    rows=(MapRow(0, 0, 1, v.RECOVERED, r1(1)),) + good_rows[1:]),
        unknown_map(v.NAMESPACE_CAPTURE, count, **passed,
                    rows=(MapRow(0, 0, 1, v.OBSERVED, r1(1)),) + good_rows[1:]),
    ]
    invalid = [
        unknown_map("scratch", count),
        unknown_map(v.NAMESPACE_CAPTURE, count, derivation=v.DERIVED_FINGERPRINT),
        unknown_map(v.NAMESPACE_CAPTURE, count, **passed,
                    rows=(MapRow(0, 0, 1, v.RECOVERED),) + good_rows[1:]),  # recovered, no parent
        unknown_map(v.NAMESPACE_CAPTURE, count, **passed,
                    rows=good_rows[:2] + (MapRow(2, 2, 3, v.CANDIDATES, candidates=(r1(3),)),)),
        unknown_map(v.NAMESPACE_CAPTURE, count, **passed,
                    rows=good_rows[:2] + (MapRow(1, 2, 3, v.RECOVERED, r1(3)),)),  # an ordinal twice
        unknown_map(v.NAMESPACE_CAPTURE, count, **passed,
                    rows=(MapRow(0, 0, 1, v.RECOVERED, r1(9)),) + good_rows[1:]),  # no such record
        unknown_map(v.NAMESPACE_CAPTURE, count, **passed,
                    rows=(MapRow(0, 3, 1, v.RECOVERED, r1(1)),) + good_rows[1:]),  # not a span
    ]
    before = table_counts(s)
    for record in stronger:
        with pytest.raises(errors.LabelStrongerThanDerivation):
            write(record)
    for record in invalid:
        with pytest.raises(errors.InvalidRecord):
            write(record)
    with pytest.raises(errors.InvalidRecord, match="not registered"):
        s.write_map("0" * 64, unknown_map(v.NAMESPACE_CAPTURE, count),
                    declared_count=count, declared_digest=digest)
    with pytest.raises(errors.InvalidRecord, match="not a shard"):
        s.write_map(world.R1, unknown_map(v.NAMESPACE_CAPTURE, count),
                    declared_count=count, declared_digest=digest)
    assert table_counts(s) == before

    good = unknown_map(v.NAMESPACE_CAPTURE, count, **passed, rows=good_rows)
    write(good)
    after = table_counts(s)
    write(good)  # the same map again: nothing changes
    assert table_counts(s) == after
    with pytest.raises(errors.InvalidRecord, match="never overwritten"):
        write(unknown_map(v.NAMESPACE_CAPTURE, count))
    assert s.map_row(shard, 0).parent == r1(1)


def test_write_map_refuses_when_the_shard_changed_on_disk(world):
    s = world.store
    shard, count, digest = fresh_shard(world)
    write_shard(world.folder / "shards/new", [[1], [2], [7]])  # other tokens, the same count
    with pytest.raises(errors.DeclaredDigestMismatch):
        s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count), declared_count=count,
                    declared_digest=digest)
    changed = ids.shard_digest(ids.file_digest(world.folder / "shards/new.bin"),
                               ids.file_digest(world.folder / "shards/new.idx"))
    with pytest.raises(errors.ArtifactDigestMismatch):  # matches the disk, not the instance
        s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count), declared_count=count,
                    declared_digest=changed)
    os.rename(world.folder / "shards/new.bin", world.folder / "shards/new.bin.away")
    with pytest.raises(errors.InvalidRecord, match="not at"):
        s.write_map(shard, unknown_map(v.NAMESPACE_CAPTURE, count), declared_count=count,
                    declared_digest=changed)
    assert s.binding(shard) is None


# --- A writer killed in the middle of a transaction ------------------------------------

KILLED_WRITER = textwrap.dedent(
    """
    import os, signal, sys
    from waterwhite import vocabulary as v
    from waterwhite.records import MapRecord, MapRow
    from waterwhite.store.store import Store

    store_path, shard, count, digest, kill_table, kill_at = sys.argv[1:7]
    store = Store.open(store_path)

    def hook(table, written):
        if table == kill_table and written == int(kill_at):
            os.kill(os.getpid(), signal.SIGKILL)

    store.set_row_hook(hook)
    rows = tuple(MapRow(n, n, n + 1, v.UNKNOWN) for n in range(int(count)))
    store.write_map(
        shard,
        MapRecord(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_NOT_RUN, v.UNKNOWN, True, True, rows),
        declared_count=int(count), declared_digest=digest,
    )
    print("finished")
    """
)


def run_writer(tmp_path: Path, script: str, *args: object) -> subprocess.CompletedProcess:
    path = tmp_path / "writer.py"
    path.write_text(script, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run([sys.executable, str(path), *map(str, args)], env=env,
                          capture_output=True, text=True, timeout=60)


def test_a_map_writer_killed_after_some_rows_leaves_no_map(tmp_path):
    store = Store.create(tmp_path / "store.db", normaliser=NORMALISER)
    count, digest = write_shard(tmp_path / "big", [[n] for n in range(500)])
    shard = store.register_shard(tmp_path / "big", name="big", step=step("tokenize"), inputs=[])
    store.close()

    done = run_writer(tmp_path, KILLED_WRITER, tmp_path / "store.db", shard, count, digest, "map_row", 100)
    assert done.returncode == -signal.SIGKILL, done.stderr
    assert "finished" not in done.stdout

    store = Store.open(tmp_path / "store.db")
    # Absent or complete; here the kill came before the commit, so: absent.
    assert store.binding(shard) is None and list(store.map_rows(shard)) == []
    assert store._conn.execute("SELECT COUNT(*) FROM map_row").fetchone()[0] == 0
    store.check_integrity()
    store.close()

    # The interrupted write, tried again, gives the whole map.
    done = run_writer(tmp_path, KILLED_WRITER, tmp_path / "store.db", shard, count, digest, "map_row", -1)
    assert done.returncode == 0 and "finished" in done.stdout, done.stderr
    store = Store.open(tmp_path / "store.db")
    assert store.binding(shard).declared_count == 500 and store.stored_row_count(shard) == 500
    assert [r.ordinal for r in store.map_rows(shard)] == list(range(500))
    store.check_integrity()
    store.close()


KILLED_JSONL_WRITER = textwrap.dedent(
    """
    import os, signal, sys
    from waterwhite import ids, vocabulary as v
    from waterwhite.records import OccurrenceRecord
    from waterwhite.store.store import Store

    store_path, file_path, count = sys.argv[1:4]
    store = Store.open(store_path)

    def hook(table, written):
        if table == "occurrence" and written == 40:
            os.kill(os.getpid(), signal.SIGKILL)

    store.set_row_hook(hook)
    store.register_jsonl(
        file_path, name="big", role=v.ROLE_SOURCE, step=None, member_name="big/r1", inputs=(),
        occurrences=[OccurrenceRecord(f"L{n}", None) for n in range(1, int(count) + 1)],
        declared_count=int(count), declared_digest=ids.file_digest(file_path),
    )
    """
)


def test_a_jsonl_writer_killed_after_some_rows_leaves_no_artifact(tmp_path):
    Store.create(tmp_path / "store.db", normaliser=NORMALISER).close()
    write_jsonl(tmp_path / "big.jsonl", [f"line {n}" for n in range(100)])
    done = run_writer(tmp_path, KILLED_JSONL_WRITER, tmp_path / "store.db", tmp_path / "big.jsonl", 100)
    assert done.returncode == -signal.SIGKILL, done.stderr
    with Store.open(tmp_path / "store.db") as store:
        assert store.artifact("big") is None
        counts = table_counts(store)
        assert counts["artifact"] == counts["binding"] == counts["occurrence"] == 0


# --- Mixtures, runs, traces, checkpoints -----------------------------------------------


def test_mixture_reads_back_and_names_a_shard_the_store_was_never_told_of(world):
    s = world.store
    mixture = s.artifact("E/mix-1")
    assert (mixture.kind, mixture.path, mixture.instance_id) == (v.KIND_MIXTURE, None, world.mixture)
    assert s.mixture_shards(world.mixture) == (
        SnapshotShard(0, world.shard, "shards/shard-1", world.shard_digest),
        SnapshotShard(1, world.ghost_shard, "shards/shard-2", "e" * 64))
    assert s.mixtures_holding(world.shard) == (world.mixture,)
    assert s.segments_using(world.mixture) == ("E/seg-1", "E/seg-2")
    missing = views.Reference(views.REL_MIXTURE_SHARD, world.mixture, world.ghost_shard, 1, False)
    assert s.named_without_record() == (missing,)
    assert s.references_to(world.ghost_shard) == (missing,)  # named: "record missing"
    assert s.references_to("9" * 64) == ()  # never named: "not registered"
    assert s.artifact(world.ghost_shard) is None
    named = s.references_to(world.shard)
    assert [(r.relation, r.registered) for r in named] == [(views.REL_MIXTURE_SHARD, True)]

    again = s.register_mixture(
        name="E/mix-1-alias",
        shards=[SnapshotShard(1, world.ghost_shard, "elsewhere/shard-2", "e" * 64),
                SnapshotShard(0, world.shard, "elsewhere/shard-1", world.shard_digest)],
        settings={"split": "1,0,0"})
    assert again == world.mixture and s.artifact("E/mix-1").names == ("E/mix-1", "E/mix-1-alias")
    other_order = s.register_mixture(
        name="reordered",
        shards=[SnapshotShard(0, world.ghost_shard, "p", "e" * 64),
                SnapshotShard(1, world.shard, "q", world.shard_digest)],
        settings={"split": "1,0,0"})
    assert other_order != world.mixture
    for shards in (
        [],
        [SnapshotShard(1, world.shard, "p", world.shard_digest)],  # positions start at 0
        [SnapshotShard(0, world.shard, "p", "0" * 64)],  # not the registered shard's digest
        [SnapshotShard(0, world.R1, "p", s.artifact("R1").content_digest)],  # not a shard
    ):
        with pytest.raises(errors.InvalidRecord):
            s.register_mixture(name="bad", shards=shards, settings={})
    with pytest.raises(errors.InvalidRecord):
        s.register_mixture(name="bad", shards=[SnapshotShard(0, "a" * 64, "p", "b" * 64)],
                           settings={"x": object()})
    assert s.artifact("bad") is None


def test_a_mixtures_settings_are_kept_whole_and_their_digest_is_in_the_id(world):
    s = world.store
    # The settings are kept whole and read back; their digest is in the id.
    assert s.mixture_settings(world.mixture) == {"split": "1,0,0"}
    assert s.mixture_settings(world.shard) is None and s.mixture_settings("0" * 64) is None
    assert world.mixture == ids.mixture_instance_id(
        [world.shard, world.ghost_shard], [world.shard_digest, "e" * 64], {"split": "1,0,0"})
    other_settings = s.register_mixture(
        name="E/mix-1-other-settings",
        shards=[SnapshotShard(0, world.shard, "shards/shard-1", world.shard_digest),
                SnapshotShard(1, world.ghost_shard, "shards/shard-2", "e" * 64)],
        settings={"split": "98,1,1", "nested": {"b": [1, 2], "a": None}})
    assert other_settings != world.mixture
    assert s.mixture_settings(other_settings) == {"split": "98,1,1", "nested": {"b": [1, 2], "a": None}}
    with pytest.raises(errors.InvalidRecord, match="JSON object"):
        s.register_mixture(name="bad", shards=[SnapshotShard(0, "a" * 64, "p", "b" * 64)],
                           settings=["not", "an", "object"])


def test_segment_reads_back_with_launches_digests_and_steps(world):
    s = world.store
    assert s.runs() == ("E",) and s.segments() == ("E/seg-1", "E/seg-2") == s.segments("E")
    seg = s.segment("E/seg-1")
    assert (seg.run, seg.position, seg.mixture_instance_id, seg.parent_checkpoint, seg.start_kind) == (
        "E", 0, world.mixture, None, "fresh")
    assert (seg.sampler_type, seg.sampler_class, seg.data_sharding, seg.micro_batch_size,
            seg.global_batch_size, seg.sequence_length, seg.add_extra_token, seg.random_seed) == (
        "single", JOIN[2], True, 2, 2, 4, True, 1234)
    assert (seg.megatron_core_version, seg.megatron_bridge_version, seg.torch_version, seg.release_tag) == (
        "0.20.0", None, "2.0", JOIN[0])
    assert seg.launches == (views.Launch(0, 1, 1, 0, 0, 1, 3),)
    assert seg.cache_digests == tuple(views.CacheDigest(c["kind"], c["shard"], c["digest"])
                                      for c in CACHE_DIGESTS)
    assert seg.shuffle_digests == (views.ShuffleDigest(0, 0, 0, "s0"),)
    assert seg.steps == (views.SegmentStep(1, 2, None), views.SegmentStep(2, 2, 3),
                         views.SegmentStep(3, 2, None))
    assert seg.run_record["settings"]["train_iters"] == 3  # kept whole
    assert seg.run_record_digest == ids.bytes_digest(ids.canonical(seg.run_record))
    assert (seg.trace_route, seg.trace_cause) == ("cache files", None)

    second = s.segment("E/seg-2")
    assert (second.run_record, second.sampler_type, second.launches, second.global_batch_size) == (
        None, None, (), None)  # registered without a run record: nothing is guessed
    assert (second.parent_checkpoint, second.trace_route, second.trace_cause, second.trace_cause_detail) == (
        world.ckpt2, "none", v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    assert s.segment("E/seg-9") is None
    assert s.segments_started_from(world.ckpt2) == ("E/seg-2",)


def segment_args(**changes) -> dict:
    """A registration that is accepted as it stands: its run record holds the digests
    and the launch its route needs. A test that wants no record, or another one, says so."""
    args = dict(run="K", segment="K/seg-1", position=0, mixture_instance_id=None,
                parent_checkpoint=None, start_kind="fresh", steps=[(1, 2, None)],
                trace_route="cache files")
    args.update(changes)
    if "run_record" not in changes:
        launch = {"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0,
                  "step_at_start": 0, "first_step": 1, "last_step": 1}
        args["run_record"] = {**run_record(args["segment"], args["start_kind"], [launch]),
                              "run": args["run"]}
    return args


def test_register_run_segment_again_and_its_refusals(world):
    s = world.store
    before = table_counts(s)
    s.register_run_segment(
        run="E", segment="E/seg-2", position=1, mixture_instance_id=world.mixture,
        parent_checkpoint=world.ckpt2, start_kind="resume", run_record=None,
        steps=[(5, 2, None), (4, 2, None)], trace_route="none",
        trace_cause=v.NAMES_TRACE_MISSING, trace_cause_detail=v.TRACE_MISSING_NO_DIGEST)
    assert table_counts(s) == before  # the same segment again: nothing new
    bad = [
        segment_args(run="E", segment="E/seg-2"),  # the name is taken, other contents
        segment_args(run="E", segment="E/seg-3", position=1),  # the position is taken
        segment_args(start_kind="restart"),
        segment_args(start_kind="resume"),  # a resume with no parent checkpoint
        segment_args(start_kind="warm-start"),
        segment_args(parent_checkpoint=world.ckpt2),  # a fresh start with a parent
        segment_args(start_kind="resume", parent_checkpoint=world.shard),  # not a checkpoint
        segment_args(mixture_instance_id=world.shard),  # not a mixture
        segment_args(trace_route="guess"),
        segment_args(trace_route="none"),  # no named cause
        segment_args(trace_route="none", trace_cause="because"),
        segment_args(trace_cause=v.NAMES_TRACE_MISSING),  # a cause beside a trace route
        segment_args(steps=[(1, 2, None), (1, 2, None)]),
        segment_args(steps=[(1, "two", None)]),
        segment_args(run_record={"format": "something/9"}),
        segment_args(run_record=run_record("K/seg-other", "fresh", [])),  # names another segment
        segment_args(run_record=run_record("K/seg-1", "resume", [])),  # another start kind
        segment_args(run_record={**run_record("K/seg-1", "fresh", []), "run": "K",
                                 "launches": [{"position": 0}]}),
        segment_args(run_record={**run_record("K/seg-1", "fresh", []), "run": "K",
                                 "cache_digests": [{"kind": "document_index", "shard": None, "digest": "d"}]}),
    ]
    for args in bad:
        with pytest.raises(errors.InvalidRecord):
            s.register_run_segment(**args)
    assert table_counts(s) == before and s.segment("K/seg-1") is None and s.runs() == ("E",)


def test_step_traces_read_back_and_show_what_is_untraced(world):
    s = world.store
    first = s.step_trace("E/seg-1", 1)
    assert first == StepTrace(1, 2, (whole_sample(0, 0, 0), whole_sample(1, 1, 1)))
    half = s.step_trace("E/seg-1", 2).samples[1]
    assert half.uncovered == (3, 5) and half.rows == (TraceRow(0, 3, 0, 1), TraceRow(0, 0, 0, 2))
    assert s.step_trace("E/seg-1", 3) is None  # untraced, not a fault
    traces = s.step_traces("E/seg-1")
    assert next(traces).step == 1 and [t.step for t in traces] == [2]  # read lazily
    assert [t.step for t in s.step_traces("E/seg-1", first_step=2, last_step=9)] == [2]

    states = list(s.trace_states("E/seg-1"))
    assert states == [
        views.TraceState(1, 2, None, True, 2, 2, 0),
        views.TraceState(2, 2, 3, True, 2, 2, 1),  # a sample with an uncovered range
        views.TraceState(3, 2, None, False, None, 0, 0),  # no trace
    ]
    assert [x.step for x in s.trace_states("E/seg-1", upto_step=2)] == [1, 2]
    assert [x.traced for x in s.trace_states("E/seg-2")] == [False, False]

    assert list(s.trace_documents("E/seg-1")) == [
        views.ConsumedDocument(0, 0, 1), views.ConsumedDocument(0, 1, 1),
        views.ConsumedDocument(0, 2, 2), views.ConsumedDocument(0, 3, 2)]
    assert [d.ordinal for d in s.trace_documents("E/seg-1", upto_step=1)] == [0, 1]
    assert list(s.trace_hits("E/seg-1", 0, 0)) == [
        views.TraceHit(1, 0, 0, 0, 5, 5, 5), views.TraceHit(2, 1, 1, 0, 2, 5, 3)]
    assert list(s.trace_hits("E/seg-1", 0, 0, upto_step=1)) == [views.TraceHit(1, 0, 0, 0, 5, 5, 5)]
    assert list(s.trace_hits("E/seg-1", 3, 0)) == []


def test_write_step_traces_again_and_its_refusals(world):
    s = world.store
    before = table_counts(s)
    s.write_step_traces("E/seg-1", [StepTrace(1, 2, (whole_sample(1, 1, 1), whole_sample(0, 0, 0)))])
    assert table_counts(s) == before  # the same trace again: nothing new
    bad = [
        ("E/seg-1", StepTrace(3, 1, (whole_sample(0, 4, 0),))),  # one short of the batch size
        ("E/seg-1", StepTrace(3, 2, (whole_sample(0, 4, 0),))),  # counts two, holds one
        ("E/seg-1", StepTrace(9, 2, (whole_sample(0, 4, 0), whole_sample(1, 5, 1)))),  # no such step
        ("E/seg-1", StepTrace(1, 2, (whole_sample(0, 0, 0), whole_sample(1, 1, 2)))),  # another trace
        ("E/seg-1", StepTrace(3, 2, (whole_sample(0, 4, 0), whole_sample(2, 5, 1)))),  # numbering
        ("E/seg-1", StepTrace(3, 2, (whole_sample(0, 4, 0), TraceSample(1, 5, 0, 5, 6, ())))),
        ("E/seg-9", StepTrace(1, 2, (whole_sample(0, 0, 0), whole_sample(1, 1, 1)))),  # no segment
    ]
    for segment, trace in bad:
        with pytest.raises(errors.InvalidRecord):
            s.write_step_traces(segment, [trace])
    assert table_counts(s) == before
    s.write_step_traces("E/seg-1", [StepTrace(3, 2, (whole_sample(0, 4, 0), whole_sample(1, 5, 1)))])
    assert s.step_trace("E/seg-1", 3).consumed_samples == 2


def test_checkpoints_and_their_history_through_ancestors(world):
    s = world.store
    assert world.ckpt2 == checkpoint_instance_id("E/seg-1", 2)
    assert s.checkpoint("E@2") == views.Checkpoint(world.ckpt2, ("E@2",), "E/seg-1", 2)
    assert s.checkpoint(world.ckpt5).names == ("E@5",) and s.checkpoint("E@9") is None
    assert s.artifact("E@2").kind == v.KIND_CHECKPOINT
    assert s.checkpoints_of("E/seg-1") == (s.checkpoint("E@2"),)
    assert s.register_checkpoint(name="E@2-again", segment="E/seg-1", step=2) == world.ckpt2
    assert s.checkpoint("E@2").names == ("E@2", "E@2-again")

    own = s.checkpoint_history("E@2")
    assert own.complete and own.missing is None
    assert own.steps() == (("E/seg-1", 1), ("E/seg-1", 2))  # step 3 came after the checkpoint
    through = s.checkpoint_history("E@5")
    assert through.complete
    assert through.steps() == (("E/seg-1", 1), ("E/seg-1", 2), ("E/seg-2", 4), ("E/seg-2", 5))
    assert [(p.segment, p.checkpoint, p.upto_step, p.upto_registered) for p in through.segments] == [
        ("E/seg-1", world.ckpt2, 2, True), ("E/seg-2", world.ckpt5, 5, True)]
    assert s.checkpoint_history("E@9") is None

    # A checkpoint whose segment was never registered: named by value, history unknown.
    lost = s.register_checkpoint(name="Q@7", segment="Q/seg-1", step=7)
    history = s.checkpoint_history("Q@7")
    assert not history.complete and history.steps() == ()
    assert history.missing == views.Reference(views.REL_CHECKPOINT_SEGMENT, lost, "Q/seg-1", None, False)
    # A segment resumed from a checkpoint the store was never told of.
    unknown_parent = checkpoint_instance_id("W/seg-0", 3)
    s.register_run_segment(**segment_args(run="W", segment="W/seg-1", start_kind="warm-start",
                                          parent_checkpoint=unknown_parent))
    child = s.register_checkpoint(name="W@1", segment="W/seg-1", step=1)
    history = s.checkpoint_history(child)
    assert not history.complete and history.steps() == (("W/seg-1", 1),)
    assert history.missing == views.Reference(views.REL_SEGMENT_PARENT, "W/seg-1", unknown_parent, None, False)
    assert {(r.relation, r.referred) for r in s.named_without_record()} == {
        (views.REL_MIXTURE_SHARD, world.ghost_shard),
        (views.REL_CHECKPOINT_SEGMENT, "Q/seg-1"),
        (views.REL_SEGMENT_PARENT, unknown_parent)}
    assert [r.referrer for r in s.references_to("Q/seg-1")] == [lost]

    with pytest.raises(errors.InvalidRecord, match="registered steps"):
        s.register_checkpoint(name="E@7", segment="E/seg-1", step=7)
    with pytest.raises(errors.InvalidRecord, match="already names another instance"):
        s.register_checkpoint(name="E@2", segment="E/seg-1", step=1)


def test_a_history_cannot_loop(world):
    s = world.store
    # A segment that names, as its parent, a checkpoint of itself that is not registered yet.
    own = checkpoint_instance_id("L/seg-1", 1)
    s.register_run_segment(**segment_args(run="L", segment="L/seg-1", start_kind="resume",
                                          parent_checkpoint=own))
    with pytest.raises(errors.InvalidRecord, match="its own history"):
        s.register_checkpoint(name="L@1", segment="L/seg-1", step=1)
    # A segment that would start from a checkpoint of its own descendant.
    s.register_run_segment(**segment_args(run="M", segment="M/seg-2", start_kind="resume",
                                          parent_checkpoint=checkpoint_instance_id("M/seg-1", 1)))
    tip = s.register_checkpoint(name="M@tip", segment="M/seg-2", step=1)
    s.register_checkpoint(name="M@1", segment="M/seg-1", step=1)
    with pytest.raises(errors.InvalidRecord, match="its own history"):
        s.register_run_segment(**segment_args(run="M", segment="M/seg-1", position=1,
                                              start_kind="resume", parent_checkpoint=tip))


# --- The loud checks (plan 3.4) and the states that are not faults ----------------------


def edit(path: Path, *statements: str | tuple) -> None:
    """Change a store outside the write API, as a fault injection does."""
    raw = sqlite3.connect(path)
    for statement in statements:
        if isinstance(statement, tuple):
            raw.execute(*statement)
        else:
            raw.execute(statement)
    raw.commit()
    raw.close()


def artifact_row(path: Path, instance: str) -> int:
    raw = sqlite3.connect(path)
    found = raw.execute("SELECT artifact_id FROM artifact WHERE instance_id = ?", (instance,)).fetchone()[0]
    raw.close()
    return found


def test_an_unfaulted_store_passes_whole_and_scoped(world):
    s = world.store
    s.check_integrity()
    for instance in (world.A, world.R1, world.shard, world.mixture, world.ckpt2, world.ckpt5):
        s.check_integrity([instance])
    s.check_integrity(["0" * 64])  # an id the store does not hold has no closure
    s.check_integrity(iter([world.ckpt5, world.R1]))
    assert s.missing_files() == () and s.file_state(world.shard) == views.FILE_PRESENT
    assert s.file_state(world.mixture) == views.FILE_NONE == s.file_state("0" * 64)


@pytest.mark.parametrize(
    ("target", "fault", "error"),
    [
        ("shard", "UPDATE binding SET declared_count = declared_count - 1 WHERE artifact_id = ?",
         errors.MapCountMismatch),
        ("shard", "DELETE FROM map_row WHERE ordinal = 1 AND binding_id IN "
                  "(SELECT binding_id FROM binding WHERE artifact_id = ?)", errors.MapCountMismatch),
        ("R1", "UPDATE binding SET declared_count = 5 WHERE artifact_id = ?", errors.MapCountMismatch),
        ("shard", "UPDATE binding SET declared_digest = 'edited' WHERE artifact_id = ?",
         errors.MapDigestMismatch),
        ("A", "UPDATE binding SET declared_digest = 'edited' WHERE artifact_id = ?",
         errors.MapDigestMismatch),
        ("R1", "UPDATE edge_input SET occurrence_id = 987654 WHERE position = 0 AND edge_id IN "
               "(SELECT e.edge_id FROM edge e JOIN occurrence c ON c.occurrence_id = e.child_id "
               "WHERE c.artifact_id = ? AND c.locator = 'L2')", errors.EdgeParentMissing),
        ("shard", "UPDATE map_row SET parent_occurrence_id = 987654 WHERE ordinal = 0 AND binding_id IN "
                  "(SELECT binding_id FROM binding WHERE artifact_id = ?)", errors.EdgeParentMissing),
        ("shard", "UPDATE map_candidate SET occurrence_id = 987654 WHERE ordinal = 2 AND occurrence_id = "
                  "(SELECT MIN(occurrence_id) FROM map_candidate) AND binding_id IN "
                  "(SELECT binding_id FROM binding WHERE artifact_id = ?)", errors.EdgeParentMissing),
        # The registered digest edited: the binding no longer agrees with it. The map
        # check holds the two stored digests against each other and comes first.
        ("shard", "UPDATE artifact SET content_digest = 'edited' WHERE artifact_id = ?",
         errors.MapDigestMismatch),
        ("A", "UPDATE artifact SET content_digest = 'edited' WHERE artifact_id = ?",
         errors.MapDigestMismatch),
        # A snapshot entry whose digest is not the one its shard is registered with.
        ("mixture", "UPDATE mixture_shard SET content_digest = 'edited' WHERE position = 0 "
                    "AND artifact_id = ?", errors.ArtifactDigestMismatch),
    ],
)
def test_each_fault_edited_into_a_copy_stops_the_check(world, target, fault, error):
    instance = getattr(world, target)
    copy = world.copy()
    edit(copy, (fault, (artifact_row(copy, instance),)))
    with Store.open(copy, readonly=True) as s:
        with pytest.raises(error) as caught:
            s.check_integrity()
        assert caught.value.exit_code == errors.EXIT_INTEGRITY
        # Everything downstream holds the fault in its closure: the mixture and both checkpoints.
        for downstream in (world.mixture, world.ckpt2, world.ckpt5):
            with pytest.raises(error):
                s.check_integrity([downstream])
        # A source the fault does not touch is outside it, unless the fault sits on that source.
        if target != "A":
            s.check_integrity([world.A])
            s.check_integrity([world.Z])
    # The store the copy was made from is as it was.
    with Store.open(world.folder / "store.db") as s:
        s.check_integrity()


def test_a_trace_one_sample_short_stops_the_check(world):
    for name, fault in (
        ("field.db", "UPDATE step_trace SET consumed_samples = consumed_samples - 1 WHERE step = 1"),
        ("rows.db", "DELETE FROM trace_sample WHERE sample_pos = 1 AND trace_id = "
                    "(SELECT trace_id FROM step_trace WHERE step = 1)"),
        ("step.db", "DELETE FROM segment_step WHERE step = 1"),  # a trace with no batch size
    ):
        copy = world.copy(name)
        edit(copy, fault)
        with Store.open(copy, readonly=True) as s:
            with pytest.raises(errors.TraceSampleCountMismatch):
                s.check_integrity()
            for checkpoint in (world.ckpt2, world.ckpt5):  # step 1 is in both histories
                with pytest.raises(errors.TraceSampleCountMismatch):
                    s.check_integrity([checkpoint])
            s.check_integrity([world.mixture])  # a mixture has no steps
            s.check_integrity([world.shard])
        world.store = Store.open(world.folder / "store.db")


def test_a_trace_fault_after_a_checkpoints_step_is_outside_its_history(world):
    s = world.store
    s.write_step_traces("E/seg-1", [StepTrace(3, 2, (whole_sample(0, 4, 0), whole_sample(1, 5, 1)))])
    copy = world.copy()
    edit(copy, "UPDATE step_trace SET consumed_samples = 1 WHERE step = 3")
    with Store.open(copy, readonly=True) as s:
        with pytest.raises(errors.TraceSampleCountMismatch):
            s.check_integrity()
        s.check_integrity([world.ckpt2])  # the checkpoint at step 2 never held step 3
        s.check_integrity([world.ckpt5])  # nor does the segment resumed from it


def test_the_faults_are_found_in_the_fixed_order(world):
    """Count, digest, edge parent, trace count, artifact digest: with all five
    faults in one store each is named in turn as the one before it is repaired."""
    release = world.folder / "R1.jsonl"
    good = release.read_bytes()
    release.write_bytes(good.replace(b"passage X", b"passage Y", 1))  # the artifact fault
    copy = world.copy()
    row = artifact_row(copy, world.shard)
    edit(copy,
         "UPDATE step_trace SET consumed_samples = 7 WHERE step = 1",
         "UPDATE edge_input SET occurrence_id = 987654 WHERE edge_id = 1",
         ("UPDATE binding SET declared_digest = 'edited' WHERE artifact_id = ?", (row,)),
         ("UPDATE binding SET declared_count = 9 WHERE artifact_id = ?", (row,)))
    expected = [errors.MapCountMismatch, errors.MapDigestMismatch, errors.EdgeParentMissing,
                errors.TraceSampleCountMismatch, errors.ArtifactDigestMismatch]
    repairs = [
        ("UPDATE binding SET declared_count = 4 WHERE artifact_id = ?", (row,)),
        ("UPDATE binding SET declared_digest = ? WHERE artifact_id = ?", (world.shard_digest, row)),
        "UPDATE edge_input SET occurrence_id = (SELECT MIN(occurrence_id) FROM occurrence) WHERE edge_id = 1",
        "UPDATE step_trace SET consumed_samples = 2 WHERE step = 1",
    ]
    for position, error in enumerate(expected):
        with Store.open(copy, readonly=True) as s:
            with pytest.raises(error):
                s.check_integrity()
        if position < len(repairs):
            edit(copy, repairs[position])
    release.write_bytes(good)
    with Store.open(copy, readonly=True) as s:
        s.check_integrity()


def test_one_changed_byte_in_a_file_is_an_artifact_fault_and_never_a_map_fault(world):
    """DECISIONS.md D-18: with one byte of a release file changed and nothing else,
    the fault named is "artifact digest mismatch". A map fault is never named for
    a changed file: the count a binding declares is held against a file only when
    the file has its registered digest."""
    s = world.store
    release = world.folder / "R1.jsonl"
    good = release.read_bytes()
    changed = good.replace(b"passage X", b"passage Y", 1)
    assert len(changed) == len(good) and changed.count(b"\n") == good.count(b"\n")
    assert sum(a != b for a, b in zip(good, changed)) == 1  # one byte, the same lines
    release.write_bytes(changed)

    with pytest.raises(errors.ArtifactDigestMismatch) as caught:
        s.check_integrity()
    assert caught.value.name == "artifact digest mismatch"
    assert caught.value.exit_code == errors.EXIT_INTEGRITY
    assert caught.value.context["instance"] == world.R1
    assert not isinstance(caught.value, (errors.MapDigestMismatch, errors.MapCountMismatch))
    # Everything that has the release in its ancestor closure stops with the same name.
    for downstream in (world.R1, world.shard, world.mixture, world.ckpt2, world.ckpt5):
        with pytest.raises(errors.ArtifactDigestMismatch):
            s.check_integrity([downstream])
    # What the release was made from is outside the fault.
    for upstream in (world.A, world.B, world.P, world.Z):
        s.check_integrity([upstream])
    # The store's own records are as they were: the binding still agrees with them.
    binding = s.binding(world.R1)
    assert binding.declared_digest == s.artifact(world.R1).content_digest == ids.bytes_digest(good)
    assert binding.declared_count == s.stored_row_count(world.R1) == 4
    release.write_bytes(good)
    s.check_integrity()

    # A line added to a source file: the count on disk differs from the declared
    # count, and the fault is still the artifact's. A is an ancestor of R1.
    source = world.folder / "src/A.jsonl"
    source_good = source.read_bytes()
    source.write_bytes(source_good + b'{"text": "added later"}\n')
    with pytest.raises(errors.ArtifactDigestMismatch) as caught:
        s.check_integrity()
    assert caught.value.name == "artifact digest mismatch"
    assert caught.value.context["instance"] == world.A
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity([world.R1])
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity([world.ckpt5])
    s.check_integrity([world.Z])
    source.write_bytes(source_good)
    s.check_integrity()


def test_a_shard_changed_on_disk_is_an_artifact_fault_with_or_without_a_map(world):
    s = world.store
    # A shard with no map: only the artifact's own digest can catch the change.
    shard, _, _ = fresh_shard(world)
    s.check_integrity()
    write_shard(world.folder / "shards/new", [[1], [2], [9]])
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity()
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity([shard])
    s.check_integrity([world.shard])  # another shard is not in its closure
    write_shard(world.folder / "shards/new", [[1], [2], [3]])
    s.check_integrity()

    # A shard with a map, one token changed: the same fault, not a map fault.
    write_shard(world.folder / "shards/shard-1", [[1, 2, 3], [4, 5], [6], [7]])
    with pytest.raises(errors.ArtifactDigestMismatch) as caught:
        s.check_integrity([world.mixture])
    assert caught.value.context["instance"] == world.shard
    # Its index cut short, so that it cannot be read as an index: still that fault.
    write_shard(world.folder / "shards/shard-1", [[1, 2, 3], [4, 5], [6], [6]])
    s.check_integrity()
    index = world.folder / "shards/shard-1.idx"
    whole = index.read_bytes()
    index.write_bytes(whole[:-8])
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity([world.ckpt5])
    index.write_bytes(whole)
    s.check_integrity()


def test_a_snapshot_entry_is_held_against_its_shards_registered_digest(world):
    s = world.store
    # The world's mixture names a shard the store was never told of: no fault.
    assert [x.registered for x in s.references_to(world.ghost_shard)] == [False]
    s.check_integrity([world.mixture])
    copy = world.copy()
    edit(copy, ("UPDATE mixture_shard SET content_digest = ? WHERE position = 0 AND artifact_id = ?",
                ("0" * 64, artifact_row(copy, world.mixture))))
    with Store.open(copy, readonly=True) as edited:
        with pytest.raises(errors.ArtifactDigestMismatch, match="snapshot") as caught:
            edited.check_integrity([world.mixture])
        assert caught.value.context["shard"] == world.shard
        assert caught.value.context["position"] == 0
        with pytest.raises(errors.ArtifactDigestMismatch):
            edited.check_integrity([world.ckpt5])  # the checkpoint's run read this mixture
        edited.check_integrity([world.shard])  # the shard itself is whole


def test_reading_an_edge_whose_parent_row_is_gone_raises_the_same_fault(world):
    copy = world.copy()
    edit(copy, "UPDATE edge_input SET occurrence_id = 987654 WHERE edge_id = "
               "(SELECT edge_id FROM edge ORDER BY edge_id LIMIT 1)",
         "UPDATE map_row SET parent_occurrence_id = 987654 WHERE ordinal = 1")
    with Store.open(copy, readonly=True) as s:
        with pytest.raises(errors.EdgeParentMissing):
            s.edge_of(world.ref(world.P, 1))
        with pytest.raises(errors.EdgeParentMissing):
            list(s.map_rows(world.shard))
        assert s.map_row(world.shard, 0).parent == world.ref(world.R1, 1)


def test_states_the_proof_reads_are_not_faults(world):
    copy = world.copy()
    shard_row = artifact_row(copy, world.shard)
    edit(
        copy,
        # a deleted trace (step 1) and a lowered coverage mark, its sample still counted
        "DELETE FROM trace_row WHERE trace_id = (SELECT trace_id FROM step_trace WHERE step = 1)",
        "DELETE FROM trace_sample WHERE trace_id = (SELECT trace_id FROM step_trace WHERE step = 1)",
        "DELETE FROM step_trace WHERE step = 1",
        "UPDATE trace_sample SET covered_tokens = 2 WHERE sample_pos = 0 AND trace_id = "
        "(SELECT trace_id FROM step_trace WHERE step = 2)",
        # a deleted edge
        "DELETE FROM edge_input WHERE edge_id = (SELECT MAX(edge_id) FROM edge)",
        "DELETE FROM edge WHERE edge_id = (SELECT MAX(edge_id) FROM edge)",
        # a deleted tokenization map
        ("DELETE FROM map_candidate WHERE binding_id IN "
         "(SELECT binding_id FROM binding WHERE artifact_id = ?)", (shard_row,)),
        ("DELETE FROM map_row WHERE binding_id IN "
         "(SELECT binding_id FROM binding WHERE artifact_id = ?)", (shard_row,)),
        ("DELETE FROM binding WHERE artifact_id = ?", (shard_row,)),
        # labels and bound fields edited after writing: the proof's conditions (f) and (g)
        ("UPDATE edge SET label = 'file-level' WHERE child_id = (SELECT occurrence_id FROM occurrence "
         "WHERE artifact_id = ? AND locator = 'L2')", (artifact_row(copy, world.R1),)),
        "UPDATE edge SET bound_established = 1, log_id = NULL WHERE is_generation = 1",
        "INSERT INTO harness_read (log_id, seq, occurrence_id, outside_ref, refused) "
        "VALUES (1, 99, NULL, 'src/B:L2', 0)",
    )
    with Store.open(copy, readonly=True) as s:
        s.check_integrity()
        for instance in (world.R1, world.shard, world.mixture, world.ckpt2, world.ckpt5):
            s.check_integrity([instance])
        # Each state is there for the proof to read.
        assert s.step_trace("E/seg-1", 1) is None
        states = list(s.trace_states("E/seg-1"))
        assert [(x.traced, x.uncovered_samples) for x in states] == [(False, 0), (True, 2), (False, 0)]
        assert s.edge_of(world.ref(world.R1, 4)) is None
        assert s.bindings(world.shard) == () and list(s.map_rows(world.shard)) == []
        assert s.edge_of(world.ref(world.R1, 2)).label == v.FILE_LEVEL  # as stored, not corrected
        generation = s.edge_of(world.ref(world.P, 1))
        assert generation.bound_field is True and generation.harness_log is None
        log = s.harness_log("paraphrase")
        assert not log.bound_established() and log.reads[-1] == HarnessRead(None, "src/B:L2", False)


def test_an_archived_file_is_a_state_and_half_a_shard_is_a_fault(world):
    s = world.store
    (world.folder / "archive").mkdir()
    os.rename(world.folder / "src/A.jsonl", world.folder / "archive/A.jsonl")
    os.rename(world.folder / "shards/shard-1.bin", world.folder / "archive/shard-1.bin")
    os.rename(world.folder / "shards/shard-1.idx", world.folder / "archive/shard-1.idx")
    s.check_integrity()
    s.check_integrity([world.ckpt5])
    assert s.file_state(world.A) == views.FILE_ABSENT == s.file_state(world.shard)
    missing = s.missing_files()
    assert [(m.instance_id, m.state) for m in missing] == [
        (world.A, views.FILE_ABSENT), (world.shard, views.FILE_ABSENT)]
    assert missing[0].path == str(world.folder / "src/A.jsonl")
    assert [m.instance_id for m in s.missing_files([world.shard])] == [world.A, world.shard]
    assert s.missing_files([world.Z]) == ()

    # The map check needs no file: the two digests the store holds must still agree.
    copy = world.copy()
    edit(copy, ("UPDATE artifact SET content_digest = 'edited' WHERE artifact_id = ?",
                (artifact_row(copy, world.shard),)))
    with Store.open(copy, readonly=True) as edited:
        with pytest.raises(errors.MapDigestMismatch):
            edited.check_integrity()

    # One of a shard's two files back in place: neither archived nor whole.
    world.store = s = Store.open(world.folder / "store.db")
    os.rename(world.folder / "archive/shard-1.idx", world.folder / "shards/shard-1.idx")
    assert s.file_state(world.shard) == views.FILE_PARTIAL
    with pytest.raises(errors.ArtifactDigestMismatch, match="one of the two files"):
        s.check_integrity([world.mixture])
    os.rename(world.folder / "archive/shard-1.bin", world.folder / "shards/shard-1.bin")
    s.check_integrity([world.mixture])


# --- A declared count is held against the file too (plan 2.2(d), 2.3, 3.4) ---------------


def hide_records(copy: Path, instance: str, locators: tuple[str, ...]) -> None:
    """Two edits that hide each other from the store's own rows: occurrence rows of
    an artifact are removed and its binding's declared count is lowered by as many."""
    row = artifact_row(copy, instance)
    marks = ", ".join("?" * len(locators))
    edit(copy,
         (f"DELETE FROM occurrence WHERE artifact_id = ? AND locator IN ({marks})", (row, *locators)),
         ("UPDATE binding SET declared_count = declared_count - ? WHERE artifact_id = ?",
          (len(locators), row)))


def test_records_removed_with_a_lowered_count_are_found_by_the_count_of_the_file(world):
    """A's second line is removed from the store and A's binding declares one record
    fewer. The store agrees with itself, and the file still holds two lines under
    its registered digest: "map count mismatch", from the whole check and from
    every artifact that has A in its ancestor closure."""
    source = world.folder / "src/A.jsonl"
    copy = world.copy()
    hide_records(copy, world.A, ("L2",))
    with Store.open(copy, readonly=True) as s:
        assert s.binding(world.A).declared_count == s.stored_row_count(world.A) == 1
        on_disk = artifact_files.read_jsonl(source)
        assert (on_disk.line_count, on_disk.digest) == (2, s.artifact(world.A).content_digest)

        with pytest.raises(errors.MapCountMismatch) as caught:
            s.check_integrity()
        fault = caught.value
        assert (fault.name, fault.exit_code) == ("map count mismatch", errors.EXIT_INTEGRITY)
        assert fault.context == {
            "declared": 1, "on_disk": 2, "instance": world.A, "path": str(source),
            "namespace": v.NAMESPACE_CAPTURE, "derivation": None,
        }
        assert "declares 1 records" in fault.detail and "holds 2" in fault.detail
        # What a query runs: the check over the closure of the artifact it asks about.
        for holder in (world.A, world.P, world.R1, world.shard, world.mixture, world.ckpt2, world.ckpt5):
            with pytest.raises(errors.MapCountMismatch):
                s.check_integrity([holder])
        for outside in (world.B, world.Z):
            s.check_integrity([outside])

        # A line appended as well: the file is no longer the registered one, and the
        # fault named is the artifact's. The count is held against a file only when
        # the file has its registered digest.
        good = source.read_bytes()
        source.write_bytes(good + b'{"text": "added later"}\n')
        for scope in (None, [world.R1]):
            with pytest.raises(errors.ArtifactDigestMismatch) as caught:
                s.check_integrity(scope)
            assert caught.value.name == "artifact digest mismatch"
        source.write_bytes(good)

        # The file gone from its recorded place: not a fault (DECISIONS.md D-18).
        # The count is then held against the store's rows only, which agree.
        os.rename(source, world.folder / "A.archived")
        s.check_integrity()
        s.check_integrity([world.R1])
        assert [m.instance_id for m in s.missing_files([world.R1])] == [world.A]
        os.rename(world.folder / "A.archived", source)
        with pytest.raises(errors.MapCountMismatch):
            s.check_integrity([world.R1])
    # The store the copy was made from is whole.
    with Store.open(world.folder / "store.db") as s:
        s.check_integrity()


def test_map_rows_removed_with_a_lowered_count_are_found_by_the_count_of_the_index(world):
    """The same pair of edits on a shard's map: its last row is removed and the map
    declares three documents. The shard's index still holds four."""
    prefix = world.folder / "shards/shard-1"
    copy = world.copy()
    row = artifact_row(copy, world.shard)
    of_the_map = "binding_id IN (SELECT binding_id FROM binding WHERE artifact_id = ?)"
    edit(copy,
         (f"DELETE FROM map_candidate WHERE ordinal = 3 AND {of_the_map}", (row,)),
         (f"DELETE FROM map_row WHERE ordinal = 3 AND {of_the_map}", (row,)),
         ("UPDATE binding SET declared_count = declared_count - 1 WHERE artifact_id = ?", (row,)))
    with Store.open(copy, readonly=True) as s:
        assert s.binding(world.shard).declared_count == s.stored_row_count(world.shard) == 3
        assert artifact_files.read_shard_once(prefix).document_count == world.shard_count == 4

        with pytest.raises(errors.MapCountMismatch) as caught:
            s.check_integrity()
        fault = caught.value
        assert fault.context == {
            "declared": 3, "on_disk": 4, "instance": world.shard, "path": str(prefix),
            "namespace": v.NAMESPACE_CAPTURE, "derivation": v.DERIVED_POSITION,
        }
        assert "declares 3 documents" in fault.detail and "holds 4" in fault.detail
        for holder in (world.shard, world.mixture, world.ckpt2, world.ckpt5):
            with pytest.raises(errors.MapCountMismatch):
                s.check_integrity([holder])
        for upstream in (world.A, world.R1):  # the shard is in no closure of these
            s.check_integrity([upstream])

        # One token changed on disk as well: the pair is not the registered one, and
        # the fault named is the artifact's.
        write_shard(prefix, [[1, 2, 3], [4, 5], [6], [7]])
        with pytest.raises(errors.ArtifactDigestMismatch):
            s.check_integrity([world.mixture])
        write_shard(prefix, [[1, 2, 3], [4, 5], [6], [6]])

        # Both files gone: not a fault, and the count is held against the store's
        # rows only. One of the two back: the artifact's fault. Both back: the count.
        (world.folder / "archive").mkdir()
        for suffix in (".bin", ".idx"):
            os.rename(f"{prefix}{suffix}", world.folder / f"archive/shard-1{suffix}")
        s.check_integrity()
        s.check_integrity([world.ckpt5])
        assert [m.instance_id for m in s.missing_files()] == [world.shard]
        os.rename(world.folder / "archive/shard-1.idx", f"{prefix}.idx")
        with pytest.raises(errors.ArtifactDigestMismatch, match="one of the two files"):
            s.check_integrity([world.mixture])
        os.rename(world.folder / "archive/shard-1.bin", f"{prefix}.bin")
        with pytest.raises(errors.MapCountMismatch):
            s.check_integrity([world.mixture])


def test_a_registered_shard_whose_bin_is_not_the_size_its_index_gives_stops_the_check(world):
    """The outside review of 504bf64, finding 2, for a store that registered the pair before
    registration refused it: the digest the store was edited to agree with is of the short
    file, and the map's count cannot be held against a shard the token reader refuses. That
    is the count fault, not a pass and not a negative proof."""
    prefix = world.folder / "shards/shard-1"
    bin_path = Path(f"{prefix}.bin")
    bin_path.write_bytes(bin_path.read_bytes()[:-4])
    cut = artifact_files.shard_digest_only(prefix)
    copy = world.copy()
    row = artifact_row(copy, world.shard)
    edit(copy,
         ("UPDATE artifact SET content_digest = ? WHERE artifact_id = ?", (cut, row)),
         ("UPDATE binding SET declared_digest = ? WHERE artifact_id = ?", (cut, row)))
    with Store.open(copy, readonly=True) as s:
        with pytest.raises(errors.MapCountMismatch, match="cannot be read to count them") as caught:
            s.check_integrity([world.shard])
        assert "its index gives" in caught.value.context["unreadable"]


def test_a_registered_shard_whose_index_cannot_be_counted_stops_the_check(world):
    """An index that cannot be read as an index, under a digest the store was
    edited to agree with: the map's count cannot be held against the file, and
    that is the count fault, not a pass."""
    prefix = world.folder / "shards/shard-1"
    index = Path(f"{prefix}.idx")
    index.write_bytes(index.read_bytes()[:-8])
    cut = artifact_files.shard_digest_only(prefix)
    copy = world.copy()
    row = artifact_row(copy, world.shard)
    edit(copy,
         ("UPDATE artifact SET content_digest = ? WHERE artifact_id = ?", (cut, row)),
         ("UPDATE binding SET declared_digest = ? WHERE artifact_id = ?", (cut, row)))
    with Store.open(copy, readonly=True) as s:
        with pytest.raises(errors.MapCountMismatch, match="cannot be read to count them") as caught:
            s.check_integrity([world.shard])
        assert caught.value.context["declared"] == 4 and caught.value.context["on_disk"] is None
        assert caught.value.context["unreadable"]
    # With the store unedited the same file is the artifact's fault, as before.
    with Store.open(world.folder / "store.db", readonly=True) as s:
        with pytest.raises(errors.ArtifactDigestMismatch):
            s.check_integrity([world.shard])
