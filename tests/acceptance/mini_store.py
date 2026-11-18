"""A small store written by hand, with the names of the fixture, for the tests of the
faults (``test_faults.py``).

The faults of ``acceptance_faults`` are put on records the replica's build and the
run wrapper make. A fault written against those records and never run against any
would be untested until the runs exist, so this file writes the few records a fault
reads, through the store's own write methods and with no Megatron, in seconds:

- the sources ``src/A`` and ``src/B`` (X@A is line 1 of A, X@B line 1 of B), the
  generation output ``gen/P`` (made from X@A by the job ``paraphrase``) and the
  release ``R2`` (two lines of B);
- the shards ``E/shard-1``, ``E/shard-2`` and ``E/shard-3``, each with its capture map
  (position-derived, verified, label recovered);
- the mixture ``E/mix-1`` (shards 1 and 2);
- run E: the segment ``E/seg-1`` of six steps of four samples each, 33 tokens a
  sample, with a trace for every step.

The values are this file's own; no expected value of a row is taken from here.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from waterwhite import ids
from waterwhite import vocabulary as v
from waterwhite.records import (
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
    StepTrace,
    TraceRow,
    TraceSample,
)
from waterwhite.store import views
from waterwhite.store.store import Store

NORMALISER = "control-chars/1"
SAMPLER = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
SAMPLE_TOKENS = 33
BATCH = 4
STEPS = 6
DOCUMENTS = 3


def _step(name: str, params: object = None) -> StepRef:
    return StepRef(name, "1", ids.param_digest(params))


def _occurrences(texts: list[str]) -> list[OccurrenceRecord]:
    return [OccurrenceRecord(f"L{n}", ids.fingerprint(t)) for n, t in enumerate(texts, start=1)]


def _write_jsonl(path: Path, texts: list[str]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return len(texts), ids.file_digest(path)


def _write_shard(prefix: Path, documents: list[list[int]]) -> tuple[int, str]:
    """A shard of one-token documents in the index layout the store reads."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for document in documents:
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
    return len(lengths), ids.shard_digest(ids.file_digest(f"{prefix}.bin"),
                                          ids.file_digest(f"{prefix}.idx"))


def _ref(instance: str, line: int) -> OccurrenceRef:
    return OccurrenceRef(instance, f"L{line}")


@dataclass
class Mini:
    folder: Path
    ids: dict[str, str]

    @property
    def store(self) -> Path:
        return self.folder / "store.sqlite"


def _log(job: str, handed: list[OccurrenceRef]) -> HarnessLog:
    return HarnessLog(job, f"log-digest-{job}", tuple(handed),
                      tuple(HarnessRead(occurrence=h) for h in handed), "none",
                      harness=v.HARNESS_CONFINED)


def build(folder: Path) -> Mini:
    folder = Path(folder)
    folder.mkdir(parents=True)
    store = Store.create(folder / "store.sqlite", normaliser=NORMALISER,
                         verified_joins=[("core_v0.20.0", "single", SAMPLER)])
    made: dict[str, str] = {}

    def ingest(name: str, source: str, texts: list[str]) -> str:
        count, digest = _write_jsonl(folder / f"{name}.jsonl", texts)
        instance = store.register_jsonl(
            folder / f"{name}.jsonl", name=name, role=v.ROLE_SOURCE, step=None,
            member_name=f"{source}/r1", inputs=(), occurrences=_occurrences(texts),
            declared_count=count, declared_digest=digest)
        store.add_source(source)
        store.add_member(source, f"{source}/r1", "r1", views.PART_FILE, [(instance, "")])
        return made.setdefault(name, instance)

    a = ingest("src/A", "A", ["passage X", "passage SA"])
    b = ingest("src/B", "B", ["passage X", "b two"])

    handed = [_ref(a, 1)]
    edge = EdgeRecord("L1", (EdgeInput(handed[0]),), _step("generate", "P"), is_generation=True,
                      harness_log=_log("paraphrase", handed), job_name="paraphrase")
    count, digest = _write_jsonl(folder / "gen/P.jsonl", ["paraphrase of X"])
    made["gen/P"] = store.register_jsonl(
        folder / "gen/P.jsonl", name="gen/P", role=v.ROLE_INTERMEDIATE,
        step=_step("generate", "P"), member_name=None, inputs=[a],
        occurrences=_occurrences(["paraphrase of X"]), edges=[edge],
        declared_count=count, declared_digest=digest)

    texts = ["passage X", "b two"]
    copy = [EdgeRecord(f"L{n}", (EdgeInput(_ref(b, n)),), _step("merge")) for n in (1, 2)]
    count, digest = _write_jsonl(folder / "R2.jsonl", texts)
    made["R2"] = store.register_jsonl(
        folder / "R2.jsonl", name="R2", role=v.ROLE_RELEASE, step=_step("release", "R2"),
        member_name=None, inputs=[b], occurrences=_occurrences(texts), edges=copy,
        declared_count=count, declared_digest=digest)

    snapshot = []
    for position, name in enumerate(("E/shard-1", "E/shard-2", "E/shard-3")):
        prefix = folder / "shards" / name.split("/")[1]
        count, digest = _write_shard(prefix, [[n + 1] for n in range(DOCUMENTS)])
        instance = store.register_shard(prefix, name=name, step=_step("tokenize", name),
                                        inputs=[made["R2"]])
        rows = [MapRow(n, n, n + 1, v.RECOVERED, _ref(made["R2"], 1 + n % 2))
                for n in range(DOCUMENTS)]
        store.write_map(
            instance, MapRecord(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED,
                                v.RECOVERED, True, True, tuple(rows)),
            declared_count=count, declared_digest=digest)
        made[name] = instance
        snapshot.append(SnapshotShard(position, instance, f"shards/{name.split('/')[1]}", digest))
    made["E/mix-1"] = store.register_mixture(name="E/mix-1", shards=snapshot[:2],
                                             settings={"mix": 1})

    record = {
        "format": "waterwhite.run-record/1", "run": "E", "segment": "E/seg-1",
        "start_kind": "fresh", "parent_checkpoint": None,
        "settings": {"dataloader_type": "single", "sampler_class": SAMPLER,
                     "data_sharding": True, "micro_batch_size": BATCH,
                     "global_batch_size": BATCH, "sequence_length": SAMPLE_TOKENS - 1,
                     "add_extra_token_to_sequence": True, "random_seed": 578, "split": "1,0,0",
                     "train_iters": STEPS, "train_samples": None, "batch_schedule": None},
        "versions": {"release_tag": "core_v0.20.0", "megatron_core": "0.20.0",
                     "megatron_bridge": None, "torch": "2.0"},
        "shards": [],
        "launches": [{"position": 0, "world_size": 1, "data_parallel_size": 1,
                      "consumed_at_start": 0, "step_at_start": 0, "first_step": 1,
                      "last_step": STEPS}],
        # A route of "cache files" rests on recorded digests (plan 3.5): the three
        # arrays of one shard, with digests no file here has; the rows that use this
        # store read the traces it was handed and compare no array.
        "cache_digests": [{"kind": kind, "shard": 0, "digest": f"{n}" * 64}
                          for n, kind in enumerate(("document_index", "sample_index",
                                                    "shuffle_index"), 1)],
        "shuffle_digests": [],
    }
    store.register_run_segment(
        run="E", segment="E/seg-1", position=0, mixture_instance_id=made["E/mix-1"],
        parent_checkpoint=None, start_kind="fresh", run_record=record,
        steps=[(n, BATCH, None) for n in range(1, STEPS + 1)], trace_route="cache files")
    traces = []
    for step in range(1, STEPS + 1):
        samples = tuple(
            TraceSample(pos, (step - 1) * BATCH + pos, 0, SAMPLE_TOKENS, SAMPLE_TOKENS,
                        (TraceRow(0, pos % DOCUMENTS, 0, SAMPLE_TOKENS),))
            for pos in range(BATCH))
        traces.append(StepTrace(step, BATCH, samples))
    store.write_step_traces("E/seg-1", traces)
    made["E@6"] = store.register_checkpoint(name="E@6", segment="E/seg-1", step=6)
    store.close()
    return Mini(folder, made)


def build_scratch(folder: Path) -> Mini:
    """A small stand-in for the table's store ``scratch-500`` (fixture/table/README.md
    section 4), with no Megatron: the source A, a source ``c04`` ingested from a file of
    500 lines as ``c04/shard-500.input``, and the shard ``c04/shard-500`` of 500
    one-token documents, registered with no map. The shard's tokens are this file's own
    (the writes of Check 4 ask what the store holds, not what the tokens are)."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    store = Store.create(folder / "store.sqlite", normaliser=NORMALISER)
    made: dict[str, str] = {}

    def ingest(name: str, source: str, texts: list[str]) -> str:
        count, digest = _write_jsonl(folder / f"{name}.jsonl", texts)
        instance = store.register_jsonl(
            folder / f"{name}.jsonl", name=name, role=v.ROLE_SOURCE, step=None,
            member_name=f"{source}/r1", inputs=(), occurrences=_occurrences(texts),
            declared_count=count, declared_digest=digest)
        store.add_source(source)
        store.add_member(source, f"{source}/r1", "r1", views.PART_FILE, [(instance, "")])
        made[name] = instance
        return instance

    ingest("src/A", "A", ["passage X", "passage SA"])
    texts = [f"line {n}" for n in range(500)]
    count, digest = _write_jsonl(folder / "c04" / "input.jsonl", texts)
    made["c04/shard-500.input"] = store.register_jsonl(
        folder / "c04" / "input.jsonl", name="c04/shard-500.input", role=v.ROLE_SOURCE, step=None,
        member_name="c04/r1", inputs=(), occurrences=_occurrences(texts),
        declared_count=count, declared_digest=digest)
    store.add_source("c04")
    store.add_member("c04", "c04/r1", "r1", views.PART_FILE,
                     [(made["c04/shard-500.input"], "")])
    prefix = folder / "shards" / "shard-500"
    _write_shard(prefix, [[n + 1] for n in range(500)])
    made["c04/shard-500"] = store.register_shard(
        prefix, name="c04/shard-500", step=_step("tokenize"),
        inputs=[made["c04/shard-500.input"]])
    store.close()
    return Mini(folder, made)
