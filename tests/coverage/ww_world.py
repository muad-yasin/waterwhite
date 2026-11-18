"""A small hand-built store for the developer tests of the proof, the answers, the
listing and the command line.

Everything is written through the store's write methods: tiny JSONL files, tiny
hand-made .bin/.idx pairs, and runs, traces and checkpoints given directly. No
Megatron, no replica. Faults are then applied to the SQLite file from outside the
write API, as the acceptance checks do.

The world (the names follow the fixture's, the contents are this file's own):

- sources A, B, C, Z, one member each (``A/r1`` and so on);
- ``gen/P``: a paraphrase of A's line 1, made by the job "paraphrase";
- ``gen/T``: a translation of P, made by the job "translation";
- ``gen/S``: a summary of A's line 2 and C's line 1, made by the job "summary";
- ``R1``: X (dedup kept A's copy, B's is the loser), P, T, S and two lines of Z;
- ``R2``: B's two lines and one line of Z;
- ``shard-1``: the three records of R2; ``shard-3``: T and the two Z lines of R1;
- ``mix-1`` = [shard-1], ``mix-2`` = [shard-1, shard-3];
- run E: ``E/seg-1`` steps 1 to 3 on mix-1, checkpoint ``E@3``; ``E/seg-2`` steps 4
  to 6 on mix-2 from ``E@3``: step 4 reads T, the operator's step record gives 3 of
  the 5 tokens of step 5's sample as read; checkpoints ``E@4`` and ``E@6``;
- run D: ``D/seg-1`` steps 1 and 2 on mix-1 from ``E@6``, checkpoint ``D@2``;
- run Y: ``Y/seg-1`` steps 1 and 2 on mix-2 from ``E@3``, never reading T,
  checkpoint ``Y@2``.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import struct
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
SAMPLER_CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
JOIN = ("core_v0.20.0", "single", SAMPLER_CLASS)
SAMPLE_TOKENS = 5


def fp(text: str) -> bytes:
    return ids.fingerprint(text)


def step(name: str, params: object = None) -> StepRef:
    return StepRef(name, "1", ids.param_digest(params))


def ref(instance: str, line: int) -> OccurrenceRef:
    return OccurrenceRef(instance, f"L{line}")


def write_jsonl(path: Path, texts: list[str]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return len(texts), ids.file_digest(path)


# The text of every record this module wrote, by (instance id, locator): a job "handed" a
# record is handed its text, so the edges ``generation_edge`` builds carry the texts and the
# store marks their logs compared (a log without its texts establishes no bound).
TEXTS: dict[tuple[str, str], str] = {}


def occurrences(texts: list[str]) -> list[OccurrenceRecord]:
    return [OccurrenceRecord(f"L{n}", fp(t) if t else None) for n, t in enumerate(texts, start=1)]


def write_shard(prefix: Path, documents: list[list[int]]) -> tuple[int, str]:
    """A hand-made shard in the index layout the store reads."""
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
    digest = ids.shard_digest(ids.file_digest(f"{prefix}.bin"), ids.file_digest(f"{prefix}.idx"))
    return len(lengths), digest


def ingest(store: Store, path: Path, texts: list[str], source: str, name: str,
           member: str | None = None) -> str:
    count, digest = write_jsonl(path, texts)
    member = member or f"{source}/r1"
    instance = store.register_jsonl(
        path, name=name, role=v.ROLE_SOURCE, step=None, member_name=member, inputs=(),
        occurrences=occurrences(texts), declared_count=count, declared_digest=digest,
    )
    store.add_source(source)
    store.add_member(source, member, "r1", views.PART_FILE, [(instance, "")])
    TEXTS.update({(instance, f"L{n}"): t for n, t in enumerate(texts, start=1)})
    return instance


def record(store: Store, path: Path, texts: list[str], name: str, role: str, step_ref: StepRef,
           inputs: list[str], edges: list[EdgeRecord]) -> str:
    count, digest = write_jsonl(path, texts)
    instance = store.register_jsonl(
        path, name=name, role=role, step=step_ref, member_name=None, inputs=inputs,
        occurrences=occurrences(texts), edges=edges, declared_count=count, declared_digest=digest,
    )
    TEXTS.update({(instance, f"L{n}"): t for n, t in enumerate(texts, start=1)})
    return instance


def copy_edge(line: int, parent: OccurrenceRef, name: str = "merge") -> EdgeRecord:
    return EdgeRecord(f"L{line}", (EdgeInput(parent),), step(name))


def generation_edge(line: int, job: str, handed: list[OccurrenceRef],
                    used: list[OccurrenceRef], reads: list[HarnessRead] | None = None) -> EdgeRecord:
    """One output of a job run in the harness: every handed record is an input, the
    used ones are parents, the log lists one read per handed record."""
    log = HarnessLog(
        job, f"log-digest-{job}", tuple(handed),
        tuple(reads if reads is not None else [HarnessRead(occurrence=h) for h in handed]), "none",
        harness=v.HARNESS_CONFINED,
    )
    texts = tuple((h, TEXTS[(h.instance_id, h.locator)]) for h in handed
                  if (h.instance_id, h.locator) in TEXTS)
    return EdgeRecord(
        f"L{line}", tuple(EdgeInput(h, used=h in used) for h in handed), step("generate"),
        is_generation=True, harness_log=log, job_name=job, handed_texts=texts,
    )


def shard(store: Store, prefix: Path, name: str, inputs: list[str], rows: list[MapRow] | None,
          *, namespace: str = v.NAMESPACE_CAPTURE, verification: str = v.VERIFIED_PASSED,
          label: str = v.RECOVERED) -> tuple[str, str]:
    """Register a shard of one-token documents and, unless ``rows`` is None, its
    map. Returns (instance id, content digest)."""
    documents = [[n + 1] for n in range(len(rows) if rows is not None else 3)]
    count, digest = write_shard(prefix, documents)
    instance = store.register_shard(prefix, name=name, step=step("tokenize", name), inputs=inputs)
    if rows is not None:
        store.write_map(
            instance,
            MapRecord(namespace, v.DERIVED_POSITION, verification, label, True, True, tuple(rows)),
            declared_count=count, declared_digest=digest,
        )
    return instance, digest


def recovered(ordinal: int, parent: OccurrenceRef) -> MapRow:
    return MapRow(ordinal, ordinal, ordinal + 1, v.RECOVERED, parent)


def twins(ordinal: int, *candidates: OccurrenceRef) -> MapRow:
    return MapRow(ordinal, ordinal, ordinal + 1, v.CANDIDATES, candidates=tuple(candidates))


def sample(pos: int, index: int, shard_position: int, ordinal: int, *,
           covered: int = SAMPLE_TOKENS) -> TraceSample:
    """A sample of five tokens that reads one document."""
    return TraceSample(pos, index, 0, SAMPLE_TOKENS, covered,
                       (TraceRow(shard_position, ordinal, 0, SAMPLE_TOKENS),))


def run_record(run: str, segment: str, start_kind: str, first: int, last: int, *,
               sampler: str = "single", release: str = JOIN[0],
               sampler_class: str = SAMPLER_CLASS, shuffle: bool = False) -> dict:
    """A run record of one launch. ``shuffle`` records the digest of one epoch's order
    for rank 0, which a run read with the cyclic sampler needs to have a trace route."""
    return {
        "format": "waterwhite.run-record/1",
        "run": run,
        "segment": segment,
        "start_kind": start_kind,
        "parent_checkpoint": None,
        "settings": {
            "dataloader_type": sampler,
            "sampler_class": sampler_class,
            "data_sharding": True,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "sequence_length": SAMPLE_TOKENS - 1,
            "add_extra_token_to_sequence": True,
            "random_seed": 1234,
            "split": "1,0,0",
            "train_iters": last,
            "train_samples": None,
            "batch_schedule": None,
        },
        "versions": {"release_tag": release, "megatron_core": "0.20.0", "megatron_bridge": None,
                     "torch": "2.0"},
        "shards": [],
        "launches": [{"position": 0, "world_size": 1, "data_parallel_size": 1,
                      "consumed_at_start": 0, "step_at_start": first - 1, "first_step": first,
                      "last_step": last}],
        # A route of "cache files" or "replay" rests on recorded digests (plan 3.5): these
        # are the three arrays of one shard, with digests no file here has. The tests that
        # use this record read traces the store was handed; none compares an array.
        "cache_digests": [{"kind": kind, "shard": 0, "digest": f"{n}" * 64}
                          for n, kind in enumerate(("document_index", "sample_index", "shuffle_index"), 1)],
        "shuffle_digests": ([{"launch": 0, "epoch": 0, "rank": 0, "digest": "0" * 64, "count": 1}]
                            if shuffle else []),
    }


def segment(store: Store, run: str, name: str, position: int, mixture: str | None,
            parent: str | None, steps: list[tuple[int, int, int | None]],
            reads: dict[int, tuple[int, int]] | None, *, start_kind: str | None = None,
            route: str = "cache files", cause: str | None = None, detail: str | None = None,
            **record_args: str) -> None:
    """Register a segment whose steps each read one document: ``reads`` maps a step
    to (shard position, ordinal). None for ``reads`` writes no trace at all."""
    kind = start_kind or ("fresh" if parent is None else "resume")
    numbers = [s[0] for s in steps]
    store.register_run_segment(
        run=run, segment=name, position=position, mixture_instance_id=mixture,
        parent_checkpoint=parent, start_kind=kind,
        run_record=run_record(run, name, kind, numbers[0], numbers[-1], **record_args),
        steps=steps, trace_route=route, trace_cause=cause, trace_cause_detail=detail,
    )
    if reads is not None:
        notes = {s[0]: s[2] for s in steps}
        traces = []
        for number, (position_, ordinal) in sorted(reads.items()):
            covered = SAMPLE_TOKENS if notes[number] is None else notes[number]
            traces.append(StepTrace(number, 1, (sample(0, number, position_, ordinal, covered=covered),)))
        store.write_step_traces(name, traces)


@dataclasses.dataclass
class World:
    folder: Path
    store: Store
    ids: dict[str, str]

    @property
    def path(self) -> Path:
        return self.folder / "store.db"

    def __getitem__(self, name: str) -> str:
        return self.ids[name]

    def ref(self, name: str, line: int) -> OccurrenceRef:
        return ref(self.ids[name], line)

    def sql(self, *statements: str | tuple[str, tuple]) -> None:
        """Edit the store file from outside the write API: each statement is text,
        or (text, parameters). The store is closed first and opened again
        afterwards, as a fault on a copy would be read. A statement that changes
        no row is an error of the test: the fault was not applied."""
        self.store.close()
        raw = sqlite3.connect(self.path)
        try:
            for statement in statements:
                text, params = statement if isinstance(statement, tuple) else (statement, ())
                if raw.execute(text, params).rowcount == 0:
                    raise AssertionError(f"the edit changed no row: {text}")
            raw.commit()
        finally:
            raw.close()
        self.store = Store.open(self.path)

    def scalar(self, query: str, params: tuple = ()):
        row = self.store._conn.execute(query, params).fetchone()
        if row is None:
            raise AssertionError(f"no row: {query} {params}")
        return row[0]

    def occurrence_id(self, name: str, line: int) -> int:
        return self.scalar(
            "SELECT o.occurrence_id FROM occurrence o JOIN artifact a ON a.artifact_id = o.artifact_id "
            "WHERE a.instance_id = ? AND o.locator = ?", (self.ids[name], f"L{line}"))

    def edge_id(self, name: str, line: int) -> int:
        return self.scalar("SELECT edge_id FROM edge WHERE child_id = ?",
                           (self.occurrence_id(name, line),))

    def binding_id(self, name: str, namespace: str = "capture") -> int:
        return self.scalar(
            "SELECT b.binding_id FROM binding b JOIN artifact a ON a.artifact_id = b.artifact_id "
            "WHERE a.instance_id = ? AND b.namespace = ?", (self.ids[name], namespace))

    def trace_id(self, segment_name: str, step_number: int) -> int:
        return self.scalar(
            "SELECT t.trace_id FROM step_trace t JOIN run_segment s ON s.segment_id = t.segment_id "
            "WHERE s.name = ? AND t.step = ?", (segment_name, step_number))

    def log_id(self, job: str) -> int:
        return self.scalar("SELECT log_id FROM harness_log WHERE job_name = ?", (job,))

    # --- The faults the tests apply, each from outside the write API ---------------

    def delete_edge(self, name: str, line: int) -> None:
        edge = self.edge_id(name, line)
        self.sql(("DELETE FROM edge_input WHERE edge_id = ?", (edge,)),
                 ("DELETE FROM edge WHERE edge_id = ?", (edge,)))

    def delete_map(self, name: str) -> None:
        binding = self.binding_id(name)
        self.sql(("DELETE FROM map_row WHERE binding_id = ?", (binding,)),
                 ("DELETE FROM binding WHERE binding_id = ?", (binding,)))

    def delete_trace(self, segment_name: str, step_number: int) -> None:
        trace = self.trace_id(segment_name, step_number)
        self.sql(("DELETE FROM trace_row WHERE trace_id = ?", (trace,)),
                 ("DELETE FROM trace_sample WHERE trace_id = ?", (trace,)),
                 ("DELETE FROM step_trace WHERE trace_id = ?", (trace,)))

    def remove_log(self, name: str, line: int, job: str) -> None:
        """Remove a generation edge's harness log and leave its bound field as it is."""
        log = self.log_id(job)
        self.sql(("UPDATE edge SET log_id = NULL WHERE edge_id = ?", (self.edge_id(name, line),)),
                 ("DELETE FROM harness_read WHERE log_id = ?", (log,)),
                 ("DELETE FROM harness_handed WHERE log_id = ?", (log,)),
                 ("DELETE FROM harness_log WHERE log_id = ?", (log,)))


def build(folder: Path) -> World:
    store = Store.create(folder / "store.db", normaliser=NORMALISER, verified_joins=[JOIN])
    made: dict[str, str] = {}
    a = made["src/A"] = ingest(store, folder / "src/A.jsonl", ["passage X", "passage SA"], "A", "src/A")
    b = made["src/B"] = ingest(store, folder / "src/B.jsonl", ["passage X", "b two"], "B", "src/B")
    c = made["src/C"] = ingest(store, folder / "src/C.jsonl", ["passage SC"], "C", "src/C")
    z = made["src/Z"] = ingest(store, folder / "src/Z.jsonl", ["z one", "z two", "z three"], "Z", "src/Z")

    p = made["gen/P"] = record(
        store, folder / "gen/P.jsonl", ["paraphrase of X"], "gen/P", v.ROLE_INTERMEDIATE,
        step("generate", "P"), [a],
        [generation_edge(1, "paraphrase", [ref(a, 1)], [ref(a, 1)])],
    )
    t = made["gen/T"] = record(
        store, folder / "gen/T.jsonl", ["Uebersetzung von P"], "gen/T", v.ROLE_INTERMEDIATE,
        step("generate", "T"), [p],
        [generation_edge(1, "translation", [ref(p, 1)], [ref(p, 1)])],
    )
    s = made["gen/S"] = record(
        store, folder / "gen/S.jsonl", ["summary of SA and SC"], "gen/S", v.ROLE_INTERMEDIATE,
        step("generate", "S"), [a, c],
        [generation_edge(1, "summary", [ref(a, 2), ref(c, 1)], [ref(a, 2), ref(c, 1)])],
    )
    dedup = EdgeRecord(
        "L1", (EdgeInput(ref(a, 1), selected=True), EdgeInput(ref(b, 1))), step("dedup"),
        parent_mode=v.MODE_ANY,
    )
    r1 = made["R1"] = record(
        store, folder / "R1.jsonl",
        ["passage X", "paraphrase of X", "Uebersetzung von P", "summary of SA and SC", "z one", "z two"],
        "R1", v.ROLE_RELEASE, step("release", "R1"), [a, b, p, t, s, z],
        [dedup, copy_edge(2, ref(p, 1)), copy_edge(3, ref(t, 1)), copy_edge(4, ref(s, 1)),
         copy_edge(5, ref(z, 1)), copy_edge(6, ref(z, 2))],
    )
    r2 = made["R2"] = record(
        store, folder / "R2.jsonl", ["passage X", "b two", "z three"], "R2", v.ROLE_RELEASE,
        step("release", "R2"), [b, z],
        [copy_edge(1, ref(b, 1)), copy_edge(2, ref(b, 2)), copy_edge(3, ref(z, 3))],
    )

    s1, d1 = shard(store, folder / "shards/shard-1", "shard-1", [r2],
                   [recovered(0, ref(r2, 1)), recovered(1, ref(r2, 2)), recovered(2, ref(r2, 3))])
    s3, d3 = shard(store, folder / "shards/shard-3", "shard-3", [r1],
                   [recovered(0, ref(r1, 3)), recovered(1, ref(r1, 5)), recovered(2, ref(r1, 6))])
    made["shard-1"], made["shard-3"] = s1, s3
    m1 = made["mix-1"] = store.register_mixture(
        name="mix-1", shards=[SnapshotShard(0, s1, "shards/shard-1", d1)], settings={"mix": 1})
    m2 = made["mix-2"] = store.register_mixture(
        name="mix-2",
        shards=[SnapshotShard(0, s1, "shards/shard-1", d1), SnapshotShard(1, s3, "shards/shard-3", d3)],
        settings={"mix": 2},
    )

    # Run E. Segment 1 reads shard-1 only. Segment 2 reads T (shard position 1,
    # ordinal 0) at step 4; step 5 has an operator note of 3 of 5 tokens.
    segment(store, "E", "E/seg-1", 0, m1, None, [(1, 1, None), (2, 1, None), (3, 1, None)],
            {1: (0, 0), 2: (0, 1), 3: (0, 2)})
    e3 = made["E@3"] = store.register_checkpoint(name="E@3", segment="E/seg-1", step=3)
    segment(store, "E", "E/seg-2", 1, m2, e3, [(4, 1, None), (5, 1, 3), (6, 1, None)],
            {4: (1, 0), 5: (1, 1), 6: (0, 0)})
    made["E@4"] = store.register_checkpoint(name="E@4", segment="E/seg-2", step=4)
    e6 = made["E@6"] = store.register_checkpoint(name="E@6", segment="E/seg-2", step=6)
    # D descends from E@6 and reads shard-1 only. Y branches from E@3, trains on
    # mix-2 and never reads T.
    segment(store, "D", "D/seg-1", 0, m1, e6, [(1, 1, None), (2, 1, None)], {1: (0, 1), 2: (0, 2)},
            start_kind="warm-start")
    made["D@2"] = store.register_checkpoint(name="D@2", segment="D/seg-1", step=2)
    segment(store, "Y", "Y/seg-1", 0, m2, e3, [(1, 1, None), (2, 1, None)], {1: (1, 1), 2: (1, 2)},
            start_kind="warm-start")
    made["Y@2"] = store.register_checkpoint(name="Y@2", segment="Y/seg-1", step=2)
    return World(folder, store, made)
