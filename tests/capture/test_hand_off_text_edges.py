"""The corners of the hand-off comparison (docs/capture.md section 3, docs/store.md "The
write methods"), driven through the real harness, the real capture wrappers and the real
store. Nothing is stood in for.

The sentences the expectations rest on:

- the comparison is made wherever the store takes an edge that carries the texts its job
  was handed: when an artifact is registered, when the same history is handed again, and
  when an edge found afterwards is added (`add_recovered_edges`); a text that is not the
  record its reference names (compared by fingerprint after the store's normaliser) is
  refused with "invalid record", naming the job and the reference, and nothing is written;
- a reference into the artifact being written names a line of that file: the text is
  compared with the line the file holds, not with a record of another artifact whose
  locator reads the same;
- a handed text that has no fingerprint (a lone surrogate has no UTF-8 form) is not the
  record: the registration is refused, it does not fail with an encoding error;
- an edge that carries texts is a generation edge, each text is held under a reference of
  its log's hand-off, and each is a (reference, text) pair: anything else is refused with
  "invalid record" before the store is read (docs/store.md, the list that begins
  "register_jsonl refuses the following").

Only a hand-built edge reaches `add_recovered_edges` with texts: `generation_edge` builds a
stated edge, which that method refuses, and no recovery in the package builds a generation
edge. The tests of that method therefore change the derivation of a built generation edge,
which is what a person who adds a found generation edge does.

The tests that name the new field of an edge assert first that the field exists, so that a
build without it fails at that assertion and not at a constructor. The tests that hold a
text that is the record are pins: they pass with and without the comparison, and say so in
their names ("as before", "is accepted").
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, errors
from waterwhite import vocabulary as v
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.records import OccurrenceRef
from waterwhite.store import Store

NORMALISER = "control-chars/1"

TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."
OWN_TEXT = "The first line of a file that a second job then extends."
FIXED_TEXT = "A text that the same history always holds."
LONE_SURROGATE = "A text with a lone surrogate \ud800 in it."

STEP = capture.step_ref("generate", "1", {"style": "plain"})
REFUSAL = "invalid record"
NO_FIELD = "the edge has no handed_texts field"


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def build_sources(tmp_path: Path) -> SimpleNamespace:
    """Two sources of one record each, ingested the way a team ingests them."""
    w = SimpleNamespace(folder=tmp_path, db=tmp_path / "evidence.db")
    w.store = Store.create(w.db, normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", [{"id": "A-001", "text": TEXT_A}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    w.ref_b = capture.occurrence(w.src_b, 1)
    return w


def paraphrase(ctx):
    (only,) = ctx.handed()
    return JobOutput(text=ctx.read(only).upper(), used=(only,))


def keeps_out_of_the_text(ctx):
    """A job whose output and log do not depend on the text it was handed."""
    (only,) = ctx.handed()
    ctx.read(only)
    return JobOutput(text=FIXED_TEXT, used=(only,))


def run(w: SimpleNamespace, handed, job=paraphrase, job_name: str = "paraphrase", tag: str = "run"):
    return run_job(job, job_name=job_name, handed=handed, log_path=w.folder / f"logs/{tag}.jsonl")


def row_counts(db: Path) -> dict[str, int]:
    """The number of rows of every table of the store file."""
    raw = sqlite3.connect(db)
    try:
        tables = [r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {table: raw.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
    finally:
        raw.close()


def attempt(action):
    """None when the action went through, otherwise the error it raised. Only an error of
    the package counts as a refusal: any other exception is a crash and fails the test."""
    try:
        action()
    except errors.WaterwhiteError as error:
        return error
    return None


def assert_refused_naming(error, job: str, key: str) -> None:
    assert error is not None, "the registration was accepted"
    assert error.name == REFUSAL, str(error)
    assert job in str(error), str(error)
    assert key in str(error), str(error)
    assert error.context.get("job") == job, str(error.context)
    assert error.context.get("ref") == key, str(error.context)


def with_texts(edge, texts):
    """The edge with the texts its job was handed; fails at an assertion when the build
    has no such field."""
    assert "handed_texts" in {f.name for f in dataclasses.fields(edge)}, NO_FIELD
    return dataclasses.replace(edge, handed_texts=texts)


# --- a history handed again -----------------------------------------------------------------


def register_fixed_file(w: SimpleNamespace, edges, name: str = "gen/S") -> str:
    """The same file every time: the same bytes, so the same history, so the same
    instance id whatever the texts of the job behind its edge."""
    path = w.folder / "gen/shared.jsonl"
    write_rows(path, [{"id": "S-1", "text": FIXED_TEXT}])
    return capture.record_jsonl(
        w.store, path, name=name, role=capture.ROLE_INTERMEDIATE, step=STEP,
        inputs=[w.src_a], edges=edges)


def first_registration(w: SimpleNamespace) -> str:
    right = run(w, [Handed(w.ref_a, TEXT_A)], job=keeps_out_of_the_text, tag="first")
    return register_fixed_file(w, [capture.generation_edge(1, right.outputs[0], right.log, STEP)])


def test_the_same_history_handed_again_with_a_wrong_text_is_refused_and_nothing_is_added(tmp_path):
    w = build_sources(tmp_path)
    try:
        instance = first_registration(w)
        before = row_counts(w.db)
        wrong = run(w, [Handed(w.ref_a, TEXT_B)], job=keeps_out_of_the_text, tag="again")
        edge = capture.generation_edge(1, wrong.outputs[0], wrong.log, STEP)
        refusal = attempt(lambda: register_fixed_file(w, [edge], name="gen/S-again"))
        assert_refused_naming(refusal, "paraphrase", w.ref_a.key)
        assert row_counts(w.db) == before
        assert w.store.artifact("gen/S-again") is None
        assert w.store.artifact("gen/S").instance_id == instance
    finally:
        w.store.close()


def test_the_same_history_handed_again_with_the_right_text_is_accepted_as_before(tmp_path):
    """A pin: it passes with and without the comparison."""
    w = build_sources(tmp_path)
    try:
        instance = first_registration(w)
        again = run(w, [Handed(w.ref_a, TEXT_A)], job=keeps_out_of_the_text, tag="again")
        edge = capture.generation_edge(1, again.outputs[0], again.log, STEP)
        assert register_fixed_file(w, [edge], name="gen/S-again") == instance
        assert w.store.artifact("gen/S-again").instance_id == instance
    finally:
        w.store.close()


# --- an edge added after the artifact was registered ----------------------------------------


def found_generation_edge(result, derivation: str = v.DERIVED_FUNCTION):
    """A generation edge of a job's result with a derivation that is not "stated": the
    form of an edge that is found after the file was registered (a stated edge is part of
    the artifact's identity and is refused by `add_recovered_edges`)."""
    built = capture.generation_edge(1, result.outputs[0], result.log, STEP)
    return dataclasses.replace(
        built, derivation=derivation, verification=v.VERIFIED_PASSED, label=v.RECOVERED)


def import_unlinked_file(w: SimpleNamespace) -> str:
    write_rows(w.folder / "legacy/U.jsonl", [{"id": "U-1", "text": FIXED_TEXT}])
    return capture.import_jsonl(
        w.store, w.folder / "legacy/U.jsonl", name="import/U", member_name="legacy/U")


def test_a_found_generation_edge_with_a_wrong_text_is_refused_and_nothing_is_written(tmp_path):
    w = build_sources(tmp_path)
    try:
        instance = import_unlinked_file(w)
        before = row_counts(w.db)
        wrong = run(w, [Handed(w.ref_a, TEXT_B)], job=keeps_out_of_the_text)
        edge = found_generation_edge(wrong)
        refusal = attempt(lambda: w.store.add_recovered_edges(instance, [edge]))
        assert_refused_naming(refusal, "paraphrase", w.ref_a.key)
        assert row_counts(w.db) == before
        assert w.store.harness_log("paraphrase") is None
        assert w.store.edge_of(OccurrenceRef(instance, "L1")) is None
    finally:
        w.store.close()


def test_a_found_generation_edge_with_the_right_text_is_added_as_before(tmp_path):
    """A pin: it passes with and without the comparison."""
    w = build_sources(tmp_path)
    try:
        instance = import_unlinked_file(w)
        right = run(w, [Handed(w.ref_a, TEXT_A)], job=keeps_out_of_the_text)
        w.store.add_recovered_edges(instance, [found_generation_edge(right)])
        stored = w.store.edge_of(OccurrenceRef(instance, "L1"))
        assert stored is not None
        assert (stored.is_generation, stored.label, stored.job_name) == (True, v.RECOVERED, "paraphrase")
        assert w.store.harness_log("paraphrase") is not None
    finally:
        w.store.close()


# --- a handed reference into the artifact being written --------------------------------------


def register_extended_file(w: SimpleNamespace, name: str, own_text: str) -> str:
    """A file of two lines. Line 1 was made by a job handed the record of A; line 2 by a
    job handed line 1 of the file itself, with `own_text` as that line's text."""
    own_line_1 = OccurrenceRef(capture.OWN_ARTIFACT, capture.line_locator(1))
    first = run(w, [Handed(w.ref_a, TEXT_A)], job=keeps_out_of_the_text, job_name="first-job", tag=f"{name}-1")
    second = run(
        w, [Handed(own_line_1, own_text)], job=keeps_out_of_the_text, job_name="second-job", tag=f"{name}-2")
    path = w.folder / f"gen/{name}.jsonl"
    write_rows(path, [{"id": f"{name}-1", "text": OWN_TEXT}, {"id": f"{name}-2", "text": FIXED_TEXT}])
    return capture.record_jsonl(
        w.store, path, name=f"gen/{name}", role=capture.ROLE_INTERMEDIATE, step=STEP, inputs=[w.src_a],
        edges=[capture.generation_edge(1, first.outputs[0], first.log, STEP),
               capture.generation_edge(2, second.outputs[0], second.log, STEP)])


def test_a_handed_reference_into_the_artifact_being_written_with_the_right_text_is_accepted(tmp_path):
    """A pin: it passes with and without the comparison."""
    w = build_sources(tmp_path)
    try:
        instance = register_extended_file(w, "own-right", OWN_TEXT)
        assert w.store.artifact("gen/own-right").instance_id == instance
        stored = w.store.edge_of(OccurrenceRef(instance, "L2"))
        assert stored is not None and stored.parents == (OccurrenceRef(instance, "L1"),)
    finally:
        w.store.close()


def test_a_handed_reference_into_the_artifact_being_written_is_compared_with_its_own_line(tmp_path):
    """The text handed under line 1 of the file being written is the text of line 1 of
    A's record (a record whose locator reads the same): it is not the line the file holds."""
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        refusal = attempt(lambda: register_extended_file(w, "own-wrong", TEXT_A))
        assert refusal is not None, "the registration was accepted"
        assert refusal.name == REFUSAL, str(refusal)
        assert "second-job" in str(refusal), str(refusal)
        assert refusal.context.get("job") == "second-job", str(refusal.context)
        key = refusal.context.get("ref", "")
        assert key in str(refusal) and key.endswith(":L1"), str(refusal.context)
        assert key.split(":")[0] not in (w.src_a, w.src_b), str(refusal.context)
        assert row_counts(w.db) == before
        assert w.store.artifact("gen/own-wrong") is None
    finally:
        w.store.close()


# --- a text that has no fingerprint -------------------------------------------------------------


def test_a_lone_surrogate_in_a_handed_text_is_refused_not_a_crash(tmp_path):
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        result = run(w, [Handed(w.ref_a, LONE_SURROGATE)], job=keeps_out_of_the_text)
        edge = capture.generation_edge(1, result.outputs[0], result.log, STEP)
        refusal = attempt(lambda: register_fixed_file(w, [edge]))
        assert_refused_naming(refusal, "paraphrase", w.ref_a.key)
        assert row_counts(w.db) == before
    finally:
        w.store.close()


# --- the refusals of the shape of the texts ---------------------------------------------------


def generation_edge_of(w: SimpleNamespace, texts_handed: str = TEXT_A):
    result = run(w, [Handed(w.ref_a, texts_handed)], job=keeps_out_of_the_text)
    return capture.generation_edge(1, result.outputs[0], result.log, STEP)


def test_texts_on_an_edge_that_is_not_a_generation_edge_are_refused(tmp_path):
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        copy = capture.copy_edge(1, w.ref_a, STEP)
        edge = with_texts(copy, ((w.ref_a, TEXT_A),))
        assert not edge.is_generation
        refusal = attempt(lambda: register_fixed_file(w, [edge]))
        assert refusal is not None, "the registration was accepted"
        assert refusal.name == REFUSAL, str(refusal)
        assert "generation edge" in str(refusal) and "text" in str(refusal), str(refusal)
        assert row_counts(w.db) == before
    finally:
        w.store.close()


def test_a_text_handed_under_a_reference_outside_its_logs_hand_off_is_refused(tmp_path):
    """The job was handed A only. The text of B, which is B's record, is held under B's
    reference: it would match its record, but B is not in the log's hand-off. The edge's
    inputs stay inside the hand-off."""
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        built = generation_edge_of(w)
        edge = with_texts(built, ((w.ref_a, TEXT_A), (w.ref_b, TEXT_B)))
        assert tuple(item.parent for item in edge.inputs) == (w.ref_a,)
        refusal = attempt(lambda: register_fixed_file(w, [edge]))
        assert_refused_naming(refusal, "paraphrase", w.ref_b.key)
        assert row_counts(w.db) == before
    finally:
        w.store.close()


@pytest.mark.parametrize("shape", ["three-items", "text-is-not-text", "a-bare-reference"])
def test_a_handed_text_that_is_not_a_reference_and_text_pair_is_refused(tmp_path, shape):
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        malformed = {
            "three-items": (w.ref_a, TEXT_A, "another item"),
            "text-is-not-text": (w.ref_a, b"bytes are not a text"),
            "a-bare-reference": w.ref_a,
        }[shape]
        edge = with_texts(generation_edge_of(w), (malformed,))
        refusal = attempt(lambda: register_fixed_file(w, [edge]))
        assert refusal is not None, "the registration was accepted"
        assert refusal.name == REFUSAL, str(refusal)
        assert "pair" in str(refusal), str(refusal)
        assert row_counts(w.db) == before
    finally:
        w.store.close()
