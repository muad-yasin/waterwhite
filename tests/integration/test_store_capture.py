"""The capture wrappers, the harness and the store, run together (plan 3.1, 3.3, 3.4).

A small pipeline is written here as a team would write it. It imports four things
of the tool and no more: ``waterwhite.capture``, ``waterwhite.harness``,
``waterwhite.normalise`` and ``waterwhite.store.Store``. It writes its own small
files, calls the wrappers at its capture points, and the tests read the result
back through the store's read methods.

The tests then hold the rules these modules share: one definition of a line, one
identity (plan Decision 29), and every refusal that DECISIONS.md D-14, D-15, D-18,
D-20 and D-21 fix. What a query answers from these records is the work of other
modules and is not tested here. Nothing here imports torch or Megatron.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, harness, normalise
from waterwhite.store import Store

# What the tests, not the pipeline, read with: the error names, the id functions
# an expected id is written out with, and the result types.
from waterwhite import errors, ids
from waterwhite import vocabulary as v
from waterwhite.store import views

NORMALISER = "control-chars/1"

X = "The square behind the old station fills with vans before dawn."
SA = "A repair workshop meets once a month."
SC = "The library lends donated tools."
U = "An unrelated record of source B."
P = "Before dawn the square behind the old station is full of vans."
S = "Two schemes keep household goods in use."
T = "Der Platz hinter dem alten Bahnhof fuellt sich vor der Daemmerung."
FAMILY = "file family: translations made from source A"


# --- the pipeline ----------------------------------------------------------------------


def write_rows(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(data)
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table_counts(store: Store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def paraphrase(ctx):
    """Handed one record. It reads it and states nothing about what it used."""
    (only,) = ctx.handed()
    ctx.read(only)
    return harness.JobOutput(text=P, used=())


def summary(ctx):
    """Handed two records. It reads both and states the first only."""
    first, second = ctx.handed()
    ctx.read(first)
    ctx.read(second)
    return harness.JobOutput(text=S, used=(first,))


RELEASE_STEP = ("release", "1", {"dedup": "normalised text"})


def release(store, path, name, world, kept):
    """The release step: line 1 is the passage three records supply, of which dedup
    keeps ``kept``; line 2 is the paraphrase; line 3 an unrelated record of B."""
    step = capture.step_ref(*RELEASE_STEP)
    group = [world.x_a, world.x_b, world.x_b_dirty]
    selection, drops = capture.selection_edge(1, group, kept, step)
    edges = [
        selection,
        capture.copy_edge(2, capture.occurrence(world.gen_p, 1), step),
        capture.copy_edge(3, capture.occurrence(world.src_b, 2), step),
    ]
    return capture.record_jsonl(
        store, path, name=name, role=capture.ROLE_RELEASE, step=step,
        inputs=[world.src_a, world.src_b, world.gen_p], edges=edges, drops=drops,
    )


@pytest.fixture
def world(tmp_path):
    """Everything the pipeline makes, up to the first release."""
    w = SimpleNamespace(folder=tmp_path)
    w.store = store = Store.create(tmp_path / "evidence.db", normaliser=NORMALISER)

    # Two sources hold the passage X, word for word. B also holds it once more with
    # a control character and a Windows line end, which the normaliser removes.
    w.a_bytes = write_rows(tmp_path / "sources/A.jsonl", [
        {"id": "A-001", "text": X}, {"id": "A-002", "text": SA}])
    w.b_bytes = write_rows(tmp_path / "sources/B.jsonl", [
        {"id": "B-001", "text": X}, {"id": "B-002", "text": U},
        {"id": "B-003", "text": X.replace("vans", "va\u0000ns") + "\r\n"}])
    write_rows(tmp_path / "sources/C.jsonl", [{"id": "C-001", "text": SC}])
    w.src_a = capture.ingest_jsonl(store, tmp_path / "sources/A.jsonl", source="A", member="A/r1",
                                   revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(store, tmp_path / "sources/B.jsonl", source="B", member="B/r1",
                                   revision="r1", name="src/B")
    w.src_c = capture.ingest_jsonl(store, tmp_path / "sources/C.jsonl", source="C", member="C/r1",
                                   revision="r1", name="src/C")
    w.x_a = capture.occurrence(w.src_a, 1)
    w.s_a = capture.occurrence(w.src_a, 2)
    w.x_b = capture.occurrence(w.src_b, 1)
    w.x_b_dirty = capture.occurrence(w.src_b, 3)
    w.s_c = capture.occurrence(w.src_c, 1)

    # A generation job in the harness, and its output file with the job's edge.
    w.p_log_path = tmp_path / "logs/paraphrase.jsonl"
    w.p_result = harness.run_job(paraphrase, job_name="paraphrase",
                                 handed=[harness.Handed(w.x_a, X)], log_path=w.p_log_path)
    w.p_step = capture.step_ref("paraphrase", "1", {"style": "plain"})
    w.p_bytes = write_rows(tmp_path / "gen/P.jsonl", [{"id": "P", "text": w.p_result.outputs[0].text}])
    w.gen_p = capture.record_jsonl(
        store, tmp_path / "gen/P.jsonl", name="gen/P", role=capture.ROLE_INTERMEDIATE,
        step=w.p_step, inputs=[w.src_a],
        edges=capture.generation_edges(w.p_result, [1], w.p_step))

    # The release, with the dedup selection that kept A's copy of X.
    w.r1_bytes = write_rows(tmp_path / "releases/R1.jsonl", [{"text": X}, {"text": P}, {"text": U}])
    w.r1 = release(store, tmp_path / "releases/R1.jsonl", "R1", w, kept=w.x_a)
    yield w
    store.close()


# --- ingest ------------------------------------------------------------------------------


def test_two_sources_with_an_identical_record_are_two_occurrences_with_one_fingerprint(world):
    s = world.store
    assert world.src_a == ids.ingest_instance_id(sha256(world.a_bytes), "A/r1")
    assert world.src_b == ids.ingest_instance_id(sha256(world.b_bytes), "B/r1")
    a, b = s.artifact("src/A"), s.artifact("src/B")
    assert (a.kind, a.role, a.step, a.member_name, a.path) == (
        v.KIND_JSONL, v.ROLE_SOURCE, None, "A/r1", "sources/A.jsonl")
    assert (s.occurrence_count(world.src_a), s.occurrence_count(world.src_b)) == (2, 3)

    # Equal after the store's normaliser means one fingerprint, in whichever source.
    fingerprint = normalise.text_fingerprint(s.normaliser, X)
    assert fingerprint == hashlib.blake2b(X.encode("utf-8"), digest_size=16).digest()
    twins = [o.ref for o in s.occurrences_by_fingerprint(fingerprint)]
    assert twins[:3] == [world.x_a, world.x_b, world.x_b_dirty]
    assert s.occurrence(world.x_a).fingerprint == s.occurrence(world.x_b).fingerprint
    # The same text in two sources is two records with two keys and two origins.
    assert world.x_a.key != world.x_b.key
    assert [m.name for m in s.members_covering(world.x_a)] == ["A/r1"]
    assert [m.name for m in s.members_covering(world.x_b)] == ["B/r1"]
    assert s.member("A/r1") == views.Member("A/r1", "A", "r1", v.PART_FILE, ((world.src_a, ""),))
    assert s.sources() == ("A", "B", "C")
    assert s.edge_of(world.x_a) is None  # a source record has no edge

    # The ingest run again is the same instance and writes nothing.
    before = table_counts(s)
    again = capture.ingest_jsonl(s, world.folder / "sources/A.jsonl", source="A", member="A/r1",
                                 revision="r1", name="src/A")
    assert again == world.src_a and table_counts(s) == before
    # The same bytes under another member are another instance.
    other = capture.ingest_jsonl(s, world.folder / "sources/A.jsonl", source="A2", member="A2/r1",
                                 revision="r1", name="src/A2")
    assert other != world.src_a and s.artifact(other).content_digest == a.content_digest
    # A member name that is taken by another source or revision: the ingest is
    # refused before its first write, so no file is left that no member covers.
    before = table_counts(s)
    for changes in (dict(source="B"), dict(revision="r2")):
        with pytest.raises(errors.InvalidRecord, match="registered already"):
            capture.ingest_jsonl(s, world.folder / "sources/B.jsonl", name="src/B-again",
                                 **{"source": "A", "member": "A/r1", "revision": "r1", **changes})
        assert table_counts(s) == before and s.artifact("src/B-again") is None
    s.check_integrity()


def test_capture_and_the_store_agree_on_the_lines_of_an_awkward_file(world):
    """The wrapper declares a line count, the store counts the file itself, and both
    use one definition of a line. Empty lines, a carriage return, an unparseable
    line and a last line with no line feed must all come out the same."""
    s = world.store
    data = (b'{"text": "one"}\r\n'          # the carriage return stays in line 1
            b"\n"                            # an empty line is a line
            b'{"text": "a\rb"}\r{"text": "c"}\n'  # a lone carriage return ends no line
            b"not json at all\n"
            b'{"text": "last line, no line feed"}')
    path = world.folder / "sources/awkward.jsonl"
    path.write_bytes(data)
    instance = capture.ingest_jsonl(s, path, source="W", member="W/r1", revision="r1")
    binding = s.binding(instance)
    assert (binding.declared_count, binding.declared_digest) == (5, sha256(data))
    assert s.stored_row_count(instance) == 5
    assert [o.ref.locator for o in s.occurrences(instance)] == ["L1", "L2", "L3", "L4", "L5"]
    assert [o.fingerprint is None for o in s.occurrences(instance)] == [False, True, True, True, False]
    assert s.occurrence(capture.occurrence(instance, 1)).fingerprint == normalise.text_fingerprint(
        NORMALISER, "one")
    empty = world.folder / "sources/empty.jsonl"
    empty.write_bytes(b"")
    nothing = capture.ingest_jsonl(s, empty, source="W", member="W/empty", revision="r1")
    assert s.binding(nothing).declared_count == 0 and s.occurrence_count(nothing) == 0
    s.check_integrity()


# --- generation --------------------------------------------------------------------------


def test_a_generation_job_through_the_harness_and_its_edge_in_the_store(world):
    s = world.store
    # The id is the one plan 3.1 fixes: content, step, inputs, the stated edge.
    edge_digest = ids.edge_digest([ids.edge_record("L1", [(world.src_a, "L1")], None)])
    assert world.gen_p == ids.instance_id(
        sha256(world.p_bytes), "paraphrase", "1", ids.param_digest({"style": "plain"}),
        [world.src_a], edge_digest)
    artifact = s.artifact("gen/P")
    assert (artifact.role, artifact.step, artifact.inputs, artifact.edge_digest) == (
        v.ROLE_INTERMEDIATE, world.p_step, (world.src_a,), edge_digest)

    edge = s.edge_of(capture.occurrence(world.gen_p, 1))
    assert (edge.is_generation, edge.parent_mode, edge.derivation, edge.label, edge.transform) == (
        True, v.MODE_ALL, v.DERIVED_STATED, v.OBSERVED, world.p_step)
    # The job was handed one record and its output states nothing: the sole handed
    # record is the parent all the same (DECISIONS.md D-21).
    assert world.p_result.outputs[0].used == ()
    assert edge.parents == (world.x_a,) and edge.in_scope_only == ()

    # The stored log is the file the harness wrote: its digest, its hand-off, its reads.
    log = s.harness_log("paraphrase")
    assert log == edge.harness_log == world.p_result.log == harness.read_log(world.p_log_path)
    assert log.log_digest == sha256(world.p_log_path.read_bytes())
    assert log.handed == (world.x_a,) and [r.occurrence for r in log.reads] == [world.x_a]
    # An in-process log: its reads lie within the hand-off, and it establishes no
    # bound, so the stored field says so (Decision 13: only a confined log does).
    assert log.model == "none" and log.reads_within_hand_off()
    assert not log.bound_established() and edge.bound_field is False
    assert X.encode() not in world.p_log_path.read_bytes()  # no record text in the log

    assert [e.child for e in s.edges_naming(world.x_a)] == [
        capture.occurrence(world.gen_p, 1), capture.occurrence(world.r1, 1)]
    s.check_integrity([world.gen_p])


def test_a_job_handed_two_records_names_the_stated_one_as_parent_and_the_other_in_scope(world):
    s = world.store
    result = harness.run_job(
        summary, job_name="summary",
        handed=[harness.Handed(world.s_a, SA), harness.Handed(world.s_c, SC)],
        log_path=world.folder / "logs/summary.jsonl")
    step = capture.step_ref("summary", "1")
    write_rows(world.folder / "gen/S.jsonl", [{"id": "S", "text": result.outputs[0].text}])
    gen_s = capture.record_jsonl(
        s, world.folder / "gen/S.jsonl", name="gen/S", role=capture.ROLE_INTERMEDIATE, step=step,
        inputs=[world.src_a, world.src_c], edges=capture.generation_edges(result, [1], step))
    edge = s.edge_of(capture.occurrence(gen_s, 1))
    assert edge.parents == (world.s_a,) and edge.in_scope_only == (world.s_c,)
    assert edge.harness_log.reads_within_hand_off()
    # The record in scope only is not part of the id: the edge digest names SA alone.
    assert s.artifact(gen_s).edge_digest == ids.edge_digest(
        [ids.edge_record("L1", [(world.src_a, "L2")], None)])
    # Both records are found from below, each edge saying which it is.
    assert [e.child for e in s.edges_naming(world.s_c)] == [capture.occurrence(gen_s, 1)]


# --- selection ---------------------------------------------------------------------------


def test_a_selection_is_an_any_edge_with_the_kept_copy_selected_and_a_drop_per_loser(world):
    s = world.store
    edge = s.edge_of(capture.occurrence(world.r1, 1))
    assert (edge.parent_mode, edge.derivation, edge.label, edge.is_generation) == (
        v.MODE_ANY, v.DERIVED_STATED, v.OBSERVED, False)
    assert edge.parents == (world.x_a, world.x_b, world.x_b_dirty)
    assert edge.selected == world.x_a

    losers = s.drops_in_group("L1")
    assert losers == (
        views.StoredDrop(world.x_b, v.DROP_DEDUP_LOSER, "release", world.r1, "L1"),
        views.StoredDrop(world.x_b_dirty, v.DROP_DEDUP_LOSER, "release", world.r1, "L1"))
    assert tuple(s.drops_by_output(world.r1)) == losers
    assert s.drops_of(world.x_b) == losers[:1] and s.drops_of(world.x_a) == ()
    assert [d.input for d in s.drops_from(world.src_b)] == [world.x_b, world.x_b_dirty]

    # The other lines are plain copies.
    assert s.edge_of(capture.occurrence(world.r1, 2)).parents == (capture.occurrence(world.gen_p, 1),)
    assert s.edge_of(capture.occurrence(world.r1, 3)).parents == (capture.occurrence(world.src_b, 2),)
    assert s.artifact("R1").inputs == (world.src_a, world.src_b, world.gen_p)
    s.check_integrity([world.r1])


# --- identity ----------------------------------------------------------------------------


def test_a_release_registered_twice_is_one_instance(world):
    s = world.store
    before = table_counts(s)
    again = release(s, world.folder / "releases/R1.jsonl", "R1", world, kept=world.x_a)
    assert again == world.r1 and table_counts(s) == before
    # Under a second name: one more name and nothing else.
    alias = release(s, world.folder / "releases/R1.jsonl", "R1-alias", world, kept=world.x_a)
    after = table_counts(s)
    assert alias == world.r1
    assert after.pop("artifact_name") == before.pop("artifact_name") + 1 and after == before
    assert s.artifact("R1").names == ("R1", "R1-alias")
    # The same bytes at another place, the same history: still that instance.
    (world.folder / "releases/R1-copy.jsonl").write_bytes(world.r1_bytes)
    assert release(s, world.folder / "releases/R1-copy.jsonl", "R1", world, kept=world.x_a) == world.r1
    assert s.artifact("R1").path == "releases/R1.jsonl"


def test_the_same_release_with_the_other_copy_selected_is_a_new_instance_with_the_same_bytes(world):
    s = world.store
    (world.folder / "releases/R1-swap.jsonl").write_bytes(world.r1_bytes)
    swap = release(s, world.folder / "releases/R1-swap.jsonl", "R1-swap", world, kept=world.x_b)
    one, two = s.artifact("R1"), s.artifact("R1-swap")
    assert swap != world.r1
    assert one.content_digest == two.content_digest == sha256(world.r1_bytes)
    assert (one.step, one.inputs) == (two.step, two.inputs)  # only the kept copy differs
    assert one.edge_digest != two.edge_digest
    # The occurrences of X carry different keys and different selected parents.
    assert capture.occurrence(world.r1, 1).key != capture.occurrence(swap, 1).key
    assert s.edge_of(capture.occurrence(world.r1, 1)).selected == world.x_a
    assert s.edge_of(capture.occurrence(swap, 1)).selected == world.x_b
    assert s.edge_of(capture.occurrence(swap, 1)).parents == (world.x_a, world.x_b, world.x_b_dirty)

    # Each build has its own losers. B's dirty copy is dropped by the same step
    # name once for each release: the output is part of what makes a drop one drop.
    assert [d.input for d in s.drops_by_output(swap)] == [world.x_a, world.x_b_dirty]
    assert [d.output_instance_id for d in s.drops_of(world.x_b_dirty)] == [world.r1, swap]
    assert [d.output_instance_id for d in s.drops_in_group("L1")] == [world.r1, world.r1, swap, swap]

    # Two instances share their bytes, so the digest alone names neither.
    with pytest.raises(errors.AmbiguousArtifact) as caught:
        s.resolve(one.content_digest)
    assert caught.value.name == "ambiguous artifact"
    assert s.resolve("R1-swap").instance_id == swap
    # The name R1 names the first history for good.
    with pytest.raises(errors.InvalidRecord, match="already names another instance"):
        release(s, world.folder / "releases/R1.jsonl", "R1", world, kept=world.x_b)
    s.check_integrity()


# --- import ------------------------------------------------------------------------------


def test_an_import_with_a_file_family_gets_one_file_level_link_and_no_edge(world):
    s = world.store
    data = write_rows(world.folder / "legacy/T.jsonl", [{"text": T}, {"text": X}])
    imported = capture.import_jsonl(
        s, world.folder / "legacy/T.jsonl", name="import/T", member_name="legacy/T",
        file_family=("A/r1", FAMILY))
    assert imported == ids.ingest_instance_id(sha256(data), "legacy/T")
    artifact = s.artifact("import/T")
    assert (artifact.role, artifact.step, artifact.member_name, artifact.inputs) == (
        v.ROLE_IMPORT, None, "legacy/T", ())

    link = views.SourceLink(imported, None, "A/r1", "A", v.FILE_LEVEL, FAMILY, v.DERIVED_FILE, None)
    assert s.file_source_links(imported) == (link,)
    for line in (1, 2):
        ref = capture.occurrence(imported, line)
        assert s.source_links(ref) == (link,)
        assert s.edge_of(ref) is None  # no record-level link: which item it came from is unknown
        assert s.members_covering(ref) == ()  # an import is no member of a source
    assert list(s.record_source_links(imported)) == []
    # Line 2 equals X after normalisation. That is a fingerprint, not a parent.
    assert capture.occurrence(imported, 2) in [
        o.ref for o in s.occurrences_by_fingerprint(s.occurrence(world.x_a).fingerprint)]

    # Again: the same instance, the same one link.
    before = table_counts(s)
    assert capture.import_jsonl(
        s, world.folder / "legacy/T.jsonl", name="import/T", member_name="legacy/T",
        file_family=("A/r1", FAMILY)) == imported
    assert table_counts(s) == before
    # The same bytes under another member name: another instance, with no link.
    other = capture.import_jsonl(s, world.folder / "legacy/T.jsonl", name="import/T2",
                                 member_name="legacy/T2")
    assert other != imported and s.file_source_links(other) == ()
    # A file family must name a registered member.
    before = table_counts(s)
    with pytest.raises(errors.InvalidRecord, match="not registered"):
        capture.import_jsonl(s, world.folder / "legacy/T.jsonl", name="import/T3",
                             member_name="legacy/T3", file_family=("Q/r1", FAMILY))
    # The refusal comes before the first write: no import is left without its link.
    assert table_counts(s) == before and s.artifact("import/T3") is None
    s.check_integrity()


# --- the harness refuses, and the bound stays ------------------------------------------------


def test_a_refused_outside_read_is_in_the_stored_log_and_is_no_read_outside_the_hand_off(world):
    s = world.store
    refusals = []

    def curious(ctx):
        (only,) = ctx.handed()
        text = ctx.read(only)
        try:
            ctx.read(world.x_b)  # B's copy was not handed
        except errors.HarnessRefused as refusal:
            refusals.append(refusal)
        return harness.JobOutput(text=text.upper(), used=(only,))

    log_path = world.folder / "logs/curious.jsonl"
    result = harness.run_job(curious, job_name="curious", handed=[harness.Handed(world.x_a, X)],
                             log_path=log_path)
    (refusal,) = refusals
    assert (refusal.name, refusal.exit_code) == ("read outside the hand-off", errors.EXIT_REFUSED)
    assert refusal.context == {"job": "curious", "ref": world.x_b.key}

    step = capture.step_ref("curious", "1")
    write_rows(world.folder / "gen/Q.jsonl", [{"text": result.outputs[0].text}])
    made = capture.record_jsonl(
        s, world.folder / "gen/Q.jsonl", name="gen/Q", role=capture.ROLE_INTERMEDIATE, step=step,
        inputs=[world.src_a], edges=capture.generation_edges(result, [1], step))

    edge = s.edge_of(capture.occurrence(made, 1))
    log = s.harness_log("curious")
    assert log == edge.harness_log == result.log
    assert [(r.occurrence, r.outside_ref, r.refused) for r in log.reads] == [
        (world.x_a, None, False), (None, world.x_b.key, True)]
    # A refused attempt delivered nothing, so it is no read outside the hand-off
    # (DECISIONS.md D-07). The log is an in-process one: no bound (Decision 13).
    assert log.reads_within_hand_off() and edge.bound_field is False
    assert log.log_digest == sha256(log_path.read_bytes())
    assert edge.parents == (world.x_a,)  # B's copy is no input of the edge
    assert [e.child for e in s.edges_naming(world.x_b)] == [capture.occurrence(world.r1, 1)]

    # A job that does not catch the refusal fails; its log still shows the attempt
    # and still gives an established bound.
    def careless(ctx):
        ctx.read(world.x_b)
        return harness.JobOutput(text="never written", used=())

    failed_path = world.folder / "logs/careless.jsonl"
    with pytest.raises(errors.HarnessRefused):
        harness.run_job(careless, job_name="careless", handed=[harness.Handed(world.x_a, X)],
                        log_path=failed_path)
    failed = harness.read_log(failed_path)
    assert [(r.occurrence, r.outside_ref, r.refused) for r in failed.reads] == [
        (None, world.x_b.key, True)]
    assert failed.reads_within_hand_off() and failed.model == "none"
    assert json.loads(failed_path.read_bytes().splitlines()[-1])["status"] == "failed"
    assert s.harness_log("careless") is None  # nothing of it reached the store
    s.check_integrity()


# --- refusals: a second registration that disagrees (D-14) ------------------------------------


def test_registering_the_same_output_with_another_log_is_refused(world):
    """The job is run again and this time asks for a record it was not handed. The
    output file, the step, the inputs and the parent are the same, so the instance
    id is the same. The harness log is not covered by the id and differs: the
    second registration is refused, and the stored record stays as it was."""
    s = world.store

    def paraphrase_again(ctx):
        (only,) = ctx.handed()
        ctx.read(only)
        try:
            ctx.read(world.x_b)
        except errors.HarnessRefused:
            pass
        return harness.JobOutput(text=P, used=())

    second = harness.run_job(paraphrase_again, job_name="paraphrase",
                             handed=[harness.Handed(world.x_a, X)],
                             log_path=world.folder / "logs/paraphrase-2.jsonl")
    assert second.log.log_digest != world.p_result.log.log_digest
    before = table_counts(s)
    with pytest.raises(errors.InvalidRecord, match="registered already .* the harness log") as caught:
        capture.record_jsonl(
            s, world.folder / "gen/P.jsonl", name="gen/P", role=capture.ROLE_INTERMEDIATE,
            step=world.p_step, inputs=[world.src_a],
            edges=capture.generation_edges(second, [1], world.p_step))
    assert (caught.value.name, caught.value.exit_code) == ("invalid record", errors.EXIT_WRITE_REFUSED)
    assert caught.value.context["instance"] == world.gen_p
    assert table_counts(s) == before
    assert s.harness_log("paraphrase") == world.p_result.log

    # The first log again, read back from its file: everything agrees.
    first = harness.JobResult(world.p_result.outputs, harness.read_log(world.p_log_path))
    assert capture.record_jsonl(
        s, world.folder / "gen/P.jsonl", name="gen/P", role=capture.ROLE_INTERMEDIATE,
        step=world.p_step, inputs=[world.src_a],
        edges=capture.generation_edges(first, [1], world.p_step)) == world.gen_p
    assert table_counts(s) == before


def test_registering_the_same_output_with_other_in_scope_records_a_label_or_a_role_is_refused(world):
    s = world.store
    step = capture.step_ref("summary", "1")
    path = world.folder / "gen/S.jsonl"
    write_rows(path, [{"id": "S", "text": S}])

    def run(name, handed):
        return harness.run_job(summary_of_first, job_name="summary", handed=handed,
                               log_path=world.folder / f"logs/{name}.jsonl")

    def summary_of_first(ctx):
        first = ctx.handed()[0]
        ctx.read(first)
        return harness.JobOutput(text=S, used=(first,))

    def register(result, **changes):
        fields = dict(name="gen/S", role=capture.ROLE_INTERMEDIATE, step=step,
                      inputs=[world.src_a, world.src_c],
                      edges=capture.generation_edges(result, [1], step))
        fields.update(changes)
        return capture.record_jsonl(s, path, **fields)

    two = run("summary-two", [harness.Handed(world.s_a, SA), harness.Handed(world.s_c, SC)])
    gen_s = register(two)
    before = table_counts(s)

    # Handed a third record, which the output does not state: the parents are the
    # same, so the id is the same, and the records in scope only differ.
    three = run("summary-three", [harness.Handed(world.s_a, SA), harness.Handed(world.s_c, SC),
                                  harness.Handed(world.x_a, X)])
    edges = capture.generation_edges(three, [1], step)
    assert [i.used for i in edges[0].inputs] == [True, False, False]
    with pytest.raises(errors.InvalidRecord, match="registered already .* the edge of L1"):
        register(three)
    # Another role for the same history.
    with pytest.raises(errors.InvalidRecord, match="registered already .* the role"):
        register(two, role=capture.ROLE_RELEASE)
    # A drop the stored instance does not hold.
    with pytest.raises(errors.InvalidRecord, match="registered already .* the drop"):
        register(two, drops=[capture.drop(world.x_a, capture.DROP_FILTER, step)])
    # A fingerprint taken from another text field.
    with pytest.raises(errors.InvalidRecord, match="another fingerprint"):
        capture.record_jsonl(s, path, name="gen/S", role=capture.ROLE_INTERMEDIATE, step=step,
                             inputs=[world.src_a, world.src_c],
                             edges=capture.generation_edges(two, [1], step), text_field="id")
    assert table_counts(s) == before
    assert s.edge_of(capture.occurrence(gen_s, 1)).in_scope_only == (world.s_c,)

    # A label is not covered by the id either: a recovered link handed again as
    # "unknown" disagrees with the stored one.
    write_rows(world.folder / "legacy/X.jsonl", [{"text": X}])
    recovered = capture.EdgeRecord(
        "L1", (capture.EdgeInput(world.x_a),), capture.step_ref("recover-by-fingerprint", "1"),
        derivation=v.DERIVED_FINGERPRINT, verification=v.VERIFIED_PASSED, label=v.RECOVERED)
    scan = capture.scan_jsonl(world.folder / "legacy/X.jsonl", normaliser=s.normaliser)

    def register_import(edge):
        return s.register_jsonl(
            world.folder / "legacy/X.jsonl", name="import/X", role=v.ROLE_IMPORT, step=None,
            member_name="legacy/X", inputs=(), occurrences=scan.occurrences, edges=[edge],
            declared_count=scan.count, declared_digest=scan.digest)

    imported = register_import(recovered)
    assert register_import(recovered) == imported
    with pytest.raises(errors.InvalidRecord, match="registered already .* the edge of L1"):
        register_import(dataclasses.replace(recovered, verification=v.VERIFIED_NOT_RUN, label=v.UNKNOWN))
    assert s.edge_of(capture.occurrence(imported, 1)).label == v.RECOVERED


# --- refusals: parents (D-15, D-21) ------------------------------------------------------------


def test_a_stated_parent_outside_the_declared_inputs_is_refused(world):
    s = world.store
    step = capture.step_ref("merge", "1")
    path = world.folder / "work/merged.jsonl"
    write_rows(path, [{"text": X}, {"text": SC}])
    edges = [capture.copy_edge(1, world.x_a, step), capture.copy_edge(2, world.s_c, step)]
    before = table_counts(s)
    # C is registered, and the step did not declare it as an input.
    with pytest.raises(errors.InvalidRecord, match="no declared input") as caught:
        capture.record_jsonl(s, path, name="merged", role=capture.ROLE_INTERMEDIATE, step=step,
                             inputs=[world.src_a], edges=edges)
    assert caught.value.context["parent"] == world.s_c.key
    assert table_counts(s) == before and s.artifact("merged") is None
    merged = capture.record_jsonl(s, path, name="merged", role=capture.ROLE_INTERMEDIATE, step=step,
                                  inputs=[world.src_a, world.src_c], edges=edges)

    # A parent in the artifact itself is named with the instance id "".
    path = world.folder / "work/expanded.jsonl"
    write_rows(path, [{"text": X}, {"text": X + " " + X}])
    step = capture.step_ref("expand", "1")
    own = capture.OccurrenceRef(capture.OWN_ARTIFACT, capture.line_locator(1))
    expanded = capture.record_jsonl(
        s, path, name="expanded", role=capture.ROLE_INTERMEDIATE, step=step, inputs=[merged],
        edges=[capture.copy_edge(1, capture.occurrence(merged, 1), step),
               capture.copy_edge(2, own, step)])
    assert s.edge_of(capture.occurrence(expanded, 2)).parents == (capture.occurrence(expanded, 1),)
    assert s.artifact(expanded).edge_digest == ids.edge_digest([
        ids.edge_record("L1", [(merged, "L1")], None), ids.edge_record("L2", [(None, "L1")], None)])

    # A recovered edge is written after registration and is exempt: it may name
    # any registered record.
    write_rows(world.folder / "legacy/U.jsonl", [{"text": U}])
    imported = capture.import_jsonl(s, world.folder / "legacy/U.jsonl", name="import/U",
                                    member_name="legacy/U")
    s.add_recovered_edges(imported, [capture.EdgeRecord(
        "L1", (capture.EdgeInput(capture.occurrence(world.src_b, 2)),),
        capture.step_ref("recover-by-fingerprint", "1"), derivation=v.DERIVED_FINGERPRINT,
        verification=v.VERIFIED_PASSED, label=v.RECOVERED)])
    assert s.edge_of(capture.occurrence(imported, 1)).parents == (capture.occurrence(world.src_b, 2),)
    assert s.artifact(imported).inputs == ()
    s.check_integrity()


def test_an_edge_that_marks_a_sole_handed_record_in_scope_only_is_refused_by_the_store(world):
    s = world.store
    result = harness.run_job(paraphrase, job_name="paraphrase-3", handed=[harness.Handed(world.x_a, X)],
                             log_path=world.folder / "logs/paraphrase-3.jsonl")
    step = capture.step_ref("paraphrase", "3")
    path = world.folder / "gen/P3.jsonl"
    write_rows(path, [{"text": P}])
    # The builder names the sole handed record as a parent although the output states nothing.
    (built,) = capture.generation_edges(result, [1], step)
    assert built.inputs == (capture.EdgeInput(world.x_a, used=True),)
    # An edge written by hand from the stated list alone contradicts D-21.
    by_hand = dataclasses.replace(built, inputs=(capture.EdgeInput(world.x_a, used=False),))
    before = table_counts(s)
    with pytest.raises(errors.InvalidRecord, match="sole handed record"):
        capture.record_jsonl(s, path, name="gen/P3", role=capture.ROLE_INTERMEDIATE, step=step,
                             inputs=[world.src_a], edges=[by_hand])
    # A parent outside the hand-off of its job is refused as before.
    outside = dataclasses.replace(built, inputs=(capture.EdgeInput(world.x_a), capture.EdgeInput(world.x_b)))
    with pytest.raises(errors.InvalidRecord, match="outside the hand-off"):
        capture.record_jsonl(s, path, name="gen/P3", role=capture.ROLE_INTERMEDIATE, step=step,
                             inputs=[world.src_a, world.src_b], edges=[outside])
    assert table_counts(s) == before and s.harness_log("paraphrase-3") is None
    capture.record_jsonl(s, path, name="gen/P3", role=capture.ROLE_INTERMEDIATE, step=step,
                         inputs=[world.src_a], edges=[built])
    assert s.harness_log("paraphrase-3") == result.log


# --- refusals: source links, drops, the store file ------------------------------------------------


def test_the_label_rule_holds_for_a_source_link(world):
    s = world.store
    ref = capture.occurrence(world.r1, 3)
    before = table_counts(s)
    with pytest.raises(errors.LabelStrongerThanDerivation) as caught:  # a pointer nobody verified
        s.add_source_link(world.r1, "L3", "B/r1", v.MAPPED, "publisher's pointer", v.DERIVED_POINTER,
                          v.VERIFIED_NOT_RUN)
    assert (caught.value.name, caught.value.exit_code) == (
        "label stronger than derivation", errors.EXIT_WRITE_REFUSED)
    with pytest.raises(errors.LabelStrongerThanDerivation):  # file knowledge is never "mapped"
        s.add_source_link(world.r1, "L3", "B/r1", v.MAPPED, "file family", v.DERIVED_FILE)
    with pytest.raises(errors.LabelStrongerThanDerivation):  # no parent record: never "observed"
        s.add_source_link(world.r1, "L3", "B/r1", v.OBSERVED, "claimed", v.DERIVED_FILE)
    with pytest.raises(errors.InvalidRecord):  # nor stated by a step
        s.add_source_link(world.r1, "L3", "B/r1", v.OBSERVED, "claimed", v.DERIVED_STATED)
    assert table_counts(s) == before and s.source_links(ref) == ()

    s.add_source_link(world.r1, "L3", "B/r1", v.MAPPED, "publisher's pointer, verified on the row",
                      v.DERIVED_POINTER, v.VERIFIED_PASSED)
    s.add_source_link(world.r1, None, "C/r1", v.FILE_LEVEL, "file family: built beside C", v.DERIVED_FILE)
    assert [(x.member, x.label, x.derivation, x.verification) for x in s.source_links(ref)] == [
        ("B/r1", v.MAPPED, v.DERIVED_POINTER, v.VERIFIED_PASSED),
        ("C/r1", v.FILE_LEVEL, v.DERIVED_FILE, None)]
    # The observed edge of the record stands beside both links, untouched.
    assert s.edge_of(ref).label == v.OBSERVED
    assert s.artifact("R1").instance_id == world.r1  # a link never changes an id


def test_a_drop_is_one_per_record_step_name_and_output(world):
    s = world.store
    step = capture.step_ref(*RELEASE_STEP)
    loser = s.drops_of(world.x_b)[0]
    # The same drop again: nothing new.
    before = table_counts(s)
    s.add_drops(world.r1, [capture.DropRecord(world.x_b, v.DROP_DEDUP_LOSER, "release", "L1")])
    assert table_counts(s) == before
    # The same record, step name and output with another reason: refused.
    with pytest.raises(errors.InvalidRecord, match="one drop per record, step name and output"):
        s.add_drops(world.r1, [capture.drop(world.x_b, capture.DROP_FILTER, step)])
    with pytest.raises(errors.InvalidRecord):
        s.add_drops(world.r1, [capture.DropRecord(world.x_b, "lost", "release")])
    assert s.drops_of(world.x_b) == (loser,)
    # Another output of the same step name may drop the same record.
    s.add_drops(world.gen_p, [capture.drop(world.x_b, capture.DROP_FILTER, step)])
    assert [(d.output_instance_id, d.reason) for d in s.drops_of(world.x_b)] == [
        (world.r1, v.DROP_DEDUP_LOSER), (world.gen_p, v.DROP_FILTER)]


def test_what_is_not_a_store_is_refused_by_name(world):
    folder = world.folder
    with pytest.raises(errors.StoreFileExists) as caught:  # over the store itself
        Store.create(folder / "evidence.db", normaliser=NORMALISER)
    assert caught.value.name == "a file already exists at the store path"
    with pytest.raises(errors.StoreFileExists):  # over any file
        Store.create(folder / "sources/A.jsonl", normaliser=NORMALISER)
    assert (folder / "sources/A.jsonl").read_bytes() == world.a_bytes
    with pytest.raises(errors.NotAStore) as caught:
        Store.open(folder / "sources/A.jsonl")
    assert (caught.value.name, caught.value.exit_code) == ("not a store", errors.EXIT_REFUSED)
    with pytest.raises(errors.NotAStore):
        Store.open(folder / "no-such-file.db")
    with pytest.raises(errors.UnknownNormaliser) as caught:
        capture.scan_jsonl(folder / "sources/A.jsonl", normaliser="control-chars/9")
    assert caught.value.name == "unknown normaliser"

    world.store.close()
    copy = folder / "newer.db"
    copy.write_bytes((folder / "evidence.db").read_bytes())
    raw = sqlite3.connect(copy)
    raw.execute("UPDATE meta SET value = '7' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()
    with pytest.raises(errors.NewerSchemaVersion) as caught:
        Store.open(copy, readonly=True)
    assert caught.value.name == "store of a newer schema version"
    # The store itself opens again, read-only, and names its normaliser.
    world.store = Store.open(folder / "evidence.db", readonly=True)
    assert world.store.normaliser == NORMALISER == capture.store_normaliser(world.store)
    with pytest.raises(errors.WriteRefused):
        capture.ingest_jsonl(world.store, folder / "sources/C.jsonl", source="C", member="C/r2",
                             revision="r2")


# --- the loud checks on what capture wrote (D-18) -------------------------------------------------


def edit(path: Path, statement: str, parameters: tuple = ()) -> None:
    """Change a store outside the write API, as a fault injection does."""
    raw = sqlite3.connect(path)
    raw.execute(statement, parameters)
    raw.commit()
    raw.close()


def test_one_byte_of_a_release_changed_is_an_artifact_fault_and_not_a_map_fault(world):
    s = world.store
    release_file = world.folder / "releases/R1.jsonl"
    changed = world.r1_bytes.replace(b"unrelated", b"unrelatad", 1)
    assert sum(a != b for a, b in zip(world.r1_bytes, changed)) == 1
    assert changed.count(b"\n") == world.r1_bytes.count(b"\n")  # the line count stays
    release_file.write_bytes(changed)  # nothing else changes; the store is not edited

    with pytest.raises(errors.ArtifactDigestMismatch) as caught:
        s.check_integrity()
    assert (caught.value.name, caught.value.exit_code) == (
        "artifact digest mismatch", errors.EXIT_INTEGRITY)
    assert caught.value.context["instance"] == world.r1
    assert not isinstance(caught.value, (errors.MapDigestMismatch, errors.MapCountMismatch))
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity([world.r1])
    # The release's ancestors are whole and answer.
    for upstream in (world.src_a, world.src_b, world.gen_p):
        s.check_integrity([upstream])
    release_file.write_bytes(world.r1_bytes)
    s.check_integrity()


def test_the_map_faults_come_from_the_stores_own_records_and_a_gone_file_is_a_state(world):
    s = world.store
    s.close()
    folder = world.folder

    def copy_with(name: str, statement: str) -> Path:
        target = folder / name
        target.write_bytes((folder / "evidence.db").read_bytes())
        row = sqlite3.connect(target)
        artifact_id = row.execute(
            "SELECT artifact_id FROM artifact WHERE instance_id = ?", (world.r1,)).fetchone()[0]
        row.close()
        edit(target, statement, (artifact_id,))
        return target

    faults = [
        ("count.db", "UPDATE binding SET declared_count = declared_count - 1 WHERE artifact_id = ?",
         errors.MapCountMismatch, "map count mismatch"),
        ("rows.db", "DELETE FROM occurrence WHERE locator = 'L3' AND artifact_id = ?",
         errors.MapCountMismatch, "map count mismatch"),  # a stored row gone: 2 rows, 3 declared
        ("digest.db", "UPDATE binding SET declared_digest = 'edited' WHERE artifact_id = ?",
         errors.MapDigestMismatch, "map digest mismatch"),
        ("registered.db", "UPDATE artifact SET content_digest = 'edited' WHERE artifact_id = ?",
         errors.MapDigestMismatch, "map digest mismatch"),
        ("parent.db", "UPDATE edge_input SET occurrence_id = 987654 WHERE position = 0 AND edge_id IN "
                      "(SELECT e.edge_id FROM edge e JOIN occurrence c ON c.occurrence_id = e.child_id "
                      "WHERE c.artifact_id = ? AND c.locator = 'L2')",
         errors.EdgeParentMissing, "edge parent missing"),
    ]
    for name, statement, error, error_name in faults:
        target = copy_with(name, statement)
        with Store.open(target, readonly=True) as edited:
            with pytest.raises(error) as caught:
                edited.check_integrity([world.r1])
            assert (caught.value.name, caught.value.exit_code) == (error_name, errors.EXIT_INTEGRITY)
            edited.check_integrity([world.src_b])  # outside the faulted artifact's closure

    # The file gone from its place is a state the store reports, not a fault.
    world.store = s = Store.open(folder / "evidence.db")
    (folder / "releases/R1.jsonl").rename(folder / "releases/R1.archived")
    s.check_integrity()
    assert s.file_state(world.r1) == views.FILE_ABSENT
    assert [m.instance_id for m in s.missing_files()] == [world.r1]
    assert s.edge_of(capture.occurrence(world.r1, 1)).selected == world.x_a  # the records still answer


# --- a shard of the release: one map per namespace and derivation (D-20) ----------------------------


def write_shard(prefix: Path, documents: list[list[int]]) -> tuple[int, str]:
    """A hand-made shard in the documented index layout (docs/store.md)."""
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
    return len(lengths), capture.shard_digest(capture.file_digest(f"{prefix}.bin"),
                                              capture.file_digest(f"{prefix}.idx"))


def test_a_shard_of_the_release_holds_its_maps_side_by_side_and_none_is_overwritten(world):
    s = world.store
    count, digest = write_shard(world.folder / "shards/shard-1", [[11, 12], [21], [31, 32, 33]])
    shard = s.register_shard(world.folder / "shards/shard-1", name="E/shard-1",
                             step=capture.step_ref("tokenize", "1"), inputs=[world.r1])
    assert s.binding(shard) is None  # no map yet: the shard has no record of its own
    line = lambda n: capture.occurrence(world.r1, n)

    def a_map(namespace, derivation, verification, lines):
        label = capture.RECOVERED if verification == capture.VERIFIED_PASSED else capture.UNKNOWN
        spans = [(0, 2), (2, 3), (3, 6)]
        rows = tuple(
            capture.MapRow(n, *spans[n], label, line(parent) if label == capture.RECOVERED else None)
            for n, parent in enumerate(lines))
        return capture.MapRecord(namespace, derivation, verification, label, True, True, rows)

    def write(record, **changes):
        s.write_map(shard, record, **{"declared_count": count, "declared_digest": digest, **changes})

    position = a_map(capture.NAMESPACE_RECOVERY, capture.DERIVED_POSITION, capture.VERIFIED_FAILED, (1, 2, 3))
    matching = a_map(capture.NAMESPACE_RECOVERY, capture.DERIVED_TOKEN_MATCHING, capture.VERIFIED_PASSED,
                     (1, 2, 3))
    write(position)
    write(matching)
    # Both recovery maps are stored; the reader gets the verified one.
    assert {(b.derivation, b.verification) for b in s.bindings(shard)} == {
        (capture.DERIVED_POSITION, capture.VERIFIED_FAILED),
        (capture.DERIVED_TOKEN_MATCHING, capture.VERIFIED_PASSED)}
    assert s.binding(shard).derivation == capture.DERIVED_TOKEN_MATCHING
    assert [r.parent for r in s.map_rows(shard)] == [line(1), line(2), line(3)]
    assert [r.parent for r in s.map_rows(shard, derivation=capture.DERIVED_POSITION)] == [None] * 3
    assert [(d.shard_instance_id, d.ordinal) for d in s.documents_naming(line(2))] == [(shard, 1)]

    before = table_counts(s)
    write(matching)  # the same map again
    assert table_counts(s) == before
    # Another map of a namespace and derivation that hold one: never overwritten.
    with pytest.raises(errors.InvalidRecord, match="never overwritten"):
        write(a_map(capture.NAMESPACE_RECOVERY, capture.DERIVED_TOKEN_MATCHING, capture.VERIFIED_PASSED,
                    (3, 2, 1)))
    # A map that says more than its derivation allows.
    with pytest.raises(errors.LabelStrongerThanDerivation):
        write(dataclasses.replace(position, namespace=capture.NAMESPACE_CAPTURE, label=capture.RECOVERED))
    # A declared count or digest that is not the shard's.
    with pytest.raises(errors.DeclaredCountMismatch):
        write(dataclasses.replace(position, namespace=capture.NAMESPACE_CAPTURE, rows=position.rows[:2]),
              declared_count=2)
    with pytest.raises(errors.DeclaredDigestMismatch):
        write(dataclasses.replace(position, namespace=capture.NAMESPACE_CAPTURE),
              declared_digest=sha256(b"another shard"))
    assert table_counts(s) == before and s.binding(shard, capture.NAMESPACE_CAPTURE) is None

    # A mixture snapshot of the shard, with the settings the trainer used.
    mixture = s.register_mixture(
        name="E/mix-1", shards=[capture.SnapshotShard(0, shard, "shards/shard-1", digest)],
        settings={"split": "1,0,0"})
    assert s.mixture_settings(mixture) == {"split": "1,0,0"}
    s.check_integrity([mixture])
    # One token of the shard changed on disk: the artifact's fault, for the mixture too.
    write_shard(world.folder / "shards/shard-1", [[11, 12], [21], [31, 32, 34]])
    with pytest.raises(errors.ArtifactDigestMismatch):
        s.check_integrity([mixture])
    write_shard(world.folder / "shards/shard-1", [[11, 12], [21], [31, 32, 33]])
    s.check_integrity()
