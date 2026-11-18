"""Recovering a copy step (plan 3.3 "Copy steps", 3.1 "Fingerprint", 3.2, L3, L5).

Each test writes small source files, ingests them through capture, makes a later
file by shuffling, dealing and merging in ``recovery_kit``, registers it as a file
found afterwards and recovers the step. The expected parents come from the kit's
own bookkeeping of which line was copied from where.
"""

from __future__ import annotations

import json

import pytest
import recovery_kit as kit

from waterwhite import capture, errors, ids, normalise, recovery
from waterwhite import vocabulary as v
from waterwhite.recovery._files import RegisteredFile

STEP = capture.step_ref("shuffle-deal-merge", "not recorded")


def recover(store, later, earlier, **more):
    return recovery.recover_copy_step(store, later, earlier, step=STEP, **more)


# --- Registering a file found afterwards ------------------------------------------------


def test_a_found_file_is_registered_from_its_digest_and_its_name(store, folder):
    path = kit.write_lines(
        folder / "pipeline/found.jsonl",
        [json.dumps({"text": "one"}), "", "this line is no record", json.dumps({"text": "two"})],
    )
    instance = recovery.register_found_file(store, path, name="found")

    artifact = store.artifact("found")
    assert instance == ids.ingest_instance_id(ids.file_digest(path), "found")
    # A found file keeps its place in the pipeline and has no producing step.
    assert (artifact.kind, artifact.role, artifact.step, artifact.inputs) == (
        v.KIND_JSONL, v.ROLE_INTERMEDIATE, None, ())
    assert [o.fingerprint for o in store.occurrences(instance)] == [
        ids.fingerprint("one"), None, None, ids.fingerprint("two")]
    # No member and no source: the file's records have no recorded origin yet.
    assert store.members() == () and store.sources() == ()
    assert kit.rows_held(store, "found") == (0, 0, 0)
    # The same file under the same name again is the same instance.
    assert recovery.register_found_file(store, path, name="found") == instance
    assert store.occurrence_count(instance) == 4

    kit.source(store, folder, "C", ["c one"])
    answer = kit.ask(store, "C", "found")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert (answer["failing"]["artifact"], answer["failing"]["record"]) == ("found", "L1")


# --- A clean copy step ------------------------------------------------------------------


def test_a_clean_copy_step_is_recovered_record_by_record(store, folder):
    a = kit.source(store, folder, "A", ["a one", "a two", "a three", "a four", "a five"])
    b = kit.source(store, folder, "B", ["b one", "b two", "b three", "b four"])
    chunks = kit.chunked([a, b], 2, seed=3)
    assert sorted(len(chunk) for chunk in chunks) == [4, 5]

    for number, chunk in enumerate(chunks, 1):
        name = f"chunk-{number}"
        kit.found_file(store, folder, name, chunk)
        before = store.artifact(name)
        result = recover(store, name, ["src/A", "src/B"])

        assert (result.kind, result.outcome, result.reason) == (
            recovery.KIND_COPY_STEP, recovery.OUTCOME_RECORD_LEVEL, None)
        assert (result.artifact, result.instance_id) == (name, before.instance_id)
        assert (result.records, result.recovered, result.candidates, result.no_edge) == (
            len(chunk), len(chunk), 0, 0)
        assert (result.no_text, result.collisions_rejected, result.drops) == (0, 0, 0)
        assert (result.edges_written, result.drops_written, result.links_written) == (
            len(chunk), 0, 0)
        assert result.wrote_nothing is False
        assert result.files_read == (name, "src/A", "src/B")

        for line, record in enumerate(chunk, 1):
            assert kit.parents(store, name, line) == [record.origin], (name, line)
            edge = store.edge_of(kit.ref(store, name, line))
            assert (edge.label, edge.derivation, edge.verification, edge.parent_mode) == (
                v.RECOVERED, v.DERIVED_FINGERPRINT, v.VERIFIED_PASSED, v.MODE_ALL)
            assert (edge.transform, edge.selected, edge.is_generation) == (STEP, None, False)
        # A link derived afterwards never changes an instance (DECISIONS.md D-05).
        assert store.artifact(name) == before
    store.check_integrity()


# --- Identical texts ---------------------------------------------------------------------


def test_identical_texts_from_two_sources_give_candidates_naming_both(store, folder):
    shared = "a passage two sources supply word for word"
    a = kit.source(store, folder, "A", [shared, "a two"])
    kit.source(store, folder, "B", [shared, "b two"])
    z = kit.source(store, folder, "Z", ["z one", "z two", "z three"])
    kit.source(store, folder, "C", ["c one"])
    # The chunk holds A's copy of the passage and Z's records. B's copy went elsewhere.
    (chunk,) = kit.chunked([[a[0]], z], 1, seed=5)
    kit.found_file(store, folder, "chunk-1", chunk)
    x_line = chunk.index(a[0]) + 1

    result = recover(store, "chunk-1", ["src/A", "src/B", "src/Z"])

    assert (result.records, result.recovered, result.candidates, result.no_edge) == (4, 3, 1, 0)
    assert result.edges_written == 4
    edge = store.edge_of(kit.ref(store, "chunk-1", x_line))
    assert (edge.label, edge.derivation, edge.verification, edge.parent_mode, edge.selected) == (
        v.CANDIDATES, v.DERIVED_FINGERPRINT, v.VERIFIED_PASSED, v.MODE_ANY, None)
    # Every candidate is named, in the order the caller listed the files.
    assert kit.parents(store, "chunk-1", x_line) == [("src/A", 1), ("src/B", 1)]

    # "may contain" for each candidate's source, with both sources named (plan 3.2, L3).
    for source in ("A", "B"):
        answer = kit.ask(store, source, "chunk-1")
        assert (answer["answer"], answer["names"], answer["candidates"]) == (
            "may contain", v.NAMES_CANDIDATES, ["A", "B"]), source
        failing = answer["failing"]
        assert (failing["condition"], failing["artifact"], failing["record"]) == (
            "b", "chunk-1", f"L{x_line}")
    assert kit.ask(store, "Z", "chunk-1")["answer"] == "contains"
    assert kit.ask(store, "C", "chunk-1")["answer"] == "does not contain"


def test_identical_texts_inside_one_source_give_candidates_inside_that_source(store, folder):
    twin = "a line the source holds twice"
    z = kit.source(store, folder, "Z", ["z one", twin, "z three", twin])
    a = kit.source(store, folder, "A", ["a one", "a two"])
    kit.source(store, folder, "C", ["c one"])
    (chunk,) = kit.chunked([[z[1]], a], 1, seed=6)
    kit.found_file(store, folder, "chunk-1", chunk)
    twin_line = chunk.index(z[1]) + 1

    result = recover(store, "chunk-1", ["src/Z", "src/A"])

    assert (result.recovered, result.candidates, result.no_edge) == (2, 1, 0)
    assert kit.parents(store, "chunk-1", twin_line) == [("src/Z", 2), ("src/Z", 4)]
    assert store.edge_of(kit.ref(store, "chunk-1", twin_line)).label == v.CANDIDATES

    # Every twin belongs to Z, so the link decides "contains" for Z; none belongs to
    # C, so it counts as a link to other sources (plan 3.2).
    whole = kit.ask(store, "Z", "chunk-1")
    assert (whole["answer"], kit.hops(whole)) == (
        "contains", [f"chunk-1:L{twin_line}", "src/Z:L2"])
    assert whole["chain"][0]["link"] == v.CANDIDATES
    assert kit.ask(store, "C", "chunk-1")["answer"] == "does not contain"
    # Narrowed to one of the twins it decides nothing: which twin the line is a
    # copy of is not known.
    narrowed = kit.ask(store, "Z", "chunk-1", records=["src/Z:L2"])
    assert (narrowed["answer"], narrowed["names"]) == ("may contain", v.NAMES_CANDIDATES)


# --- No match ----------------------------------------------------------------------------


def test_a_record_with_no_match_gets_no_edge_and_the_proof_names_it(store, folder):
    a = kit.source(store, folder, "A", ["a one", "a two"])
    kit.source(store, folder, "C", ["c one"])
    stray = kit.Rec(("no file", 0), json.dumps({"text": "a line no named input holds"}))
    blank = kit.Rec(("no file", 0), "")
    kit.found_file(store, folder, "chunk-1", [a[1], stray, blank, a[0]])

    result = recover(store, "chunk-1", ["src/A"])

    assert (result.records, result.recovered, result.candidates) == (4, 2, 0)
    assert (result.no_edge, result.no_text, result.edges_written) == (2, 1, 2)
    assert kit.parents(store, "chunk-1", 1) == [("src/A", 2)]
    assert store.edge_of(kit.ref(store, "chunk-1", 2)) is None
    assert store.edge_of(kit.ref(store, "chunk-1", 3)) is None
    assert kit.parents(store, "chunk-1", 4) == [("src/A", 1)]

    # The record keeps no recorded origin, so no source is ever ruled out for the file.
    answer = kit.ask(store, "C", "chunk-1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    failing = answer["failing"]
    assert (failing["condition"], failing["artifact"], failing["record"]) == ("a", "chunk-1", "L2")
    assert kit.ask(store, "A", "chunk-1")["answer"] == "contains"


def test_only_the_files_the_caller_names_are_searched(store, folder):
    a = kit.source(store, folder, "A", ["a one"])
    b = kit.source(store, folder, "B", ["b one"])
    kit.found_file(store, folder, "chunk-1", [a[0], b[0]])

    result = recover(store, "chunk-1", ["src/A"])

    # src/B is registered and holds the very text of line 2; it was not named.
    assert (result.recovered, result.no_edge) == (1, 1)
    assert result.files_read == ("chunk-1", "src/A")
    assert kit.parents(store, "chunk-1", 2) == []


# --- A hash collision --------------------------------------------------------------------


@pytest.fixture
def colliding(monkeypatch):
    """A test double of the fingerprint function: the texts "alpha" and "omega"
    get one fingerprint. Every other text keeps its real one."""
    real = ids.fingerprint

    def double(normalised_text: str) -> bytes:
        if normalised_text in ("alpha", "omega"):
            return b"\x5a" * ids.FINGERPRINT_BYTES
        return real(normalised_text)

    monkeypatch.setattr(ids, "fingerprint", double)


def test_a_forged_collision_is_rejected_by_the_direct_comparison(store, folder, colliding):
    e = kit.source(store, folder, "E", ["omega", "beta"])
    alpha = kit.Rec(("no file", 0), json.dumps({"text": "alpha"}))
    kit.found_file(store, folder, "later", [alpha, e[1]])
    # The store holds one fingerprint for the two different texts.
    assert (store.occurrence(kit.ref(store, "later", 1)).fingerprint
            == store.occurrence(kit.ref(store, "src/E", 1)).fingerprint)

    result = recover(store, "later", ["src/E"])

    assert kit.parents(store, "later", 1) == []  # "alpha" is not a copy of "omega"
    assert kit.parents(store, "later", 2) == [("src/E", 2)]
    assert (result.recovered, result.candidates, result.no_edge) == (1, 0, 1)
    assert (result.collisions_rejected, result.edges_written) == (1, 1)


def test_a_collision_beside_a_real_match_leaves_one_parent(store, folder, colliding):
    e = kit.source(store, folder, "E", ["omega", "alpha"])
    kit.found_file(store, folder, "later", [e[1]])

    result = recover(store, "later", ["src/E"])

    # Two records share the fingerprint; the comparison leaves the one true parent,
    # so the edge is "recovered" and not a candidate set.
    assert kit.parents(store, "later", 1) == [("src/E", 2)]
    assert store.edge_of(kit.ref(store, "later", 1)).label == v.RECOVERED
    assert (result.recovered, result.candidates, result.collisions_rejected) == (1, 0, 1)


# --- The second run ----------------------------------------------------------------------


def test_the_second_run_writes_nothing_and_says_so(store, folder):
    shared = "a passage two sources supply"
    a = kit.source(store, folder, "A", [shared, "a two", "a three"])
    b = kit.source(store, folder, "B", [shared, "b two"])
    stray = kit.Rec(("no file", 0), json.dumps({"text": "held by no input"}))
    (chunk,) = kit.chunked([a, b[1:], [stray]], 1, seed=9)
    kit.found_file(store, folder, "chunk-1", chunk)

    first = recover(store, "chunk-1", ["src/A", "src/B"])
    held = kit.rows_held(store, "chunk-1")
    edges = list(store.edges(store.artifact("chunk-1").instance_id))
    second = recover(store, "chunk-1", ["src/A", "src/B"])

    assert (first.recovered, first.candidates, first.no_edge) == (3, 1, 1)
    assert (first.edges_written, first.wrote_nothing) == (4, False)
    # The second run finds the same links and adds none.
    assert (second.recovered, second.candidates, second.no_edge) == (3, 1, 1)
    assert (second.edges_written, second.drops_written, second.links_written) == (0, 0, 0)
    assert second.wrote_nothing is True
    assert second.to_json()["wrote_nothing"] is True
    assert kit.rows_held(store, "chunk-1") == held == (4, 0, 0)
    assert list(store.edges(store.artifact("chunk-1").instance_id)) == edges


# --- Refusals ----------------------------------------------------------------------------


def test_a_second_recovery_that_would_change_a_stored_edge_is_refused_whole(store, folder):
    shared = "a passage two sources supply"
    a = kit.source(store, folder, "A", [shared, "a two"])
    b = kit.source(store, folder, "B", [shared, "b two"])
    kit.found_file(store, folder, "chunk-1", [a[0], b[1]])
    recover(store, "chunk-1", ["src/A"])  # B was left out: the passage looks unique
    assert kit.parents(store, "chunk-1", 1) == [("src/A", 1)]
    held = kit.rows_held(store, "chunk-1")

    # With B named the passage has two candidates. The stored edge is not replaced,
    # and the edge of line 2, which would be new, is not written either.
    with pytest.raises(errors.InvalidRecord, match="not replaced"):
        recover(store, "chunk-1", ["src/A", "src/B"])
    assert kit.rows_held(store, "chunk-1") == held
    assert kit.parents(store, "chunk-1", 1) == [("src/A", 1)]
    assert kit.parents(store, "chunk-1", 2) == []


def test_files_that_cannot_be_trusted_are_refused_and_nothing_is_written(store, folder):
    a = kit.source(store, folder, "A", ["a one", "a two"])
    kit.found_file(store, folder, "chunk-1", [a[1], a[0]])
    kit.found_file(store, folder, "chunk-2", [a[0]])

    # Another text field than the one the files were registered with.
    with pytest.raises(errors.InvalidRecord, match="another text field"):
        recover(store, "chunk-1", ["src/A"], text_field="body")

    # The later file changed after it was registered.
    path = kit.file_path(folder, "chunk-1")
    kept = path.read_bytes()
    path.write_bytes(kept + b'{"text": "a line added afterwards"}\n')
    with pytest.raises(errors.ArtifactDigestMismatch):
        recover(store, "chunk-1", ["src/A"])
    path.write_bytes(kept.replace(b"a two", b"a 2wo"))
    with pytest.raises(errors.ArtifactDigestMismatch):
        recover(store, "chunk-1", ["src/A"])
    path.write_bytes(kept)

    # A candidate file changed after it was registered.
    source_path = folder / "sources/A.jsonl"
    source_bytes = source_path.read_bytes()
    source_path.write_bytes(source_bytes.replace(b"a one", b"a 1ne"))
    with pytest.raises(errors.ArtifactDigestMismatch):
        recover(store, "chunk-1", ["src/A"])
    source_path.write_bytes(source_bytes)

    # The later file is no longer at its recorded place.
    moved = path.with_name("chunk-1.moved")
    path.rename(moved)
    with pytest.raises(errors.WriteRefused, match="not at its recorded place"):
        recover(store, "chunk-1", ["src/A"])
    moved.rename(path)

    assert kit.rows_held(store, "chunk-1") == (0, 0, 0)
    assert recover(store, "chunk-1", ["src/A"]).edges_written == 2


def test_a_call_that_names_its_files_wrongly_is_refused(store, folder):
    a = kit.source(store, folder, "A", ["a one"])
    kit.found_file(store, folder, "chunk-1", [a[0]])
    refused = [
        lambda: recover(store, "chunk-9", ["src/A"]),  # not registered
        lambda: recover(store, "chunk-1", ["src/Q"]),  # not registered
        lambda: recover(store, "chunk-1", []),  # no possible input
        lambda: recover(store, "chunk-1", "src/A"),  # one text, not a sequence
        lambda: recover(store, "chunk-1", ["src/A", "src/A"]),  # named twice
        lambda: recover(store, "chunk-1", ["src/A", "chunk-1"]),  # its own input
        lambda: recovery.recover_copy_step(store, "chunk-1", ["src/A"], step="shuffle"),
    ]
    for call in refused:
        with pytest.raises(errors.InvalidRecord):
            call()
    assert kit.rows_held(store, "chunk-1") == (0, 0, 0)


# --- Size --------------------------------------------------------------------------------


def test_a_few_thousand_lines_are_recovered_from_streams(store, folder):
    a = kit.source(store, folder, "A", [f"record {n} of source a" for n in range(1500)])
    b = kit.source(store, folder, "B", [f"record {n} of source b" for n in range(1500)])
    (chunk,) = kit.chunked([a, b], 1, seed=21)
    kit.found_file(store, folder, "chunk-1", chunk)

    result = recover(store, "chunk-1", ["src/A", "src/B"])

    assert (result.records, result.recovered, result.edges_written) == (3000, 3000, 3000)
    for line in (1, 2, 1499, 3000):
        assert kit.parents(store, "chunk-1", line) == [chunk[line - 1].origin]


# --- Reading a file by line --------------------------------------------------------------


def test_a_line_read_by_its_number_is_the_line_the_store_counts(store, folder):
    # A blank line, a carriage return inside a line and at its end, and a last line
    # with no line feed after it: the places where two ways of counting could part.
    data = b'{"text": "one"}\n\n{"text": "tw\\ro"}\r\n{"text": "three"}'
    path = folder / "pipeline/odd.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    recovery.register_found_file(store, path, name="odd")
    expected = list(ids.split_lines([data]))
    assert len(expected) == store.occurrence_count(store.artifact("odd").instance_id) == 4

    file = RegisteredFile(store, store.artifact("odd"),
                          function=normalise.get(store.normaliser), text_field="text")
    try:
        assert list(file.stream()) == expected
        assert [file.raw_line(n) for n in (4, 2, 3, 1)] == [expected[n - 1] for n in (4, 2, 3, 1)]
        for number, raw in enumerate(expected, 1):
            stored = store.occurrence(kit.ref(store, "odd", number)).fingerprint
            assert file.record(raw, number)[1] == stored
        with pytest.raises(errors.MapCountMismatch):
            file.raw_line(5)
    finally:
        file.close()
