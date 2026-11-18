"""ingest_jsonl, record_jsonl, import_jsonl and the edge builders (plan 3.1, 3.3,
5.4, L1, Decisions 25 and 29).

The store is a recording stand-in: it has the write methods of docs/interfaces.md
section 4 with their exact signatures and only records the calls it receives.
Expected digests and fingerprints are computed here with hashlib, not with the
tool's own functions.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from waterwhite import capture, errors, ids, normalise, records
from waterwhite import vocabulary as v
from waterwhite.errors import InvalidRecord
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.normalise import UnknownNormaliser
from waterwhite.records import (
    DropRecord,
    EdgeInput,
    EdgeRecord,
    OccurrenceRecord,
    OccurrenceRef,
    StepRef,
)

N1 = "control-chars/1"


class RecordingStore:
    """Records the calls it receives. ``register_jsonl`` returns the id the real
    store would return for a file with no producing step, and a fixed made-up id
    for a file a step wrote (the real one needs the store's edge digest)."""

    def __init__(self, normaliser: str = N1) -> None:
        self.normaliser = normaliser
        self.calls: list[tuple[str, dict]] = []
        self.members: dict[str, object] = {}
        self.asked: list[str] = []

    def member(self, name):
        """What a wrapper reads before it writes: the member under this name."""
        self.asked.append(name)
        return self.members.get(name)

    def add_source(self, name):
        self.calls.append(("add_source", {"name": name}))

    def add_member(self, source, name, revision, part_kind, parts):
        self.calls.append(
            (
                "add_member",
                {
                    "source": source,
                    "name": name,
                    "revision": revision,
                    "part_kind": part_kind,
                    "parts": tuple(parts),
                },
            )
        )

    def register_jsonl(
        self,
        path,
        *,
        name,
        role,
        step,
        member_name,
        inputs,
        occurrences,
        edges=(),
        drops=(),
        declared_count,
        declared_digest,
    ):
        self.calls.append(
            (
                "register_jsonl",
                {
                    "path": path,
                    "name": name,
                    "role": role,
                    "step": step,
                    "member_name": member_name,
                    "inputs": inputs,
                    "occurrences": occurrences,
                    "edges": edges,
                    "drops": drops,
                    "declared_count": declared_count,
                    "declared_digest": declared_digest,
                },
            )
        )
        if step is None:
            return ids.ingest_instance_id(declared_digest, member_name)
        return "step-made:" + name

    def add_source_link(
        self, instance_id, locator, member, label, evidence, derivation, verification=None
    ):
        self.calls.append(
            (
                "add_source_link",
                {
                    "instance_id": instance_id,
                    "locator": locator,
                    "member": member,
                    "label": label,
                    "evidence": evidence,
                    "derivation": derivation,
                    "verification": verification,
                },
            )
        )


def fp(normalised_text: str) -> bytes:
    """BLAKE2b, 16 bytes, of an already normalised text (plan 3.1)."""
    return hashlib.blake2b(normalised_text.encode("utf-8"), digest_size=16).digest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_id(digest: str, member: str) -> str:
    """The id of a file with no producing step, written out from ids.py's rule."""
    payload = json.dumps(
        ["ww.instance.ingest.v1", digest, member], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


# Nine lines. Line 4 holds control characters as JSON escapes and is equal to
# line 1 after normalisation. The last line has no line feed after it.
SOURCE_FILE = (
    b'{"id": "A-001", "text": "The shared passage."}\n'
    b"\n"
    b'{"id": "A-003", "text": broken\n'
    b'{"id": "A-004", "text": "The shared\\u0000 passage.\\r\\n"}\n'
    b'{"id": "A-005"}\n'
    b'{"id": "A-006", "text": " \\u0007 "}\n'
    b'{"id": "A-007", "text": 7}\n'
    b'["a list, not an object"]\n'
    b'{"id": "A-009", "text": "Last line, no line feed after it."}'
)
SOURCE_FINGERPRINTS = [
    fp("The shared passage."),
    None,  # blank line
    None,  # unparseable
    fp("The shared passage."),
    None,  # no text field
    None,  # empty after normalisation
    None,  # the field holds a number
    None,  # not a JSON object
    fp("Last line, no line feed after it."),
]


def write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def release_file(tmp_path, texts: list[str], name: str = "out.jsonl"):
    data = "".join(json.dumps({"text": t}) + "\n" for t in texts).encode("utf-8")
    return write(tmp_path, name, data), data


# --- scan_jsonl ------------------------------------------------------------------------


def test_scan_gives_count_digest_and_one_occurrence_per_line(tmp_path):
    path = write(tmp_path, "A.jsonl", SOURCE_FILE)
    scan = capture.scan_jsonl(path, normaliser=N1)
    assert scan.count == 9
    assert scan.digest == sha256(SOURCE_FILE) == ids.file_digest(path)
    assert [o.locator for o in scan.occurrences] == [f"L{n}" for n in range(1, 10)]
    assert [o.fingerprint for o in scan.occurrences] == SOURCE_FINGERPRINTS


def test_scan_of_an_empty_file_has_no_lines(tmp_path):
    scan = capture.scan_jsonl(write(tmp_path, "empty.jsonl", b""), normaliser=N1)
    assert (scan.count, scan.occurrences, scan.digest) == (0, (), sha256(b""))


def test_capture_and_the_store_count_lines_with_one_function(tmp_path):
    """What a line is, is defined once (ids.split_lines). The wrapper's count and
    the count the store finds on disk come from that one function."""
    from waterwhite.store import artifact_files

    cases = {
        b"": 0,
        b"\n": 1,  # one empty line
        b"\n\n": 2,
        b"a": 1,  # a last piece with no line feed counts when it is not empty
        b"a\n": 1,
        b"a\nb": 2,
        b"a\n\nb\n": 3,
        b"a\rb\n": 1,  # a carriage return ends no line
        b"a\r\nb\r\n": 2,
        b"\r": 1,
        b"a\x0bb\x0cc\x1cd\xc2\x85e\n": 1,  # other "line breaks" of text mode end no line
    }
    for number, (data, expected) in enumerate(cases.items()):
        path = write(tmp_path, f"case-{number}.jsonl", data)
        assert len(list(ids.split_lines([data]))) == expected, data
        scan = capture.scan_jsonl(path, normaliser=N1)
        found = artifact_files.read_jsonl(path)
        assert scan.count == found.line_count == len(scan.occurrences) == expected, data
        assert scan.digest == found.digest == sha256(data), data


def test_a_line_ends_at_a_line_feed_byte_only(tmp_path):
    data = (
        b'{"text": "windows line end"}\r\n'  # the carriage return is part of line 1
        b'{"text": "one"}\r{"text": "two"}\n'  # a lone carriage return ends no line
        b'{"text": "bad utf-8 \xff"}\n'
        b"\n"
    )
    scan = capture.scan_jsonl(write(tmp_path, "f.jsonl", data), normaliser=N1)
    assert scan.count == 4
    assert [o.fingerprint for o in scan.occurrences] == [fp("windows line end"), None, None, None]


def test_the_text_field_can_be_named(tmp_path):
    data = b'{"text": "wrong field", "body": "right field"}\n'
    scan = capture.scan_jsonl(write(tmp_path, "f.jsonl", data), normaliser=N1, text_field="body")
    assert [o.fingerprint for o in scan.occurrences] == [fp("right field")]


def test_a_text_with_no_utf8_form_is_refused_and_not_passed_over(tmp_path):
    data = b'{"text": "fine"}\n{"text": "half a pair \\ud800"}\n'
    path = write(tmp_path, "f.jsonl", data)
    with pytest.raises(InvalidRecord) as caught:
        capture.scan_jsonl(path, normaliser=N1)
    assert caught.value.context["line"] == 2
    store = RecordingStore()
    with pytest.raises(InvalidRecord):
        capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    assert store.calls == []


def test_scan_with_an_unknown_normaliser_is_refused(tmp_path):
    path = write(tmp_path, "f.jsonl", b'{"text": "x"}\n')
    with pytest.raises(UnknownNormaliser):
        capture.scan_jsonl(path, normaliser="control-chars/2")
    store = RecordingStore(normaliser="control-chars/2")
    with pytest.raises(UnknownNormaliser):
        capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    assert store.calls == []


def test_a_store_that_names_no_normaliser_is_refused(tmp_path):
    path = write(tmp_path, "f.jsonl", b'{"text": "x"}\n')
    with pytest.raises(InvalidRecord):
        capture.ingest_jsonl(object(), path, source="A", member="A/r1", revision="r1")


def test_the_store_normaliser_may_be_a_method():
    class MethodStore:
        def normaliser(self):
            return N1

    assert capture.store_normaliser(MethodStore()) == N1
    assert capture.store_normaliser(RecordingStore()) == N1


# --- ingest_jsonl ------------------------------------------------------------------------


def test_ingest_registers_the_file_then_the_source_and_the_member(tmp_path):
    path = write(tmp_path, "A.jsonl", SOURCE_FILE)
    store = RecordingStore()
    instance = capture.ingest_jsonl(
        store, path, source="A", member="A/r1", revision="r1", name="src/A"
    )
    digest = sha256(SOURCE_FILE)
    assert instance == ingest_id(digest, "A/r1")
    assert [name for name, _ in store.calls] == ["register_jsonl", "add_source", "add_member"]
    assert store.asked == ["A/r1"]  # the member name was read before the first write
    registered = store.calls[0][1]
    assert registered == {
        "path": path,
        "name": "src/A",
        "role": "source",
        "step": None,
        "member_name": "A/r1",
        "inputs": (),
        "occurrences": tuple(
            OccurrenceRecord(f"L{n}", f) for n, f in enumerate(SOURCE_FINGERPRINTS, start=1)
        ),
        "edges": (),
        "drops": (),
        "declared_count": 9,
        "declared_digest": digest,
    }
    assert store.calls[1][1] == {"name": "A"}
    assert store.calls[2][1] == {
        "source": "A",
        "name": "A/r1",
        "revision": "r1",
        "part_kind": "file",
        "parts": ((instance, ""),),
    }


def test_ingest_names_the_artifact_after_its_member_when_no_name_is_given(tmp_path):
    path = write(tmp_path, "A.jsonl", SOURCE_FILE)
    store = RecordingStore()
    capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    assert store.calls[0][1]["name"] == "A/r1"
    with pytest.raises(InvalidRecord):
        capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1", name="")
    assert len(store.calls) == 3


def test_the_same_bytes_under_two_members_are_two_instances(tmp_path):
    path = write(tmp_path, "X.jsonl", b'{"text": "The shared passage."}\n')
    store = RecordingStore()
    one = capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    two = capture.ingest_jsonl(store, path, source="B", member="B/r1", revision="r1")
    assert one != two
    assert store.calls[0][1]["declared_digest"] == store.calls[3][1]["declared_digest"]


@pytest.mark.parametrize("missing", ["source", "member", "revision"])
def test_ingest_needs_source_member_and_revision(tmp_path, missing):
    path = write(tmp_path, "A.jsonl", SOURCE_FILE)
    store = RecordingStore()
    names = {"source": "A", "member": "A/r1", "revision": "r1", missing: ""}
    with pytest.raises(InvalidRecord):
        capture.ingest_jsonl(store, path, **names)
    assert store.calls == []


def test_ingest_stops_when_the_store_returns_another_id(tmp_path):
    class WrongIdStore(RecordingStore):
        def register_jsonl(self, path, **fields):
            super().register_jsonl(path, **fields)
            return "0" * 64

    store = WrongIdStore()
    path = write(tmp_path, "A.jsonl", SOURCE_FILE)
    with pytest.raises(InvalidRecord):
        capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    assert [name for name, _ in store.calls] == ["register_jsonl"]


# --- record_jsonl --------------------------------------------------------------------------

SRC_A = "a" * 64
SRC_B = "b" * 64


def test_record_with_copy_edges(tmp_path):
    path, data = release_file(tmp_path, ["first", "second", "third"])
    step = capture.step_ref("shuffle", "1", {"seed": 7})
    parents = [OccurrenceRef(SRC_A, "L3"), OccurrenceRef(SRC_A, "L1"), OccurrenceRef(SRC_B, "L2")]
    edges = [capture.copy_edge(n, parent, step) for n, parent in enumerate(parents, start=1)]
    store = RecordingStore()
    instance = capture.record_jsonl(
        store,
        path,
        name="shuffled",
        role="intermediate",
        step=step,
        inputs=[SRC_A, SRC_B],
        edges=edges,
    )
    assert instance == "step-made:shuffled"
    assert [name for name, _ in store.calls] == ["register_jsonl"]
    call = store.calls[0][1]
    assert call["declared_count"] == 3
    assert call["declared_digest"] == sha256(data)
    assert call["occurrences"] == (
        OccurrenceRecord("L1", fp("first")),
        OccurrenceRecord("L2", fp("second")),
        OccurrenceRecord("L3", fp("third")),
    )
    assert (call["name"], call["role"], call["step"], call["member_name"]) == (
        "shuffled",
        "intermediate",
        step,
        None,
    )
    assert call["inputs"] == (SRC_A, SRC_B)
    assert call["drops"] == ()
    assert call["edges"] == tuple(
        EdgeRecord(
            child_locator=f"L{n}",
            inputs=(EdgeInput(parent, used=True, selected=False),),
            transform=step,
            parent_mode="all",
            derivation="stated",
            verification=None,
            label="observed",
            is_generation=False,
            harness_log=None,
        )
        for n, parent in enumerate(parents, start=1)
    )


def test_step_ref_takes_the_digest_of_the_parameters():
    step = capture.step_ref("shuffle", "1", {"seed": 7})
    payload = json.dumps(
        ["ww.params.v1", {"seed": 7}], sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    assert step == StepRef("shuffle", "1", sha256(payload))
    assert capture.step_ref("shuffle", "1", {"seed": 8}) != step
    with pytest.raises(InvalidRecord):
        capture.step_ref("shuffle", "1", {"seed": object()})
    with pytest.raises(InvalidRecord):
        capture.step_ref("", "1")


def test_record_with_a_selection_edge_and_its_drops(tmp_path):
    # Line 1 of the release is the passage A and B both supply; dedup kept A's copy.
    path, _ = release_file(tmp_path, ["The shared passage.", "unrelated"], "R1.jsonl")
    step = capture.step_ref("dedup", "1", {"key": "normalised text"})
    x_a, x_b, u = OccurrenceRef(SRC_A, "L1"), OccurrenceRef(SRC_B, "L1"), OccurrenceRef(SRC_B, "L5")
    edge, drops = capture.selection_edge(1, [x_a, x_b], x_a, step)
    assert edge == EdgeRecord(
        child_locator="L1",
        inputs=(EdgeInput(x_a, used=True, selected=True), EdgeInput(x_b, used=True, selected=False)),
        transform=step,
        parent_mode="any",
        derivation="stated",
        label="observed",
    )
    assert drops == (DropRecord(input=x_b, reason="dedup loser", step_name="dedup", group_key="L1"),)
    store = RecordingStore()
    capture.record_jsonl(
        store,
        path,
        name="R1",
        role="release",
        step=step,
        inputs=[SRC_A, SRC_B],
        edges=[edge, capture.copy_edge(2, u, step)],
        drops=drops,
    )
    call = store.calls[0][1]
    assert call["role"] == "release"
    assert call["edges"][0] == edge
    assert call["drops"] == drops


def test_swapping_the_winner_changes_the_selected_parent_and_the_drop():
    step = capture.step_ref("dedup", "1")
    x_a, x_b = OccurrenceRef(SRC_A, "L1"), OccurrenceRef(SRC_B, "L1")
    kept_a, drops_a = capture.selection_edge(1, [x_a, x_b], x_a, step)
    kept_b, drops_b = capture.selection_edge(1, [x_a, x_b], x_b, step)
    assert [i.parent for i in kept_a.inputs] == [i.parent for i in kept_b.inputs] == [x_a, x_b]
    assert [i.selected for i in kept_a.inputs] == [True, False]
    assert [i.selected for i in kept_b.inputs] == [False, True]
    assert [d.input for d in drops_a] == [x_b]
    assert [d.input for d in drops_b] == [x_a]
    assert drops_a[0].group_key == drops_b[0].group_key == "L1"


def test_a_selection_states_its_own_group_key_and_a_group_of_one_has_no_drop():
    step = capture.step_ref("dedup", "1")
    x_a, x_b, x_c = (OccurrenceRef(i * 64, "L1") for i in "abc")
    _, drops = capture.selection_edge(4, [x_a, x_b, x_c], x_b, step, group_key="passage-x")
    assert [(d.input, d.group_key) for d in drops] == [(x_a, "passage-x"), (x_c, "passage-x")]
    edge, none = capture.selection_edge(2, [x_a], x_a, step)
    assert none == ()
    assert edge.inputs == (EdgeInput(x_a, used=True, selected=True),)


def test_a_selection_refuses_a_kept_copy_outside_its_group_and_a_copy_named_twice():
    step = capture.step_ref("dedup", "1")
    x_a, x_b = OccurrenceRef(SRC_A, "L1"), OccurrenceRef(SRC_B, "L1")
    with pytest.raises(InvalidRecord):
        capture.selection_edge(1, [x_a], x_b, step)
    with pytest.raises(InvalidRecord):
        capture.selection_edge(1, [x_a, x_a], x_a, step)
    with pytest.raises(InvalidRecord):
        capture.selection_edge(0, [x_a], x_a, step)


def test_a_drop_has_a_known_reason(tmp_path):
    step = capture.step_ref("sanitize", "1")
    gone = OccurrenceRef(SRC_A, "L2")
    assert capture.drop(gone, v.DROP_EMPTY_LINE, step) == DropRecord(
        input=gone, reason="empty line", step_name="sanitize", group_key=None
    )
    with pytest.raises(InvalidRecord):
        capture.drop(gone, "lost", step)
    path, _ = release_file(tmp_path, ["one"])
    store = RecordingStore()
    with pytest.raises(InvalidRecord):
        capture.record_jsonl(
            store,
            path,
            name="out",
            role="intermediate",
            step=step,
            inputs=[SRC_A],
            edges=[],
            drops=[DropRecord(input=gone, reason="lost", step_name="sanitize")],
        )
    assert store.calls == []


@pytest.mark.parametrize("role", ["source", "import", "shard", ""])
def test_record_refuses_a_role_a_step_cannot_write(tmp_path, role):
    path, _ = release_file(tmp_path, ["one"])
    store = RecordingStore()
    with pytest.raises(InvalidRecord):
        capture.record_jsonl(
            store, path, name="out", role=role, step=capture.step_ref("s", "1"), inputs=[], edges=[]
        )
    assert store.calls == []


@pytest.mark.parametrize("inputs", [SRC_A, [SRC_A, ""], [SRC_A, None]])
def test_record_takes_its_inputs_as_a_sequence_of_instance_ids(tmp_path, inputs):
    path, _ = release_file(tmp_path, ["one"])
    store = RecordingStore()
    with pytest.raises(InvalidRecord):
        capture.record_jsonl(
            store,
            path,
            name="out",
            role="intermediate",
            step=capture.step_ref("s", "1"),
            inputs=inputs,
            edges=[],
        )
    assert store.calls == []


def test_record_refuses_edges_that_do_not_fit_the_file(tmp_path):
    path, _ = release_file(tmp_path, ["one", "two"])
    step = capture.step_ref("merge", "1")
    parent = OccurrenceRef(SRC_A, "L1")
    stated = capture.copy_edge(1, parent, step)
    cases = {
        "a child past the last line": [capture.copy_edge(3, parent, step)],
        "two edges for one child": [stated, capture.copy_edge(1, OccurrenceRef(SRC_A, "L2"), step)],
        "a shard locator": [EdgeRecord("D0", stated.inputs, step)],
        "a locator that is not one": [EdgeRecord("L01", stated.inputs, step)],
        "an edge derived afterwards": [
            EdgeRecord(
                "L1",
                stated.inputs,
                step,
                derivation=v.DERIVED_FINGERPRINT,
                verification=v.VERIFIED_PASSED,
                label=v.RECOVERED,
            )
        ],
    }
    for what, edges in cases.items():
        store = RecordingStore()
        with pytest.raises(InvalidRecord):
            capture.record_jsonl(
                store, path, name="out", role="intermediate", step=step, inputs=[SRC_A], edges=edges
            )
        assert store.calls == [], what


# --- import_jsonl --------------------------------------------------------------------------


def test_an_ingest_the_store_would_refuse_at_the_member_writes_nothing(tmp_path):
    """The member is the third write of an ingest. A member name that is taken by
    another source, revision or part kind is refused before the first."""
    path = write(tmp_path, "A.jsonl", SOURCE_FILE)
    taken = SimpleNamespace(source="A", revision="r1", part_kind="file")
    for changes in (dict(source="B"), dict(revision="r2")):
        store = RecordingStore()
        store.members["A/r1"] = taken
        with pytest.raises(InvalidRecord, match="registered already"):
            capture.ingest_jsonl(store, path, **{"source": "A", "member": "A/r1", "revision": "r1", **changes})
        assert store.calls == []
    store = RecordingStore()
    store.members["A/r1"] = SimpleNamespace(source="A", revision="r1", part_kind="records")
    with pytest.raises(InvalidRecord, match="registered already"):
        capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    assert store.calls == []
    # The same member again is the same ingest.
    store = RecordingStore()
    store.members["A/r1"] = taken
    capture.ingest_jsonl(store, path, source="A", member="A/r1", revision="r1")
    assert [name for name, _ in store.calls] == ["register_jsonl", "add_source", "add_member"]


def test_an_import_whose_file_family_names_no_registered_member_writes_nothing(tmp_path):
    path, _ = release_file(tmp_path, ["Eine Uebersetzung."], "T.jsonl")
    store = RecordingStore()
    with pytest.raises(InvalidRecord, match="not registered"):
        capture.import_jsonl(store, path, name="import/T", member_name="legacy/T",
                             file_family=("A/r1", "file family: translations of A"))
    assert store.calls == [] and store.asked == ["A/r1"]


def test_import_with_a_file_family_writes_one_file_level_link(tmp_path):
    path, data = release_file(tmp_path, ["Eine Uebersetzung."], "T.jsonl")
    store = RecordingStore()
    store.members["A/r1"] = SimpleNamespace(source="A", revision="r1", part_kind="file")
    evidence = "file family: translations made from source A"
    instance = capture.import_jsonl(
        store, path, name="import/T", member_name="legacy/T", file_family=("A/r1", evidence)
    )
    assert instance == ingest_id(sha256(data), "legacy/T")
    assert [name for name, _ in store.calls] == ["register_jsonl", "add_source_link"]
    call = store.calls[0][1]
    assert (call["name"], call["role"], call["step"], call["member_name"]) == (
        "import/T",
        "import",
        None,
        "legacy/T",
    )
    assert (call["inputs"], call["edges"], call["drops"]) == ((), (), ())
    assert call["occurrences"] == (OccurrenceRecord("L1", fp("Eine Uebersetzung.")),)
    assert (call["declared_count"], call["declared_digest"]) == (1, sha256(data))
    assert store.calls[1][1] == {
        "instance_id": instance,
        "locator": None,
        "member": "A/r1",
        "label": "file-level",
        "evidence": evidence,
        "derivation": "file",
        "verification": None,
    }


def test_import_without_a_file_family_writes_no_link(tmp_path):
    path, _ = release_file(tmp_path, ["Eine Uebersetzung."], "T.jsonl")
    store = RecordingStore()
    capture.import_jsonl(store, path, name="import/T", member_name="legacy/T")
    assert [name for name, _ in store.calls] == ["register_jsonl"]


def test_the_same_bytes_imported_under_two_members_are_two_instances(tmp_path):
    path, _ = release_file(tmp_path, ["Eine Uebersetzung."], "T.jsonl")
    store = RecordingStore()
    one = capture.import_jsonl(store, path, name="import/T", member_name="legacy/T")
    two = capture.import_jsonl(store, path, name="import/T2", member_name="legacy/T2")
    assert one != two


@pytest.mark.parametrize(
    "family", [("A/r1", ""), ("", "evidence"), ("A/r1",), ["A/r1", "evidence"], "A/r1"]
)
def test_import_refuses_a_file_family_without_member_or_evidence(tmp_path, family):
    path, _ = release_file(tmp_path, ["Eine Uebersetzung."], "T.jsonl")
    store = RecordingStore()
    with pytest.raises(InvalidRecord):
        capture.import_jsonl(
            store, path, name="import/T", member_name="legacy/T", file_family=family
        )
    assert store.calls == []


# --- generation edges from a harness result -------------------------------------------------

SA = OccurrenceRef(SRC_A, "L2")
SC = OccurrenceRef("c" * 64, "L1")


def test_a_generation_edge_names_used_records_as_parents_and_the_rest_in_scope(tmp_path):
    def summary(ctx):
        text = ctx.read(SA)
        return JobOutput(text="A summary of " + text, used=(SA,))

    result = run_job(
        summary,
        job_name="summary",
        handed=[Handed(SA, "the passage of A"), Handed(SC, "the passage of C")],
        log_path=tmp_path / "summary.log.jsonl",
    )
    step = capture.step_ref("summary", "1")
    (edge,) = capture.generation_edges(result, [1], step)
    assert edge == EdgeRecord(
        child_locator="L1",
        inputs=(EdgeInput(SA, used=True, selected=False), EdgeInput(SC, used=False, selected=False)),
        transform=step,
        parent_mode="all",
        derivation="stated",
        verification=None,
        label="observed",
        is_generation=True,
        harness_log=result.log,
        job_name="summary",  # the job name of the log, kept by the edge (D-24)
    )
    assert edge.job_name == result.log.job_name
    # An in-process log lists the reads, and establishes no bound (Decision 13).
    assert edge.harness_log.reads_within_hand_off() and not edge.harness_log.bound_established()


def test_a_job_handed_one_record_has_it_as_a_parent_even_when_the_output_states_nothing(tmp_path):
    """DECISIONS.md D-21: the sole handed record is a parent of each output. With
    two or more handed records only the stated ones are parents."""
    def silent(ctx):
        ctx.read(SA)
        return [JobOutput(text="first output", used=()), JobOutput(text="second", used=(SA,))]

    result = run_job(
        silent, job_name="silent", handed=[Handed(SA, "the passage of A")],
        log_path=tmp_path / "silent.log.jsonl",
    )
    assert [o.used for o in result.outputs] == [(), (SA,)]  # as stated; the log keeps that
    step = capture.step_ref("silent", "1")
    edges = capture.generation_edges(result, [1, 2], step)
    assert [e.inputs for e in edges] == [(EdgeInput(SA, used=True),), (EdgeInput(SA, used=True),)]

    def silent_about_two(ctx):
        return JobOutput(text="output", used=())

    result = run_job(
        silent_about_two,
        job_name="silent-two",
        handed=[Handed(SA, "the passage of A"), Handed(SC, "the passage of C")],
        log_path=tmp_path / "two.log.jsonl",
    )
    (edge,) = capture.generation_edges(result, [1], step)
    assert edge.inputs == (EdgeInput(SA, used=False), EdgeInput(SC, used=False))


def test_a_generation_edge_goes_to_the_store_with_its_log(tmp_path):
    def both(ctx):
        return [JobOutput(text=ctx.read(SA) + " / " + ctx.read(SC), used=(SA, SC))]

    result = run_job(
        both,
        job_name="summary",
        handed=[Handed(SA, "the passage of A"), Handed(SC, "the passage of C")],
        log_path=tmp_path / "summary.log.jsonl",
    )
    path, _ = release_file(tmp_path, [result.outputs[0].text], "S.jsonl")
    step = capture.step_ref("summary", "1")
    edges = capture.generation_edges(result, [1], step)
    store = RecordingStore()
    capture.record_jsonl(
        store, path, name="gen/S", role="intermediate", step=step, inputs=[SRC_A, "c" * 64], edges=edges
    )
    sent = store.calls[0][1]["edges"]
    assert [i.used for i in sent[0].inputs] == [True, True]
    assert sent[0].harness_log is result.log
    assert sent[0].harness_log.model == "none"


def test_the_called_model_is_a_field_of_the_log_and_never_an_input(tmp_path):
    def paraphrase(ctx):
        text = ctx.call_model("stand-in-model", "paraphrase: " + ctx.read(SA))
        return JobOutput(text=text, used=(SA,))

    result = run_job(
        paraphrase,
        job_name="paraphrase",
        handed=[Handed(SA, "the passage of A")],
        log_path=tmp_path / "p.log.jsonl",
        model_client=lambda model, prompt: prompt.upper(),
    )
    edge = capture.generation_edge(1, result.outputs[0], result.log, capture.step_ref("p", "1"))
    assert result.log.model == "stand-in-model"
    assert edge.inputs == (EdgeInput(SA, used=True, selected=False),)
    assert edge.harness_log.model == "stand-in-model"


def test_a_generation_edge_refuses_a_used_record_that_was_not_handed(tmp_path):
    result = run_job(
        lambda ctx: JobOutput(text="x", used=()),
        job_name="job",
        handed=[Handed(SA, "the passage of A")],
        log_path=tmp_path / "job.log.jsonl",
    )
    with pytest.raises(InvalidRecord):
        capture.generation_edge(
            1, JobOutput(text="x", used=(SC,)), result.log, capture.step_ref("p", "1")
        )
    with pytest.raises(InvalidRecord):
        capture.generation_edges(result, [1, 2], capture.step_ref("p", "1"))
    with pytest.raises(InvalidRecord):
        capture.generation_edge(1, result.outputs[0], None, capture.step_ref("p", "1"))


# --- what a pipeline imports -----------------------------------------------------------------


def test_capture_names_what_a_pipeline_needs_beside_the_harness_the_normaliser_and_the_store():
    """A pipeline imports waterwhite.capture, waterwhite.harness, waterwhite.normalise
    and waterwhite.store.Store only. The record types, id functions and words it
    writes are named by capture, and they are the same objects, not copies."""
    for name in capture.__all__:
        assert hasattr(capture, name), name
    for name in ("StepRef", "OccurrenceRef", "EdgeRecord", "EdgeInput", "DropRecord", "HarnessLog",
                 "MapRecord", "MapRow", "SnapshotShard", "OccurrenceRecord", "OWN_ARTIFACT"):
        assert name in capture.__all__ and getattr(capture, name) is getattr(records, name), name
    for name in ("line_locator", "ordinal_locator", "param_digest", "file_digest", "shard_digest"):
        assert name in capture.__all__ and getattr(capture, name) is getattr(ids, name), name
    for name in ("MODE_ALL", "MODE_ANY", "DERIVED_STATED", "OBSERVED", "DROP_REASONS",
                 "DROP_DEDUP_LOSER", "DROP_SANITIZER_SKIP", "ROLE_RELEASE", "ROLE_INTERMEDIATE",
                 "NAMESPACE_CAPTURE", "DERIVED_POSITION", "VERIFIED_PASSED", "PART_FILE"):
        assert name in capture.__all__ and getattr(capture, name) == getattr(v, name), name
    assert capture.PART_KIND_FILE == v.PART_FILE == "file"


def test_the_named_errors_live_in_one_module():
    from waterwhite import harness
    from waterwhite.store import Store  # noqa: F401  (the fourth import of a pipeline)

    assert normalise.UnknownNormaliser is errors.UnknownNormaliser
    assert harness.HarnessRefused is errors.HarnessRefused
    assert harness.ModelClientMissing is errors.ModelClientMissing
    assert (errors.HarnessRefused.name, errors.HarnessRefused.exit_code) == (
        "read outside the hand-off", errors.EXIT_REFUSED)
    assert issubclass(errors.HarnessRefused, errors.Refused)
    assert errors.ModelClientMissing.name == "no model client"
    assert (errors.UnknownNormaliser.name, errors.UnknownNormaliser.exit_code) == (
        "unknown normaliser", errors.EXIT_WRITE_REFUSED)
