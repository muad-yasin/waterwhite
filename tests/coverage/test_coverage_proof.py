"""The coverage proof over files, shards and mixtures (plan 2.1, 2.2, 3.2, 3.3).

Each test builds the small store of ``ww_world``, applies at most one fault from
outside the write API, and reads the answer. A condition is tested failing alone,
with the link it names.
"""

from __future__ import annotations

import pytest
import ww_world

from waterwhite import capture, coverage, errors, extensions, requests
from waterwhite import vocabulary as v
from waterwhite.answers import Finding, Proof
from waterwhite.coverage import _walk
from waterwhite.records import EdgeInput, EdgeRecord, HarnessRead, SnapshotShard
from waterwhite.store.store import OWN_ARTIFACT

NOTE_PROCESS = [v.NOTE_PROCESS_EVIDENCE]
NOTE_MODEL = [v.NOTE_GENERATING_MODEL_NOT_ASSESSED]


def hops(answer: dict) -> list[str]:
    return [hop["hop"] for hop in answer["chain"]]


def link(answer: dict, *keys: str) -> dict:
    """The named keys of an answer's failing link."""
    return {key: answer["failing"].get(key) for key in keys}


# --- Unmutated ------------------------------------------------------------------------


def test_withdraw_a_unmutated(ask):
    assert (ask("A", "src/A")["answer"], ask("A", "src/A")["notes"]) == ("contains", [])
    for name in ("src/B", "src/C", "src/Z", "R2", "shard-1", "mix-1"):
        answer = ask("A", name)
        assert (answer["answer"], answer["notes"]) == ("does not contain", []), name
        assert answer["scope"]["source"] == "A" and answer["scope"]["members"] == ["A/r1"]
        assert answer["scope"]["generation_edges"] == []
    for name, chain in (
        ("gen/P", ["gen/P:L1", "src/A:L1"]),
        ("gen/T", ["gen/T:L1", "gen/P:L1", "src/A:L1"]),
        ("gen/S", ["gen/S:L1", "src/A:L2"]),
        ("shard-3", ["shard-3:D0", "R1:L3", "gen/T:L1", "gen/P:L1", "src/A:L1"]),
        ("mix-2", ["shard-3:D0", "R1:L3", "gen/T:L1", "gen/P:L1", "src/A:L1"]),
    ):
        answer = ask("A", name)
        assert (answer["answer"], hops(answer), answer["notes"]) == ("contains", chain, NOTE_PROCESS), name


def test_a_contains_carries_the_process_note_when_any_of_its_chains_passes_a_generation_edge(ask):
    # R1's first record in scope is X, kept from A by dedup: no generation edge on
    # that chain. P, T and S are in R1 too, and theirs pass one.
    answer = ask("A", "R1")
    assert (answer["answer"], hops(answer), answer["notes"]) == ("contains", ["R1:L1", "src/A:L1"], NOTE_PROCESS)
    # src/A is in scope by itself: no generation edge anywhere, no note.
    assert ask("A", "src/A")["notes"] == []


def test_the_note_counts_the_chains_of_one_record_too(world, ask):
    # One record made from A's own line and from P: the chain shown is the first
    # parent's, and the second parent's chain passes a generation edge.
    a, p = world["src/A"], world["gen/P"]
    world.ids["joined"] = ww_world.record(
        world.store, world.folder / "joined.jsonl", ["X and its paraphrase"], "joined",
        v.ROLE_INTERMEDIATE, ww_world.step("join"), [a, p],
        [EdgeRecord("L1", (EdgeInput(ww_world.ref(a, 1)), EdgeInput(ww_world.ref(p, 1))),
                    ww_world.step("join"))],
    )
    answer = ask("A", "joined")
    assert (hops(answer), answer["notes"]) == (["joined:L1", "src/A:L1"], NOTE_PROCESS)


def test_an_any_edge_follows_the_parent_its_step_selected(world, ask):
    # R1 keeps A's copy of X: a withdrawal of B does not reach R1 through the loser.
    assert ask("B", "R1")["answer"] == "does not contain"
    assert ask("B", "R2")["answer"] == "contains"
    # The same bytes with B's copy selected are attributed to B.
    a, b = world["src/A"], world["src/B"]
    swapped = ww_world.record(
        world.store, world.folder / "R1-swap.jsonl", ["passage X"], "R1-swap", v.ROLE_RELEASE,
        ww_world.step("release", "swap"), [a, b],
        [EdgeRecord("L1", (EdgeInput(ww_world.ref(a, 1)), EdgeInput(ww_world.ref(b, 1), selected=True)),
                    ww_world.step("dedup"), parent_mode=v.MODE_ANY)],
    )
    assert swapped != world["R1"]
    assert ask("A", "R1-swap")["answer"] == "does not contain"
    assert hops(ask("B", "R1-swap")) == ["R1-swap:L1", "src/B:L1"]


def test_a_request_narrowed_to_one_record(ask):
    x_of_a = ask("A", "R1", records=["src/A:L1"])
    assert (x_of_a["answer"], x_of_a["request"]["records"]) == ("contains", ["src/A:L1"])
    # S is made from A's second line: the first line alone does not put it in scope.
    assert ask("A", "gen/S", records=["src/A:L1"])["answer"] == "does not contain"
    assert hops(ask("A", "gen/S", records=["src/A:L2"])) == ["gen/S:L1", "src/A:L2"]
    scoped = ask("A", "src/A", records=["src/A:L2"])
    assert hops(scoped) == ["src/A:L2"]
    assert ask("A", "gen/S", records=["src/A:L1"])["scope"]["records"] == ["src/A:L1"]


def test_withdraw_c_past_generation_edges_rests_on_their_logs(ask):
    answer = ask("C", "shard-3")
    assert (answer["answer"], answer["notes"]) == ("does not contain", NOTE_MODEL)
    assert answer["scope"]["generation_edges"] == ["translation", "paraphrase"]
    assert [entry["job"] for entry in answer["scope"]["harness_logs"]] == ["translation", "paraphrase"]
    # The world's logs were built by hand as the confined harness writes them.
    assert answer["scope"]["harness_logs"][0] == {
        "job": "translation", "log_digest": "log-digest-translation", "handed": 1, "reads": 1,
        "refused_attempts": [], "model": "none", "harness": "confined",
    }
    r1 = ask("C", "R1")
    assert (r1["answer"], hops(r1), r1["notes"]) == ("contains", ["R1:L4", "gen/S:L1", "src/C:L1"], NOTE_PROCESS)
    # No generation edge on the way: the negative carries no note.
    assert ask("C", "shard-1")["notes"] == []


def test_a_member_that_is_a_list_of_records(world, ask):
    # Source G is two records of the file src/Z, which is also a revision of Z.
    world.store.add_source("G")
    world.store.add_member("G", "G/part", "r1", "records",
                           [(world["src/Z"], "L1"), (world["src/Z"], "L3")])
    request = requests.withdraw(world.store, "G")
    assert request.members == ("G/part",)
    assert hops(ask("G", "src/Z")) == ["src/Z:L1"]
    assert hops(ask("G", "R1")) == ["R1:L5", "src/Z:L1"]  # "z one"
    assert hops(ask("G", "R2")) == ["R2:L3", "src/Z:L3"]  # "z three"
    assert ask("G", "gen/P")["answer"] == "does not contain"
    # Z's line 2 belongs to Z alone.
    assert hops(ask("G", "shard-3")) == ["shard-3:D1", "R1:L5", "src/Z:L1"]
    narrowed = ask("G", "R1", records=["src/Z:L3"])
    assert narrowed["answer"] == "does not contain"
    with pytest.raises(requests.RecordNotInSource):
        requests.withdraw(world.store, "G", records=["src/Z:L2"])


def test_a_negative_names_the_files_whose_digest_could_not_be_checked(world, ask):
    assert ask("A", "R2")["scope"]["files_not_checked"] == []
    # src/B is archived: its records stay and still answer; the proof says which
    # digests it could not compare with a file.
    (world.folder / "src/B.jsonl").rename(world.folder / "src/B.archived")
    answer = ask("A", "R2")
    assert answer["answer"] == "does not contain"
    assert answer["scope"]["files_not_checked"] == ["src/B"]


# --- (a) every occurrence has a recorded chain ---------------------------------------


def test_a_deleted_edge_fails_condition_a_at_that_record(world, ask):
    world.delete_edge("gen/P", 1)
    answer = ask("A", "gen/P")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert link(answer, "condition", "artifact", "record") == {
        "condition": "a", "artifact": "gen/P", "record": "L1"}
    assert answer["candidates"] == ["A"]  # the sources among the inputs of gen/P's step
    for name in ("gen/T", "shard-3", "mix-2"):
        downstream = ask("A", name)
        assert downstream["answer"] == "may contain", name
        assert link(downstream, "condition", "artifact", "record") == {
            "condition": "a", "artifact": "gen/P", "record": "L1"}, name
    # No answer about the source the edge named either: the record has no origin.
    assert ask("C", "shard-3")["answer"] == "may contain"
    # R1 keeps a second recorded chain to A, through X and through S: it wins.
    assert ask("A", "R1")["answer"] == "contains"
    # An artifact whose closure does not hold the edge keeps its negative.
    assert ask("A", "shard-1")["answer"] == "does not contain"


def test_a_record_with_no_origin_never_gives_does_not_contain(world, ask):
    # An import with no file family: its record came through an input nobody registered.
    ww_world.write_jsonl(world.folder / "legacy.jsonl", ["some old text"])
    capture.import_jsonl(world.store, world.folder / "legacy.jsonl", name="import/old",
                         member_name="legacy/old")
    for source in ("A", "B", "C", "Z"):
        answer = ask(source, "import/old")
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "a", "artifact": "import/old", "record": "L1"}


def test_a_chain_that_runs_into_itself_has_no_origin(world, ask):
    looped = ww_world.record(
        world.store, world.folder / "loop.jsonl", ["one", "two", "three"], "loop", v.ROLE_INTERMEDIATE,
        ww_world.step("rewrite"), [world["src/Z"]],
        [ww_world.copy_edge(1, ww_world.ref(world["src/Z"], 1)),
         ww_world.copy_edge(2, ww_world.ref(OWN_ARTIFACT, 1)),
         ww_world.copy_edge(3, ww_world.ref(OWN_ARTIFACT, 2))],
    )
    world.ids["loop"] = looped
    assert ask("A", "loop")["answer"] == "does not contain"
    # Line 1 now names line 3 as its parent: 1 <- 3 <- 2 <- 1.
    world.sql(("UPDATE edge_input SET occurrence_id = ? WHERE edge_id = ?",
               (world.occurrence_id("loop", 3), world.edge_id("loop", 1))))
    answer = ask("A", "loop")
    assert (answer["answer"], answer["names"], answer["failing"]["condition"]) == (
        "may contain", v.NAMES_MISSING_ORIGIN, "a")


def test_a_long_chain_is_walked_without_deep_recursion(world, ask):
    lines = 3000
    edges = [ww_world.copy_edge(1, world.ref("src/A", 1))]
    edges += [ww_world.copy_edge(n, ww_world.ref(OWN_ARTIFACT, n - 1)) for n in range(2, lines + 1)]
    world.ids["long"] = ww_world.record(
        world.store, world.folder / "long.jsonl", [f"t{n}" for n in range(lines)], "long",
        v.ROLE_INTERMEDIATE, ww_world.step("rewrite"), [world["src/A"]], edges,
    )
    request = requests.withdraw(world.store, "A")
    session = coverage.Session(world.store, request)
    last = session.occurrence(world.ref("long", lines))  # the deepest record first
    assert last.verdict == _walk.HOLDS and len(last.chain) == lines + 1
    assert ask("A", "long")["answer"] == "contains"
    assert ask("C", "long")["scope"]["walked"] == lines


def test_the_memo_is_a_cache_and_its_limit_changes_no_answer(world, ask, monkeypatch):
    names = ("R1", "R2", "shard-3", "mix-2", "gen/S")
    before = [(ask(s, n)["answer"], ask(s, n)["failing"]) for s in ("A", "C") for n in names]
    monkeypatch.setattr(_walk, "MEMO_LIMIT", 1)
    assert [(ask(s, n)["answer"], ask(s, n)["failing"]) for s in ("A", "C") for n in names] == before


# --- (b) the level of each link, per record and requested source ---------------------


def test_a_file_level_edge_cannot_decide_a_record_level_request(world, ask):
    world.sql(("UPDATE edge SET label = 'file-level', derivation = 'file' WHERE edge_id = ?",
               (world.edge_id("gen/T", 1),)))
    for name in ("gen/T", "shard-3", "mix-2"):
        answer = ask("A", name, records=["src/A:L1"])
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK), name
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "b", "artifact": "gen/T", "record": "L1"}, name
    # A request about the whole source is one the link can decide (plan 3.2).
    whole = ask("A", "gen/T")
    assert (whole["answer"], hops(whole)) == ("contains", ["gen/T:L1", "gen/P:L1", "src/A:L1"])
    assert whole["chain"][0]["link"] == "file-level"
    # It shows nothing about a source it does not lead to.
    other = ask("C", "gen/T")
    assert (other["answer"], other["failing"]["condition"]) == ("may contain", "b")


def test_a_file_level_link_decides_the_whole_source_and_no_record(world, ask):
    ww_world.write_jsonl(world.folder / "legacy/T.jsonl", ["Uebersetzung von P"])
    capture.import_jsonl(
        world.store, world.folder / "legacy/T.jsonl", name="import/T", member_name="legacy/T",
        file_family=("A/r1", "file family: translations made from source A"))
    whole = ask("A", "import/T")
    assert (whole["answer"], whole["notes"], hops(whole)) == ("contains", [], ["import/T:L1"])
    assert (whole["chain"][0]["link"], whole["chain"][0]["member"]) == ("file-level link", "A/r1")
    narrowed = ask("A", "import/T", records=["src/A:L1"])
    assert (narrowed["answer"], narrowed["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK)
    assert link(narrowed, "condition", "artifact", "record") == {
        "condition": "b", "artifact": "import/T", "record": "L1"}
    # For another source the file's link says nothing, and the record's own origin
    # was never recorded.
    other = ask("C", "import/T")
    assert (other["answer"], other["names"], other["failing"]["condition"]) == (
        "may contain", v.NAMES_MISSING_ORIGIN, "a")


def test_an_observed_link_to_a_never_masks_a_file_level_link_to_b(world, ask):
    before = ask("B", "gen/P", records=["src/B:L1"])
    assert (before["answer"], before["notes"]) == ("does not contain", NOTE_MODEL)
    world.store.add_source_link(world["gen/P"], "L1", "B/r1", v.FILE_LEVEL,
                                "file family: paraphrases of passages that source B supplies",
                                derivation=v.DERIVED_FILE)
    after = ask("B", "gen/P", records=["src/B:L1"])
    assert (after["answer"], after["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK)
    assert link(after, "condition", "artifact", "record") == {
        "condition": "b", "artifact": "gen/P", "record": "L1"}
    # The link to A is as strong as before, and a request about all of B is decided.
    assert ask("A", "gen/P")["answer"] == "contains"
    assert ask("B", "gen/P")["answer"] == "contains"
    assert ask("C", "gen/P")["answer"] == "does not contain"


def test_a_mapped_edge_decides_its_parents_and_never_a_negative_for_another_source(world, ask):
    # A text-only file whose rows a publisher's pointer ties to records of A. The
    # pointer was verified on the data, so the link is "mapped" (plan 3.2).
    ww_world.write_jsonl(world.folder / "pointed.jsonl", ["made from X"])
    instance = capture.import_jsonl(world.store, world.folder / "pointed.jsonl", name="import/M",
                                    member_name="legacy/M")
    world.store.add_recovered_edges(instance, [EdgeRecord(
        "L1", (EdgeInput(world.ref("src/A", 1)),), ww_world.step("publisher-pointer"),
        derivation=v.DERIVED_POINTER, verification=v.VERIFIED_PASSED, label=v.MAPPED)])
    whole = ask("A", "import/M")
    assert (whole["answer"], hops(whole), whole["chain"][0]["link"]) == (
        "contains", ["import/M:L1", "src/A:L1"], "mapped")
    assert ask("A", "import/M", records=["src/A:L1"])["answer"] == "contains"
    # The pointer names what the row was made from and nothing about what else was
    # read: no bound, so no negative for a source it does not name.
    for request in (dict(source="C"), dict(source="A", records=["src/A:L2"])):
        source = request.pop("source")
        other = ask(source, "import/M", **request)
        assert (other["answer"], other["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
        assert link(other, "condition", "job", "artifact", "record") == {
            "condition": "g", "job": "publisher-pointer", "artifact": "import/M", "record": "L1"}


def test_a_mapped_link_to_a_member_names_no_record(world, ask):
    world.store.add_source_link(world["R2"], "L3", "A/r1", v.MAPPED, "publisher's source id",
                                derivation=v.DERIVED_POINTER, verification=v.VERIFIED_PASSED)
    whole = ask("A", "R2")
    assert (whole["answer"], hops(whole), whole["chain"][0]["link"]) == (
        "contains", ["R2:L3"], "mapped link")
    narrowed = ask("A", "R2", records=["src/A:L1"])
    assert (narrowed["answer"], narrowed["failing"]["condition"]) == ("may contain", "b")
    other = ask("C", "R2")
    assert (other["answer"], other["names"], other["failing"]["condition"]) == (
        "may contain", v.NAMES_BOUND_NOT_ESTABLISHED, "g")


def test_a_source_link_label_stronger_than_its_derivation_fails_condition_f(world, ask):
    # An unverified pointer can only be "unknown" (plan 3.2). The store refuses a
    # stronger label; one edited in afterwards shows nothing.
    world.store.add_source_link(world["R2"], "L3", "A/r1", v.UNKNOWN, "a pointer nobody verified",
                                derivation=v.DERIVED_POINTER, verification=v.VERIFIED_NOT_RUN)
    assert ask("C", "R2")["answer"] == "does not contain"
    world.sql("UPDATE source_link SET label = 'mapped' WHERE evidence = 'a pointer nobody verified'")
    for source in ("A", "C"):
        answer = ask(source, "R2")
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LABEL_STRONGER), source
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "f", "artifact": "R2", "record": "L3"}


def test_an_unknown_link_decides_nothing(world, ask):
    world.sql(("UPDATE edge SET label = 'unknown', derivation = 'none' WHERE edge_id = ?",
               (world.edge_id("R2", 3),)))
    answer = ask("A", "R2")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK)
    assert link(answer, "condition", "artifact", "record") == {
        "condition": "b", "artifact": "R2", "record": "L3"}
    assert ask("B", "R2")["answer"] == "contains"  # a recorded chain elsewhere wins


def test_candidates_contain_when_all_belong_to_the_source_and_may_contain_when_some_do(world, ask):
    r1, r2 = world["R1"], world["R2"]

    def twin_shard(name: str, first, second) -> None:
        ww_world.shard(world.store, world.folder / f"shards/{name}", name, [r1, r2],
                       [ww_world.twins(0, first, second), ww_world.twins(1, first, second)])

    twin_shard("twins-mixed", ww_world.ref(r1, 1), ww_world.ref(r2, 1))  # X from A, X from B
    twin_shard("twins-all", ww_world.ref(r1, 2), ww_world.ref(r1, 3))  # P and T: both from A
    twin_shard("twins-none", ww_world.ref(r2, 2), ww_world.ref(r2, 3))  # B and Z

    mixed = ask("A", "twins-mixed")
    assert (mixed["answer"], mixed["names"], mixed["candidates"]) == (
        "may contain", v.NAMES_CANDIDATES, ["A", "B"])
    assert link(mixed, "condition", "artifact", "record") == {
        "condition": "b", "artifact": "twins-mixed", "record": "D0"}
    assert ask("B", "twins-mixed")["answer"] == "may contain"
    assert ask("Z", "twins-mixed")["answer"] == "does not contain"

    every = ask("A", "twins-all")
    assert (every["answer"], hops(every)[:2]) == ("contains", ["twins-all:D0", "R1:L2"])
    assert every["chain"][0]["link"] == "candidates"

    assert ask("A", "twins-none")["answer"] == "does not contain"
    assert ask("B", "twins-none")["answer"] == "may contain"  # one twin is B's, one is Z's


# --- (c) and (d): the loud faults ------------------------------------------------------


def test_a_missing_edge_parent_stops_the_proof_loudly(world):
    world.sql(("UPDATE edge_input SET occurrence_id = 987654 WHERE edge_id = ?",
               (world.edge_id("gen/P", 1),)))
    request = requests.withdraw(world.store, "A")
    for name in ("gen/P", "gen/T", "R1", "shard-3", "mix-2", "E@6"):
        with pytest.raises(errors.EdgeParentMissing):
            coverage.prove(world.store, name, request)
    # An artifact outside the fault's closure still answers.
    assert isinstance(coverage.prove(world.store, "shard-1", request), Proof)


def test_a_map_digest_mismatch_stops_every_query_whose_closure_holds_the_map(world):
    world.sql(("UPDATE binding SET declared_digest = 'f' || substr(declared_digest, 2) "
               "WHERE binding_id = ? AND substr(declared_digest, 1, 1) <> 'f'",
               (world.binding_id("shard-3"),)))
    request = requests.withdraw(world.store, "A")
    for name in ("shard-3", "mix-2", "E@6", "Y@2"):
        with pytest.raises(errors.MapDigestMismatch) as fault:
            coverage.prove(world.store, name, request)
        assert fault.value.exit_code == 3
    for name in ("shard-1", "mix-1", "E@3", "R1"):
        assert isinstance(coverage.prove(world.store, name, request), (Proof, Finding))


def test_an_integrity_pass_spares_the_second_check_and_never_a_first(world, monkeypatch):
    request = requests.withdraw(world.store, "A")
    calls = []
    real = type(world.store).check_integrity
    monkeypatch.setattr(type(world.store), "check_integrity",
                        lambda self, ids=None: (calls.append(ids), real(self, ids))[1])
    coverage.prove(world.store, "R2", request)
    assert calls == [[world["R2"]]]
    whole = coverage.check_store(world.store)
    coverage.prove(world.store, "R2", request, integrity=whole)
    assert calls == [[world["R2"]], None]
    # A pass over one artifact does not cover another.
    partial = coverage.check_store(world.store, [world["R1"]])
    coverage.prove(world.store, "R2", request, integrity=partial)
    assert calls[-1] == [world["R2"]]


# --- (f) a label stronger than its derivation ----------------------------------------


def test_a_map_label_stronger_than_its_derivation_fails_condition_f(world, ask):
    world.sql(("UPDATE binding SET label = 'observed' WHERE binding_id = ?",
               (world.binding_id("shard-1"),)))
    for name in ("shard-1", "mix-1"):
        answer = ask("A", name)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LABEL_STRONGER), name
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "f", "artifact": "shard-1", "record": None}, name
    # The map exists: the shard is not "not covered".
    assert ask("A", "shard-1")["reason"] is None
    # A recorded chain through another shard still shows "contains", and so does a
    # chain through the rows of this map: only its label field was changed.
    assert ask("A", "mix-2")["answer"] == "contains"
    assert ask("C", "mix-2")["failing"]["condition"] == "f"
    assert hops(ask("B", "shard-1")) == ["shard-1:D0", "R2:L1", "src/B:L1"]


def test_an_unverified_map_and_an_edited_row_label(world, ask):
    r2 = world["R2"]
    ww_world.shard(world.store, world.folder / "shards/unverified", "unverified", [r2],
                   [ww_world.MapRow(0, 0, 1, v.UNKNOWN), ww_world.MapRow(1, 1, 2, v.UNKNOWN)],
                   namespace=v.NAMESPACE_RECOVERY, verification=v.VERIFIED_FAILED, label=v.UNKNOWN)
    answer = ask("A", "unverified")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK)
    assert link(answer, "condition", "artifact", "record") == {
        "condition": "b", "artifact": "unverified", "record": "D0"}

    world.sql(("UPDATE map_row SET label = 'observed' WHERE binding_id = ? AND ordinal = 1",
               (world.binding_id("shard-1"),)))
    edited = ask("A", "shard-1")
    assert (edited["answer"], edited["names"]) == ("may contain", v.NAMES_LABEL_STRONGER)
    assert link(edited, "condition", "artifact", "record") == {
        "condition": "f", "artifact": "shard-1", "record": "D1"}


def test_an_edge_label_stronger_than_its_derivation_fails_condition_f(world, ask):
    world.sql(("UPDATE edge SET derivation = 'fingerprint', verification = 'failed' WHERE edge_id = ?",
               (world.edge_id("R2", 2),)))
    answer = ask("A", "R2")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LABEL_STRONGER)
    assert link(answer, "condition", "artifact", "record") == {
        "condition": "f", "artifact": "R2", "record": "L2"}


# --- (g) the upper bound of a generation edge ----------------------------------------


def expect_no_bound(answer: dict, job: str) -> None:
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    assert link(answer, "condition", "job", "artifact", "record") == {
        "condition": "g", "job": job, "artifact": "gen/P", "record": "L1"}


def test_a_bound_field_marked_not_established_denies_the_negative(world, ask):
    world.sql(("UPDATE edge SET bound_established = 0 WHERE edge_id = ?", (world.edge_id("gen/P", 1),)))
    expect_no_bound(ask("C", "shard-3"), "paraphrase")
    expect_no_bound(ask("C", "gen/T"), "paraphrase")
    # The edge still names A, and "contains" needs no bound.
    assert ask("A", "shard-3")["answer"] == "contains"


def test_the_bound_is_read_from_the_stored_log_not_from_the_field(world, ask):
    # The field stays at established; the log gains a read outside the hand-off.
    world.sql(("INSERT INTO harness_read (log_id, seq, occurrence_id, outside_ref, refused) "
               "VALUES (?, 99, ?, NULL, 0)",
               (world.log_id("paraphrase"), world.occurrence_id("src/B", 1))))
    assert world.scalar("SELECT bound_established FROM edge WHERE edge_id = ?",
                        (world.edge_id("gen/P", 1),)) == 1
    expect_no_bound(ask("C", "shard-3"), "paraphrase")


def test_a_bound_field_with_its_log_removed_is_no_bound(world, ask):
    world.remove_log("gen/P", 1, "paraphrase")
    assert world.scalar("SELECT bound_established FROM edge WHERE edge_id = ?",
                        (world.edge_id("gen/P", 1),)) == 1
    answer = ask("C", "shard-3")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    # The edge keeps the name of its job (DECISIONS.md D-24), so the answer names
    # P's job although the log that held the name is gone (plan Check 5).
    expect_no_bound(answer, "paraphrase")
    assert "paraphrase" in answer["failing"]["detail"]
    expect_no_bound(ask("C", "gen/T"), "paraphrase")
    # The edge still names A, and its hop still names the job.
    shown = ask("A", "gen/P")
    assert (shown["answer"], shown["chain"][0]["job"]) == ("contains", "paraphrase")


def test_the_log_of_another_job_establishes_no_bound(world, ask):
    # The edge of P is made to name a job whose log it does not point at.
    world.sql(("UPDATE edge SET job_name = 'paraphrase-rerun' WHERE edge_id = ?",
               (world.edge_id("gen/P", 1),)))
    answer = ask("C", "shard-3")
    expect_no_bound(answer, "paraphrase-rerun")
    assert "'paraphrase'" in answer["failing"]["detail"]


def test_a_refused_read_does_not_break_the_bound_and_is_listed(world, ask):
    world.sql(("INSERT INTO harness_read (log_id, seq, occurrence_id, outside_ref, refused) "
               "VALUES (?, 99, NULL, 'src/B:L1', 1)", (world.log_id("paraphrase"),)))
    answer = ask("C", "shard-3")
    assert answer["answer"] == "does not contain"
    by_job = {entry["job"]: entry for entry in answer["scope"]["harness_logs"]}
    assert by_job["paraphrase"]["refused_attempts"] == ["src/B:L1"]
    assert by_job["paraphrase"]["reads"] == 2


def test_a_job_written_list_is_observed_for_its_parents_and_establishes_nothing_else(world, ask):
    # A generation edge with no harness log: the job wrote its own parent list.
    z = world["src/Z"]
    world.ids["gen/J"] = ww_world.record(
        world.store, world.folder / "gen/J.jsonl", ["made from z"], "gen/J", v.ROLE_INTERMEDIATE,
        ww_world.step("job-j"), [z],
        [EdgeRecord("L1", (EdgeInput(ww_world.ref(z, 1)),), ww_world.step("generate-j"),
                    is_generation=True, job_name="job-j")],
    )
    assert ask("Z", "gen/J")["answer"] == "contains"
    answer = ask("A", "gen/J")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    # The job named is the one the pipeline wrote on the edge, not the step.
    assert link(answer, "condition", "job") == {"condition": "g", "job": "job-j"}


def test_a_record_in_scope_only_gives_may_contain_and_never_contains(world, ask):
    assert ask("C", "R1")["answer"] == "contains"
    world.sql(("UPDATE edge_input SET used = 0 WHERE edge_id = ? AND occurrence_id = ?",
               (world.edge_id("gen/S", 1), world.occurrence_id("src/C", 1))))
    for name in ("gen/S", "R1"):
        answer = ask("C", name)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_IN_SCOPE_ONLY), name
        assert link(answer, "condition", "job", "artifact", "record") == {
            "condition": "g", "job": "summary", "artifact": "src/C", "record": "L1"}, name
    # S still states its use of A's passage.
    assert hops(ask("A", "gen/S")) == ["gen/S:L1", "src/A:L2"]


def test_in_scope_only_through_the_write_api(world, ask):
    # Handed A's and C's passages, the output states A's alone.
    a, c = world["src/A"], world["src/C"]
    handed = [ww_world.ref(a, 2), ww_world.ref(c, 1)]
    world.ids["gen/S2"] = ww_world.record(
        world.store, world.folder / "gen/S2.jsonl", ["another summary"], "gen/S2",
        v.ROLE_INTERMEDIATE, ww_world.step("generate", "S2"), [a, c],
        [ww_world.generation_edge(1, "summary-2", handed, [handed[0]])],
    )
    assert ask("A", "gen/S2")["answer"] == "contains"
    answer = ask("C", "gen/S2")
    assert (answer["answer"], answer["names"], answer["failing"]["job"]) == (
        "may contain", v.NAMES_IN_SCOPE_ONLY, "summary-2")
    # A source the job was never handed is bounded by the log.
    assert ask("B", "gen/S2")["answer"] == "does not contain"


def test_a_jobs_sole_handed_record_is_a_parent_even_when_it_is_not_stated(world, ask):
    # P's job was handed one record. Marking it as not stated changes nothing: the
    # sole input of a job is a parent of its output (DECISIONS.md D-21).
    world.sql(("UPDATE edge_input SET used = 0 WHERE edge_id = ?", (world.edge_id("gen/P", 1),)))
    answer = ask("A", "gen/P")
    assert (answer["answer"], hops(answer)) == ("contains", ["gen/P:L1", "src/A:L1"])
    assert ask("C", "gen/P")["answer"] == "does not contain"


# --- (h) is not applied while the generating-model extension is off -------------------


def test_the_model_entry_is_a_logged_attribute_while_the_extension_is_off(world, ask):
    world.sql(("UPDATE harness_log SET model = NULL WHERE log_id = ?", (world.log_id("paraphrase"),)))
    answer = ask("C", "shard-3")
    assert (answer["answer"], answer["notes"]) == ("does not contain", NOTE_MODEL)
    by_job = {entry["job"]: entry for entry in answer["scope"]["harness_logs"]}
    assert by_job["paraphrase"]["model"] is None and by_job["translation"]["model"] == "none"


def test_a_negative_names_the_harness_behind_each_bound_it_rests_on(world, ask):
    """The answer says which harness wrote each log a negative rests on (plan F10);
    only a confined log gives such a negative (Decision 13), so each entry says
    "confined"."""
    answer = ask("C", "shard-3")
    assert answer["answer"] == "does not contain"
    by_job = {entry["job"]: entry["harness"] for entry in answer["scope"]["harness_logs"]}
    assert by_job == {"paraphrase": "confined", "translation": "confined"}


def test_only_a_log_the_confined_harness_wrote_establishes_a_bound(world, ask):
    """Decision 13: a bound is established only by a closed-input harness whose
    access log lists every record the job read, and the job can reach no other
    record or record source (plan 3.3). The in-process harness logs what passes
    its hand-off and cannot say what else the job opened, so past an edge whose
    log it wrote the answer is "may contain" naming the job (plan F10's fallback,
    2.2(g)); a log that names no harness establishes none either."""
    jobs = ("paraphrase", "translation", "summary")
    for job in jobs:
        world.sql(("UPDATE harness_log SET harness = 'confined' WHERE log_id = ?", (world.log_id(job),)))
    assert ask("C", "shard-3")["answer"] == "does not contain"
    for written_by in ("in-process", None):
        world.sql(("UPDATE harness_log SET harness = ? WHERE log_id = ?",
                   (written_by, world.log_id("translation"))))
        answer = ask("C", "shard-3")
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
        assert link(answer, "condition", "job", "artifact", "record") == {
            "condition": "g", "job": "translation", "artifact": "gen/T", "record": "L1"}
        written_by = "the in-process harness" if written_by else "a harness it does not name"
        assert written_by in answer["failing"]["detail"]
        assert "only a log of the confined harness establishes a bound" in answer["failing"]["detail"]
        # "contains" needs no bound: the edge still names what the job used.
        assert ask("A", "gen/T")["answer"] == "contains"


def test_with_the_extension_on_a_negative_past_a_generation_edge_is_judged_by_its_model(world, ask):
    """Plan 2.2(h) (line 181 (a)) and the world's docstring: every log of the world carries the
    model "none", so condition (h) holds on both generation edges, and with the switch on the
    standing negative carries no exposure note (plan line 168: the words belong to the state in
    which the extension is off)."""
    world.store.set_setting(extensions.setting_key(extensions.GENERATING_MODEL), "on")
    try:
        shown = ask("C", "shard-3")
    except errors.Refused as refused:
        shown = {"answer": f"error: {refused.name}", "notes": None}
    assert (shown["answer"], shown["notes"]) == ("does not contain", [])
    # Answers that pass no generation edge on the way to a negative are unchanged.
    assert ask("A", "shard-3")["answer"] == "contains"
    assert ask("C", "shard-1")["answer"] == "does not contain"


# --- "not covered", and "may contain" downstream of it (Decision 21) -----------------


def test_a_switch_that_is_neither_on_nor_off_is_refused_and_not_read_as_off(world, ask):
    world.store.set_setting(extensions.setting_key(extensions.GENERATING_MODEL), "on")
    world.sql("UPDATE setting SET value = 'ON' WHERE key = 'extension.generating-model'")
    # A negative past a generation edge depends on the switch: no guess, no answer.
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        ask("C", "shard-3")
    # The switches are checked first, so an answer that does not depend on it is refused too
    # (tests/coverage/test_switch_file_checked_first.py).
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        ask("A", "shard-3")


def test_a_shard_with_no_map_is_not_covered_and_everything_downstream_may_contain(world, ask):
    world.delete_map("shard-1")
    own = ask("A", "shard-1")
    assert (own["answer"], own["reason"], own["names"], own["failing"]) == (
        "not covered", "record missing", None, None)
    downstream = ask("A", "mix-1")
    assert (downstream["answer"], downstream["names"]) == ("may contain", v.NAMES_UPSTREAM_NOT_COVERED)
    assert link(downstream, "condition", "artifact", "reason") == {
        "condition": None, "artifact": "shard-1", "reason": "record missing"}
    # A recorded "contains" the fault does not touch wins (Decision 20).
    assert ask("A", "mix-2")["answer"] == "contains"
    assert ask("C", "mix-2")["answer"] == "may contain"


def test_a_snapshot_entry_that_names_an_unknown_shard(world, ask):
    ghost = "f" * 64
    world.ids["mix-ghost"] = world.store.register_mixture(
        name="mix-ghost",
        shards=[SnapshotShard(0, world["shard-1"], "shards/shard-1",
                              world.store.artifact("shard-1").content_digest),
                SnapshotShard(1, ghost, "shards/ghost", "e" * 64)],
        settings={"mix": "ghost"},
    )
    answer = ask("A", "mix-ghost")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_UPSTREAM_NOT_COVERED)
    assert link(answer, "artifact", "instance", "reason") == {
        "artifact": "shards/ghost", "instance": ghost, "reason": "record missing"}
    # Asked by its instance id, the shard itself is "not covered": a record names it.
    named = ask("A", ghost)
    assert (named["answer"], named["reason"], named["kind"]) == ("not covered", "record missing", None)


def test_a_mixture_whose_snapshot_is_gone_is_not_covered(world, ask):
    # A mixture's own record is its snapshot (plan 2.1). With every entry gone
    # there is no shard to walk, and that is no proof: the answer is "not
    # covered", never "does not contain".
    s1 = world.store.artifact("shard-1")
    world.ids["mix-solo"] = world.store.register_mixture(
        name="mix-solo", shards=[SnapshotShard(0, s1.instance_id, "shards/shard-1", s1.content_digest)],
        settings={"mix": "solo"})
    assert ask("A", "mix-solo")["answer"] == "does not contain"
    world.sql(("DELETE FROM mixture_shard WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-solo"],)))
    for source in ("A", "C"):
        answer = ask(source, "mix-solo")
        assert (answer["answer"], answer["reason"], answer["scope"]) == (
            "not covered", "record missing", None), source


def test_a_verified_recovery_map_is_read_before_a_capture_map_that_failed(world, ask):
    """DECISIONS.md D-24: a shard whose capture map failed its verification is
    still a shard with a record of its own, answered from "unknown" links; once
    it has a verified recovery map, the reader gets that one."""
    rows = [ww_world.MapRow(n, n, n + 1, v.UNKNOWN) for n in range(2)]
    instance, digest = ww_world.shard(
        world.store, world.folder / "shards/shard-9", "shard-9", [world["R1"]], rows,
        verification=v.VERIFIED_FAILED, label=v.UNKNOWN)
    world.ids["shard-9"] = instance
    for source in ("A", "C"):
        answer = ask(source, "shard-9")
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK), source
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "b", "artifact": "shard-9", "record": "D0"}
    world.store.write_map(
        instance,
        ww_world.MapRecord(v.NAMESPACE_RECOVERY, v.DERIVED_TOKEN_MATCHING, v.VERIFIED_PASSED,
                           v.RECOVERED, True, True,
                           (ww_world.recovered(0, world.ref("R1", 5)), ww_world.recovered(1, world.ref("R1", 3)))),
        declared_count=2, declared_digest=digest)
    assert world.store.binding(instance).namespace == v.NAMESPACE_RECOVERY
    contains = ask("A", "shard-9")
    assert (contains["answer"], hops(contains)[:2]) == ("contains", ["shard-9:D1", "R1:L3"])
    assert ask("Z", "shard-9")["answer"] == "contains"
    assert ask("C", "shard-9")["answer"] == "does not contain"


def test_an_artifact_the_store_was_never_told_of(ask):
    for name in ("no-such-release", "E@12"):
        answer = ask("A", name)
        assert (answer["answer"], answer["reason"], answer["instance"]) == (
            "not covered", "not registered", None)
        assert answer["artifact"] == name


def test_a_jsonl_file_whose_occurrence_record_is_gone(world, ask):
    # The binding row is removed; the occurrence rows stay, so no loud check fires.
    world.sql(("UPDATE occurrence SET binding_id = (SELECT MIN(binding_id) FROM binding) "
               "WHERE binding_id = ?", (world.binding_id("gen/P"),)),
              ("DELETE FROM binding WHERE binding_id = ?", (world.binding_id("gen/P"),)))
    own = ask("A", "gen/P")
    assert (own["answer"], own["reason"]) == ("not covered", "record missing")
    downstream = ask("C", "gen/T")
    assert (downstream["answer"], downstream["names"]) == ("may contain", v.NAMES_UPSTREAM_NOT_COVERED)
    assert link(downstream, "artifact", "reason") == {"artifact": "gen/P", "reason": "record missing"}


def test_a_published_model_is_an_extension_that_is_off(world):
    # Plan 1.4, 3.1, Section 7: a query naming a published model while the extension is off
    # answers "not covered", reason "not registered"; the note says the extension is off.
    # A bare model row, as the store file would hold it (this build has no write path for one with the switch
    # off): turning R2 into a model would give a row that no integrity check passes.
    model_id = "d" * 64
    world.sql(("INSERT INTO artifact (instance_id, kind, role, content_digest, edge_digest) "
               "VALUES (?, 'published-model', NULL, ?, ?)", (model_id, "e" * 64, "f" * 64)),
              ("INSERT INTO artifact_name (name, artifact_id) "
               "SELECT 'model-1', artifact_id FROM artifact WHERE instance_id = ?", (model_id,)))
    found = coverage.prove(world.store, "model-1", requests.withdraw(world.store, "A"))
    assert (found.answer, found.reason, found.kind, found.notes) == (
        "not covered", "not registered", v.KIND_PUBLISHED_MODEL, (v.NOTE_PUBLISHED_MODELS_OFF,))


def test_prove_takes_an_expanded_withdrawal_only(world):
    with pytest.raises(errors.UnsupportedSelector):
        coverage.prove(world.store, "R2", {"kind": "withdraw", "source": "A"})
    request = requests.withdraw(world.store, "A")
    other = coverage.Session(world.store, requests.withdraw(world.store, "C"))
    with pytest.raises(ValueError):
        coverage.prove(world.store, "R2", request, session=other)
