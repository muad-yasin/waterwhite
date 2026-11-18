"""The two derivation modes, their verification and the labels that follow (plan
3.3 "Tokenization", Decision 4, F1; the questions of plan 5.5 Check 3 on toy
shards).

Every shard here is tiny and written by hand in the real layout; the tokenizer is
the toy of ``tm_world.py``. The expected values are written out in each test from
the texts it uses, never read back from the tool.
"""

from __future__ import annotations

import numpy as np
import pytest
import tm_world
from tm_store import CAPTURE, MATCHING, POSITION, RECOVERY, SETTINGS, STEP

from waterwhite import errors, ids, tokenmap
from waterwhite import vocabulary as v
from waterwhite.records import MapRow
from waterwhite.store import artifact_files

RECOVERED, CANDIDATES, UNKNOWN = v.RECOVERED, v.CANDIDATES, v.UNKNOWN


def recover(world, shard: str, input_name: str, **changes) -> tokenmap.ShardMaps:
    arguments = dict(shard=shard, input=input_name, settings=SETTINGS,
                     tokenize=tm_world.toy_tokenize)
    arguments.update(changes)
    return tokenmap.recover_map(world.store, **arguments)


# --- Position mode -------------------------------------------------------------------------


def test_position_mode_on_a_clean_shard(world):
    texts = ["alpha", "be", "gamma!"]
    result = world.capture("E/shard-1", texts)

    shard = world.store.artifact("E/shard-1")
    source = world.store.artifact("E/shard-1.input")
    assert shard.kind == v.KIND_SHARD and shard.step == STEP
    assert shard.inputs == (source.instance_id,)
    assert result.shard_instance_id == shard.instance_id
    assert (result.document_count, result.surviving_lines) == (3, 3)

    # One line, one document, in order. A span is the document's own size in the
    # index: the bytes of the text and the end-of-document token.
    def fingerprint(text: str) -> bytes:
        return ids.fingerprint(text)

    assert world.rows("E/shard-1", CAPTURE, POSITION) == [
        MapRow(0, 0, 6, RECOVERED, world.ref("E/shard-1.input", 1), (), fingerprint("alpha")),
        MapRow(1, 6, 9, RECOVERED, world.ref("E/shard-1.input", 2), (), fingerprint("be")),
        MapRow(2, 9, 16, RECOVERED, world.ref("E/shard-1.input", 3), (), fingerprint("gamma!")),
    ]
    outcome = result.map(POSITION)
    assert outcome.verification == tokenmap.Verification(
        result=v.VERIFIED_PASSED, spans_match=True, count_matches=True,
        documents_compared=3, failures=0, reasons=())
    assert (outcome.recovered, outcome.candidates, outcome.stored) == (3, 0, True)
    assert result.maps == (outcome,) and result.drops == () and result.verified
    assert world.drops("E/shard-1") == []
    # The spans of the map add up to the total of the shard's own index.
    index = artifact_files.read_shard_index(f"{world.prefix('E/shard-1')}.idx")
    rows = world.rows("E/shard-1", CAPTURE, POSITION)
    assert sum(r.token_end - r.token_start for r in rows) == index.token_count == 16


def test_a_line_that_gives_no_tokens_is_a_drop_and_shifts_no_ordinal(world):
    # Five input lines, the third with a text of no characters: four documents.
    texts = ["one", "two", "", "four", "five"]
    result = world.capture("c03/shard-notokens", texts)

    index = artifact_files.read_shard_index(f"{world.prefix('c03/shard-notokens')}.idx")
    # How the line shows in the shard: a document boundary with no sequence.
    assert (index.document_count, index.empty_documents) == (4, 1)
    assert [int(x) for x in index.document_index] == [0, 1, 2, 2, 3, 4]

    assert world.assigned("c03/shard-notokens", CAPTURE, POSITION) == [
        (RECOVERED, "L1", []), (RECOVERED, "L2", []), (RECOVERED, "L4", []), (RECOVERED, "L5", []),
    ]
    assert world.drops("c03/shard-notokens") == [("L3", v.DROP_NO_TOKENS, "tokenize")]
    assert result.document_count == 4 and result.surviving_lines == 5
    assert result.map(POSITION).verification.result == v.VERIFIED_PASSED

    # The position-mode recovery gives the same ordinals and the same drop.
    recover(world, "c03/shard-notokens", "c03/shard-notokens.input")
    assert (world.assigned("c03/shard-notokens", RECOVERY, POSITION)
            == world.assigned("c03/shard-notokens", CAPTURE, POSITION))
    assert world.drops("c03/shard-notokens") == [("L3", v.DROP_NO_TOKENS, "tokenize")]


def test_a_line_the_index_calls_empty_must_give_no_tokens_again(world):
    # The shard says its second line gave no tokens; the input's second line has text.
    world.input("in", ["one", "two", "three"])
    world.bare_shard("shard", "in", tm_world.documents_of(["one", "", "three"]))
    result = recover(world, "shard", "in", f1_holds=True)
    position = result.map(POSITION)
    assert position.verification.result == v.VERIFIED_FAILED
    assert "line 2 gave no tokens by the shard's index" in position.verification.reasons[0]
    assert world.drops("shard") == []  # nothing verified: no drop is recorded


def test_a_skipped_line_is_read_from_the_sanitizer_log_as_a_drop(world):
    # The file the sanitizer read: line 2 does not parse, line 4 is empty after
    # sanitizing. Line 3 carries a control character and spaces the sanitizer
    # removes, so the tokenizer saw another text than the file holds.
    raw = [
        '{"id": "Z-001", "text": "first"}',
        "this line is not JSON",
        '{"id": "Z-003", "text": "  thi\\u0007rd  "}',
        '{"id": "Z-004", "text": " \\t "}',
        '{"id": "Z-005", "text": "fifth"}',
    ]
    path = tm_world.write_raw_lines(world.folder / "sanitizer-input/shard-2.jsonl", raw)
    world.register(path, "E/shard-2.chunk")
    log = world.folder / "sanitizer.log.jsonl"
    log.write_text(
        '{"shard": "shard-1", "input": "sanitizer-input/shard-1.jsonl", "line": 1, '
        '"id": null, "reason": "sanitizer skip", "detail": "not JSON"}\n'
        '{"shard": "shard-2", "input": "sanitizer-input/shard-2.jsonl", "line": 2, '
        '"id": null, "reason": "sanitizer skip", "detail": "not JSON"}\n'
        '{"shard": "shard-2", "input": "sanitizer-input/shard-2.jsonl", "line": 4, '
        '"id": "Z-004", "reason": "empty line", "detail": "empty after sanitizing"}\n',
        encoding="utf-8")
    skips = tokenmap.read_sanitizer_log(log, shard="shard-2")
    assert skips == (tokenmap.SanitizerSkip(2, v.DROP_SANITIZER_SKIP),
                     tokenmap.SanitizerSkip(4, v.DROP_EMPTY_LINE))
    assert tokenmap.read_sanitizer_log(log, input="sanitizer-input/shard-2.jsonl") == skips

    # The run made without the wrapper tokenized the three sanitized survivors.
    world.bare_shard("E/shard-2", "E/shard-2.chunk",
                     tm_world.documents_of(["first", "third", "fifth"]))
    result = recover(world, "E/shard-2", "E/shard-2.chunk",
                     sanitizer=tokenmap.SanitizerRun(skips))

    assert world.head("E/shard-2") == [(RECOVERY, POSITION, v.VERIFIED_PASSED, RECOVERED)]
    # The skipped lines take no ordinal: the line after one gets the next ordinal.
    assert world.assigned("E/shard-2", RECOVERY, POSITION) == [
        (RECOVERED, "L1", []), (RECOVERED, "L3", []), (RECOVERED, "L5", []),
    ]
    assert world.drops("E/shard-2") == [
        ("L2", v.DROP_SANITIZER_SKIP, "sanitize"), ("L4", v.DROP_EMPTY_LINE, "sanitize"),
    ]
    assert (result.document_count, result.surviving_lines) == (3, 3)

    # Without the log the same file does not verify: five lines for three documents.
    blind = recover(world, "E/shard-2", "E/shard-2.chunk", write=False)
    assert blind.map(POSITION).verification.count_matches is False
    assert not blind.verified


def test_the_sanitizer_is_run_again_before_a_line_is_tokenized_again(world):
    # The same file read as if it were the tokenizer's own input: the raw text of
    # line 2 is not what the tokenizer saw, so its document does not match.
    raw = ['{"text": "first"}', '{"text": "  thi\\u0007rd  "}']
    path = tm_world.write_raw_lines(world.folder / "chunk.jsonl", raw)
    world.register(path, "chunk")
    world.bare_shard("shard", "chunk", tm_world.documents_of(["first", "third"]))
    as_tokenizer_input = recover(world, "shard", "chunk", write=False)
    assert as_tokenizer_input.map(POSITION).verification.result == v.VERIFIED_FAILED
    as_sanitizer_input = recover(world, "shard", "chunk", write=False,
                                 sanitizer=tokenmap.SanitizerRun(()))
    assert as_sanitizer_input.map(POSITION).verification.result == v.VERIFIED_PASSED


# --- Twins ---------------------------------------------------------------------------------


def test_twins_in_one_shard_are_candidates_with_exactly_the_two_occurrences(world):
    texts = ["solo", "the same passage", "other", "the same passage"]
    world.capture("E/shard-1", texts)
    recover(world, "E/shard-1", "E/shard-1.input", also_token_matching=True)

    expected = [
        (RECOVERED, "L1", []),
        (CANDIDATES, None, ["L2", "L4"]),
        (RECOVERED, "L3", []),
        (CANDIDATES, None, ["L2", "L4"]),
    ]
    for namespace, derivation in ((CAPTURE, POSITION), (RECOVERY, POSITION), (RECOVERY, MATCHING)):
        assert world.assigned("E/shard-1", namespace, derivation) == expected
        rows = world.rows("E/shard-1", namespace, derivation)
        pair = {world.ref("E/shard-1.input", 2), world.ref("E/shard-1.input", 4)}
        assert set(rows[1].candidates) == set(rows[3].candidates) == pair
        assert rows[1].fingerprint == ids.fingerprint("the same passage")


def test_swapping_the_twins_in_the_recovery_input_changes_nothing(world):
    # The two lines of the pair carry different ids, so swapping them makes
    # another file with the same texts at the same places.
    texts = ["solo", "the same passage", "other", "the same passage"]
    world.input("in", texts)
    world.bare_shard("shard", "in", tm_world.documents_of(texts))
    recover(world, "shard", "in")

    rows = (world.folder / "tokenizer-input/in.jsonl").read_text(encoding="utf-8").splitlines()
    rows[1], rows[3] = rows[3], rows[1]
    swapped = tm_world.write_raw_lines(world.folder / "tokenizer-input/in-swapped.jsonl", rows)
    world.register(swapped, "in-swapped")
    assert (world.store.artifact("in-swapped").content_digest
            != world.store.artifact("in").content_digest)
    world.bare_shard("shard-swapped", "in-swapped", tm_world.documents_of(texts))
    recover(world, "shard-swapped", "in-swapped")

    expected = [
        (RECOVERED, "L1", []),
        (CANDIDATES, None, ["L2", "L4"]),
        (RECOVERED, "L3", []),
        (CANDIDATES, None, ["L2", "L4"]),
    ]
    assert world.assigned("shard", RECOVERY, POSITION) == expected
    assert world.assigned("shard-swapped", RECOVERY, POSITION) == expected
    assert world.head("shard-swapped") == [(RECOVERY, POSITION, v.VERIFIED_PASSED, RECOVERED)]


def test_twins_in_two_shards_are_recovered_each_in_its_own_shard(world):
    world.capture("E/shard-1", ["the same passage", "one"])
    world.capture("E/shard-2", ["two", "the same passage"])
    recover(world, "E/shard-1", "E/shard-1.input", also_token_matching=True)
    recover(world, "E/shard-2", "E/shard-2.input", also_token_matching=True)
    for namespace, derivation in ((CAPTURE, POSITION), (RECOVERY, POSITION), (RECOVERY, MATCHING)):
        first = world.rows("E/shard-1", namespace, derivation)[0]
        second = world.rows("E/shard-2", namespace, derivation)[1]
        assert (first.label, first.parent) == (RECOVERED, world.ref("E/shard-1.input", 1))
        assert (second.label, second.parent) == (RECOVERED, world.ref("E/shard-2.input", 2))
        assert first.candidates == () and second.candidates == ()


def test_three_identical_lines_give_one_candidate_set_of_three(world):
    world.capture("shard", ["x", "same", "same", "y", "same"])
    for row in (world.assigned("shard", CAPTURE, POSITION)[i] for i in (1, 2, 4)):
        assert row == (CANDIDATES, None, ["L2", "L3", "L5"])


# --- Verification ----------------------------------------------------------------------------


def test_a_displaced_document_is_caught_and_the_position_map_is_unknown(world):
    # The tokenizer run wrote the documents of lines 2 and 3 in the other order:
    # the counts agree, and only the token comparison can tell.
    texts = ["first", "second", "third", "fourth"]
    result = world.capture(
        "shard", texts,
        run_tokenizer=lambda source, prefix: tm_world.toy_preprocess(
            source, prefix, order=[0, 2, 1, 3]))

    position = result.map(POSITION)
    assert position.verification.result == v.VERIFIED_FAILED
    assert position.verification.count_matches is True and position.verification.spans_match
    assert position.verification.failures == 2
    assert position.verification.reasons[0] == "document D1 does not hold the tokens of line 2"
    assert (position.label, position.recovered, position.candidates) == (UNKNOWN, 0, 0)
    # The stored position map is "unknown" in every row and names no record.
    assert world.assigned("shard", CAPTURE, POSITION) == [(UNKNOWN, None, [])] * 4
    assert [(r.token_start, r.token_end) for r in world.rows("shard", CAPTURE, POSITION)] == [
        (0, 6), (6, 12), (12, 19), (19, 26)]

    # Token matching took over and found each document's own line.
    assert world.assigned("shard", CAPTURE, MATCHING) == [
        (RECOVERED, "L1", []), (RECOVERED, "L3", []), (RECOVERED, "L2", []), (RECOVERED, "L4", []),
    ]
    # A reader gets the verified map, not the failed one (DECISIONS.md D-24).
    assert world.head("shard") == [
        (CAPTURE, MATCHING, v.VERIFIED_PASSED, RECOVERED),
        (CAPTURE, POSITION, v.VERIFIED_FAILED, UNKNOWN),
    ]
    assert result.verified and [m.derivation for m in result.maps] == [POSITION, MATCHING]


def test_token_matching_assigns_the_same_records_as_position_mode(world):
    texts = ["solo", "twin", "", "other", "twin", "last one"]
    world.capture("shard", texts)
    result = recover(world, "shard", "shard.input", also_token_matching=True)

    by_position = world.rows("shard", RECOVERY, POSITION)
    by_matching = world.rows("shard", RECOVERY, MATCHING)
    assert [dataclass_form(r) for r in by_matching] == [dataclass_form(r) for r in by_position]
    assert [dataclass_form(r) for r in by_position] == [
        dataclass_form(r) for r in world.rows("shard", CAPTURE, POSITION)]
    assert world.assigned("shard", RECOVERY, MATCHING) == [
        (RECOVERED, "L1", []), (CANDIDATES, None, ["L2", "L5"]), (RECOVERED, "L4", []),
        (CANDIDATES, None, ["L2", "L5"]), (RECOVERED, "L6", []),
    ]
    matching = result.map(MATCHING)
    assert matching.verification.result == v.VERIFIED_PASSED
    assert matching.verification.count_matches is None  # token matching counts no lines
    assert matching.verification.documents_compared == 5
    assert world.drops("shard") == [("L3", v.DROP_NO_TOKENS, "tokenize")]


def dataclass_form(row: MapRow) -> tuple:
    return (row.ordinal, row.token_start, row.token_end, row.label, row.parent,
            frozenset(row.candidates), row.fingerprint)


def test_a_line_no_document_holds_fails_both_modes_and_both_maps_are_unknown(world):
    # The recovery input has one line replaced by a text no document holds.
    world.input("true-input", ["first", "second", "third"])
    world.input("wrong-input", ["first", "something else", "third"])
    world.bare_shard("shard", "true-input", tm_world.documents_of(["first", "second", "third"]))
    result = recover(world, "shard", "wrong-input")

    assert world.head("shard") == [
        (RECOVERY, POSITION, v.VERIFIED_FAILED, UNKNOWN),
        (RECOVERY, MATCHING, v.VERIFIED_FAILED, UNKNOWN),
    ]
    for derivation in (POSITION, MATCHING):
        assert world.assigned("shard", RECOVERY, derivation) == [(UNKNOWN, None, [])] * 3
    assert result.map(MATCHING).verification.reasons == (
        "no document holds the tokens of line 2",
        "1 documents hold the tokens of no input line, the first is D1",
    )
    assert not result.verified and result.drops == ()
    assert "no map verified: no drop is recorded" in result.notes


def test_position_mode_fails_on_a_count_that_does_not_match(world):
    world.input("in", ["first", "second", "third"])
    world.bare_shard("shard", "in", tm_world.documents_of(["first", "second"]))
    result = recover(world, "shard", "in", write=False)
    position = result.map(POSITION).verification
    assert (position.result, position.count_matches) == (v.VERIFIED_FAILED, False)
    assert position.reasons == (
        "3 surviving input lines and 2 documents in the shard's index (2 with tokens, 0 without)",)
    # Token matching fails too: a line with tokens that no document holds.
    assert result.map(MATCHING).verification.result == v.VERIFIED_FAILED


def test_token_matching_wants_as_many_lines_as_documents_for_twins(world):
    # Two identical documents and one such line: the second twin has no line.
    world.input("in", ["same", "other"])
    world.bare_shard("shard", "in", tm_world.documents_of(["same", "other", "same"]))
    result = recover(world, "shard", "in", f1_holds=False, write=False)
    assert result.map(POSITION) is None
    matching = result.map(MATCHING).verification
    assert matching.result == v.VERIFIED_FAILED
    assert matching.reasons == ("1 documents hold the tokens of no input line, the first is D2",)

    # Two such lines and one document: one line too many.
    world.input("in2", ["same", "other", "same"])
    world.bare_shard("shard2", "in2", tm_world.documents_of(["same", "other"]))
    result = recover(world, "shard2", "in2", f1_holds=False, write=False)
    assert result.map(MATCHING).verification.reasons == (
        "line 3 is one line more than the 1 documents that hold its tokens",)


def test_the_end_of_document_convention_is_held_against_every_document(world):
    world.input("in", ["first", "second"])
    world.bare_shard("with-eod", "in", tm_world.documents_of(["first", "second"]))
    world.bare_shard("without-eod", "in",
                     tm_world.documents_of(["first", "second"], append_eod=False))
    in_span = SETTINGS
    not_in_span = tokenmap.TokenizerSettings(append_eod=True, eod_in_span=False)
    no_eod = tokenmap.TokenizerSettings(append_eod=False, eod_in_span=False)

    def result(shard, settings):
        return recover(world, shard, "in", settings=settings, write=False,
                       f1_holds=True).map(POSITION).verification.result

    assert result("with-eod", in_span) == v.VERIFIED_PASSED
    assert result("with-eod", not_in_span) == v.VERIFIED_FAILED
    assert result("with-eod", no_eod) == v.VERIFIED_FAILED
    assert result("without-eod", in_span) == v.VERIFIED_FAILED
    assert result("without-eod", no_eod) == v.VERIFIED_PASSED


def test_without_a_tokenize_function_the_verification_is_not_run(world):
    result = world.capture("shard", ["first", "second"], tokenize=None)
    position = result.map(POSITION)
    assert position.verification.result == v.VERIFIED_NOT_RUN
    assert position.verification.documents_compared == 0
    assert world.head("shard") == [(CAPTURE, POSITION, v.VERIFIED_NOT_RUN, UNKNOWN)]
    assert world.assigned("shard", CAPTURE, POSITION) == [(UNKNOWN, None, [])] * 2
    assert "no tokenize function was given: the verification was not run" in result.notes
    assert not result.verified and world.drops("shard") == []


def test_a_failed_count_without_a_tokenize_function_leaves_one_failed_map(world):
    # Three lines, two documents, and no tokenizer to match tokens with: position
    # mode fails on the count alone, and token matching cannot take over.
    world.input("in", ["first", "second", "third"])
    world.bare_shard("shard", "in", tm_world.documents_of(["first", "second"]))
    result = recover(world, "shard", "in", tokenize=None)
    assert world.head("shard") == [(RECOVERY, POSITION, v.VERIFIED_FAILED, UNKNOWN)]
    assert result.map(MATCHING) is None
    assert ("position mode failed its verification and token matching was not run: "
            "no tokenize function was given") in result.notes
    with pytest.raises(errors.InvalidRecord, match="token matching needs the tokenize function"):
        recover(world, "shard", "in", tokenize=None, also_token_matching=True)


def test_a_sanitizer_this_build_does_not_hold_is_a_named_error(world):
    world.input("in", ["first"])
    world.bare_shard("shard", "in", tm_world.documents_of(["first"]))
    with pytest.raises(errors.UnknownNormaliser):
        recover(world, "shard", "in",
                sanitizer=tokenmap.SanitizerRun((), normaliser="team-sanitizer/7"))
    assert world.head("shard") == []


def test_where_f1_does_not_hold_the_wrapper_builds_by_token_matching(world):
    result = world.capture(
        "shard", ["first", "second", "third"], f1_holds=False,
        run_tokenizer=lambda source, prefix: tm_world.toy_preprocess(
            source, prefix, order=[2, 0, 1]))
    assert world.head("shard") == [(CAPTURE, MATCHING, v.VERIFIED_PASSED, RECOVERED)]
    assert world.assigned("shard", CAPTURE, MATCHING) == [
        (RECOVERED, "L3", []), (RECOVERED, "L1", []), (RECOVERED, "L2", []),
    ]
    assert result.notes == ("F1 is said not to hold: position mode was not run",)
    # Token matching cannot run without the tokenizer, and says so.
    world.input("other.input", ["x"])
    with pytest.raises(errors.InvalidRecord, match="token matching needs the tokenize function"):
        tokenmap.tokenize_shard(
            world.store, input="other.input", prefix=world.prefix("other"), name="other",
            step=STEP, run_tokenizer=tm_world.toy_preprocess, settings=SETTINGS,
            tokenize=None, f1_holds=False)


def test_check_spans_holds_a_map_against_the_index(world):
    tm_world.write_shard(world.prefix("s"), [[1, 2, 3], [4], [5, 6]])
    index = artifact_files.read_shard_index(f"{world.prefix('s')}.idx")

    def check(starts, ends):
        return tokenmap.check_spans(np.array(starts, dtype=np.int64),
                                    np.array(ends, dtype=np.int64), index)

    assert check([0, 3, 4], [3, 4, 6]) is None
    assert check([0, 3, 4], [3, 5, 6]) == (
        "document D1 spans 2 tokens and the index records 1 (1 documents differ)")
    assert check([1, 4, 5], [4, 5, 7]) == "the first span starts at token 1, not 0"
    assert check([0, 4, 5], [3, 5, 7]) == (
        "document D1 starts at token 4, the one before it ends at 3")
    assert check([0, 3], [3, 4]) == "2 spans for the 3 documents of the index"


def test_every_document_is_compared_on_a_larger_shard(world):
    # 3,000 documents with a pair of twins: every document is read and compared
    # on its own, and both modes agree.
    texts = [f"record number {n}" for n in range(3000)]
    texts[2500] = texts[10]
    result = world.capture("big", texts)
    position = result.map(POSITION)
    assert position.verification.documents_compared == 3000
    assert (position.recovered, position.candidates) == (2998, 2)
    again = recover(world, "big", "big.input", f1_holds=False, write=False)
    matching = again.map(MATCHING)
    assert matching.verification.result == v.VERIFIED_PASSED
    assert [dataclass_form(r) for r in matching.record.rows] == [
        dataclass_form(r) for r in position.record.rows]
