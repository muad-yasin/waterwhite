"""What the tokenization-map wrappers leave in the store, and what they leave when
a check refuses: nothing (plan 3.4; DECISIONS.md D-18, D-20 and its amendment in
D-24; plan 5.5 Check 3, the questions about the map's fields).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import tm_world
from tm_store import CAPTURE, MATCHING, POSITION, RECOVERY, SETTINGS, STEP

from waterwhite import errors, tokenmap
from waterwhite import vocabulary as v
from waterwhite.records import DropRecord, MapRecord, MapRow
from waterwhite.store import artifact_files, views

TEXTS = ["first", "twin", "", "third", "twin"]


def recover(world, shard: str, input_name: str, **changes) -> tokenmap.ShardMaps:
    arguments = dict(shard=shard, input=input_name, settings=SETTINGS,
                     tokenize=tm_world.toy_tokenize)
    arguments.update(changes)
    return tokenmap.recover_map(world.store, **arguments)


def shard_on_disk(world, shard: str) -> artifact_files.ShardFiles:
    return artifact_files.read_shard(world.prefix(shard))


# --- The fields of a stored map ---------------------------------------------------------------


def test_every_map_stores_its_derivation_verification_label_and_settings(world):
    world.capture("E/shard-1", TEXTS)
    recover(world, "E/shard-1", "E/shard-1.input", also_token_matching=True)
    instance = world.store.artifact("E/shard-1").instance_id
    disk = shard_on_disk(world, "E/shard-1")

    def binding(namespace, derivation):
        return views.Binding(instance, namespace, 4, disk.digest, derivation,
                             v.VERIFIED_PASSED, v.RECOVERED, True, True)

    # Three maps side by side, in the order a reader gets them.
    assert world.store.bindings(instance) == (
        binding(CAPTURE, POSITION), binding(RECOVERY, POSITION), binding(RECOVERY, MATCHING))
    assert disk.document_count == 4
    for namespace, derivation in ((CAPTURE, POSITION), (RECOVERY, POSITION), (RECOVERY, MATCHING)):
        assert world.store.stored_row_count(instance, namespace, derivation=derivation) == 4
        assert {r.label for r in world.rows("E/shard-1", namespace, derivation)} == {
            v.RECOVERED, v.CANDIDATES}
    world.store.check_integrity()


def test_the_append_eod_setting_is_recorded_as_given(world):
    world.input("in", ["first", "second"])
    tm_world.toy_preprocess(world.folder / "tokenizer-input/in.jsonl", world.prefix("plain"),
                            append_eod=False)
    settings = tokenmap.TokenizerSettings(append_eod=False, eod_in_span=False)
    tokenmap.map_tokenized_shard(world.store, input="in", prefix=world.prefix("plain"),
                                 name="plain", step=STEP, settings=settings,
                                 tokenize=tm_world.toy_tokenize)
    held = world.store.binding(world.store.artifact("plain").instance_id)
    assert (held.append_eod, held.eod_in_span, held.verification, held.label) == (
        False, False, v.VERIFIED_PASSED, v.RECOVERED)
    # A span is then the bytes of the text and nothing more.
    assert [(r.token_start, r.token_end) for r in world.rows("plain", CAPTURE, POSITION)] == [
        (0, 5), (5, 11)]


def test_a_recovery_map_stands_beside_a_capture_map_and_never_replaces_it(world):
    world.capture("E/shard-1", TEXTS)
    before = world.rows("E/shard-1", CAPTURE, POSITION)
    assert world.head("E/shard-1") == [(CAPTURE, POSITION, v.VERIFIED_PASSED, v.RECOVERED)]

    # A recovery against another file that does not fit the shard: its maps fail,
    # are stored in their own namespace, and the capture map is untouched.
    world.input("wrong", ["first", "twin", "", "something else", "twin"])
    result = recover(world, "E/shard-1", "wrong")
    assert [m.namespace for m in result.maps] == [RECOVERY, RECOVERY]
    assert world.head("E/shard-1") == [
        (CAPTURE, POSITION, v.VERIFIED_PASSED, v.RECOVERED),
        (RECOVERY, POSITION, v.VERIFIED_FAILED, v.UNKNOWN),
        (RECOVERY, MATCHING, v.VERIFIED_FAILED, v.UNKNOWN),
    ]
    assert world.rows("E/shard-1", CAPTURE, POSITION) == before
    # A reader still gets the capture map.
    instance = world.store.artifact("E/shard-1").instance_id
    assert world.store.binding(instance).namespace == CAPTURE
    assert [r.parent for r in world.store.map_rows(instance)] == [r.parent for r in before]


def test_the_same_call_again_changes_nothing(world):
    first = world.capture("E/shard-1", TEXTS)
    recover(world, "E/shard-1", "E/shard-1.input", also_token_matching=True)
    counts = table_counts(world.store)
    again = recover(world, "E/shard-1", "E/shard-1.input", also_token_matching=True)
    assert [m.stored for m in again.maps] == [True, True]
    second = tokenmap.map_tokenized_shard(
        world.store, input="E/shard-1.input", prefix=world.prefix("E/shard-1"),
        name="E/shard-1", step=STEP, settings=SETTINGS, tokenize=tm_world.toy_tokenize)
    assert second.shard_instance_id == first.shard_instance_id
    assert table_counts(world.store) == counts


def table_counts(store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def test_write_false_returns_the_result_and_writes_nothing(world):
    world.input("in", TEXTS)
    world.bare_shard("shard", "in", tm_world.documents_of(TEXTS))
    counts = table_counts(world.store)
    result = recover(world, "shard", "in", write=False)
    position = result.map(POSITION)
    assert position.verification.result == v.VERIFIED_PASSED and position.stored is False
    assert [r.label for r in position.record.rows] == [
        v.RECOVERED, v.CANDIDATES, v.RECOVERED, v.CANDIDATES]
    assert result.drops == (DropRecord(world.ref("in", 3), v.DROP_NO_TOKENS, "tokenize"),)
    assert "nothing was written: the call asked for the result only" in result.notes
    assert table_counts(world.store) == counts
    assert world.store.bindings(world.store.artifact("shard").instance_id) == ()


# --- A refused write leaves no map -------------------------------------------------------------


def test_a_shard_changed_since_registration_gets_no_map(world):
    world.input("in", TEXTS)
    instance = world.bare_shard("shard", "in", tm_world.documents_of(TEXTS))
    # One token of the .bin file changes after the shard was registered.
    path = Path(f"{world.prefix('shard')}.bin")
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))
    counts = table_counts(world.store)
    with pytest.raises(errors.ArtifactDigestMismatch):
        recover(world, "shard", "in")
    assert world.store.bindings(instance) == () and world.drops("shard") == []
    assert table_counts(world.store) == counts


def test_a_shard_that_changes_while_its_map_is_derived_is_refused_by_the_store(world):
    world.input("in", ["first", "second", "third"])
    instance = world.bare_shard("shard", "in", tm_world.documents_of(["first", "second", "third"]))
    path = Path(f"{world.prefix('shard')}.bin")

    def tokenize_and_tamper(text: str) -> list[int]:
        # While the last line is tokenized again, the first document, which was
        # already compared, changes on disk. The derived map still says "passed";
        # the count and digest it is declared with were read before the change.
        if text == "third":
            data = bytearray(path.read_bytes())
            data[0] ^= 1
            path.write_bytes(bytes(data))
        return tm_world.toy_tokenize(text)

    counts = table_counts(world.store)
    with pytest.raises(errors.DeclaredDigestMismatch):
        recover(world, "shard", "in", tokenize=tokenize_and_tamper)
    assert world.store.bindings(instance) == () and world.drops("shard") == []
    assert table_counts(world.store) == counts


def test_a_write_the_store_stops_half_way_leaves_no_row(world):
    world.input("in", TEXTS)
    instance = world.bare_shard("shard", "in", tm_world.documents_of(TEXTS))

    class Stopped(Exception):
        pass

    def stop(table: str, written: int) -> None:
        if table == "map_row" and written == 2:
            raise Stopped

    world.store.set_row_hook(stop)
    counts = table_counts(world.store)
    with pytest.raises(Stopped):
        recover(world, "shard", "in")
    world.store.set_row_hook(None)
    assert world.store.bindings(instance) == () and world.drops("shard") == []
    assert table_counts(world.store) == counts


def test_a_verified_map_is_never_overwritten_and_nothing_else_is_written(world):
    # The shard already holds a verified token-matching recovery map.
    world.input("true-input", ["first", "second", "third"])
    world.input("wrong", ["first", "something else", "third"])
    instance = world.bare_shard("shard", "true-input",
                                tm_world.documents_of(["first", "second", "third"]))
    disk = shard_on_disk(world, "shard")
    held = MapRecord(RECOVERY, MATCHING, v.VERIFIED_PASSED, v.RECOVERED, True, True,
                     tuple(MapRow(k, a, b, v.RECOVERED, world.ref("true-input", k + 1))
                           for k, (a, b) in enumerate([(0, 6), (6, 13), (13, 19)])))
    world.store.write_map(instance, held, declared_count=3, declared_digest=disk.digest)

    # The recovery fails in position mode, goes on to token matching, and its
    # token-matching map differs from the held, verified one. The store refuses
    # it, and with it the position map of the same call: neither is written.
    counts = table_counts(world.store)
    with pytest.raises(errors.InvalidRecord, match="a map is never overwritten"):
        recover(world, "shard", "wrong")
    assert world.head("shard") == [(RECOVERY, MATCHING, v.VERIFIED_PASSED, v.RECOVERED)]
    assert table_counts(world.store) == counts


def test_a_drop_the_store_holds_with_another_reason_refuses_before_any_map(world):
    world.input("in", TEXTS)
    instance = world.bare_shard("shard", "in", tm_world.documents_of(TEXTS))
    world.store.add_drops(instance, [DropRecord(world.ref("in", 3), v.DROP_FILTER, "tokenize")])
    # The store refuses the drop inside the one transaction, and the map with it.
    with pytest.raises(errors.InvalidRecord, match="with another reason"):
        recover(world, "shard", "in")
    assert world.store.bindings(instance) == ()


def test_an_input_that_changed_on_disk_is_refused(world):
    world.input("in", TEXTS)
    instance = world.bare_shard("shard", "in", tm_world.documents_of(TEXTS))
    tm_world.write_rows(world.folder / "tokenizer-input/in.jsonl", TEXTS, id_prefix="V")
    with pytest.raises(errors.ArtifactDigestMismatch, match="not the registered records"):
        recover(world, "shard", "in")
    assert world.store.bindings(instance) == ()

    # The capture wrapper does not even run the tokenizer on other bytes.
    ran = []
    with pytest.raises(errors.ArtifactDigestMismatch, match="not run on other bytes"):
        tokenmap.tokenize_shard(
            world.store, input="in", prefix=world.prefix("never"), name="never", step=STEP,
            run_tokenizer=lambda source, prefix: ran.append(source), settings=SETTINGS,
            tokenize=tm_world.toy_tokenize)
    assert ran == [] and world.store.artifact("never") is None


# --- The capture wrapper and its tokenizer run --------------------------------------------------


def test_the_wrapper_runs_the_tokenizer_as_a_process_of_its_own(world):
    world.input("E/shard-1.input", TEXTS)
    script = Path(tm_world.__file__)
    calls = []

    def run_tokenizer(input_path: Path, prefix: Path) -> None:
        calls.append((input_path, prefix))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(script), str(input_path), str(prefix)],
                       check=True, timeout=30)

    result = tokenmap.tokenize_shard(
        world.store, input="E/shard-1.input", prefix=world.prefix("E/shard-1"),
        name="E/shard-1", step=STEP, run_tokenizer=run_tokenizer, settings=SETTINGS,
        tokenize=tm_world.toy_tokenize)
    assert calls == [(world.folder / "tokenizer-input/E/shard-1.input.jsonl",
                      world.prefix("E/shard-1"))]
    assert result.verified and result.document_count == 4
    assert world.head("E/shard-1") == [(CAPTURE, POSITION, v.VERIFIED_PASSED, v.RECOVERED)]


def test_the_wrapper_refuses_a_pair_that_lies_at_the_prefix_already(world):
    world.input("in", TEXTS)
    tm_world.write_shard(world.prefix("stale"), tm_world.documents_of(TEXTS))
    ran = []
    with pytest.raises(errors.InvalidRecord, match="a file already lies at"):
        tokenmap.tokenize_shard(
            world.store, input="in", prefix=world.prefix("stale"), name="stale", step=STEP,
            run_tokenizer=lambda source, prefix: ran.append(source), settings=SETTINGS,
            tokenize=tm_world.toy_tokenize)
    assert ran == [] and world.store.artifact("stale") is None


def test_a_tokenizer_run_that_leaves_no_pair_registers_nothing(world):
    world.input("in", TEXTS)
    with pytest.raises(errors.InvalidRecord, match="no file at"):
        tokenmap.tokenize_shard(
            world.store, input="in", prefix=world.prefix("none"), name="none", step=STEP,
            run_tokenizer=lambda source, prefix: None, settings=SETTINGS,
            tokenize=tm_world.toy_tokenize)
    assert world.store.artifact("none") is None


def test_names_that_are_not_registered_or_of_the_wrong_kind_are_refused(world):
    world.input("in", TEXTS)
    world.bare_shard("shard", "in", tm_world.documents_of(TEXTS))
    with pytest.raises(errors.InvalidRecord, match="the shard 'nowhere' is not registered"):
        recover(world, "nowhere", "in")
    with pytest.raises(errors.InvalidRecord, match="the shard 'in' is a jsonl, not a shard"):
        recover(world, "in", "in")
    with pytest.raises(errors.InvalidRecord, match="the input 'shard' is a shard, not a jsonl"):
        recover(world, "shard", "shard")


# --- Settings, the sanitizer's log, the shard reader --------------------------------------------


def test_settings_that_contradict_each_other_are_refused():
    with pytest.raises(errors.InvalidRecord, match="appends no end-of-document token"):
        tokenmap.TokenizerSettings(append_eod=False, eod_in_span=True, eod_token_id=1)
    with pytest.raises(errors.InvalidRecord, match="no eod_token_id is given"):
        tokenmap.TokenizerSettings(append_eod=True, eod_in_span=True)
    with pytest.raises(errors.InvalidRecord, match="true or false"):
        tokenmap.TokenizerSettings(append_eod=1, eod_in_span=True, eod_token_id=1)
    with pytest.raises(errors.InvalidRecord, match="not a token id"):
        tokenmap.TokenizerSettings(append_eod=True, eod_in_span=True, eod_token_id=-1)
    assert tokenmap.tokenize_step("1", {"a": 1}).name == "tokenize"


def test_the_sanitizer_log_reader_refuses_what_it_cannot_read(world, tmp_path):
    def log(*rows: str) -> Path:
        path = tmp_path / "log.jsonl"
        path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
        return path

    two = log('{"shard": "a", "line": 1, "reason": "empty line"}',
              '{"shard": "b", "line": 1, "reason": "sanitizer skip"}')
    with pytest.raises(errors.InvalidRecord, match="those of several files"):
        tokenmap.read_sanitizer_log(two)
    assert tokenmap.read_sanitizer_log(two, shard="b") == (
        tokenmap.SanitizerSkip(1, v.DROP_SANITIZER_SKIP),)
    assert tokenmap.read_sanitizer_log(two, shard="c") == ()
    with pytest.raises(errors.InvalidRecord, match="row 1: line 1: a sanitizer's log gives the reason"):
        tokenmap.read_sanitizer_log(log('{"line": 1, "reason": "filter"}'))
    with pytest.raises(errors.InvalidRecord, match="row 1: a line number is a whole number"):
        tokenmap.read_sanitizer_log(log('{"line": "1", "reason": "empty line"}'))
    with pytest.raises(errors.InvalidRecord, match="row 2: line 4 is logged twice"):
        tokenmap.read_sanitizer_log(log('{"line": 4, "reason": "empty line"}',
                                        '{"line": 4, "reason": "empty line"}'))
    with pytest.raises(errors.InvalidRecord, match="row 1: not a JSON object"):
        tokenmap.read_sanitizer_log(log("not JSON"))

    # A log of another file names a line the input does not have.
    world.input("in", ["first", "second"])
    world.bare_shard("shard", "in", tm_world.documents_of(["first"]))
    with pytest.raises(errors.InvalidRecord, match="it is the log of another file"):
        recover(world, "shard", "in", sanitizer=tokenmap.SanitizerRun(
            (tokenmap.SanitizerSkip(3, v.DROP_EMPTY_LINE),)))


def test_the_shard_reader_reads_one_document_at_a_time_in_the_shards_own_type(tmp_path):
    documents = [[1, 2, 3], [], [65535], [7, 8]]
    tm_world.write_shard(tmp_path / "s16", documents, tm_world.UINT16)
    tm_world.write_shard(tmp_path / "s32", documents, tm_world.INT32)
    for name in ("s16", "s32"):
        with tokenmap.ShardTokens(tmp_path / name) as shard:
            assert (shard.count, shard.token_count) == (3, 6)
            assert [shard.span(k) for k in range(3)] == [(0, 3), (3, 4), (4, 6)]
            assert [shard.read(k).tolist() for k in (2, 0, 1)] == [[7, 8], [1, 2, 3], [65535]]
            assert shard.read(1).dtype.name == "int64"
            with pytest.raises(IndexError):
                shard.read(3)

    # A .bin file that does not have the size its index gives is refused.
    with open(tmp_path / "s32.bin", "ab") as f:
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(errors.UnsupportedFormat, match="its index gives 24"):
        tokenmap.ShardTokens(tmp_path / "s32")


def test_a_tokenize_function_must_return_token_ids(world):
    world.input("in", ["first"])
    world.bare_shard("shard", "in", tm_world.documents_of(["first"]))
    with pytest.raises(errors.InvalidRecord, match="token ids are whole numbers"):
        recover(world, "shard", "in", tokenize=lambda text: [0.5, 1.5], write=False)
    with pytest.raises(errors.InvalidRecord, match="one flat sequence of token ids"):
        recover(world, "shard", "in", tokenize=lambda text: [[1, 2], [3, 4]], write=False)
