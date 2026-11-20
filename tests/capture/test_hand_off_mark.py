"""A log whose handed texts were not shown and compared establishes no bound (the owner's
answer of 18 Nov 2026: "may contain until the original texts are shown and compared, then the
clean answer returns"), through the real capture wrappers, store, proof and command line.

The sentences the expectations rest on (docs/store.md "The write methods", docs/capture.md
section 3, docs/cli.md "compare-hand-off"):

- a stored harness log carries a mark, `hand_off_checked`: set by the store when the registration
  carried a text for every reference of the log's hand-off and each text matched its record;
  a log registered with none, with only some, or read back from its file is not marked;
- a stored log that is not marked establishes no bound: a negative past its edge reads
  "may contain" naming "upper bound not established" and the job, with its own reason sentence;
- `Store.verify_hand_off` (the command `compare-hand-off`) takes the texts for exactly the
  references of the log's hand-off, compares each as at registration (a record that holds no text
  takes only a text empty after the normaliser) and marks the log; a refusal changes nothing and
  never prints a text; an in-process log stays unbound whatever is compared;
- a value a caller puts in `HarnessLog.hand_off_checked` is not read; registering a job again
  never lifts the mark; a store of schema version 3 is refused read-only until migrated, and
  every log it holds is then not marked.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, cli, coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.store import Store

NORMALISER = "control-chars/1"
TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."
STEP = capture.step_ref("generate", "1", {"style": "plain"})
JOB = "paraphrase"


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def build(tmp_path: Path, rows_a: list[dict] | None = None) -> SimpleNamespace:
    w = SimpleNamespace(folder=tmp_path, db=tmp_path / "evidence.db")
    w.store = Store.create(w.db, normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", rows_a if rows_a is not None else [{"id": "A-001", "text": TEXT_A}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    w.ref_b = capture.occurrence(w.src_b, 1)
    return w


def paraphrase(ctx):
    (only, *_rest) = ctx.handed()
    return JobOutput(text=ctx.read(only).upper(), used=(only,))


def edge_for(w, handed, *, harness=v.HARNESS_CONFINED, tag="run", job=JOB):
    """The generation edge of a job that read only its first handed record. The log is written
    by the in-process harness and stamped with ``harness`` (a unit test has no sandbox here)."""
    result = run_job(paraphrase, job_name=job, handed=handed, log_path=w.folder / f"logs/{tag}.jsonl")
    (edge,) = capture.generation_edges(result, [1], STEP)
    log = dataclasses.replace(edge.harness_log, harness=harness)
    return dataclasses.replace(edge, harness_log=log)


def without_texts(edge):
    """The edge of a log read back from its file: the same log and edge, no text anywhere."""
    return dataclasses.replace(
        edge, handed_texts=(), harness_log=dataclasses.replace(edge.harness_log, handed_texts=()))


def register(w, edge, name="W"):
    path = w.folder / f"gen/{name}.jsonl"
    write_rows(path, [{"id": f"{name}-1", "text": "An output."}])
    capture.record_jsonl(w.store, path, name=f"gen/{name}", role=capture.ROLE_INTERMEDIATE, step=STEP,
                         inputs=[w.src_a, w.src_b], edges=[edge])


def answer(w, name="gen/W", source="B"):
    request = requests.withdraw(w.store, source)
    return coverage.answer(w.store, name, request).to_json()


def mark(w, job=JOB):
    return w.store.harness_log(job).hand_off_checked


def row_counts(db: Path) -> dict[str, int]:
    raw = sqlite3.connect(db)
    try:
        tables = [r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: raw.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    finally:
        raw.close()


def run_cli(w, *argv: str) -> tuple[int, str, str]:
    w.store.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv, "--store", str(w.db)], stdout=out, stderr=err)
    w.store = Store.open(w.db)
    return code, out.getvalue(), err.getvalue()


def texts_file(w, pairs, name="texts.json"):
    path = w.folder / name
    path.write_text(json.dumps({"texts": [{"record": r, "text": t} for r, t in pairs]}), encoding="utf-8")
    return str(path)


# --- registration marks a log the texts cover ---------------------------------------------------


def test_a_log_registered_with_the_text_of_every_handed_record_is_marked_and_gives_the_clean_answer(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)]))
        found = answer(w)
        assert found["answer"] == "does not contain", found
        assert mark(w) is True
    finally:
        w.store.close()


@pytest.mark.parametrize("how", ["no-text-anywhere", "log-only-no-edge-text", "texts-on-the-edge-only"])
def test_a_log_registered_with_the_texts_on_either_carrier_is_marked(tmp_path, how):
    w = build(tmp_path)
    try:
        edge = edge_for(w, [Handed(w.ref_a, TEXT_A)])
        if how == "log-only-no-edge-text":
            edge = dataclasses.replace(edge, handed_texts=())
        elif how == "texts-on-the-edge-only":
            edge = dataclasses.replace(edge, harness_log=dataclasses.replace(edge.harness_log, handed_texts=()))
        else:
            edge = without_texts(edge)
        register(w, edge)
        want = "may contain" if how == "no-text-anywhere" else "does not contain"
        assert answer(w)["answer"] == want
        assert mark(w) is (how != "no-text-anywhere")
    finally:
        w.store.close()


# --- a log whose texts were not shown establishes no bound ---------------------------------------


def test_a_log_registered_without_texts_gives_may_contain_naming_the_bound_and_the_job(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        found = answer(w)
        assert (found["answer"], found["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
        assert mark(w) is False
        assert JOB in found["failing"]["detail"] and "never compared" in found["failing"]["detail"]
        # "contains" needs no bound: a withdrawal of A is still answered from the edge
        assert answer(w, source="A")["answer"] == "contains"
    finally:
        w.store.close()


def test_a_log_whose_texts_cover_only_part_of_its_hand_off_is_not_marked(tmp_path):
    w = build(tmp_path)
    try:
        edge = edge_for(w, [Handed(w.ref_a, TEXT_A), Handed(w.ref_b, TEXT_B)])
        edge = dataclasses.replace(edge, handed_texts=((w.ref_a, TEXT_A),),
                                   harness_log=dataclasses.replace(edge.harness_log, handed_texts=((w.ref_a, TEXT_A),)))
        register(w, edge)
        assert answer(w)["answer"] == "may contain"
        assert mark(w) is False
    finally:
        w.store.close()


def test_a_value_handed_in_for_the_mark_is_not_read(tmp_path):
    w = build(tmp_path)
    try:
        edge = without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)]))
        edge = dataclasses.replace(edge, harness_log=dataclasses.replace(edge.harness_log, hand_off_checked=True))
        register(w, edge)
        assert answer(w)["answer"] == "may contain"
        assert mark(w) is False
    finally:
        w.store.close()


def test_registering_a_job_again_with_texts_does_not_lift_the_mark(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)], tag="again"), name="W2")
        assert answer(w, "gen/W2")["answer"] == "may contain"
        assert mark(w) is False
    finally:
        w.store.close()


def test_an_in_process_log_stays_unbound_whatever_is_compared(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)], harness=v.HARNESS_IN_PROCESS))
        assert mark(w) is True
        found = answer(w)
        assert (found["answer"], found["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
        assert "in-process" in found["failing"]["detail"]
        w.store.verify_hand_off(JOB, [(w.ref_a, TEXT_A)])
        assert answer(w)["answer"] == "may contain"
    finally:
        w.store.close()


# --- showing the texts later --------------------------------------------------------------------


def test_the_texts_shown_later_lift_the_mark_and_the_clean_answer_returns(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        assert answer(w)["answer"] == "may contain"
        w.store.verify_hand_off(JOB, [(w.ref_a, TEXT_A)])
        assert mark(w) is True
        assert answer(w)["answer"] == "does not contain"
        w.store.verify_hand_off(JOB, [(w.ref_a, TEXT_A)])  # again: compared again, still marked
        assert mark(w) is True
    finally:
        w.store.close()


@pytest.mark.parametrize("case", ["wrong-text", "missing-reference", "extra-reference", "no-such-job", "empty-list"])
def test_a_refused_comparison_changes_nothing_and_never_names_a_text(tmp_path, case):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        before = row_counts(w.db)
        job, texts = JOB, [(w.ref_a, TEXT_A)]
        if case == "wrong-text":
            texts = [(w.ref_a, TEXT_B)]
        elif case == "missing-reference":
            texts = [(w.ref_b, TEXT_B)]
        elif case == "extra-reference":
            texts = [(w.ref_a, TEXT_A), (w.ref_b, TEXT_B)]
        elif case == "no-such-job":
            job = "no-such-job"
        elif case == "empty-list":
            texts = []
        with pytest.raises(errors.InvalidRecord) as caught:
            w.store.verify_hand_off(job, texts)
        assert TEXT_A not in str(caught.value) and TEXT_B not in str(caught.value)
        assert job in str(caught.value)
        assert mark(w) is False and row_counts(w.db) == before
        assert answer(w)["answer"] == "may contain"
    finally:
        w.store.close()


def test_a_record_that_holds_no_text_takes_only_a_text_that_is_empty_after_the_normaliser(tmp_path):
    w = build(tmp_path, rows_a=[{"id": "A-001"}])  # a line with no text field: no fingerprint
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, "")])))
        assert mark(w) is False
        with pytest.raises(errors.InvalidRecord):
            w.store.verify_hand_off(JOB, [(w.ref_a, TEXT_A)])
        assert mark(w) is False
        w.store.verify_hand_off(JOB, [(w.ref_a, "\x07 \r\n")])  # empty after the normaliser
        assert mark(w) is True
    finally:
        w.store.close()


def test_a_readonly_store_cannot_be_shown_texts(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
    finally:
        w.store.close()
    with Store.open(w.db, readonly=True) as store:
        with pytest.raises(errors.WriteRefused):
            store.verify_hand_off(JOB, [(w.ref_a, TEXT_A)])


# --- the command line ------------------------------------------------------------------------------


def test_compare_hand_off_marks_the_log_and_prints_no_text(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        path = texts_file(w, [("src/A:L1", TEXT_A)])
        code, out, err = run_cli(w, "compare-hand-off", "--job", JOB, "--texts", path)
        assert (code, err) == (0, "")
        assert json.loads(out) == {"job": JOB, "references": 1, "hand_off_checked": True}
        assert TEXT_A not in out
        assert answer(w)["answer"] == "does not contain"
    finally:
        w.store.close()


def test_compare_hand_off_with_a_wrong_text_refuses_prints_nothing_on_standard_output_and_names_no_text(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        path = texts_file(w, [("src/A:L1", TEXT_B)])
        code, out, err = run_cli(w, "compare-hand-off", "--job", JOB, "--texts", path)
        assert (code, out) == (errors.EXIT_WRITE_REFUSED, "")
        assert JOB in err and w.ref_a.key in err
        assert TEXT_A not in err and TEXT_B not in err
        assert mark(w) is False
    finally:
        w.store.close()


# --- what show prints ---------------------------------------------------------------------------------


def test_show_says_whether_the_log_establishes_a_bound_and_prints_no_new_key(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)]))
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)], job="paraphrase-2", tag="two")), name="W3")
        code, checked, _ = run_cli(w, "show", "gen/W", "--record", "L1")
        assert code == 0
        code, unchecked, _ = run_cli(w, "show", "gen/W3", "--record", "L1")
        assert code == 0
        one, two = json.loads(checked)["edge"], json.loads(unchecked)["edge"]
        assert (one["bound_field"], one["bound_from_log"]) == (True, True)
        assert (two["bound_field"], two["bound_from_log"]) == (True, False)
        assert set(one) == set(two)
    finally:
        w.store.close()


# --- a store of schema version 3 --------------------------------------------------------------------------


def _step_back_to_version_3(db: Path) -> None:
    raw = sqlite3.connect(db)
    try:
        raw.execute("DROP INDEX trace_row_segment_document")  # step 5 taken back first
        raw.execute("ALTER TABLE trace_row DROP COLUMN segment_id")
        raw.execute("DROP TABLE shard_files_check")
        raw.execute("ALTER TABLE harness_log DROP COLUMN hand_off_checked")
        raw.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
        raw.commit()
    finally:
        raw.close()


def test_a_store_of_schema_version_3_is_refused_read_only_until_migrated_and_then_every_log_is_not_marked(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)]))
        assert answer(w)["answer"] == "does not contain"
    finally:
        w.store.close()
    _step_back_to_version_3(w.db)
    with pytest.raises(errors.Refused, match="must be migrated to 5 first"):
        Store.open(w.db, readonly=True)
    out, err = io.StringIO(), io.StringIO()
    assert cli.main(["migrate", "--store", str(w.db)], stdout=out, stderr=err) == 0
    said = json.loads(out.getvalue())
    assert (said["schema_version_found"], said["schema_version"], said["steps_run"]) == (3, 5, 2)
    assert said["migrated"]["from"] == 3 and "compare-hand-off" in said["migrated"]["note"]
    with Store.open(w.db, readonly=True) as store:
        assert store.harness_log(JOB).hand_off_checked is False
        found = coverage.answer(store, "gen/W", requests.withdraw(store, "B")).to_json()
        assert (found["answer"], found["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    with Store.open(w.db) as store:
        store.verify_hand_off(JOB, [(w.ref_a, TEXT_A)])
        assert coverage.answer(store, "gen/W", requests.withdraw(store, "B")).to_json()["answer"] == "does not contain"


def test_a_command_that_opens_a_version_3_store_for_writing_says_that_it_migrated_it(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)]))
    finally:
        w.store.close()
    _step_back_to_version_3(w.db)
    path = texts_file(w, [("src/A:L1", TEXT_A)])
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["compare-hand-off", "--job", JOB, "--texts", path, "--store", str(w.db)], stdout=out, stderr=err)
    assert code == 0, err.getvalue()
    said = json.loads(out.getvalue())
    assert said["hand_off_checked"] is True
    assert said["migrated"]["from"] == 3 and said["migrated"]["to"] == 5
    assert "may contain" in said["migrated"]["note"]
    # a command on a store that needs no migration says nothing of one
    out2 = io.StringIO()
    assert cli.main(["compare-hand-off", "--job", JOB, "--texts", path, "--store", str(w.db)],
                    stdout=out2, stderr=io.StringIO()) == 0
    assert "migrated" not in json.loads(out2.getvalue())


def test_a_refused_command_that_migrated_a_version_3_store_still_says_it_migrated_it(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)]))
    finally:
        w.store.close()
    _step_back_to_version_3(w.db)
    path = texts_file(w, [("src/A:L1", TEXT_B)])  # a wrong text: the command is refused
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["compare-hand-off", "--job", JOB, "--texts", path, "--store", str(w.db)], stdout=out, stderr=err)
    assert (code, out.getvalue()) == (errors.EXIT_WRITE_REFUSED, "")
    said = json.loads(err.getvalue())
    assert said["migrated"]["from"] == 3 and said["migrated"]["to"] == 5 and "may contain" in said["migrated"]["note"]
    assert TEXT_A not in err.getvalue() and TEXT_B not in err.getvalue()
    assert Store.open(w.db, readonly=True).schema_version == 5  # the step is committed: the store stays migrated


def test_compare_hand_off_names_a_badly_written_record_as_an_invalid_record_not_a_selector(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A)])))
        path = texts_file(w, [("src/A", TEXT_A), ("src/A:L1", TEXT_A)])  # the first is not ARTIFACT:L<n>
        code, out, err = run_cli(w, "compare-hand-off", "--job", JOB, "--texts", path)
        assert (code, out) == (errors.EXIT_WRITE_REFUSED, "")
        assert json.loads(err)["error"] == "invalid record" and "ARTIFACT:L<n>" in err
        assert mark(w) is False
    finally:
        w.store.close()


def test_a_log_with_an_empty_hand_off_is_never_marked_and_cannot_be_shown_texts(tmp_path):
    """A job handed nothing has no text to compare: no registration stores such a log as
    compared, and `verify_hand_off` refuses it (an empty hand-off: nothing to show)."""
    w = build(tmp_path)
    try:
        def nothing(ctx):
            return JobOutput(text="made from nothing", used=())
        result = run_job(nothing, job_name="nothing", handed=[], log_path=w.folder / "logs/nothing.jsonl")
        edges = capture.generation_edges(result, [1], STEP)
        edge = dataclasses.replace(edges[0], harness_log=dataclasses.replace(edges[0].harness_log, harness=v.HARNESS_CONFINED))
        refused = None
        try:
            register(w, edge, name="N")
        except errors.InvalidRecord as error:
            refused = error
        if refused is None:
            assert mark(w, "nothing") is False
            with pytest.raises(errors.InvalidRecord, match="is empty"):
                w.store.verify_hand_off("nothing", [])
        else:
            assert w.store.harness_log("nothing") is None  # nothing was stored
    finally:
        w.store.close()


def test_query_through_the_command_line_on_a_migrated_store_says_may_contain_until_the_texts_are_shown(tmp_path):
    w = build(tmp_path)
    try:
        register(w, edge_for(w, [Handed(w.ref_a, TEXT_A)]))
    finally:
        w.store.close()
    _step_back_to_version_3(w.db)

    def query() -> dict:
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["query", "withdraw", "B", "--artifact", "gen/W", "--store", str(w.db)], stdout=out, stderr=err)
        assert (code, err.getvalue()) == (0, ""), err.getvalue()
        return json.loads(out.getvalue())

    refused = io.StringIO()
    assert cli.main(["query", "withdraw", "B", "--artifact", "gen/W", "--store", str(w.db)],
                    stdout=io.StringIO(), stderr=refused) == errors.EXIT_REFUSED  # version 3: refused until migrated
    assert "must be migrated to 5 first" in refused.getvalue()
    assert cli.main(["migrate", "--store", str(w.db)], stdout=io.StringIO(), stderr=io.StringIO()) == 0
    before = query()
    assert (before["answer"], before["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    path = texts_file(w, [("src/A:L1", TEXT_A)])
    code, said, err = run_cli(w, "compare-hand-off", "--job", JOB, "--texts", path)
    assert code == 0, err
    assert query()["answer"] == "does not contain"


def test_compare_hand_off_with_only_some_of_a_logs_texts_leaves_the_mark_off(tmp_path):
    w = build(tmp_path)
    try:
        register(w, without_texts(edge_for(w, [Handed(w.ref_a, TEXT_A), Handed(w.ref_b, TEXT_B)])))
        assert mark(w) is False
        path = texts_file(w, [("src/A:L1", TEXT_A)])  # the text of src/B:L1 is missing
        code, out, err = run_cli(w, "compare-hand-off", "--job", JOB, "--texts", path)
        assert (code, out) == (errors.EXIT_WRITE_REFUSED, "")
        assert w.ref_b.key in err and TEXT_A not in err
        assert mark(w) is False
        assert answer(w, source="B")["answer"] == "may contain"
        both = texts_file(w, [("src/A:L1", TEXT_A), ("src/B:L1", TEXT_B)], name="both.json")
        code, out, err = run_cli(w, "compare-hand-off", "--job", JOB, "--texts", both)
        assert code == 0, err
        assert mark(w) is True
    finally:
        w.store.close()
