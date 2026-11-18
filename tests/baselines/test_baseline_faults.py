"""The fault applicator on a baseline's own records (plan 5.5 Check 10; table README,
section "The faults of Checks 4 and 5", and the ``fault`` text of each row of
``check10.toml``).

Each injected fault of Checks 4 and 5 is put to the subject's own record where the
subject has the corresponding record. This file checks, on records written by
hand, that each fault changes exactly what its row says and nothing else, that it
leaves the record it was given alone (it returns a changed copy), that a target the
record does not hold is an error and never a silent no-op, and that a fault for
which a subject has no record is refused.

What this file asks of ``baselines.faults``:

- ``apply_fault(fault_id, kind, record, **target)``: a changed copy of ``record``.
  ``kind`` is the subject whose own record it is: ``manifest``, ``trace``,
  ``dolma`` or ``originblame`` (a combination applies the half of the fault that
  belongs to a record of one of these). Every target the table names is the
  function's default; only the chunk file of the Dolma rows is an argument, named
  ``chunk_file``, because its name depends on the layout.
- ``FaultError``: raised for a fault id the applicator does not hold, for a fault
  that a kind of record has no counterpart for, for a fault that is not a change
  to a record (the writer faults), and for a target the record does not hold. Its
  message names the fault id.

The records are plain data, with the field names of the README:

- ``manifest``: a list of rows ``{"name", "files", "built_from"}``.
- ``trace``: the document ``show SEGMENT --trace`` prints (docs/cli.md):
  ``{"segment", "steps": [{"step", "consumed_samples", "samples": [{"sample_pos",
  "global_index", "rank", "sample_tokens", "covered_tokens", "rows": [{"shard",
  "shard_position", "ordinal", "token_start", "token_end"}]}]}], ...}``. Token
  ranges are half open. A step's ``consumed_samples`` is its own count of samples
  (the store checks it against the step's batch size and its sample rows). The
  faults name segment ``E/seg-1``.
- ``dolma``: the companion rows, a list of ``{"start", "end", "id", "src", "loc"}``
  (the four numbers are ints, ``src`` and ``id`` strings), in any order: the
  toolkit does not keep the order of its input, so no rule here goes by a row's
  place; "the last row of a file" is the row with the greatest ``loc`` of its
  ``src``.
- ``originblame``: the records of the store that tracked one shard, ``{"sections",
  "documents", "entries"}``, where ``entries`` is the token-index entries in the
  order they were tracked, each ``{"sources", "token_count", "tokenizer",
  "revoked"}``. Here the last entry is the last in that order.

The faults this file does not cover: ``w.kill-after-100`` and ``w.retry`` (a
process stopped by its id and run again, a matter of the tool's writer: here they
are only refused) and ``l.artifact-digest`` (one byte of one file changed on disk,
a change to a file and not to a record).

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import tomllib

import pytest

MODULE = "baselines.faults"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
    "subprocess",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")

# --- Records written by hand -------------------------------------------------------------

MANIFEST = [
    {"name": "src/A", "files": ["sources/A.jsonl"], "built_from": []},
    {"name": "src/B", "files": ["sources/B.jsonl"], "built_from": []},
    {"name": "gen/P", "files": ["gen/P.jsonl"], "built_from": ["src/A", "src/B"]},
    {"name": "E/shard-1", "files": ["pipeline/main/shards/shard-1.bin"],
     "built_from": ["src/A", "gen/P"]},
    {"name": "E/shard-2", "files": ["pipeline/main/shards/shard-2.bin"],
     "built_from": ["src/B"]},
    {"name": "E/mix-1", "files": ["pipeline/main/mixtures/mix-1.json"],
     "built_from": ["E/shard-1", "E/shard-2"]},
]


def trace_sample(pos: int, ranges: list[tuple[int, int]], *, ordinal: int = 0,
                 tokens: int = 33, covered: int = 33) -> dict:
    return {
        "sample_pos": pos, "global_index": 100 + pos, "rank": 0,
        "sample_tokens": tokens, "covered_tokens": covered,
        "rows": [{"shard": "E/shard-1", "shard_position": 0, "ordinal": ordinal + n,
                  "token_start": start, "token_end": end}
                 for n, (start, end) in enumerate(ranges)],
    }


def trace_step(step: int, last_ranges: list[tuple[int, int]]) -> dict:
    """A step of four samples; the last has the token ranges given."""
    return {
        "step": step, "consumed_samples": 4,
        "samples": [
            trace_sample(0, [(0, 33)], ordinal=0),
            trace_sample(1, [(0, 20), (20, 33)], ordinal=2),
            trace_sample(2, [(0, 33)], ordinal=5),
            trace_sample(3, last_ranges, ordinal=7),
        ],
    }


# Step 4's last sample holds four rows and none of them ends or starts at the cut (16).
HALF_RANGES = [(0, 5), (5, 12), (12, 25), (25, 33)]

TRACE = {
    "segment": "E/seg-1",
    "route": "cache files",
    "launches": [{"position": 0, "first_step": 1, "last_step": 6}],
    "steps": [
        trace_step(3, [(0, 33)]),
        trace_step(4, HALF_RANGES),
        trace_step(5, [(0, 10), (10, 33)]),
        trace_step(6, [(0, 33)]),
    ],
    "untraced_steps": [],
}

C1 = "pipeline/main/tokenizer-input/shard-1.jsonl"
C3 = "pipeline/main/tokenizer-input/shard-3.jsonl"


def row(end: int, src: str, loc: int) -> dict:
    return {"start": end - 40, "end": end, "id": f"id-{loc}", "src": src, "loc": loc}


# Eight rows of two chunk files, in an order that is neither file order nor line order.
DOLMA = [
    row(3333, C3, 3), row(2222, C1, 2), row(1234, C3, 1), row(4444, C1, 4),
    row(1111, C1, 1), row(4040, C3, 4), row(3030, C1, 3), row(2020, C3, 2),
]

OB_STORE = {
    "sections": [{"hash": "h1", "path": "sources/A.jsonl"}, {"hash": "h2", "path": "sources/B.jsonl"}],
    "documents": [{"line_hash": "l1", "sources": ["h1"]}, {"line_hash": "l2", "sources": ["h2"]}],
    "entries": [
        {"sources": ["h1"], "token_count": 2345, "tokenizer": "gpt2", "revoked": False},
        {"sources": ["h2"], "token_count": 345, "tokenizer": "gpt2", "revoked": False},
        {"sources": ["h1"], "token_count": 5120, "tokenizer": "gpt2", "revoked": True},
        {"sources": ["h2"], "token_count": 77, "tokenizer": "gpt2", "revoked": False},
    ],
}

RECORDS = {"manifest": MANIFEST, "trace": TRACE, "dolma": DOLMA, "originblame": OB_STORE}
TARGETS = {"dolma": {"chunk_file": C1}}


def put(need_attribute, fault_id: str, kind: str, record=None, **target):
    """The fault put on a record; the record it was given is checked to be unchanged."""
    apply_fault = need_attribute(MODULE, "apply_fault")
    given = copy.deepcopy(RECORDS[kind] if record is None else record)
    snapshot = copy.deepcopy(given)
    target = target or TARGETS.get(kind, {})
    changed = apply_fault(fault_id, kind, given, **target)
    assert given == snapshot, "the fault changed the record it was given"
    assert changed is not given
    return changed


def first_digit_changed(old: int, new: int) -> bool:
    """The first digit of ``old`` replaced by another digit that is not 0, the rest the
    same: the number keeps its length."""
    old_text, new_text = str(old), str(new)
    return (isinstance(new, int) and len(new_text) == len(old_text)
            and new_text[1:] == old_text[1:] and new_text[0] != old_text[0]
            and new_text[0] in "123456789")


# --- The manifest ------------------------------------------------------------------------


def test_l_edge_parent_replaces_the_entry_of_p_by_a_name_no_row_has(need_attribute):
    changed = put(need_attribute, "l.edge-parent", "manifest")
    names = {r["name"] for r in MANIFEST}
    replaced = changed[2]["built_from"]
    assert len(replaced) == 2 and replaced[1] == "src/B"
    assert replaced[0] != "src/A" and replaced[0] not in names
    expected = copy.deepcopy(MANIFEST)
    expected[2]["built_from"] = replaced
    assert changed == expected  # nothing else moved; the row of gen/P stays


def test_f_edge_deleted_removes_the_entry_of_p_and_keeps_the_row(need_attribute):
    changed = put(need_attribute, "f.edge-deleted", "manifest")
    expected = copy.deepcopy(MANIFEST)
    expected[2]["built_from"] = ["src/B"]
    assert changed == expected
    assert any(r["name"] == "gen/P" for r in changed)


def test_f_map_deleted_deletes_the_row_of_the_shard_and_leaves_the_names_of_it(
        need_attribute):
    changed = put(need_attribute, "f.map-deleted", "manifest")
    expected = [copy.deepcopy(r) for r in MANIFEST if r["name"] != "E/shard-1"]
    assert changed == expected
    # The mixture's row still names the shard that has no row now.
    assert "E/shard-1" in changed[-1]["built_from"]


# --- The trace ---------------------------------------------------------------------------


def test_f_trace_deleted_takes_step_3_out_of_steps_in_the_documented_form(need_attribute):
    """docs/cli.md ("show --trace"): `steps` holds one entry per traced step, so the
    step whose trace is deleted has no entry; no entry with an empty `samples` stands
    for it. The rest of the document is unchanged."""
    changed = put(need_attribute, "f.trace-deleted", "trace")
    expected = copy.deepcopy(TRACE)
    expected["steps"] = [s for s in expected["steps"] if s["step"] != 3]
    assert changed == expected
    assert [s["step"] for s in changed["steps"]] == [4, 5, 6]


def test_f_trace_half_cuts_the_last_sample_of_step_4_at_16_of_33(need_attribute):
    changed = put(need_attribute, "f.trace-half", "trace")
    expected = copy.deepcopy(TRACE)
    sample = expected["steps"][1]["samples"][3]
    # The first two rows end within the first 16 tokens and stay; the row that holds
    # the 17th token ends at the cut; the row after it is removed; the sample stays
    # counted and its covered tokens are 16 of its 33.
    sample["rows"] = [
        {"shard": "E/shard-1", "shard_position": 0, "ordinal": 7, "token_start": 0, "token_end": 5},
        {"shard": "E/shard-1", "shard_position": 0, "ordinal": 8, "token_start": 5, "token_end": 12},
        {"shard": "E/shard-1", "shard_position": 0, "ordinal": 9, "token_start": 12, "token_end": 16},
    ]
    sample["covered_tokens"] = 16
    assert changed == expected
    kept = changed["steps"][1]
    assert len(kept["samples"]) == 4 and kept["consumed_samples"] == 4
    assert kept["samples"][3]["sample_tokens"] == 33


def doc_row(ordinal: int, start: int, end: int) -> dict:
    return {"shard": "E/shard-1", "shard_position": 0, "ordinal": ordinal,
            "token_start": start, "token_end": end}


def trace_with_last_sample_rows(rows: list[dict]) -> dict:
    """The trace of this file whose step 4 ends in a sample of 33 tokens made of ``rows``
    (token ranges inside their documents, as the trace holds them)."""
    record = copy.deepcopy(TRACE)
    record["steps"][1]["samples"][3]["rows"] = copy.deepcopy(rows)
    return record


def trace_half_case(need_attribute, rows: list[dict], kept: list[dict]):
    """The fault put on a trace whose last sample of step 4 holds ``rows``; ``kept`` are
    the rows it must hold after. Nothing else of the document moves."""
    record = trace_with_last_sample_rows(rows)
    changed = put(need_attribute, "f.trace-half", "trace", record=record)
    expected = copy.deepcopy(record)
    sample = expected["steps"][1]["samples"][3]
    sample["rows"] = kept
    sample["covered_tokens"] = 16
    assert changed == expected
    assert changed["steps"][1]["consumed_samples"] == 4
    assert len(changed["steps"][1]["samples"]) == 4


def test_f_trace_half_counts_the_16_tokens_across_the_rows_of_the_sample(need_attribute):
    """Ranges are inside their documents: the rows of 21 and 12 tokens make 33, so the
    first row holds the 17th token and ends at the cut and the second row goes."""
    trace_half_case(need_attribute,
                    [doc_row(4, 0, 21), doc_row(5, 0, 12)],
                    [doc_row(4, 0, 16)])


def test_f_trace_half_a_row_ending_at_the_16th_token_stays_whole_and_the_next_goes(
        need_attribute):
    """The row that ends exactly at the sample's 16th token stays as it is; the row whose
    first token is the 17th would end empty and is removed with the later rows."""
    trace_half_case(need_attribute,
                    [doc_row(3, 4, 20), doc_row(4, 0, 17)],
                    [doc_row(3, 4, 20)])


def test_f_trace_half_cuts_a_row_that_starts_at_a_document_offset(need_attribute):
    """The first row covers document tokens 30 to 50 (20 tokens): its 16th token is
    document token 45, so it ends at 46 and the row after it goes."""
    trace_half_case(need_attribute,
                    [doc_row(9, 30, 50), doc_row(10, 0, 13)],
                    [doc_row(9, 30, 46)])


def test_f_trace_half_keeps_whole_rows_before_the_cut_at_document_offsets(need_attribute):
    """A row of 10 tokens at a document offset stays; the second row (23 tokens) holds the
    sample's 11th to 33rd tokens, so it ends 6 tokens after its start."""
    trace_half_case(need_attribute,
                    [doc_row(2, 7, 17), doc_row(3, 40, 63)],
                    [doc_row(2, 7, 17), doc_row(3, 40, 46)])


def test_f_trace_half_cuts_the_one_row_sample_of_the_fixture_spec_at_6_to_22(
        need_attribute, facts):
    """The half fault's own sample as the fixture spec logs it (positions.shared_ordinal,
    the first log line): E/seg-1, step 4, sample_pos 3, one row of document 1 at token 6
    to 39, which covers all 33 tokens of the sample. The 16 tokens are the sample's, so
    the row keeps 16 of them: 6 to 22."""
    spec_path = facts.repo / "fixture" / "spec" / "fixture_spec.toml"
    with open(spec_path, "rb") as handle:
        spec = tomllib.load(handle)
    logged = json.loads(spec["positions"]["shared_ordinal"]["log_lines"][0])
    assert (logged["segment"], logged["step"], logged["sample_pos"]) == ("E/seg-1", 4, 3)
    assert logged["token_end"] - logged["token_start"] == 33
    record = trace_with_last_sample_rows([
        {"shard": "E/shard-1", "shard_position": logged["shard_position"],
         "ordinal": logged["ordinal"], "token_start": logged["token_start"],
         "token_end": logged["token_end"]}])
    changed = put(need_attribute, "f.trace-half", "trace", record=record)
    expected = copy.deepcopy(record)
    sample = expected["steps"][1]["samples"][3]
    sample["rows"][0]["token_end"] = 22
    sample["covered_tokens"] = 16
    assert changed == expected
    assert changed["steps"][1]["samples"][3]["rows"][0]["token_start"] == 6
    assert changed["steps"][1]["samples"][3]["sample_tokens"] == 33


def test_l_trace_count_deletes_the_last_sample_of_step_5_and_sets_the_count_to_3(
        need_attribute):
    changed = put(need_attribute, "l.trace-count", "trace")
    expected = copy.deepcopy(TRACE)
    expected["steps"][2]["samples"] = expected["steps"][2]["samples"][:3]
    expected["steps"][2]["consumed_samples"] = 3
    assert changed == expected


def test_a_trace_fault_needs_the_segment_and_the_step_it_names(need_attribute):
    apply_fault = need_attribute(MODULE, "apply_fault")
    fault_error = need_attribute(MODULE, "FaultError")
    other_segment = copy.deepcopy(TRACE)
    other_segment["segment"] = "D/seg-1"
    no_step_3 = copy.deepcopy(TRACE)
    no_step_3["steps"] = [s for s in no_step_3["steps"] if s["step"] != 3]
    no_step_4 = copy.deepcopy(TRACE)
    no_step_4["steps"] = [s for s in no_step_4["steps"] if s["step"] != 4]
    short_step_5 = copy.deepcopy(TRACE)
    short_step_5["steps"][2]["samples"] = short_step_5["steps"][2]["samples"][:3]
    short_step_5["steps"][2]["consumed_samples"] = 3
    cases = [
        ("f.trace-deleted", other_segment), ("f.trace-deleted", no_step_3),
        ("f.trace-half", other_segment), ("f.trace-half", no_step_4),
        ("l.trace-count", other_segment), ("l.trace-count", short_step_5),
    ]
    for fault_id, document in cases:
        with pytest.raises(fault_error, match=re.escape(fault_id)):
            apply_fault(fault_id, "trace", copy.deepcopy(document))


# --- The Dolma toolkit's companion rows --------------------------------------------------


def test_f_map_deleted_deletes_every_companion_row_of_the_chunk_file(need_attribute):
    changed = put(need_attribute, "f.map-deleted", "dolma", chunk_file=C1)
    assert changed == [r for r in DOLMA if r["src"] != C1]
    assert len(changed) == 4


def test_f_digest_changes_the_first_digit_of_the_end_of_the_row_at_loc_1(need_attribute):
    changed = put(need_attribute, "f.digest", "dolma", chunk_file=C3)
    # The row at loc 1 of the chunk file is the third in this file, and the other chunk
    # file has a row at loc 1 too.
    assert len(changed) == len(DOLMA)
    differing = [n for n, (a, b) in enumerate(zip(DOLMA, changed)) if a != b]
    assert differing == [2]
    old, new = DOLMA[2], changed[2]
    assert {k for k in old if old[k] != new[k]} == {"end"}
    assert first_digit_changed(old["end"], new["end"])


def test_l_count_499_deletes_the_row_of_the_last_document_of_the_chunk_file(need_attribute):
    changed = put(need_attribute, "l.count-499", "dolma", chunk_file=C1)
    # The row at the greatest loc of the chunk file goes, wherever the file holds it.
    assert changed == [r for r in DOLMA if not (r["src"] == C1 and r["loc"] == 4)]
    assert len(changed) == len(DOLMA) - 1
    assert DOLMA[-1]["src"] == C3  # the last row of the file is not the one that goes


def test_a_dolma_fault_on_a_chunk_file_with_no_rows_is_an_error(need_attribute):
    apply_fault = need_attribute(MODULE, "apply_fault")
    fault_error = need_attribute(MODULE, "FaultError")
    for fault_id in ("f.map-deleted", "f.digest", "l.count-499"):
        with pytest.raises(fault_error, match=re.escape(fault_id)):
            apply_fault(fault_id, "dolma", copy.deepcopy(DOLMA),
                        chunk_file="pipeline/main/tokenizer-input/shard-9.jsonl")


def test_f_digest_needs_a_row_at_loc_1(need_attribute):
    apply_fault = need_attribute(MODULE, "apply_fault")
    fault_error = need_attribute(MODULE, "FaultError")
    no_first = [r for r in DOLMA if not (r["src"] == C3 and r["loc"] == 1)]
    with pytest.raises(fault_error, match=re.escape("f.digest")):
        apply_fault("f.digest", "dolma", no_first, chunk_file=C3)


# --- OriginBlame's records ---------------------------------------------------------------


def test_f_map_deleted_deletes_the_token_index_entries_and_keeps_the_sections(
        need_attribute):
    changed = put(need_attribute, "f.map-deleted", "originblame")
    expected = copy.deepcopy(OB_STORE)
    expected["entries"] = []
    assert changed == expected
    assert changed["sections"] == OB_STORE["sections"]


def test_f_digest_changes_the_first_digit_of_token_count_of_the_first_entry(need_attribute):
    changed = put(need_attribute, "f.digest", "originblame")
    old, new = OB_STORE["entries"][0], changed["entries"][0]
    assert {k for k in old if old[k] != new[k]} == {"token_count"}
    assert first_digit_changed(old["token_count"], new["token_count"])
    expected = copy.deepcopy(OB_STORE)
    expected["entries"][0] = new
    assert changed == expected


def test_l_count_499_deletes_the_last_entry_in_the_order_they_were_tracked(need_attribute):
    changed = put(need_attribute, "l.count-499", "originblame")
    expected = copy.deepcopy(OB_STORE)
    expected["entries"] = expected["entries"][:-1]
    assert changed == expected


def test_an_originblame_fault_on_a_store_with_no_entries_is_an_error(need_attribute):
    apply_fault = need_attribute(MODULE, "apply_fault")
    fault_error = need_attribute(MODULE, "FaultError")
    empty = copy.deepcopy(OB_STORE)
    empty["entries"] = []
    for fault_id in ("f.map-deleted", "f.digest", "l.count-499"):
        with pytest.raises(fault_error, match=re.escape(fault_id)):
            apply_fault(fault_id, "originblame", copy.deepcopy(empty))


# --- A target the record does not hold --------------------------------------------------


def test_a_manifest_fault_on_an_artifact_the_manifest_does_not_hold_is_an_error(
        need_attribute):
    apply_fault = need_attribute(MODULE, "apply_fault")
    fault_error = need_attribute(MODULE, "FaultError")
    no_p = [r for r in MANIFEST if r["name"] != "gen/P"]
    no_entry = copy.deepcopy(MANIFEST)
    no_entry[2]["built_from"] = ["src/B"]
    no_shard = [r for r in MANIFEST if r["name"] != "E/shard-1"]
    cases = [
        ("l.edge-parent", no_p), ("l.edge-parent", no_entry),
        ("f.edge-deleted", no_p), ("f.edge-deleted", no_entry),
        ("f.map-deleted", no_shard),
    ]
    for fault_id, manifest in cases:
        with pytest.raises(fault_error, match=re.escape(fault_id)):
            apply_fault(fault_id, "manifest", copy.deepcopy(manifest))


# --- A fault the subject has no record for, and what is no fault of a record ------------

NO_COUNTERPART = (
    [("f.label", kind) for kind in RECORDS]
    + [("w.declare-499", kind) for kind in RECORDS]
    + [("f.edge-file-level", kind) for kind in RECORDS]
    + [("g.bound-field", "manifest")]
    + [("f.digest", "manifest"), ("l.count-499", "manifest"),
       ("f.digest", "trace"), ("l.count-499", "trace"),
       ("f.edge-deleted", "dolma"), ("f.edge-deleted", "originblame"),
       ("l.edge-parent", "dolma"), ("l.edge-parent", "originblame"),
       ("f.map-deleted", "trace"),
       ("f.trace-deleted", "manifest"), ("f.trace-deleted", "dolma"),
       ("f.trace-half", "originblame"), ("l.trace-count", "manifest")]
    # The writer faults are made by stopping the tool's own writer, not on a record.
    + [("w.kill-after-100", kind) for kind in ("dolma", "originblame")]
    + [("w.retry", kind) for kind in ("dolma", "originblame")]
    + [("x.no-such-fault", "manifest")]
)


@pytest.mark.parametrize("fault_id, kind", NO_COUNTERPART,
                         ids=[f"{f}-on-{k}" for f, k in NO_COUNTERPART])
def test_a_fault_with_no_record_to_put_it_to_is_refused(need_attribute, fault_id, kind):
    apply_fault = need_attribute(MODULE, "apply_fault")
    fault_error = need_attribute(MODULE, "FaultError")
    target = TARGETS.get(kind, {})
    with pytest.raises(fault_error, match=re.escape(fault_id)):
        apply_fault(fault_id, kind, copy.deepcopy(RECORDS[kind]), **target)


# --- What the module is -------------------------------------------------------------------


def test_the_module_imports_none_of_the_forbidden_modules_and_holds_no_path(facts):
    path = facts.repo / "baselines" / "faults.py"
    assert path.exists(), "baselines/faults.py does not exist"
    source = path.read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, sorted(roots & FORBIDDEN_IMPORT_ROOTS)
    for text in FORBIDDEN_TEXT:
        assert text not in source, f"baselines/faults.py holds {text!r}"
