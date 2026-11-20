"""The fixture's texts (plan 5.1): the files hold what the spec and the plan say."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from waterwhite import normalise as tool_normalise

from replica.steps import expand_ids

CONTROL_CHARS = tool_normalise.get("control-chars/1")


def rows_of(repo, name, read_rows):
    return read_rows(repo / "fixture" / "sources" / f"{name}.jsonl")


def test_every_source_holds_the_ids_the_spec_lists(repo, fixture_plan, read_rows):
    for source in fixture_plan.sources:
        rows = read_rows(source.path)
        assert [r["id"] for r in rows] == list(source.records)
        assert all(set(r) == {"id", "text"} and isinstance(r["text"], str) for r in rows)
    counts = {s.key: len(s.records) for s in fixture_plan.sources}
    assert counts == {"A": 2, "B": 13, "C": 4, "Z": 90}


def test_the_shared_passage_is_the_same_bytes_in_a_and_b(repo, read_rows):
    a = rows_of(repo, "A", read_rows)[0]
    b = rows_of(repo, "B", read_rows)[0]
    assert (a["id"], b["id"]) == ("A-001", "B-001")
    assert a["text"].encode("utf-8") == b["text"].encode("utf-8")
    # X is one short sentence (fixture/sources/README.md, "X, P and T are short").
    assert len(a["text"].split()) == 14 and a["text"].count(".") == 1
    # The raw lines agree too, apart from the id.
    raw_a = (repo / "fixture" / "sources" / "A.jsonl").read_bytes().split(b"\n")[0]
    raw_b = (repo / "fixture" / "sources" / "B.jsonl").read_bytes().split(b"\n")[0]
    assert raw_a.replace(b"A-001", b"B-001") == raw_b


def test_no_other_text_repeats_after_normalisation(repo, read_rows):
    texts = {}
    for name in ("A", "B", "C", "Z", "generation"):
        for row in rows_of(repo, name, read_rows):
            texts.setdefault(CONTROL_CHARS(row["text"]), []).append(row["id"])
    repeated = sorted(ids for ids in texts.values() if len(ids) > 1)
    assert repeated == [["A-001", "B-001"]]


def test_every_text_passes_the_sanitizer_unchanged(repo, read_rows):
    # No source record is a skipped line any more: the fixture's one skipped line
    # is the spec's [sanitizer_skip], which belongs to no source. And no text
    # holds a control character or white space at an end, so the fixture does not
    # depend on which characters a normaliser counts as control characters.
    for name in ("A", "B", "C", "Z", "generation"):
        for row in rows_of(repo, name, read_rows):
            text = row["text"]
            assert text != "" and CONTROL_CHARS(text) == text, row["id"]
            assert not any(ord(ch) < 32 or 127 <= ord(ch) < 160 for ch in text), row["id"]
            assert "  " not in text, row["id"]


def test_z_046_is_an_ordinary_passage(repo, read_rows):
    row = {r["id"]: r for r in rows_of(repo, "Z", read_rows)}["Z-046"]
    assert 12 <= len(row["text"].split()) <= 70
    assert row["text"].isascii() and "\\u00" not in json.dumps(row["text"])


def test_the_one_skipped_line_is_the_specs_and_it_does_not_parse(spec_doc, repo, read_rows):
    skip = spec_doc["sanitizer_skip"]
    assert (skip["shard"], skip["line"]) == ("shard-2", 6)
    assert skip["layouts"] == ["partitioned", "round-robin"]
    try:
        json.loads(skip["text"])
    except ValueError:
        pass
    else:
        raise AssertionError("the spec's skipped line parses")
    # It is no line of any source file: it is never ingested.
    for name in ("A", "B", "C", "Z", "generation"):
        raw = (repo / "fixture" / "sources" / f"{name}.jsonl").read_text(encoding="utf-8")
        assert skip["text"] not in raw.split("\n")


def test_the_unrelated_passages_vary_in_length(repo, read_rows):
    words = {r["id"]: len(r["text"].split()) for r in rows_of(repo, "Z", read_rows)}
    assert len(words) == 90
    assert min(words.values()) >= 12 and max(words.values()) <= 70
    short = [i for i, n in words.items() if n <= 20]
    assert len(short) >= 8
    # Short passages sit in every shard of run E, where packed samples need them.
    for first, last in (("Z-041", "Z-052"), ("Z-053", "Z-064"), ("Z-065", "Z-090")):
        in_range = set(expand_ids([f"{first}..{last}"]))
        assert len([i for i in short if i in in_range]) >= 3


def test_the_generation_rows_are_the_ones_the_spec_describes(repo, read_rows):
    rows = read_rows(repo / "fixture" / "sources" / "generation.jsonl")
    assert [(r["id"], r["job"], r["handed"], r["used"]) for r in rows] == [
        ("P", "paraphrase", ["A-001"], ["A-001"]),
        ("T", "translation", ["P"], ["P"]),
        ("S", "summary", ["A-002", "C-001"], ["A-002", "C-001"]),
    ]
    assert all(set(r) == {"id", "job", "handed", "used", "text"} for r in rows)
    german = rows[1]["text"]
    assert any(ord(ch) in (0xE4, 0xF6, 0xFC, 0xC4) for ch in german)  # umlauts, in UTF-8
    # P and T are one short sentence each; S keeps its length.
    assert [len(r["text"].split()) for r in rows[:2]] == [12, 9]
    assert all(r["text"].count(".") == 1 and r["text"].endswith(".") for r in rows[:2])
    assert len(rows[2]["text"].split()) > 60


# The limits on the three short texts, in tokens with the end-of-document token
# (DECISIONS.md D-24; fixture/sources/README.md), and the counts the texts have.
TOKEN_LIMITS = {"X": 30, "P": 30, "T": 22}
TOKEN_COUNTS = {"X": 16, "P": 15, "T": 22}


@pytest.mark.megatron
def test_x_p_and_t_are_short_in_tokens(repo, spec_doc, read_rows):
    """Counts X, P and T with the fixture's tokenizer: GPT-2 BPE, loaded from the
    snapshot folder of the Megatron environment that holds the spec's vocabulary
    file, plus one for the end-of-document token. Needs the environment and
    imports transformers, so it carries the ``megatron`` marker: on a shared
    machine run it through the CPU lock. Without the environment it fails when a
    run selects it on purpose and is not skipped; a run that does not name the marker has
    it deselected by tests/conftest.py."""
    env = os.environ.get("WATERWHITE_MEGATRON_ENV")
    assert env, "WATERWHITE_MEGATRON_ENV is not set: the token counts cannot be checked"
    vocab = Path(env) / spec_doc["tokenizer"]["vocab_file"]
    assert vocab.is_file(), f"the spec's vocabulary file is not in the environment: {vocab.name}"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(vocab.parent), local_files_only=True)
    assert tokenizer.eos_token_id == spec_doc["tokenizer"]["eod_token_id"]
    generation = {r["id"]: r["text"] for r in read_rows(repo / "fixture" / "sources" / "generation.jsonl")}
    texts = {"X": rows_of(repo, "A", read_rows)[0]["text"], "P": generation["P"], "T": generation["T"]}
    counts = {name: len(tokenizer.encode(text)) + 1 for name, text in texts.items()}
    assert counts == TOKEN_COUNTS
    assert all(counts[name] <= TOKEN_LIMITS[name] for name in counts)
    # the spec records the same counts with the seeds they belong to
    sizes = spec_doc["sizes"]
    assert (sizes["tokens_of_X"], sizes["tokens_of_P"], sizes["tokens_of_T"]) == (16, 15, 22)
    # a sample of run E is 33 tokens: T leaves room for a row before it and one after it
    assert counts["T"] + 2 <= spec_doc["runs"]["E"]["sequence_length"] + 1


def test_the_files_are_plain_jsonl_without_long_dashes(repo):
    folder = repo / "fixture" / "sources"
    for path in sorted(folder.glob("*.jsonl")):
        data = path.read_bytes()
        assert data.endswith(b"\n") and b"\r" not in data
        text = data.decode("utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text
        for line in text.split("\n")[:-1]:
            json.loads(line)
