"""The tie show of the Dolma toolkit's rows to the documents of a shard (plan 5.5
Check 10; table README, section "The Dolma tie to an ordinal").

The tie is shown for one chunk file when, for every document of its shard, the
shard's tokens at ordinal n equal the tokenization of line n + 1 of the chunk file
(ordinal counted from 0, line counted from 1). Where no result is recorded the
reading is "not shown". The chunk file of a shard is its tokenizer input,
``pipeline/<layout>/tokenizer-input/<shard>.jsonl``, one ``{"id", "text"}`` row a
line; the shard is the pair ``<prefix>.bin`` and ``<prefix>.idx`` Megatron's
preprocessing wrote.

Every input here is written by hand: a chunk file of a few lines, a shard written
in the layout of the index format (version 1, token type code 8, two bytes a
token, one sequence per document that gives tokens, a document boundary with no
sequence for one that gives none), and a tokenizer that is a table of words. No
tokenizer is started, no Megatron, no Dolma.

What this file asks of ``baselines.tie`` (the module does not run anything itself:
the tokenizer is handed to it as a function from one text to its token ids, the
way ``replica.megatron_tokenizer.LineTokenizer`` is):

- ``show_chunk_file(chunk_file, tokenizer_input, shard_prefix, tokenizer)``: the
  result for one chunk file, a dict with ``chunk_file`` (the name given, a
  relative path, recorded as given), ``shown`` (a bool), ``documents`` (how many
  documents the shard holds), ``lines`` (how many lines the chunk file holds) and
  ``first_mismatch`` (the ordinal of the first document that differs from its line,
  or None). Each line's ``text`` is tokenized (not the raw line); the shard holds
  the end-of-document token, id 50256, after the tokens of every document that has
  any, and the tokenizer's answer does not hold it; a document that gives no tokens
  holds nothing. A shard that holds another number of documents than the file has
  lines is not shown. Every document is compared, not a sample of them.
- ``show_shard(files, tokenizer)``: the same for a ``replica.shard_files.ShardFiles``
  (its tokenizer input and its prefix); ``chunk_file`` is the files'
  ``tokenizer_input_relative``.
- ``TieError``: raised when a file the show needs cannot be read (it is no result,
  and so no reading of "shown").
- ``write_result(path, results, run)``: writes the record of a run to ``path``: a
  JSON file ``{"run": <run>, "chunk_files": {<chunk_file>: {"shown", "documents",
  "lines", "first_mismatch"}}}``, one entry per result. ``path`` is an argument. A run or
  a chunk file name that holds a local path, a session name or the host name of the
  machine is refused with a ``ValueError`` and no file is made, as the writers of
  ``baselines.record`` refuse it.
- ``read_result(path)``: the record read back; a ``path`` that does not exist
  gives a record that holds no result (it does not raise).
- ``is_shown(record, chunk_file)``: True only for a chunk file the record holds as
  shown; a chunk file the record does not hold is not shown.

These are the one writer and the one reader of the result. ``baselines.record`` holds no
tie writer or reader of its own; ``baselines.record.tie_shown_for(path, chunk_file)`` reads
the file at ``path`` through ``read_result`` and ``is_shown`` (False for a file that does not
exist).

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path

import pytest

MODULE = "baselines.tie"
EOD = 50256

# The words of the toy tokenizer, and the one id it gives a word it does not know.
VOCAB = {"alpha": 11, "beta": 12, "gamma": 13, "delta": 14, "epsilon": 15, "zeta": 16}
UNKNOWN = 0

# A chunk file of six lines. The third holds a newline inside its text, so a reader
# that tokenizes the raw line of the file, not the text of its row, gives other tokens.
TEXTS = ["alpha beta", "gamma", "delta\nepsilon zeta", "beta alpha", "zeta", "alpha gamma delta"]

# What a tokenizer gives each of them, by hand (no end-of-document token).
TOKENS = [[11, 12], [13], [14, 15, 16], [12, 11], [16], [11, 13, 14]]

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
    "subprocess",
    "dolma",
    "originblame",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")

# What a written result may not hold, in pieces so that this file holds none of it: a path
# of a machine, a folder of a project, a session name.
PRIVATE_PATH = "/" + "ho" + "me/" + "someone/work/build"
PRIVATE_FOLDER = "Proj" + "ects/" + "thing"
PRIVATE_SESSION = "C" + "&" + "C"


def toy_tokenizer(text: str) -> list[int]:
    return [VOCAB.get(word, UNKNOWN) for word in text.split()]


def write_chunk_file(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"id": f"id-{n}", "text": text}) for n, text in enumerate(texts)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def with_eod(tokens: list[int]) -> list[int]:
    """What a shard holds for a document: its tokens and the end-of-document token, or
    nothing for a document that gives no tokens."""
    return tokens + [EOD] if tokens else []


def write_shard(prefix: Path, documents: list[list[int]]) -> None:
    """A shard in the layout the index format has: one sequence per document that
    holds tokens, a document boundary after each document."""
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for document in documents:
        if document:
            pointers.append(2 * len(tokens))
            lengths.append(len(document))
            tokens.extend(document)
        boundaries.append(len(lengths))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    Path(f"{prefix}.idx").write_bytes(
        b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 8, len(lengths), len(boundaries))
        + struct.pack(f"<{len(lengths)}i", *lengths)
        + struct.pack(f"<{len(pointers)}q", *pointers)
        + struct.pack(f"<{len(boundaries)}q", *boundaries))


def world(tmp_path: Path, texts: list[str], documents: list[list[int]], name: str = "a"):
    """A chunk file and a shard written by hand: (chunk file path, shard prefix)."""
    chunk = tmp_path / name / "tokenizer-input" / "shard.jsonl"
    prefix = tmp_path / name / "shards" / "shard"
    write_chunk_file(chunk, texts)
    write_shard(prefix, documents)
    return chunk, prefix


def matching_documents() -> list[list[int]]:
    return [with_eod(tokens) for tokens in TOKENS]


def show(need_attribute, tmp_path, texts, documents, name="a"):
    show_chunk_file = need_attribute(MODULE, "show_chunk_file")
    chunk, prefix = world(tmp_path, texts, documents, name)
    return show_chunk_file(f"pipeline/main/tokenizer-input/{name}.jsonl", chunk, prefix,
                           toy_tokenizer)


# --- The helpers hold what they say ------------------------------------------------------


def test_the_hand_written_shard_reads_back_as_written(tmp_path):
    from replica.shard_index import read_shard_index

    documents = [with_eod(t) for t in TOKENS[:2]] + [[]] + [with_eod(TOKENS[2])]
    write_shard(tmp_path / "s", documents)
    index = read_shard_index(tmp_path / "s.idx")
    assert index.documents == 4
    assert index.document_tokens == (3, 2, 0, 4)
    assert (tmp_path / "s.bin").stat().st_size == 2 * sum(map(len, documents))


def test_the_hand_written_tokens_are_the_tokenizers_answer():
    # The expected tokens of this file are the table's words by hand, and the toy
    # tokenizer agrees with them: a mistake in either shows here, not as a verdict on
    # the code under test.
    assert [toy_tokenizer(text) for text in TEXTS] == TOKENS


# --- Shown ---------------------------------------------------------------------------------


def test_a_shard_whose_every_document_is_its_line_is_shown(need_attribute, tmp_path):
    # Line 3 of TEXTS holds a newline, which the file writes as the two characters
    # backslash and n: a reader that tokenizes the raw line, not the text of its row,
    # reads it as one unknown word and does not show the tie.
    result = show(need_attribute, tmp_path, TEXTS, matching_documents())
    assert result["shown"] is True
    assert result["documents"] == 6
    assert result["lines"] == 6
    assert result["first_mismatch"] is None


def test_the_result_names_the_chunk_file_as_it_was_given(need_attribute, tmp_path):
    result = show(need_attribute, tmp_path, TEXTS, matching_documents(), name="b")
    assert result["chunk_file"] == "pipeline/main/tokenizer-input/b.jsonl"
    # A relative name: nothing of this machine reaches the result.
    assert str(tmp_path) not in json.dumps(result)


def test_a_line_that_gives_no_tokens_is_a_document_that_holds_nothing(need_attribute, tmp_path):
    texts = ["alpha", "   ", "beta"]
    documents = [with_eod([11]), [], with_eod([12])]
    result = show(need_attribute, tmp_path, texts, documents)
    assert result["shown"] is True
    assert result["documents"] == 3
    assert result["lines"] == 3


# --- Not shown -----------------------------------------------------------------------------


def test_two_lines_of_equal_length_swapped_in_the_shard_are_not_shown(need_attribute, tmp_path):
    # Lines 1 and 4 both give two tokens: a check of the counts alone sees nothing.
    documents = matching_documents()
    documents[0], documents[3] = documents[3], documents[0]
    result = show(need_attribute, tmp_path, TEXTS, documents)
    assert result["shown"] is False
    assert result["first_mismatch"] == 0
    assert result["documents"] == result["lines"] == 6


def test_one_changed_document_among_many_makes_the_file_not_shown(need_attribute, tmp_path):
    documents = matching_documents()
    assert documents[4] == [16, EOD]
    documents[4] = [13, EOD]  # the same length as the line's
    result = show(need_attribute, tmp_path, TEXTS, documents)
    assert result["shown"] is False
    assert result["first_mismatch"] == 4


def test_the_last_document_is_compared_too(need_attribute, tmp_path):
    documents = matching_documents()
    documents[5] = [11, 13, 15, EOD]
    result = show(need_attribute, tmp_path, TEXTS, documents)
    assert result["shown"] is False
    assert result["first_mismatch"] == 5


def test_a_document_that_ends_in_another_token_than_the_end_of_document_is_not_shown(
        need_attribute, tmp_path):
    documents = matching_documents()
    documents[1] = [13, 99]
    result = show(need_attribute, tmp_path, TEXTS, documents)
    assert result["shown"] is False
    assert result["first_mismatch"] == 1


def test_a_shard_with_fewer_documents_than_the_file_has_lines_is_not_shown(
        need_attribute, tmp_path):
    result = show(need_attribute, tmp_path, TEXTS, matching_documents()[:5])
    assert result["shown"] is False
    assert result["documents"] == 5
    assert result["lines"] == 6


def test_a_shard_with_more_documents_than_the_file_has_lines_is_not_shown(
        need_attribute, tmp_path):
    result = show(need_attribute, tmp_path, TEXTS[:5], matching_documents())
    assert result["shown"] is False
    assert result["documents"] == 6
    assert result["lines"] == 5


def test_a_document_one_line_late_is_not_shown(need_attribute, tmp_path):
    # Document n is line n + 2: the whole shard is shifted by one line.
    documents = matching_documents()[1:] + [with_eod(TOKENS[0])]
    result = show(need_attribute, tmp_path, TEXTS, documents)
    assert result["shown"] is False
    assert result["first_mismatch"] == 0


# --- The shard files of a build ------------------------------------------------------------


def test_the_show_of_a_shards_files_names_its_chunk_file_relative_to_the_build(
        need_attribute, tmp_path):
    from replica.shard_files import ShardFiles

    show_shard = need_attribute(MODULE, "show_shard")
    files = ShardFiles(out=tmp_path, layout="main", shard="shard-1", store_name="E/shard-1",
                       listed=())
    write_chunk_file(files.tokenizer_input, TEXTS)
    write_shard(files.prefix, matching_documents())
    result = show_shard(files, toy_tokenizer)
    assert result["chunk_file"] == "pipeline/main/tokenizer-input/shard-1.jsonl"
    assert result["shown"] is True
    assert str(tmp_path) not in json.dumps(result)


def test_a_shard_whose_files_are_missing_is_no_result(need_attribute, tmp_path):
    show_chunk_file = need_attribute(MODULE, "show_chunk_file")
    tie_error = need_attribute(MODULE, "TieError")
    chunk = tmp_path / "tokenizer-input" / "shard.jsonl"
    write_chunk_file(chunk, TEXTS)
    with pytest.raises(tie_error):
        show_chunk_file("pipeline/main/tokenizer-input/shard.jsonl", chunk,
                        tmp_path / "shards" / "shard", toy_tokenizer)


# --- The record of a run -------------------------------------------------------------------


def test_each_chunk_file_has_its_own_result_in_the_record(need_attribute, tmp_path):
    write_result = need_attribute(MODULE, "write_result")
    read_result = need_attribute(MODULE, "read_result")
    is_shown = need_attribute(MODULE, "is_shown")
    good = show(need_attribute, tmp_path, TEXTS, matching_documents(), name="one")
    swapped = matching_documents()
    swapped[0], swapped[3] = swapped[3], swapped[0]
    bad = show(need_attribute, tmp_path, TEXTS, swapped, name="two")
    run = {"command": "tie show", "commit": "abc1234"}
    out = tmp_path / "out" / "tie.json"
    out.parent.mkdir()
    write_result(out, [good, bad], run)

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["run"] == run
    assert set(document["chunk_files"]) == {good["chunk_file"], bad["chunk_file"]}
    one = document["chunk_files"][good["chunk_file"]]
    two = document["chunk_files"][bad["chunk_file"]]
    assert (one["shown"], one["documents"], one["lines"], one["first_mismatch"]) == (
        True, 6, 6, None)
    assert (two["shown"], two["documents"], two["lines"], two["first_mismatch"]) == (
        False, 6, 6, 0)
    # Nothing of this machine reaches the file.
    assert str(tmp_path) not in out.read_text(encoding="utf-8")

    record = read_result(out)
    assert is_shown(record, good["chunk_file"]) is True
    assert is_shown(record, bad["chunk_file"]) is False


def test_a_chunk_file_with_no_recorded_result_reads_not_shown(need_attribute, tmp_path):
    write_result = need_attribute(MODULE, "write_result")
    read_result = need_attribute(MODULE, "read_result")
    is_shown = need_attribute(MODULE, "is_shown")
    good = show(need_attribute, tmp_path, TEXTS, matching_documents(), name="one")
    out = tmp_path / "tie.json"
    write_result(out, [good], {"command": "tie show"})
    record = read_result(out)
    assert is_shown(record, good["chunk_file"]) is True
    assert is_shown(record, "pipeline/main/tokenizer-input/other.jsonl") is False


def test_no_record_at_all_reads_not_shown(need_attribute, tmp_path):
    read_result = need_attribute(MODULE, "read_result")
    is_shown = need_attribute(MODULE, "is_shown")
    record = read_result(tmp_path / "no-such-file.json")
    assert is_shown(record, "pipeline/main/tokenizer-input/shard-1.jsonl") is False


# --- One writer and one reader of the result ------------------------------------------------


def test_the_result_has_one_writer_and_one_reader_in_the_tie_module_and_record_calls_them(
        need_attribute, need_module, tmp_path, facts):
    # The result of the show is written by ``baselines.tie.write_result`` and read by
    # ``baselines.tie.read_result`` and nowhere else. ``baselines.record`` holds no second
    # pair (the earlier write_tie, read_tie and tie_shown are gone) and reads the result
    # through the one: ``tie_shown_for(path, chunk_file)`` is True only for a chunk file the
    # file at ``path`` holds as shown, and False for a chunk file it does not hold and for a
    # file that does not exist.
    tie_shown_for = need_attribute("baselines.record", "tie_shown_for")
    write_result = need_attribute(MODULE, "write_result")
    record = need_module("baselines.record")
    for name in ("write_tie", "read_tie", "tie_shown"):
        assert not hasattr(record, name), f"baselines.record still has {name}"
    good = show(need_attribute, tmp_path, TEXTS, matching_documents(), name="one")
    swapped = matching_documents()
    swapped[0], swapped[3] = swapped[3], swapped[0]
    bad = show(need_attribute, tmp_path, TEXTS, swapped, name="two")
    out = tmp_path / "tie.json"
    write_result(out, [good, bad], {"command": "tie show"})
    assert tie_shown_for(out, good["chunk_file"]) is True
    assert tie_shown_for(out, bad["chunk_file"]) is False
    assert tie_shown_for(out, "pipeline/main/tokenizer-input/other.jsonl") is False
    assert tie_shown_for(tmp_path / "no-such-file.json", good["chunk_file"]) is False
    # The module reads the result through baselines.tie: it imports it.
    source = (facts.repo / "baselines" / "record.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
            if node.module == "baselines":
                imported.update(f"baselines.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "baselines.tie" in imported, sorted(imported)


@pytest.mark.parametrize("private", [PRIVATE_PATH, PRIVATE_FOLDER, PRIVATE_SESSION])
def test_a_result_that_holds_a_local_path_or_a_session_name_is_not_written(
        need_attribute, tmp_path, private):
    # The one writer of the result keeps out of the file what the writers of the record
    # keep out of theirs: it is refused with a ValueError, before any file is made, whether
    # the private text is in the run's own words (also deep inside them) or in a chunk
    # file's name.
    write_result = need_attribute(MODULE, "write_result")
    good = show(need_attribute, tmp_path, TEXTS, matching_documents(), name="one")
    runs = ({"command": f"tie show {private}", "commit": "abc1234"},
            {"command": "tie show", "where": [{"x": f"ran in {private}"}]})
    for number, run in enumerate(runs):
        out = tmp_path / f"run-{number}.json"
        with pytest.raises(ValueError):
            write_result(out, [good], run)
        assert not out.exists()
    named = dict(good, chunk_file=f"pipeline/{private}/tokenizer-input/one.jsonl")
    out = tmp_path / "named.json"
    with pytest.raises(ValueError):
        write_result(out, [named], {"command": "tie show"})
    assert not out.exists()
    # The same result without the private text is written.
    write_result(out, [good], {"command": "tie show"})
    assert out.exists()


def test_a_result_that_holds_the_host_name_of_the_machine_is_not_written(
        need_attribute, tmp_path, monkeypatch):
    write_result = need_attribute(MODULE, "write_result")
    good = show(need_attribute, tmp_path, TEXTS, matching_documents(), name="one")
    # A made-up host name: the writer reads the name of the machine it runs on.
    host = "madeup" + "-box-7"
    monkeypatch.setattr("socket.gethostname", lambda: host)
    out = tmp_path / "host.json"
    with pytest.raises(ValueError):
        write_result(out, [good], {"command": "tie show", "where": f"ran on {host}"})
    assert not out.exists()
    write_result(out, [good], {"command": "tie show"})
    assert out.exists()


# --- What the module is -------------------------------------------------------------------


def test_the_module_starts_nothing_and_holds_no_path_of_this_machine(facts):
    path = facts.repo / "baselines" / "tie.py"
    assert path.exists(), "baselines/tie.py does not exist"
    source = path.read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, sorted(roots & FORBIDDEN_IMPORT_ROOTS)
    for text in FORBIDDEN_TEXT:
        assert text not in source, f"baselines/tie.py holds {text!r}"
