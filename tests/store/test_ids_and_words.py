"""Developer tests of the shared definitions the store and capture both rest on:
the one definition of a line, the streaming content hasher, the ids of the two
artifact kinds that have no file, and the words the schema spells in its CHECK
constraints. Expected digests are computed here with hashlib, not with the tool.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from waterwhite import errors, ids
from waterwhite import vocabulary as v
from waterwhite.store import _migrate, views

# --- one definition of a line ---------------------------------------------------------

LINE_CASES = [
    (b"", []),
    (b"\n", [b""]),
    (b"\n\n", [b"", b""]),
    (b"a", [b"a"]),
    (b"a\n", [b"a"]),
    (b"a\nb", [b"a", b"b"]),
    (b"a\n\nb\n", [b"a", b"", b"b"]),
    (b"a\rb\n", [b"a\rb"]),
    (b"a\r\nb\r\n", [b"a\r", b"b\r"]),
    (b"\r", [b"\r"]),
    (b" \n", [b" "]),
]


@pytest.mark.parametrize(("data", "lines"), LINE_CASES)
def test_a_line_ends_at_a_line_feed_and_a_last_piece_counts_when_not_empty(data, lines):
    assert list(ids.split_lines([data])) == lines
    # The pieces the bytes arrive in never change the lines.
    for size in (1, 2, 3):
        pieces = [data[i:i + size] for i in range(0, len(data), size)]
        assert list(ids.split_lines(pieces)) == lines
        assert list(ids.split_lines([b"", *pieces, b""])) == lines


def test_a_file_is_digested_and_split_in_one_read(tmp_path):
    data = b'{"text": "one"}\n\n{"text": "three"}'
    path = tmp_path / "f.jsonl"
    path.write_bytes(data)
    hasher = ids.ContentHasher()
    lines = list(ids.read_lines(path, hasher))
    assert lines == [b'{"text": "one"}', b"", b'{"text": "three"}']
    assert hasher.hexdigest() == hashlib.sha256(data).hexdigest() == ids.file_digest(path)
    assert hasher.size == len(data)
    assert list(ids.read_lines(path)) == lines  # no hasher: the lines alone

    # A file larger than one read is split the same way, across the reads.
    big = b"".join(b"line %d\n" % n for n in range(200_000)) + b"last, no line feed"
    assert len(big) > 2 * (1 << 20)
    (tmp_path / "big.jsonl").write_bytes(big)
    hasher = ids.ContentHasher()
    count = 0
    last = None
    for last in ids.read_lines(tmp_path / "big.jsonl", hasher):
        count += 1
    assert (count, last) == (200_001, b"last, no line feed")
    assert hasher.hexdigest() == hashlib.sha256(big).hexdigest()


def test_the_hasher_is_sha256_of_everything_taken():
    hasher = ids.ContentHasher()
    assert hasher.hexdigest() == hashlib.sha256(b"").hexdigest() and hasher.size == 0
    hasher.update(b"water")
    hasher.update(b"white")
    assert hasher.hexdigest() == hashlib.sha256(b"waterwhite").hexdigest() == ids.bytes_digest(b"waterwhite")


# --- ids of a mixture and of a checkpoint ------------------------------------------------


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def blake32(value) -> str:
    return hashlib.blake2b(canonical(value), digest_size=32).hexdigest()


def sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def no_edges() -> str:
    h = hashlib.blake2b(digest_size=32)
    h.update(b"ww.edges.v1")
    return h.hexdigest()


def test_a_mixture_id_covers_its_shards_in_order_their_digests_and_its_settings():
    shards, digests, settings = ["s" * 64, "t" * 64], ["1" * 64, "2" * 64], {"split": "1,0,0"}
    content = sha(["ww.mixture.v1", digests])
    assert ids.mixture_content_digest(digests) == content
    expected = blake32(["ww.instance.v1", content, "waterwhite.mixture", "1",
                        sha(["ww.params.v1", settings]), shards, no_edges()])
    assert ids.mixture_instance_id(shards, digests, settings) == expected
    assert ids.no_edges_digest() == no_edges()
    others = [
        ids.mixture_instance_id(shards[::-1], digests[::-1], settings),  # another order
        ids.mixture_instance_id(shards, ["1" * 64, "3" * 64], settings),  # another digest
        ids.mixture_instance_id(shards, digests, {"split": "98,1,1"}),  # other settings
        ids.mixture_instance_id(["u" * 64, "t" * 64], digests, settings),  # another instance
    ]
    assert len({expected, *others}) == 5


def test_a_checkpoint_id_covers_its_segment_and_step_and_nothing_else():
    content = sha(["ww.checkpoint.v1", "E/seg-1", 6])
    assert ids.checkpoint_content_digest("E/seg-1", 6) == content
    expected = blake32(["ww.instance.v1", content, "waterwhite.checkpoint", "1",
                        sha(["ww.params.v1", {"segment": "E/seg-1", "step": 6}]), [], no_edges()])
    assert ids.checkpoint_instance_id("E/seg-1", 6) == expected
    assert ids.checkpoint_instance_id("E/seg-1", 7) != expected
    assert ids.checkpoint_instance_id("E/seg-2", 6) != expected


# --- words ---------------------------------------------------------------------------------


def check_words(table: str, column: str) -> set[str]:
    """The words a CHECK (column IN (...)) of the schema allows for a column."""
    schema = _migrate.schema_sql()
    body = schema[schema.index(f"CREATE TABLE {table} ("):]
    body = body[: body.index("\n);") if "\n);" in body else len(body)]
    found = re.search(rf"\b{column}\s+TEXT[^\n]*?CHECK \({column} IN\s*\(([^)]*)\)", body, re.S)
    if not found:  # a column a later step adds (schema_2.sql)
        found = re.search(rf"ALTER TABLE {table} ADD COLUMN {column}\s+TEXT\s*CHECK \({column} IN\s*\(([^)]*)\)",
                          _migrate.schema_sql("schema_2.sql"), re.S)
    assert found, (table, column)
    return set(re.findall(r"'([^']*)'", found.group(1)))


def test_the_vocabulary_spells_what_the_schema_checks():
    assert check_words("member", "part_kind") == set(v.PART_KINDS)
    assert check_words("run_segment", "start_kind") == set(v.START_KINDS)
    assert check_words("run_segment", "trace_route") == set(v.TRACE_ROUTES)
    assert check_words("source_link", "derivation") == {v.DERIVED_POINTER, v.DERIVED_FILE, v.DERIVED_NONE}
    assert check_words("source_link", "verification") == set(v.VERIFICATIONS)
    assert check_words("drop_record", "reason") == set(v.DROP_REASONS)
    assert check_words("binding", "namespace") == set(v.NAMESPACES)
    # The names the store's views carried before the vocabulary did are the same objects.
    assert views.PART_KINDS is v.PART_KINDS and views.START_KINDS is v.START_KINDS
    assert views.TRACE_ROUTES is v.TRACE_ROUTES and views.ROUTE_NONE is v.ROUTE_NONE


def test_the_trace_missing_reasons_are_the_two_of_the_plan():
    assert v.TRACE_MISSING_REASONS == ("no recorded digest", "digest mismatch")
    assert not hasattr(v, "TRACE_MISSING_NO_CACHE")
    assert v.GENERATING_MODEL_CAUSES == (
        "generating-model entry missing", "generating model exposed", "generating model not registered")
    names = {value for key, value in vars(v).items() if key.startswith("NAMES_")}
    assert set(v.GENERATING_MODEL_CAUSES) <= names


def test_the_list_of_named_causes_holds_every_cause_once():
    spelled = [value for key, value in vars(v).items() if key.startswith("NAMES_")]
    assert sorted(v.NAMED_CAUSES) == sorted(spelled)
    assert len(set(v.NAMED_CAUSES)) == len(v.NAMED_CAUSES)
    # The three causes of the generating-model rule exist while the rule is off.
    assert (v.NAMES_MODEL_ENTRY_MISSING, v.NAMES_MODEL_EXPOSED, v.NAMES_MODEL_NOT_REGISTERED) == (
        v.GENERATING_MODEL_CAUSES)
    assert set(v.GENERATING_MODEL_CAUSES) <= set(v.NAMED_CAUSES)


def test_the_words_of_a_hop():
    assert (v.HOP_MEMBER, v.HOP_CONSUMED) == ("member", "consumed")
    assert v.hop_source_link(v.FILE_LEVEL) == "file-level link"
    assert v.hop_source_link(v.MAPPED) == "mapped link"


def test_no_named_cause_is_spelled_outside_the_vocabulary():
    """Every cause, reason and note an answer carries is a constant of the
    vocabulary: no module of the tool spells one of those strings itself."""
    words = set(v.NAMED_CAUSES) | set(v.NOT_COVERED_REASONS) | set(v.TRACE_MISSING_REASONS) | set(
        v.TRACE_MISSING_RECORD_REASONS) | {
        v.NOTE_PROCESS_EVIDENCE, v.NOTE_GENERATING_MODEL_NOT_ASSESSED, v.NOTE_PUBLISHED_MODELS_OFF,
        v.BASIS_WHOLE_SAMPLE, v.BASIS_OPERATOR_NOTE}
    package = Path(v.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        if path.name in ("vocabulary.py", "errors.py"):
            continue  # the words themselves; an error name may equal a cause
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in words:
                found.append((path.name, node.lineno, node.value))
    assert found == []


def test_every_error_name_is_fixed_and_unique():
    classes = [c for c in vars(errors).values()
               if isinstance(c, type) and issubclass(c, errors.WaterwhiteError)]
    names = [c.name for c in classes]
    assert len(names) == len(set(names))
    wanted = {
        errors.HarnessRefused: ("read outside the hand-off", errors.EXIT_REFUSED),
        errors.NotAStore: ("not a store", errors.EXIT_REFUSED),
        errors.NewerSchemaVersion: ("store of a newer schema version", errors.EXIT_REFUSED),
        errors.StoreFileExists: ("a file already exists at the store path", errors.EXIT_WRITE_REFUSED),
        errors.UnknownNormaliser: ("unknown normaliser", errors.EXIT_WRITE_REFUSED),
        errors.ModelClientMissing: ("no model client", errors.EXIT_USAGE),
        errors.SandboxUnavailable: ("sandbox unavailable", errors.EXIT_USAGE),
        errors.ConfinedJobFailed: ("confined job failed", errors.EXIT_USAGE),
        errors.JobProtocolViolation: ("job protocol violation", errors.EXIT_USAGE),
        errors.TokenizerFailed: ("tokenizer failed", errors.EXIT_USAGE),
        errors.DropAlreadyRecorded: ("drop already recorded", errors.EXIT_WRITE_REFUSED),
    }
    for cls, (name, code) in wanted.items():
        assert (cls.name, cls.exit_code) == (name, code)


def test_no_module_of_the_tool_defines_a_named_error_of_its_own():
    """Every named error is a class of ``waterwhite.errors``, whichever package
    raises it: no other module of the tool subclasses one of them."""
    named = {name for name, value in vars(errors).items()
             if isinstance(value, type) and issubclass(value, errors.WaterwhiteError)}
    package = Path(v.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        if path == package / "errors.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                     for base in node.bases}
            if bases & named:
                found.append((path.name, node.name))
    assert found == []
    assert {"SandboxUnavailable", "ConfinedJobFailed", "JobProtocolViolation"} <= named


def test_the_words_the_packages_share_are_in_the_vocabulary():
    assert v.SANITIZER_DROP_REASONS == ("sanitizer skip", "empty line")
    assert set(v.SANITIZER_DROP_REASONS) < set(v.DROP_REASONS)
    assert (v.STEP_SANITIZE, v.STEP_TOKENIZE) == ("sanitize", "tokenize")
    assert check_words("harness_log", "harness") == set(v.HARNESS_KINDS) == {"in-process", "confined"}
    assert v.RECOVERY_OUTCOMES == ("record-level", "file-level", "unknown")
