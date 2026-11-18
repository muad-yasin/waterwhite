"""What the printed verification says about a mixture it could not answer for
(docs/cli.md "verify"; docs/interfaces.md section 7.3 "caches").

The cache entry of a run segment has ``stale`` null where the verification has no
answer: the report rewrites the mixture and the store holds no mixture registered as
its successor, or the report leaves the mixture unresolved. The JSON form says so
with ``stale`` null and a ``reason``. The text form (``--format text``) must name each
such mixture with its reason too, and print what it printed before for the rest. The
JSON form is unchanged, and the exit code stays 0.

The report is the one the tool makes for the store (``report withdraw A``); the store
is the hand-built world, with no successor registered (the first case) or with the
shard of the mixture the report leaves unresolved made unreadable by dropping its map
(the second case, through the world's own fault helper).
"""

from __future__ import annotations

import json

from waterwhite import vocabulary as v

from rb_kit import build, report_through_cli, run, write_json

JSON_KEYS = ["form", "request", "mode", "unresolved_records", "multisets", "differing_records",
             "differences", "releases", "per_shard_digests", "shards", "kept", "successors",
             "history", "caches", "restarts"]
CACHE_KEYS = ["run", "segment", "mixture", "corrected_mixture", "digest", "corrected_digest",
              "stale", "rebuilt", "rebuilt_from", "reason"]


def verified(world, tmp_path):
    """The JSON verification and the text one of the world's own report."""
    report = write_json(tmp_path / "report.json", report_through_cli(world))
    code, out, err = run(world, "verify", "--report", str(report))
    assert code == 0, err
    document = json.loads(out)
    code, text, err = run(world, "verify", "--report", str(report), "--format", "text")
    assert code == 0, err
    return document, text


def unanswered(document: dict) -> list[dict]:
    return [entry for entry in document["caches"] if entry["stale"] is None]


def lines_naming(text: str, entry: dict) -> list[str]:
    """The lines of the text that name the entry's mixture and carry its reason."""
    return [line for line in text.splitlines() if entry["mixture"] in line and entry["reason"] in line]


def test_a_rewritten_mixture_with_no_successor_in_the_store_is_named_with_its_reason(tmp_path):
    world = build(tmp_path / "w")
    try:
        document, text = verified(world, tmp_path)
        found = unanswered(document)
        assert {e["segment"] for e in found} == {"E/seg-2", "Y/seg-1"}
        for entry in found:
            assert entry["mixture"] == "mix-2" and entry["reason"], entry
            assert lines_naming(text, entry), (
                f"the text names no line with {entry['mixture']} and the reason {entry['reason']!r}")
    finally:
        world.store.close()


def test_a_mixture_the_report_leaves_unresolved_is_named_with_its_reason(tmp_path):
    world = build(tmp_path / "w", successors=True)
    try:
        world.delete_map("shard-1")  # without its map shard-1 answers nothing, so mix-1 cannot be answered
        document, text = verified(world, tmp_path)
        found = [e for e in unanswered(document) if e["mixture"] == "mix-1"]
        assert found, [(e["segment"], e["mixture"], e["stale"]) for e in document["caches"]]
        for entry in found:
            assert entry["reason"], entry
            assert lines_naming(text, entry), (
                f"the text names no line with {entry['mixture']} and the reason {entry['reason']!r}")
    finally:
        world.store.close()


def test_the_json_form_and_the_exit_code_are_what_they_were(tmp_path):
    world = build(tmp_path / "w")
    try:
        report = write_json(tmp_path / "report.json", report_through_cli(world))
        code, out, err = run(world, "verify", "--report", str(report))
        assert code == 0, err
        document = json.loads(out)
        assert list(document) == JSON_KEYS
        found = unanswered(document)
        assert found and all(list(e) == CACHE_KEYS for e in document["caches"])
        for entry in found:
            assert entry["stale"] is None and entry["rebuilt"] is False
            assert entry["corrected_mixture"] is None and isinstance(entry["reason"], str)
        code, text, err = run(world, "verify", "--report", str(report), "--format", "text")
        assert code == 0, err
    finally:
        world.store.close()


def test_a_successor_the_store_does_not_hold_is_still_printed_as_none_registered(tmp_path):
    world = build(tmp_path / "w")
    try:
        document, text = verified(world, tmp_path)
        assert [e["registered"] for e in document["successors"]] == [False] * 3
        for entry in document["successors"]:
            assert [line for line in text.splitlines()
                    if entry["successor"] in line and "(none registered)" in line], entry["successor"]
        assert document["form"] == v.FORM_VERIFICATION
    finally:
        world.store.close()
