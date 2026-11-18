"""The file-level manifest of the build of the key ``main+generating-model`` (docs/replica.md
section 16: a build made with capture on writes one row per artifact, generation outputs and
releases included; the table row c10.c07.R3-release.manifest follows built_from from the row of
R3 down to the rows of source files).

Expected values come from that section's table and from fixture/sources/generation_r3.jsonl (read
with ``json``): a generation output is built from the files of the records its jobs were handed
(here the records of Z, so the artifact ``src/Z``), a release from the files of the copies it kept
(here the generated file). The build needs the confined harness (the sandbox) and no Megatron
(tokenization is off), so the tests skip where the sandbox cannot be made.
"""

from __future__ import annotations

import json

import pytest

import e2_kit as kit
from replica.fixture import build_table_store

KEY = "main+generating-model"
R3_NAMES = ("R3", "gen/R3-models")


def sandbox_or_skip() -> None:
    from waterwhite import confine
    from waterwhite.errors import SandboxUnavailable

    try:
        confine.find_sandbox()
    except SandboxUnavailable as unavailable:
        pytest.skip(str(unavailable))


def rows_of(built) -> list[dict]:
    text = (built.out / "built-from.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line != ""]


def test_the_build_of_the_key_writes_the_rows_of_r3(tmp_path):
    """Section 16: the generation output ``gen/R3-models`` has a row whose built_from names the
    artifacts of the files its jobs were handed, and the release R3 a row whose built_from names
    the file of the copies it kept."""
    sandbox_or_skip()
    handed = {name for row in kit.r3_fixture_rows().values() for name in row["handed"]}
    assert all(name.startswith("Z-") for name in handed)  # every handed record is a record of Z
    by_name = {row["name"]: row for row in rows_of(build_table_store(tmp_path / "key", KEY))}
    assert (by_name.get("R3"), by_name.get("gen/R3-models")) == (
        {"name": "R3", "files": ["releases/R3.jsonl"], "built_from": ["gen/R3-models"]},
        {"name": "gen/R3-models", "files": ["gen/R3-models.jsonl"], "built_from": ["src/Z"]},
    )


def test_the_rows_of_the_main_build_are_the_same_in_the_build_of_the_key(tmp_path):
    """The key is the main build and then R3: its manifest is main's, rows and order, with the two
    rows of R3 added. A guard: green on a tree with no R3 row."""
    sandbox_or_skip()
    main = rows_of(build_table_store(tmp_path / "main", "main"))
    key = rows_of(build_table_store(tmp_path / "key", KEY))
    assert [row for row in key if row["name"] not in R3_NAMES] == main
