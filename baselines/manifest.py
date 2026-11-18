"""The reader of the file-level manifest (plan 5.5 Check 10; table README, section "Check
10": the subject ``manifest``; the reading rules of its rows in ``check10.toml``).

The manifest is the file ``built-from.jsonl`` that a build with capture on writes into its
build folder: one JSON row per artifact, ``{"name", "files", "built_from"}``. A build made
with capture off has no such file. The rows of the checkpoints of a run are appended to
that file by the run wrapper (their file paths lead from the build folder to the run
record), so one folder holds every row of one build. The repaired side of a repair is a
build of its own with a manifest of its own; ``join_manifests`` makes one list of the
rows of the two (``c09.history`` reads the historical R1 after the successor build's rows
are added). ``GROUP_READINGS`` and ``read_group`` give the reading of each group of
``check10.toml`` that the table routes to the manifest.

The reader reads and grades nothing. It answers each question from the rows alone:

- ``contains``: the walk from the artifact's row, row by row through ``built_from``,
  reaches a name the request names (the file of a member of the request, as an artifact
  name such as ``src/A``); the artifact itself counts. A name in a ``built_from`` list is
  reached whether or not it has a row of its own, and a name with no row ends its branch.
  A cycle ends the walk.
- ``does not contain``: the artifact has a row and the walk reaches no such name.
- ``no record``: the artifact has no row.
- ``absent``: there is no manifest at all (a build made with capture off). It is a word of
  its own, neither a reading of the rows nor ``no record``, and every question gives it
  for a manifest that is ``None``. A manifest with no row is not absent.

The module imports nothing of the tool, of the replica or of the world helpers of the
acceptance tests: the file name is spelled here once more.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

MANIFEST_FILE = "built-from.jsonl"

CONTAINS = "contains"
DOES_NOT_CONTAIN = "does not contain"
NO_RECORD = "no record"
ABSENT = "manifest absent"

KEEP = "keep"
REWRITE = "rewrite"

SUFFIX_CHUNK_FILE = ".input"


class ManifestError(ValueError):
    """A file that is not a manifest: a line that is not JSON, a row without the three
    keys of the right types, or two rows of one name."""


def _checked(row: object, where: str) -> dict:
    if not isinstance(row, dict):
        raise ManifestError(f"{where}: a row is a JSON object")
    if not isinstance(row.get("name"), str):
        raise ManifestError(f"{where}: the row has no name")
    for key in ("files", "built_from"):
        value = row.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ManifestError(f"{where}: {key} of {row['name']!r} is not a list of names")
    return row


def load_manifest(folder: Path | str) -> list[dict] | None:
    """The rows of ``<folder>/built-from.jsonl`` in file order; None where the file does
    not exist (capture off); ``[]`` for a file with no row."""
    path = Path(folder) / MANIFEST_FILE
    if not path.is_file():
        return None
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f.read().split("\n"), 1):
            if line == "":
                continue
            where = f"{MANIFEST_FILE} line {number}"
            try:
                row = _checked(json.loads(line), where)
            except json.JSONDecodeError as error:
                raise ManifestError(f"{where}: not JSON ({error.msg})") from None
            if row["name"] in seen:
                raise ManifestError(f"{where}: two rows for {row['name']!r}")
            seen.add(row["name"])
            rows.append(row)
    return rows


def _by_name(rows: Sequence[dict]) -> dict[str, dict]:
    return {row["name"]: row for row in rows}


def _walk(by_name: dict[str, dict], artifact: str, targets: Collection[str]) -> str:
    row = by_name.get(artifact)
    if row is None:
        return NO_RECORD
    seen = {artifact}
    todo = [artifact]
    while todo:
        name = todo.pop()
        if name in targets:
            return CONTAINS
        for parent in by_name[name]["built_from"]:
            if parent in seen:
                continue
            seen.add(parent)
            if parent in targets:
                return CONTAINS
            if parent in by_name:
                todo.append(parent)
    return DOES_NOT_CONTAIN


def read_artifact(rows: Sequence[dict] | None, artifact: str, targets: Collection[str]) -> str:
    """The word for ``artifact`` under a request whose member files are ``targets``."""
    if rows is None:
        return ABSENT
    return _walk(_by_name(rows), artifact, targets)


def _readings(rows: Sequence[dict], artifacts: Iterable[str],
              targets: Collection[str]) -> dict[str, str]:
    by_name = _by_name(rows)
    return {a: _walk(by_name, a, targets) for a in artifacts}


def mixture_lists(rows: Sequence[dict] | None, mixtures: Iterable[str],
                  targets: Collection[str]):
    """``{"needs_successor": [...], "kept": [...]}``, sorted: a mixture read ``contains``
    is in the first, ``does not contain`` in the second, ``no record`` in neither."""
    if rows is None:
        return ABSENT
    read = _readings(rows, mixtures, targets)
    return {"needs_successor": sorted(m for m, w in read.items() if w == CONTAINS),
            "kept": sorted(m for m, w in read.items() if w == DOES_NOT_CONTAIN)}


def artifact_actions(rows: Sequence[dict] | None, artifacts: Iterable[str],
                     targets: Collection[str]):
    """The actions of Check 9 read from the walk by the plan's own definition: contains is
    rewrite, does not contain is keep; an artifact read ``no record`` is in neither list."""
    if rows is None:
        return ABSENT
    read = _readings(rows, artifacts, targets)
    keep = sorted(a for a, w in read.items() if w == DOES_NOT_CONTAIN)
    rewrite = sorted(a for a, w in read.items() if w == CONTAINS)
    return {KEEP: keep, REWRITE: rewrite, "counts": {KEEP: len(keep), REWRITE: len(rewrite)}}


def exposure_free_restart(rows: Sequence[dict] | None, checkpoints: Sequence[str],
                          targets: Collection[str]):
    """The latest checkpoint, of ``checkpoints`` in step order, that is read does not
    contain with every checkpoint before it (and it) read the same; None where the first
    is read otherwise. A checkpoint with no row is read otherwise."""
    if rows is None:
        return ABSENT
    by_name = _by_name(rows)
    latest = None
    for name in checkpoints:
        if _walk(by_name, name, targets) != DOES_NOT_CONTAIN:
            break
        latest = name
    return latest


def import_family(rows: Sequence[dict] | None, name: str):
    """The ``built_from`` list of the row of ``name`` as the row has it."""
    if rows is None:
        return ABSENT
    row = _by_name(rows).get(name)
    return NO_RECORD if row is None else list(row["built_from"])


def join_manifests(*manifests: Sequence[dict] | None) -> list[dict] | None:
    """The rows of several manifests, read from several folders (the historical build and
    the repaired one, for ``c09.history``), as one list in the order given, joined by
    artifact name. A name in two of them is one row, at its first place, when the two rows
    are equal; two rows of one name that differ are refused. A manifest that is ``None`` (a
    build made with capture off) makes the join ``None``, since the rows of the others do
    not make a whole manifest."""
    if any(m is None for m in manifests):
        return None
    joined: list[dict] = []
    seen: dict[str, dict] = {}
    for rows in manifests:
        for row in rows:
            earlier = seen.get(row["name"])
            if earlier is None:
                seen[row["name"]] = row
                joined.append(row)
            elif dict(earlier) != dict(row):
                raise ManifestError(f"two manifests hold different rows for {row['name']!r}")
    return joined


# The groups of ``check10.toml`` whose manifest row has a record, each with the function of
# this module its row names (its route and its rule). Every other group has a manifest row
# that records none: the manifest holds no step, ordinal, record or count, and cannot answer
# it. Twenty-one groups are the walk of one artifact (``read_artifact``: one answer word per
# artifact; the set-size rows ``c09.10k.successor-shards`` and ``c09.10k.R1-history`` are
# in ``c09.successors`` and ``c09.history``); five have a reading of their own.
GROUP_READINGS: dict[str, str] = {
    **{group: "read_artifact" for group in (
        "c01.E@9", "c02.checkpoints", "c02.mixtures", "c04.edge-parent", "c04.artifact-digest",
        "c05.unmutated.files", "c05.unmutated.shards", "c05.unmutated.mixtures",
        "c05.unmutated.checkpoints", "c05.f.edge-deleted", "c05.f.map-deleted",
        "c05.withdraw-C.unmutated", "c05.unregistered-checkpoint", "c06.import.withdraw-A",
        "c07.release-answers", "c08.checkpoints", "c08.Y-mixture", "c09.successors",
        "c09.history",
        # The groups of the generating-model extension whose manifest row has a record (asked only while that extension is on): shard 3
        # under withdraw C with the rule on, and the release R3 as a whole.
        "c05.withdraw-C-on", "c07.R3-release")},
    "c05.listing.map-deleted": "listing",
    "c06.import.file-family": "import_family",
    "c09.mixture-successors": "mixture_lists",
    # The fixture's counts of shards, and the set-size rows ``c09.10k.partitioned.
    # kept-shards`` (the list of shards read keep) and ``c09.10k.round-robin.all-rewrite``
    # (the number of chunks read rewrite): the lists and counts of ``artifact_actions``.
    "c09.artifact-actions": "artifact_actions",
    "c09.exposure-free-restart": "exposure_free_restart",
}


def read_group(group: str, rows: Sequence[dict] | None, *, targets: Collection[str] = (),
               artifact: str | None = None, artifacts: Iterable[str] = (),
               checkpoints: Sequence[str] = ()):
    """The reading of a routed group through its function: ``artifact`` for a walk and for
    the import's file family, ``artifacts`` for the two lists and the actions,
    ``checkpoints`` (in step order) for the restart, nothing for the listing. ``targets``
    are the files of the members the request names. A group the manifest cannot answer is
    refused."""
    reading = GROUP_READINGS.get(group)
    if reading is None:
        raise ManifestError(f"the manifest has no reading for {group!r}: its row records none")
    if reading == "listing":
        return listing(rows)
    if reading == "mixture_lists":
        return mixture_lists(rows, artifacts, targets)
    if reading == "artifact_actions":
        return artifact_actions(rows, artifacts, targets)
    if reading == "exposure_free_restart":
        return exposure_free_restart(rows, checkpoints, targets)
    if artifact is None:
        raise ManifestError(f"{group!r} is read for one artifact: none was given")
    if reading == "import_family":
        return import_family(rows, artifact)
    return read_artifact(rows, artifact, targets)


def _kind(name: str) -> str:
    if name.startswith("src/"):
        return "sources"
    if name.startswith("gen/"):
        return "generated"
    if name.startswith("import/"):
        return "imports"
    if name.endswith(SUFFIX_CHUNK_FILE):
        return "chunk_files"
    if "/mix-" in name:
        return "mixtures"
    if "@" in name:
        return "checkpoints"
    if "/" in name:
        return "shards"
    return "releases"


def listing(rows: Sequence[dict] | None):
    """Every name the manifest holds, as a row of its own or inside a ``built_from``
    list, by kind, each list sorted. A name with no row of its own is in ``no_record`` and
    in no kind. The manifest holds no run, so ``runs`` is empty."""
    if rows is None:
        return ABSENT
    named = {row["name"] for row in rows}
    mentioned = {parent for row in rows for parent in row["built_from"]}
    out: dict[str, list[str]] = {kind: [] for kind in (
        "sources", "generated", "releases", "imports", "chunk_files", "shards", "mixtures",
        "checkpoints", "runs")}
    for name in sorted(named):
        out[_kind(name)].append(name)
    out["no_record"] = sorted(mentioned - named)
    return out
