"""The counts, the mixtures and the groups of the rebuild report (plan 4.1, 4.5;
docs/interfaces.md section 7.1, docs/rebuild.md section 2.3).

Every function reads the entries of ``artifacts.py`` and the store; none reads a
verdict or an answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .. import vocabulary as v
from ..errors import UsageError
from ..store import views


def check_layouts(layouts: Mapping[str, str] | None) -> dict[str, str]:
    """The layouts the caller named: a name and the prefix of the store names of its
    shards. Both must be given; the store records no layout (plan 4.5)."""
    held: dict[str, str] = {}
    for name, prefix in (layouts or {}).items():
        if not isinstance(name, str) or not name or not isinstance(prefix, str) or not prefix:
            raise UsageError(f"a layout is NAME=PREFIX, both given; got {name!r}={prefix!r}")
        held[name] = prefix
    return held


# --- Bytes -----------------------------------------------------------------------------


def size_of(store, artifact) -> int | None:
    """The size of the file or files of an artifact, read from the disk now: the file
    of a JSONL artifact, the two files of a shard. None where a file is not at its
    recorded place (the store holds no size, and a file that is gone has none)."""
    if store.file_state(artifact.instance_id) != views.FILE_PRESENT:
        return None
    path = store.disk_path(artifact.instance_id)
    if path is None:
        return None
    try:
        if artifact.kind == v.KIND_SHARD:
            return sum(Path(f"{path}{suffix}").stat().st_size for suffix in (".bin", ".idx"))
        return Path(path).stat().st_size
    except OSError:
        return None


def add_bytes(store, artifacts_by_instance: Mapping[str, object], entries: list[dict]) -> None:
    """Fill ``bytes_to_rewrite`` of every file that is rewritten. A mixture and a
    checkpoint have no file and are not asked."""
    for entry in entries:
        if entry["action"] != v.ACTION_REWRITE or entry["kind"] not in (v.KIND_JSONL, v.KIND_SHARD):
            continue
        entry["bytes_to_rewrite"] = size_of(store, artifacts_by_instance[entry["instance"]])


# --- The counts ------------------------------------------------------------------------


def _of_kind(entries: list[dict], kind: str) -> list[dict]:
    return [entry for entry in entries if entry["kind"] == kind]


def _keep_and_rewrite(entries: list[dict]) -> dict[str, int]:
    return {
        v.ACTION_KEEP: sum(1 for e in entries if e["action"] == v.ACTION_KEEP),
        v.ACTION_REWRITE: sum(1 for e in entries if e["action"] == v.ACTION_REWRITE),
    }


def _named(entries: list[dict], kind: str, action: str) -> list[str]:
    return [e["artifact"] for e in entries if e["kind"] == kind and e["action"] == action]


def totals(entries: list[dict]) -> dict:
    """The ``counts`` of the report. Only the JSONL entries feed the record counts."""
    files = _of_kind(entries, v.KIND_JSONL)
    words = (v.ACTION_KEEP, v.ACTION_RE_SOURCE, v.ACTION_PROMOTE, v.ACTION_RETIRE, v.UNRESOLVED)
    sizes = [e for e in entries if e["action"] == v.ACTION_REWRITE
             and e["kind"] in (v.KIND_JSONL, v.KIND_SHARD)]
    return {
        "records": {word: sum(e["counts"][word] for e in files) for word in words},
        "records_removed": sum(e["records_removed"] for e in files),
        "shards": _keep_and_rewrite(_of_kind(entries, v.KIND_SHARD)),
        "mixtures": _keep_and_rewrite(_of_kind(entries, v.KIND_MIXTURE)),
        "bytes_to_rewrite": sum(e["bytes_to_rewrite"] for e in sizes
                                if e["bytes_to_rewrite"] is not None),
        "bytes_unknown": [e["artifact"] for e in sizes if e["bytes_to_rewrite"] is None],
        "unresolved_artifacts": [e["artifact"] for e in entries if e["action"] == v.UNRESOLVED],
        # Neither kept nor rewritten: clean in themselves, and a copy that returns may land
        # in them (the fourth word). A sibling of the two counts, which keep their words.
        "place_not_stated": {
            "shards": _named(entries, v.KIND_SHARD, v.ACTION_CLEAN_PLACE_NOT_STATED),
            "mixtures": _named(entries, v.KIND_MIXTURE, v.ACTION_CLEAN_PLACE_NOT_STATED),
        },
    }


def mixtures(entries: list[dict]) -> dict:
    """Which mixtures need a successor, which are kept, which cannot be told and which
    read the fourth word: every mixture is in exactly one of the four lists."""
    return {
        "needs_successor": _named(entries, v.KIND_MIXTURE, v.ACTION_REWRITE),
        "kept": _named(entries, v.KIND_MIXTURE, v.ACTION_KEEP),
        "unresolved": _named(entries, v.KIND_MIXTURE, v.UNRESOLVED),
        "place_not_stated": _named(entries, v.KIND_MIXTURE, v.ACTION_CLEAN_PLACE_NOT_STATED),
    }


# --- The groups -------------------------------------------------------------------------


def _group(members: list[tuple[str, str]]) -> dict:
    """A group of shards given as (name, action) in the order they are met."""
    names = lambda action: [name for name, held in members if held == action]  # noqa: E731
    keep, rewrite = names(v.ACTION_KEEP), names(v.ACTION_REWRITE)
    return {
        "counts": {v.ACTION_KEEP: len(keep), v.ACTION_REWRITE: len(rewrite)},
        "keep": keep,
        "rewrite": rewrite,
        "unresolved": names(v.UNRESOLVED),
    }


def by_run(store, shard_entries: Mapping[str, dict]) -> dict:
    """The shards grouped by the runs whose segments' mixtures hold them. A shard a
    snapshot names with no row of its own is listed as unresolved under its path."""
    out: dict[str, dict] = {}
    for run in store.runs():
        members: list[tuple[str, str]] = []
        seen: set[str] = set()
        for segment_name in store.segments(run):
            segment = store.segment(segment_name)
            if segment is None or segment.mixture_instance_id is None:
                continue
            for member in store.mixture_shards(segment.mixture_instance_id):
                if member.shard_instance_id in seen:
                    continue
                seen.add(member.shard_instance_id)
                held = shard_entries.get(member.shard_instance_id)
                members.append((member.path, v.UNRESOLVED) if held is None
                               else (held["artifact"], held["action"]))
        if members:
            out[run] = _group(members)
    return out


def _matching(names: list[str], layouts: Mapping[str, str]) -> list[str]:
    """The layouts whose prefix some name of a shard begins with."""
    return [layout for layout, prefix in layouts.items()
            if any(name.startswith(prefix) for name in names)]


LAYOUT_NOT_COUNTED = (
    "no layout is recorded in the store, so shards are not counted per layout; "
    "give each layout with --layout NAME=PREFIX to count them")


def layout_statement(shard_names: Mapping[str, list[str]], shard_entries: Mapping[str, dict],
                     layouts: Mapping[str, str]) -> dict:
    """Whether the per-layout count was made (``groups.layouts``): with no layout named
    it was not, and the reason says why and names the flag; with layouts named it was,
    and every shard that no named prefix matches is listed."""
    if not layouts:
        return {"counted": False, "reason": LAYOUT_NOT_COUNTED, "unmatched_shards": []}
    unmatched = [shard_entries[instance]["artifact"] for instance, names in shard_names.items()
                 if not _matching(names, layouts)]
    return {"counted": True, "reason": None, "unmatched_shards": unmatched}


def membership(shard_names: Mapping[str, list[str]],
               layouts: Mapping[str, str]) -> tuple[dict[str, list[str]], list[str]]:
    """The shards (by instance id, in the order they are given) of each named layout, and
    the shards no named layout matches, which are one group of their own. A shard that two
    prefixes match is a usage error: the store records no layout, so the caller's list
    must not overlap."""
    named: dict[str, list[str]] = {name: [] for name in layouts}
    rest: list[str] = []
    for instance, names in shard_names.items():
        matching = _matching(names, layouts)
        if len(matching) > 1:
            raise UsageError(
                f"the shard {names[0]} matches the layouts {' and '.join(matching)}; "
                "a shard belongs to one layout")
        if matching:
            named[matching[0]].append(instance)
        else:
            rest.append(instance)
    return named, rest


def by_layout(shard_names: Mapping[str, list[str]], shard_entries: Mapping[str, dict],
              layouts: Mapping[str, str]) -> dict:
    """The shards grouped by the prefixes the caller names."""
    named, _rest = membership(shard_names, layouts)
    return {name: _group([(shard_entries[i]["artifact"], shard_entries[i]["action"])
                          for i in members])
            for name, members in named.items()}
