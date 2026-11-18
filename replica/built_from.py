"""The file-level manifest a build with capture on records while it runs.

One row per artifact, in the file ``built-from.jsonl`` at the root of the build
folder (not the ``<file>.manifest.jsonl`` that sits beside each record file):

    {"name": "R1", "files": ["releases/R1.jsonl"], "built_from": ["src/A", "gen/P"]}

``name`` is the name the build registers the artifact under, ``files`` are its
paths relative to the build folder, and ``built_from`` names the artifacts whose
files went into it. Nothing lies below a file: no record, line, ordinal, step,
label, count or digest.

The recorder is handed only what the build itself reads and writes: names the
build chose, paths it wrote, the files of the records it read. It imports nothing
of the tool and reads nothing of the tool's capture or store, so the manifest is
the build's own account and not a view of what the store knows. A build with
capture off makes no recorder and writes no file.

The run wrapper appends the rows of the checkpoints of a run to the same file, in the
build folder the runs were made from (``record_checkpoints``): a checkpoint's ``files``
path is the path from the build folder to the run record in the runs folder.

What ``built_from`` lists, by kind of artifact: a release, the files of the copies it kept
(the file of a copy its dedup dropped did not go into it); a chunk file, the own files
(source files and generation outputs) of the records its lines hold, in a generated set as
well, where the records were cut from a release and the release is not named.
"""

from __future__ import annotations

import json
import os
import posixpath
from collections.abc import Iterable, Sequence
from pathlib import Path

from .errors import ReplicaError

MANIFEST_FILE = "built-from.jsonl"


def _row_text(name: str, files: Sequence[str], built_from: Sequence[str]) -> str:
    """One row as it is written: ASCII only, the three keys in a fixed order."""
    return json.dumps({"name": name, "files": list(files), "built_from": list(built_from)},
                      ensure_ascii=True) + "\n"


class BuiltFrom:
    """The recorder of one build. The file is created empty with the recorder and
    each artifact's row is appended when the build has made the artifact."""

    def __init__(self, folder: Path) -> None:
        self.path = Path(folder) / MANIFEST_FILE
        self._names: set[str] = set()
        self._name_of_file: dict[str, str] = {}  # path relative to the build folder -> name
        with open(self.path, "x", encoding="utf-8", newline="\n"):
            pass

    def names_of(self, files: Iterable[str]) -> list[str]:
        """The names of the artifacts the given files belong to, each once, in the
        order they are first met. A file no row has (a generation output the build
        did not capture) names nothing."""
        out: dict[str, None] = {}
        for relative in files:
            name = self._name_of_file.get(relative)
            if name is not None:
                out[name] = None
        return list(out)

    def add(self, name: str, files: Sequence[str], built_from: Sequence[str]) -> None:
        """Append the row of one artifact. ``files`` are relative to the build
        folder; ``built_from`` are names, never the artifact's own."""
        if name in self._names:
            raise ReplicaError(f"the manifest holds two rows for {name!r}")
        if not files:
            raise ReplicaError(f"the manifest row of {name!r} names no file")
        if name in built_from:
            raise ReplicaError(f"{name!r} is listed as built from itself")
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(_row_text(name, files, built_from))
        self._names.add(name)
        for relative in files:
            self._name_of_file.setdefault(relative, name)

    def add_from_files(self, name: str, files: Sequence[str],
                       read_files: Iterable[str]) -> None:
        """Append the row of an artifact built from the files ``read_files`` (paths
        relative to the build folder): its ``built_from`` is the names of those
        files' artifacts."""
        self.add(name, files, self.names_of(read_files))


def checkpoint_rows(segments: Sequence, shard_names: dict[str, str],
                    runs_prefix: str) -> list[dict]:
    """The rows of the checkpoints of run segments, from the segments alone.

    ``segments`` are ``replica.run_plan.SegmentRun`` in the spec's order.
    ``shard_names`` maps ``(layout, shard file)`` joined as ``"<layout>/<shard>"`` to
    the name the shard is registered under. ``runs_prefix`` is the path of the runs
    folder relative to the build folder, written with slashes. A checkpoint's row
    names its segment's run record as its file (a path relative to the build folder),
    the shards the segment was handed in the trainer's order, and then the parent
    checkpoint of the segment when it has one."""
    rows = []
    for segment in segments:
        record = posixpath.normpath(f"{runs_prefix}/{segment.run}/{segment.stem}.run.json")
        handed = [shard_names[f"{segment.layout}/{shard}"] for shard in segment.shard_files]
        built_from = handed + ([segment.parent_checkpoint] if segment.parent_checkpoint else [])
        for name, _step in segment.checkpoints:
            rows.append({"name": name, "files": [record], "built_from": list(built_from)})
    return rows


def record_checkpoints(build_folder: Path, runs_folder: Path, segments: Sequence,
                       layouts: dict) -> list[str]:
    """Append the rows of the checkpoints of ``segments`` to the manifest of the build
    folder the runs were made from, and return the names appended.

    The runs are made after the build, in a folder of their own, so a row's ``files``
    path leads from the build folder to the run record there (it may begin with ``..``).
    A build folder with no manifest (made with capture off) is left as it is and gets no
    row. A checkpoint that already has a row with the same ``built_from`` is left as it
    was (the same runs recorded twice); a row of the same name with another list is
    refused before anything is written."""
    path = Path(build_folder) / MANIFEST_FILE
    if not path.is_file():
        return []
    shard_names = {f"{layout.name}/{shard}": f"{layout.store_prefix}/{shard}"
                   for layout in layouts.values() for shard in layout.shard_names()}
    prefix = Path(os.path.relpath(Path(runs_folder).resolve(),
                                  Path(build_folder).resolve())).as_posix()
    existing: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f.read().split("\n"):
            if line != "":
                row = json.loads(line)
                existing[row["name"]] = row["built_from"]
    appended = []
    for row in checkpoint_rows(segments, shard_names, prefix):
        if row["name"] not in existing:
            appended.append(row)
            existing[row["name"]] = row["built_from"]
        elif existing[row["name"]] != row["built_from"]:
            raise ReplicaError(f"the manifest already holds a row for {row['name']!r} "
                               f"built from {existing[row['name']]}, not {row['built_from']}")
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in appended:
            f.write(_row_text(row["name"], row["files"], row["built_from"]))
    return [row["name"] for row in appended]
