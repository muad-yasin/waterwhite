"""The two set-ups of Check 3 that change a tokenizer input before a recovery is
run (fixture/table/README.md section 7; plan 5.5 Check 3).

Both act on the files of a build made without capture, after tokenization and
before anything of the build is registered. They change the tokenizer input of a
shard and its manifest, and nothing else: the shard's two files stay as the
tokenizer wrote them from the unchanged input. A caller that needs the unchanged
build as well copies the build folder first (``replica.recover.Uncaptured.moved_to``).

- ``swap_twin_lines`` (``t.twin-swap``): the two lines of an identical pair that
  lie in one shard change places. The two texts are equal, so only the ids move.
- ``change_first_line`` (``t.verification-fails``): line 1 of a shard's tokenizer
  input is replaced by a record whose text no document of the shard holds. The
  line count stays.

Which pairs lie in one shard is the caller's knowledge: it comes from the plant
list and the ids of the chunk-file lines, counted by a script outside the replica.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .errors import InputError
from .records import split_lines

# The text the table gives the changed line (README section 7, t.verification-fails).
CHANGED_TEXT = "This line was changed after the shard was tokenized."
CHANGED_ID = "changed-after-tokenization"
HANDLE_CHANGED = "changed_after_tokenization"


def _tokenizer_input(out: Path, layout: str, shard: str) -> Path:
    path = Path(out) / "pipeline" / layout / "tokenizer-input" / f"{shard}.jsonl"
    if not path.is_file():
        raise InputError(f"{path}: the build holds no tokenizer input for {shard}")
    return path


def _read(path: Path) -> list[str]:
    return split_lines(path.read_bytes().decode("utf-8"))


def _write(path: Path, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def _manifest(path: Path) -> Path:
    return Path(f"{path}.manifest.jsonl")


def swap_twin_lines(out: Path, layout: str, pairs: Sequence[tuple[str, int, int]]) -> list[dict]:
    """Swap the two lines of each pair in its shard's tokenizer input.

    ``pairs`` holds ``(shard, line, line)`` entries, lines counted from 1. The two
    lines must hold the same text: the set-up swaps twins and nothing else. The
    manifest rows swap with their lines, so each row still names the record on
    its line. Returns what was swapped, with the ids that changed places."""
    done = []
    by_shard: dict[str, list[tuple[int, int]]] = {}
    for shard, first, second in pairs:
        by_shard.setdefault(shard, []).append((first, second))
    for shard, swaps in by_shard.items():
        path = _tokenizer_input(out, layout, shard)
        lines = _read(path)
        manifest = _read(_manifest(path))
        for first, second in swaps:
            if not (1 <= first <= len(lines) and 1 <= second <= len(lines)) or first == second:
                raise InputError(f"{path}: lines {first} and {second} are not two lines of "
                                 f"a file of {len(lines)} lines")
            one, other = json.loads(lines[first - 1]), json.loads(lines[second - 1])
            if one["text"] != other["text"]:
                raise InputError(f"{path}: lines {first} and {second} hold different texts; "
                                 "the set-up swaps the two lines of an identical pair")
            lines[first - 1], lines[second - 1] = lines[second - 1], lines[first - 1]
            a, b = json.loads(manifest[first - 1]), json.loads(manifest[second - 1])
            a["line"], b["line"] = second, first
            manifest[first - 1], manifest[second - 1] = (
                json.dumps(b, ensure_ascii=True), json.dumps(a, ensure_ascii=True))
            done.append({"shard": shard, "lines": [first, second],
                         "ids": [one.get("id"), other.get("id")]})
        _write(path, lines)
        _write(_manifest(path), manifest)
    return done


def change_first_line(out: Path, layout: str, shard: str | None = None) -> dict:
    """Replace line 1 of one shard's tokenizer input by a record whose text is
    ``CHANGED_TEXT``. ``shard`` None takes the first shard of the layout in name
    order that has a tokenizer input. The manifest row of the line is replaced
    too: the line is no copy of any record now. Returns the shard and the id the
    line carried."""
    folder = Path(out) / "pipeline" / layout / "tokenizer-input"
    if shard is None:
        names = sorted(p.name[:-len(".jsonl")] for p in folder.glob("*.jsonl")
                       if not p.name.endswith(".manifest.jsonl"))
        if not names:
            raise InputError(f"{folder}: the build holds no tokenizer input")
        shard = names[0]
    path = _tokenizer_input(out, layout, shard)
    lines = _read(path)
    was = json.loads(lines[0])
    lines[0] = json.dumps({"id": CHANGED_ID, "text": CHANGED_TEXT}, ensure_ascii=True)
    _write(path, lines)
    manifest = _read(_manifest(path))
    manifest[0] = json.dumps({"line": 1, "handle": HANDLE_CHANGED, "origin": []},
                             ensure_ascii=True)
    _write(_manifest(path), manifest)
    return {"shard": shard, "line": 1, "was": was.get("id")}
