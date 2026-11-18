"""A tiny pipeline for the recovery tests, written here and not taken from the tool
or the replica.

It keeps, for every line it writes, which line of which file the line is a copy
of. That bookkeeping is the tests' expected answer: the recovery never sees it.

The steps are the copy steps of a pretraining pipeline in miniature: shuffle (with
a seeded generator, so a test is the same on every run), deal round-robin into
piles, merge. A copy step here writes a row again as ``{"text": ...}``, so the ids
of the source rows are gone after the first step, as in a run made without
capture.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from waterwhite import capture, coverage, normalise, recovery, requests
from waterwhite.store import Store

NORMALISER = "control-chars/1"


@dataclass(frozen=True)
class Rec:
    """One line on its way through the test pipeline.

    ``origin`` is (registered name of the file, line) of the line this one is a
    copy of in the file the step read; ``raw`` is the line as it is written."""

    origin: tuple[str, int]
    raw: str

    @property
    def text(self) -> str | None:
        try:
            row = json.loads(self.raw)
        except ValueError:
            return None
        text = row.get("text") if isinstance(row, dict) else None
        return text if isinstance(text, str) else None


def new_store(folder: Path, normaliser: str = NORMALISER) -> Store:
    return Store.create(folder / "store.sqlite", normaliser=normaliser)


def write_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return path


def source(store: Store, folder: Path, key: str, texts: list[str]) -> list[Rec]:
    """Write and ingest one source file, ``src/<key>`` under the member
    ``<key>/r1``, and return its lines as the pipeline reads them."""
    name = f"src/{key}"
    rows = [json.dumps({"id": f"{key}-{n}", "text": text}) for n, text in enumerate(texts, 1)]
    path = write_lines(folder / f"sources/{key}.jsonl", rows)
    capture.ingest_jsonl(store, path, source=key, member=f"{key}/r1", revision="r1", name=name)
    return [Rec((name, n), raw) for n, raw in enumerate(rows, 1)]


def shuffled(records: list[Rec], seed: int) -> list[Rec]:
    out = list(records)
    random.Random(seed).shuffle(out)
    return out


def dealt(records: list[Rec], piles: int) -> list[list[Rec]]:
    """Deal round-robin, starting at the first pile."""
    out: list[list[Rec]] = [[] for _ in range(piles)]
    for place, record in enumerate(records):
        out[place % piles].append(record)
    return out


def merged(piles: list[list[Rec]]) -> list[Rec]:
    return [record for pile in piles for record in pile]


def chunked(files: list[list[Rec]], chunks: int, seed: int) -> list[list[Rec]]:
    """The round-robin layout in miniature: every file is shuffled and dealt over
    the chunks, the piles of all files are merged chunk by chunk, and each chunk
    is shuffled once more."""
    piles = [dealt(shuffled(file, seed + place), chunks) for place, file in enumerate(files)]
    return [
        shuffled(merged([pile[chunk] for pile in piles]), seed + 100 + chunk)
        for chunk in range(chunks)
    ]


def found_file(store: Store, folder: Path, name: str, records: list[Rec]) -> list[Rec]:
    """Write the records as one file a copy step left behind, register the file
    as found afterwards, and return its lines as the next step reads them. A line
    with a text is written again as ``{"text": ...}``; any other line is passed on
    as it is."""
    lines = [
        record.raw if record.text is None else json.dumps({"text": record.text})
        for record in records
    ]
    path = write_lines(folder / f"pipeline/{name}.jsonl", lines)
    recovery.register_found_file(store, path, name=name)
    return [Rec((name, n), raw) for n, raw in enumerate(lines, 1)]


def file_path(folder: Path, name: str) -> Path:
    return folder / f"pipeline/{name}.jsonl"


def sanitize(records: list[Rec], input_name: str) -> tuple[list[Rec], list[dict]]:
    """The test pipeline's sanitizer: the store's normaliser on every text, a
    blank or emptied line skipped as "empty line", an unreadable line skipped as
    "sanitizer skip". ``records`` are the lines of the sanitizer's input file and
    ``input_name`` is what the log calls that file. Returns the surviving records
    with their sanitized text and the rows of the sanitizer's log."""
    kept: list[Rec] = []
    log: list[dict] = []
    for position, record in enumerate(records, 1):
        row = {"shard": "shard-1", "input": input_name, "line": position, "id": None,
               "origin_file": None, "origin_line": None}
        if record.raw.strip() == "":
            log.append({**row, "reason": "empty line", "detail": "blank line"})
            continue
        if record.text is None:
            log.append({**row, "reason": "sanitizer skip", "detail": "not a record"})
            continue
        clean = normalise.apply(NORMALISER, record.text)
        if clean == "":
            log.append({**row, "reason": "empty line", "detail": "empty after sanitizing"})
            continue
        kept.append(Rec(record.origin, json.dumps({"text": clean})))
    return kept, log


def write_log(path: Path, rows: list[dict]) -> Path:
    return write_lines(path, [json.dumps(row, sort_keys=True) for row in rows])


def ref(store: Store, name: str, line: int) -> capture.OccurrenceRef:
    return capture.OccurrenceRef(store.artifact(name).instance_id, f"L{line}")


def parents(store: Store, name: str, line: int) -> list[tuple[str, int]]:
    """The parents the store holds for one line, as (registered name, line)."""
    edge = store.edge_of(ref(store, name, line))
    if edge is None:
        return []
    out = []
    for parent in edge.parents:
        artifact = store.artifact(parent.instance_id)
        out.append((artifact.names[0], int(parent.locator[1:])))
    return out


def ask(store: Store, source_name: str, artifact: str, **narrow) -> dict:
    """Put "withdraw <source>" to one artifact and return the answer's JSON form."""
    request = requests.withdraw(store, source_name, **narrow)
    return coverage.answer(store, artifact, request).to_json()


def hops(answer: dict) -> list[str]:
    return [hop["hop"] for hop in answer["chain"]]


def rows_held(store: Store, name: str) -> tuple[int, int, int]:
    """How many edges, drops and file links the store holds for one artifact."""
    instance = store.artifact(name).instance_id
    return (
        sum(1 for _ in store.edges(instance)),
        sum(1 for _ in store.drops_by_output(instance)),
        len(store.file_source_links(instance)),
    )
