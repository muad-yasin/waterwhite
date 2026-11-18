"""Reading what the search conditions are checked on, from files: the replica's
manifests and sanitizer log of a built fixture, and the output folder of the
fixture runs (``oracle/fixture_runs.py``).

Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path

from oracle import fixture_runs, positions
from oracle.digests import file_sha256
from oracle.errors import ConfigError
from oracle.fixture_spec import FixtureSpec, ShardRef
from oracle.search_conditions import BuildEvidence, Evidence, SegmentEvidence, ShardEvidence

# The request the fixture's checks make: the records it puts in scope are the
# "affected" records of plan 5.1.
WITHDRAWN_SOURCE = "A"


def _json_lines(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ConfigError(f"{path.name} line {number}: not a JSON object")
                rows.append(row)
    except FileNotFoundError as e:
        raise ConfigError(f"the built fixture has no file {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path.name}: not JSON lines: {e}") from e
    return rows


def _json_file(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"missing file of the fixture runs: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: not JSON: {e}") from e


def spec_shards(spec: FixtureSpec) -> list[ShardRef]:
    """Every shard a segment of the spec lists, each once, in the order met."""
    seen: dict[tuple[str, str], ShardRef] = {}
    for run in spec.runs.values():
        for segment in run.segments:
            for ref in segment.shards:
                seen.setdefault((ref.layout, ref.file), ref)
    return list(seen.values())


def load_shard(fixture: Path, ref: ShardRef, skips: list[dict]) -> ShardEvidence:
    """One shard's tokenizer input with its manifest. The manifest row of line n
    must name line n and one record, and that record must be the id the input
    line carries; anything else is refused, since a row's record would then be
    a guess."""
    input_file = fixture_runs.tokenizer_input(fixture, ref)
    if not input_file.is_file():
        raise ConfigError(f"the built fixture has no tokenizer input {input_file}")
    ids = positions.read_ids(input_file, "id")
    manifest = _json_lines(input_file.with_name(input_file.name + ".manifest.jsonl"))
    if len(manifest) != len(ids):
        raise ConfigError(f"{ref.name}: {len(ids)} input lines, {len(manifest)} manifest rows")
    for number, (row, record) in enumerate(zip(manifest, ids), start=1):
        origin = row.get("origin")
        if row.get("line") != number or not isinstance(origin, list) or origin != [record]:
            raise ConfigError(
                f"{ref.name}: manifest row {number} does not name line {number} and the record "
                f"{record!r} of that input line"
            )
    # A row of the sanitizer's log, as the replica writes it (docs/replica.md, "The
    # folder one build writes"): "input" names the sanitizer-input file of the
    # shard and "line" the line there, counted from 1.
    sanitizer_input = f"pipeline/{ref.layout}/sanitizer-input/{ref.file}.jsonl"
    places = []
    for skip in skips:
        if skip.get("shard") != ref.file:
            continue
        if skip.get("input") != sanitizer_input:
            raise ConfigError(
                f"{ref.name}: a sanitizer log row names the input {skip.get('input')!r}, "
                f"not {sanitizer_input!r}"
            )
        place = skip.get("line")
        if isinstance(place, bool) or not isinstance(place, int) or place < 1:
            raise ConfigError(f"{ref.name}: a sanitizer log row without a line in its input")
        places.append(place)
    if len(set(places)) != len(places) or any(place > len(ids) + len(places) for place in places):
        raise ConfigError(f"{ref.name}: the sanitizer log's places {places} do not fit its input")
    return ShardEvidence(
        name=ref.name,
        layout=ref.layout,
        file=ref.file,
        records=tuple(ids),
        manifest=tuple(manifest),
        skipped=tuple(sorted(places)),
    )


def load_build(spec: FixtureSpec, fixture: str | Path, base: str | Path) -> BuildEvidence:
    """The replica's evidence of one built fixture, for every shard the spec's
    segments list. ``base`` is the repository root, where the spec's generation
    file is found."""
    fixture = Path(fixture)
    skips: dict[str, list[dict]] = {}
    shards = {}
    for ref in spec_shards(spec):
        if ref.layout not in skips:
            log = fixture_runs.sanitizer_log(fixture, ref.layout)
            skips[ref.layout] = _json_lines(log)
        shards[(ref.layout, ref.file)] = load_shard(fixture, ref, skips[ref.layout])
    return BuildEvidence(shards=shards, affected=spec.derived_from(WITHDRAWN_SOURCE, Path(base)))


def evidence_from_runs(
    spec: FixtureSpec, build: BuildEvidence, fixture: str | Path, runs_folder: str | Path
) -> Evidence:
    """The oracle's output as the fixture runs wrote it. A run with no folder is
    left out; a run with some of its segment folders missing is refused. Each
    run record's shard digests must be those of the built fixture's shard
    files, so logs of another build cannot be read against this one's manifests."""
    fixture = Path(fixture)
    root = Path(runs_folder) / fixture_runs.RUNS_FOLDER
    if not root.is_dir():
        raise ConfigError(f"not an output folder of the fixture runs (no {fixture_runs.RUNS_FOLDER}/): {runs_folder}")
    runs: dict[str, list[SegmentEvidence]] = {}
    for run in spec.runs.values():
        folders = [root / run.name / segment.folder for segment in run.segments]
        present = [folder.is_dir() for folder in folders]
        if not any(present):
            continue
        if not all(present):
            raise ConfigError(f"run {run.name!r}: some of its segment folders are missing under {root}")
        segments = []
        for segment, folder in zip(run.segments, folders):
            record = _json_file(folder / fixture_runs.RUN_RECORD_FILE)
            if record.get("segment") != segment.name or len(record.get("shards", [])) != len(segment.shards):
                raise ConfigError(f"{folder}: the run record is not that of segment {segment.name!r}")
            for entry, ref in zip(record["shards"], segment.shards):
                prefix = fixture / "pipeline" / ref.layout / "shards" / ref.file
                digests = (file_sha256(str(prefix) + ".bin"), file_sha256(str(prefix) + ".idx"))
                if (entry.get("bin_sha256"), entry.get("idx_sha256")) != digests:
                    raise ConfigError(
                        f"segment {segment.name!r}: shard {ref.name!r} of the run record is not the "
                        "shard of this built fixture (the digests differ)"
                    )
            index = _json_file(folder / fixture_runs.DIGESTS_FILE)
            if not (folder / fixture_runs.LOG_FILE).is_file():
                raise ConfigError(f"missing file of the fixture runs: {folder / fixture_runs.LOG_FILE}")
            segments.append(
                SegmentEvidence(
                    name=segment.name,
                    shards=tuple(ref.name for ref in segment.shards),
                    rows=positions.read_log([folder / fixture_runs.LOG_FILE]),
                    summary=_json_file(folder / fixture_runs.SUMMARY_FILE),
                    shuffle_digests=record["shuffle_digests"],
                    step_record=_json_file(folder / fixture_runs.STEP_RECORD_FILE),
                    shard_documents=tuple(entry["documents"] for entry in index["shards"]),
                )
            )
        runs[run.name] = segments
    return Evidence(spec=spec, build=build, runs=runs)
