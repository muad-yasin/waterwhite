"""Positions report for the seed search: read one or more oracle logs with the
tokenizer-input file of each shard and print, per step, which records were read
with their token ranges. It flags the read positions the fixture must hold (plan
5.1). Config and output: docs/oracle.md, "Positions report".

A record is named by the ``id`` of its input line: line i of a shard's input file
(counted from 0) is document i of the shard. That holds when the F1 probe says so
for the shard; this report does not check it.

Words used here. A sample's rows are the documents it read, in order. With the
extra token on, the last token of the last row is the sample's extra token: the
same token is the first token of the next sample in sample-index order. The body
of a row is its token range without that extra token. A row whose body is empty
read its document through the extra token only.

Standard library only: this report runs without the Megatron environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.cli import run_tool  # noqa: E402
from oracle.config import load_json, new_output_path, resolve_path, write_json  # noqa: E402
from oracle.errors import ConfigError  # noqa: E402

REPORT_FORMAT = "waterwhite.oracle-positions/1"

_LOG_INTS = ("step", "sample_pos", "global_index", "rank", "shard_position", "ordinal", "token_start", "token_end")


def read_log(paths: list[Path]) -> list[dict]:
    """The rows of the logs, in file order and line order."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ConfigError(f"{path.name} line {number}: not JSON: {e}") from e
                if not isinstance(row, dict) or not isinstance(row.get("segment"), str):
                    raise ConfigError(f"{path.name} line {number}: not a log row")
                for key in _LOG_INTS:
                    if isinstance(row.get(key), bool) or not isinstance(row.get(key), int):
                        raise ConfigError(f"{path.name} line {number}: {key!r} is not an integer")
                if not 0 <= row["token_start"] < row["token_end"]:
                    raise ConfigError(f"{path.name} line {number}: an empty or negative token range")
                rows.append(row)
    return rows


def read_ids(path: Path, id_field: str) -> list[str]:
    """The record id of every line of a tokenizer-input file, in order."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            try:
                ids.append(str(json.loads(line)[id_field]))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ConfigError(f"{path.name} line {number}: no JSON object with {id_field!r}: {e}") from e
    return ids


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def build_report(
    rows: list[dict],
    segments: dict[str, list[str]],
    ids: dict[str, list[str]],
    *,
    extra_token: bool = True,
) -> dict:
    """Make the report from log rows.

    ``segments`` maps a segment to the labels of its shards, by shard position;
    ``ids`` maps a shard label to the record ids of its lines. A shard that two
    segments both list under one label is one shard."""
    # Group the rows into samples. A sample's rows are consecutive in the log.
    samples: list[dict] = []
    seen_keys = set()
    for row in rows:
        key = (row["segment"], row["step"], row["sample_pos"])
        if not samples or samples[-1]["key"] != key:
            if key in seen_keys:
                raise ConfigError(f"the rows of sample {key} are not consecutive in the log")
            seen_keys.add(key)
            samples.append(
                {"key": key, "global_index": row["global_index"], "rank": row["rank"], "rows": []}
            )
        sample = samples[-1]
        if (row["global_index"], row["rank"]) != (sample["global_index"], sample["rank"]):
            raise ConfigError(f"sample {key} has rows with two global indices or ranks")
        if row["segment"] not in segments:
            raise ConfigError(f"segment {row['segment']!r} of the log is not in the config's 'segments'")
        labels = segments[row["segment"]]
        if row["shard_position"] >= len(labels):
            raise ConfigError(
                f"segment {row['segment']!r}: the log names shard position "
                f"{row['shard_position']}, the config lists {len(labels)} shards"
            )
        label = labels[row["shard_position"]]
        if row["ordinal"] >= len(ids[label]):
            raise ConfigError(
                f"shard {label!r} has {len(ids[label])} input lines, the log names document "
                f"{row['ordinal']}"
            )
        sample["rows"].append(
            {
                "shard": label,
                "ordinal": row["ordinal"],
                "record": ids[label][row["ordinal"]],
                "token_start": row["token_start"],
                "token_end": row["token_end"],
            }
        )

    # Mark each row: is it the row that carries the sample's extra token, and what
    # is its body.
    reads: list[dict] = []
    for sample in samples:
        segment, step, sample_pos = sample["key"]
        for i, row in enumerate(sample["rows"]):
            carries_extra = extra_token and i == len(sample["rows"]) - 1
            body_end = row["token_end"] - 1 if carries_extra else row["token_end"]
            row["carries_extra_token"] = carries_extra
            row["extra_token_only"] = carries_extra and body_end == row["token_start"]
            reads.append(
                {
                    **row,
                    "segment": segment,
                    "step": step,
                    "sample_pos": sample_pos,
                    "body": (row["token_start"], body_end),
                }
            )

    def where(read: dict) -> dict:
        return {
            "segment": read["segment"],
            "step": read["step"],
            "sample_pos": read["sample_pos"],
            "token_start": read["token_start"],
            "token_end": read["token_end"],
        }

    by_place: dict[tuple[str, int], list[dict]] = {}
    by_record: dict[str, list[dict]] = {}
    for read in reads:
        by_place.setdefault((read["shard"], read["ordinal"]), []).append(read)
        by_record.setdefault(read["record"], []).append(read)

    # 1. Documents split across samples.
    split = []
    for (shard, ordinal), place_reads in by_place.items():
        bodies = sorted({r["body"] for r in place_reads if not r["extra_token_only"]})
        starts_inside = any(r["token_start"] > 0 for r in place_reads)
        several_parts = any(
            not _overlap(a, b) for i, a in enumerate(bodies) for b in bodies[i + 1 :]
        )
        if starts_inside or several_parts:
            split.append(
                {
                    "record": place_reads[0]["record"],
                    "shard": shard,
                    "ordinal": ordinal,
                    # true: two samples that read different parts are both in the log
                    "several_parts_read": several_parts,
                    "reads": [where(r) for r in place_reads],
                }
            )

    # 2. Reads through the extra token only.
    extra_only = []
    for read in reads:
        if not read["extra_token_only"]:
            continue
        others = [
            r for r in by_place[(read["shard"], read["ordinal"])] if not r["extra_token_only"]
        ]
        extra_only.append(
            {
                "record": read["record"],
                "shard": read["shard"],
                "ordinal": read["ordinal"],
                **where(read),
                # false: in these logs the document is read through extra tokens only
                "document_read_otherwise": bool(others),
                "first_other_read": (
                    {"segment": others[0]["segment"], "step": others[0]["step"]} if others else None
                ),
            }
        )

    # 3. Records read in more than one step.
    repeated = []
    for record, record_reads in by_record.items():
        steps = []
        for r in record_reads:
            if (r["segment"], r["step"]) not in steps:
                steps.append((r["segment"], r["step"]))
        if len(steps) < 2:
            continue
        with_body = [r for r in record_reads if not r["extra_token_only"]]
        same_tokens = any(
            (a["segment"], a["step"]) != (b["segment"], b["step"]) and _overlap(a["body"], b["body"])
            for i, a in enumerate(with_body)
            for b in with_body[i + 1 :]
        )
        repeated.append(
            {
                "record": record,
                "places": sorted({(r["shard"], r["ordinal"]) for r in record_reads}),
                "steps": [list(s) for s in steps],
                # true: some token (extra tokens left aside) was read in two steps
                "same_tokens_read_again": same_tokens,
                "reads": [where(r) for r in record_reads],
            }
        )

    # 4. The same ordinal read in two shards.
    by_ordinal: dict[int, dict[str, list[dict]]] = {}
    for read in reads:
        by_ordinal.setdefault(read["ordinal"], {}).setdefault(read["shard"], []).append(read)
    same_ordinal = []
    for ordinal in sorted(by_ordinal):
        shards = by_ordinal[ordinal]
        if len(shards) < 2:
            continue
        same_ordinal.append(
            {
                "ordinal": ordinal,
                "shards": [
                    {
                        "shard": shard,
                        "record": shard_reads[0]["record"],
                        "first_read": {
                            "segment": shard_reads[0]["segment"],
                            "step": shard_reads[0]["step"],
                        },
                    }
                    for shard, shard_reads in shards.items()
                ],
            }
        )

    steps_out = []
    for sample in samples:
        segment, step, sample_pos = sample["key"]
        if not steps_out or (steps_out[-1]["segment"], steps_out[-1]["step"]) != (segment, step):
            steps_out.append({"segment": segment, "step": step, "samples": []})
        steps_out[-1]["samples"].append(
            {
                "sample_pos": sample_pos,
                "global_index": sample["global_index"],
                "rank": sample["rank"],
                "tokens": sum(r["token_end"] - r["token_start"] for r in sample["rows"]),
                "rows": sample["rows"],
            }
        )
    return {
        "format": REPORT_FORMAT,
        "extra_token": extra_token,
        "steps": steps_out,
        "flags": {
            "split_across_samples": split,
            "read_through_extra_token_only": extra_only,
            "read_in_more_than_one_step": repeated,
            "same_ordinal_in_two_shards": same_ordinal,
        },
    }


def render(report: dict) -> str:
    """The report as text for a person."""
    lines = [
        "Positions report. A row is record[token_start:token_end], inside the document.",
        "~ after a row: its last token is the sample's extra token. !extra-only: the row is that token alone.",
        "",
    ]
    for step in report["steps"]:
        lines.append(f"{step['segment']} step {step['step']}")
        for sample in step["samples"]:
            parts = []
            for row in sample["rows"]:
                text = f"{row['record']}[{row['token_start']}:{row['token_end']}]"
                if row["extra_token_only"]:
                    text += "!extra-only"
                elif row["carries_extra_token"]:
                    text += "~"
                parts.append(text)
            lines.append(
                f"  sample {sample['sample_pos']} (global {sample['global_index']}, rank "
                f"{sample['rank']}, {sample['tokens']} tokens): " + " ".join(parts)
            )

    def at(read: dict) -> str:
        return (
            f"[{read['token_start']}:{read['token_end']}] at {read['segment']} step "
            f"{read['step']} sample {read['sample_pos']}"
        )

    flags = report["flags"]
    lines += ["", "Flags", "", f"Documents split across samples: {len(flags['split_across_samples'])}"]
    for item in flags["split_across_samples"]:
        note = "" if item["several_parts_read"] else " (one part in the log)"
        lines.append(
            f"  {item['record']} ({item['shard']}, document {item['ordinal']}){note}: "
            + "; ".join(at(r) for r in item["reads"])
        )
    lines += ["", f"Reads through a sample's extra token only: {len(flags['read_through_extra_token_only'])}"]
    for item in flags["read_through_extra_token_only"]:
        if item["document_read_otherwise"]:
            other = item["first_other_read"]
            note = f"document also read at {other['segment']} step {other['step']}"
        else:
            note = "the document's only read in these logs"
        lines.append(f"  {item['record']} ({item['shard']}, document {item['ordinal']}) {at(item)}: {note}")
    lines += ["", f"Records read in more than one step: {len(flags['read_in_more_than_one_step'])}"]
    for item in flags["read_in_more_than_one_step"]:
        note = (
            "tokens read again"
            if item["same_tokens_read_again"]
            else "no token read again, extra tokens left aside"
        )
        lines.append(f"  {item['record']} ({note}): " + "; ".join(at(r) for r in item["reads"]))
    lines += ["", f"Ordinals read in two shards: {len(flags['same_ordinal_in_two_shards'])}"]
    for item in flags["same_ordinal_in_two_shards"]:
        lines.append(
            f"  document {item['ordinal']}: "
            + "; ".join(
                f"{s['shard']} {s['record']} (first at {s['first_read']['segment']} step "
                f"{s['first_read']['step']})"
                for s in item["shards"]
            )
        )
    return "\n".join(lines)


def work(config_path: str) -> dict:
    raw, base = load_json(config_path)
    unknown = sorted(set(raw) - {"logs", "segments", "id_field", "add_extra_token_to_sequence", "outputs"})
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in the config")
    logs = raw.get("logs")
    if not isinstance(logs, list) or not logs or not all(isinstance(p, str) for p in logs):
        raise ConfigError("'logs' must be a non-empty list of oracle log paths")
    segments_raw = raw.get("segments")
    if not isinstance(segments_raw, dict) or not segments_raw:
        raise ConfigError("'segments' must map each segment to its shards' input files, in order")
    id_field = raw.get("id_field", "id")
    extra_token = raw.get("add_extra_token_to_sequence", True)
    if not isinstance(extra_token, bool):
        raise ConfigError("'add_extra_token_to_sequence' must be true or false")
    outputs = raw.get("outputs", {})
    if not isinstance(outputs, dict) or set(outputs) - {"report"}:
        raise ConfigError("'outputs' may hold 'report' only")
    out = None
    if outputs.get("report") is not None:
        out = resolve_path(outputs["report"], base)
        new_output_path(out)

    segments: dict[str, list[str]] = {}
    ids: dict[str, list[str]] = {}
    for segment, files in segments_raw.items():
        if not isinstance(files, list) or not files or not all(isinstance(p, str) for p in files):
            raise ConfigError(f"segments[{segment!r}] must be a non-empty list of input files")
        labels = []
        for text in files:
            path = resolve_path(text, base)
            if not path.is_file():
                raise ConfigError(f"segments[{segment!r}]: not a file: {path}")
            label = Path(os.path.relpath(path.resolve(), base)).as_posix()
            if label not in ids:
                ids[label] = read_ids(path, id_field)
            labels.append(label)
        segments[segment] = labels
    log_paths = [resolve_path(text, base) for text in logs]
    for path in log_paths:
        if not path.is_file():
            raise ConfigError(f"log is not a file: {path}")
    report = build_report(read_log(log_paths), segments, ids, extra_token=extra_token)
    if out is not None:
        write_json(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv, render=render)


if __name__ == "__main__":
    sys.exit(main())
