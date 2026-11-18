"""Replay for Check 2: rebuild the index arrays of a segment from its settings and
seed into another new, empty folder and print their digests, with a comparison
against recorded digests when the config names a record. Config and output:
docs/oracle.md, "Replay".

A difference is a result, not a failure: the script exits 0 and says which arrays
differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.cli import run_tool  # noqa: E402
from oracle.config import (  # noqa: E402
    load_data_order,
    load_json,
    new_output_path,
    resolve_path,
    write_json,
)
from oracle.digests import compare_digests  # noqa: E402
from oracle.errors import ConfigError  # noqa: E402

REPLAY_FORMAT = "waterwhite.oracle-replay/1"


def work(config_path: str) -> dict:
    data, raw, base = load_data_order(config_path)
    recorded = None
    recorded_name = None
    if raw.get("compare_with") is not None:
        if not isinstance(raw["compare_with"], str):
            raise ConfigError("'compare_with' must be the path of a run record or a digests file")
        compare_path = resolve_path(raw["compare_with"], base)
        document, _ = load_json(compare_path)
        recorded = document.get("cache_digests")
        if not isinstance(recorded, list):
            raise ConfigError(f"{compare_path.name} holds no 'cache_digests' list")
        recorded_name = compare_path.name
    out = None
    outputs = raw.get("outputs", {})
    if isinstance(outputs, dict) and outputs.get("digests") is not None:
        out = resolve_path(outputs["digests"], base)
        new_output_path(out)

    from oracle import pipeline  # imports Megatron

    _built, document = pipeline.build_index(data)
    result = {
        "format": REPLAY_FORMAT,
        "settings": document["settings"],
        "dataset_samples": document["dataset_samples"],
        "cache_digests": document["cache_digests"],
        "comparison": None,
    }
    if recorded is not None:
        comparison = compare_digests(document["cache_digests"], recorded)
        comparison["with"] = recorded_name
        result["comparison"] = comparison
    if out is not None:
        write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv)


if __name__ == "__main__":
    sys.exit(main())
