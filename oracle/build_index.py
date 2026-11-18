"""Index build: build Megatron's datasets for one run segment into a new, empty
cache folder and write the digest of each index array, keyed by kind and shard
position. Config and output: docs/oracle.md, "Index build".
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.cli import run_tool  # noqa: E402
from oracle.config import load_data_order, new_output_path, resolve_path, write_json  # noqa: E402
from oracle.errors import ConfigError  # noqa: E402


def work(config_path: str) -> dict:
    data, raw, base = load_data_order(config_path)
    outputs = raw.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("digests"), str):
        raise ConfigError("missing key 'outputs.digests'")
    out = resolve_path(outputs["digests"], base)
    new_output_path(out)

    from oracle import pipeline  # imports Megatron

    _built, document = pipeline.build_index(data)
    write_json(out, document)
    return document


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv)


if __name__ == "__main__":
    sys.exit(main())
