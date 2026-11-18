"""Reading the oracle's log by sample: grouping its rows, and cutting a sample
after a number of tokens.

A log row is one read of one document (docs/interfaces.md 8.3). The rows of one
sample are consecutive in the log, in the order of the reads, and their token
counts add up to the sample's ``sequence_length + 1`` tokens.

Standard library only.
"""

from __future__ import annotations

from oracle.errors import ConfigError

# The fields of one log line, in the order docs/interfaces.md 8.3 shows them.
LOG_FIELDS = (
    "segment",
    "step",
    "sample_pos",
    "global_index",
    "rank",
    "shard_position",
    "ordinal",
    "token_start",
    "token_end",
)


def samples_of(rows: list[dict]) -> list[dict]:
    """Group log rows into samples, in the order of the log.

    A sample is ``{"segment", "step", "sample_pos", "global_index", "rank",
    "rows"}``. Rows of one sample that are not consecutive, or that disagree on
    the global index or the rank, are refused: the log would not be the oracle's."""
    samples: list[dict] = []
    seen = set()
    for row in rows:
        key = (row["segment"], row["step"], row["sample_pos"])
        if not samples or (samples[-1]["segment"], samples[-1]["step"], samples[-1]["sample_pos"]) != key:
            if key in seen:
                raise ConfigError(f"the rows of sample {key} are not consecutive in the log")
            seen.add(key)
            samples.append(
                {
                    "segment": row["segment"],
                    "step": row["step"],
                    "sample_pos": row["sample_pos"],
                    "global_index": row["global_index"],
                    "rank": row["rank"],
                    "rows": [],
                }
            )
        sample = samples[-1]
        if (row["global_index"], row["rank"]) != (sample["global_index"], sample["rank"]):
            raise ConfigError(f"sample {key} has rows with two global indices or ranks")
        sample["rows"].append(row)
    return samples


def row_tokens(row: dict) -> int:
    return row["token_end"] - row["token_start"]


def sample_tokens(sample: dict) -> int:
    """The number of tokens a sample's rows cover."""
    return sum(row_tokens(row) for row in sample["rows"])


def cut_sample(sample: dict, covered: int) -> dict:
    """Cut a sample after its first ``covered`` tokens (plan 3.5: a step record
    lowers the coverage of a step's last sample).

    Returns

    - ``covered_rows``: the rows of the covered part. The rows before the cut
      stay as they are; the row the cut falls in ends at the cut; no row after it
      is kept;
    - ``cut``: where the uncovered range starts, in document terms: the shard
      position, the ordinal and the token inside the document;
    - ``starts_a_row``: true when the first uncovered token is the first token of
      a row, so that no row had to be cut;
    - ``sample_tokens``: the tokens of the whole sample.

    ``covered`` must leave at least one token uncovered; a sample read whole has
    no cut."""
    total = sample_tokens(sample)
    if isinstance(covered, bool) or not isinstance(covered, int) or not 0 <= covered < total:
        raise ConfigError(
            f"cannot cut a sample of {total} tokens after {covered!r}: the count must be "
            "from 0 to one less than the sample's tokens"
        )
    kept: list[dict] = []
    before = 0
    for row in sample["rows"]:
        length = row_tokens(row)
        if before + length <= covered:
            kept.append(dict(row))
            before += length
            continue
        inside = covered - before  # tokens of this row that are covered
        if inside > 0:
            kept.append({**row, "token_end": row["token_start"] + inside})
        return {
            "covered_rows": kept,
            "cut": {
                "shard_position": row["shard_position"],
                "ordinal": row["ordinal"],
                "token": row["token_start"] + inside,
            },
            "starts_a_row": inside == 0,
            "sample_tokens": total,
        }
    raise ConfigError("the sample's rows end before the cut")  # not reached: covered < total
