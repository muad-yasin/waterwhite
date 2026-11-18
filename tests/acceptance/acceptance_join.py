"""Check 1's comparison, as pure functions: the rows the join reports for a segment
against the oracle's log of that segment (plan 5.5 Check 1, 5.3; docs/interfaces.md
section 8.3).

Nothing here starts a process or reads a file, so the comparison can be shown to
fail on rows written by hand (``test_join.py``). A row is compared by what the plan
names: step, shard, ordinal, token start and token end, and the place of its sample
in the step (``sample_pos``), in reading order. The oracle's other fields
(``global_index``, ``rank``) are not part of the question put to the join.

The one cut the plan makes is the half-covered sample of a lowered step (plan 3.5,
5.1): the oracle's log is not changed by a lowering, and the join reports the rows of
the covered part only, the row that holds the first uncovered token ending at the
cut. ``cut_sample`` applies the spec's rule to the oracle's rows to give the rows the
join must report for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# What the join reports for one row: (shard position, ordinal, token start, token end).
Span = tuple[int, int, int, int]
Sample = tuple[str, int, int]  # (segment, step, sample_pos)

FIELDS = ("shard_position", "ordinal", "token_start", "token_end")


def span(row: Mapping) -> Span:
    return tuple(int(row[name]) for name in FIELDS)  # type: ignore[return-value]


def group(rows: Iterable[Mapping]) -> dict[Sample, list[Span]]:
    """The rows of each sample, in the order they were given (reading order)."""
    out: dict[Sample, list[Span]] = {}
    for row in rows:
        out.setdefault((row["segment"], int(row["step"]), int(row["sample_pos"])), []).append(
            span(row))
    return out


def rows_naming(reported: Iterable[Mapping], shard_position: int, ordinal: int, *,
                segments: Sequence[str] | None = None, steps: range | None = None) -> list[Mapping]:
    """The reported rows of one document (shard position and ordinal), among the
    segments and steps given (all when None)."""
    return [row for row in reported
            if int(row["shard_position"]) == shard_position and int(row["ordinal"]) == ordinal
            and (segments is None or row["segment"] in segments)
            and (steps is None or int(row["step"]) in steps)]


def cut_sample(spans: Sequence[Span], covered_tokens: int) -> list[Span]:
    """The rows of a sample the join reports when only its first ``covered_tokens``
    tokens were read: the rows that end within them, the row that holds the next token
    ending at the cut, no row after it (docs/interfaces.md 8.2, plan 3.5)."""
    out: list[Span] = []
    left = covered_tokens
    for shard, ordinal, start, end in spans:
        if left <= 0:
            break
        length = end - start
        if length <= left:
            out.append((shard, ordinal, start, end))
            left -= length
        else:
            out.append((shard, ordinal, start, start + left))
            left = 0
    return out


def expected_rows(oracle: Iterable[Mapping], cuts: Mapping[Sample, int] | None = None
                  ) -> dict[Sample, list[Span]]:
    """The rows the join must report: the oracle's, with the rows of each lowered
    sample cut (``cuts`` maps the sample to the tokens read)."""
    out = group(oracle)
    for sample, covered in (cuts or {}).items():
        if sample not in out:
            raise ValueError(f"the oracle's log holds no sample {sample} to cut")
        out[sample] = cut_sample(out[sample], covered)
    return out


def diff(reported: Iterable[Mapping], oracle: Iterable[Mapping],
         cuts: Mapping[Sample, int] | None = None, *, within: Sample | None = None
         ) -> list[str]:
    """The findings where the join's rows differ from the oracle's: a sample the join
    does not report, a sample it reports that the oracle's log has none of, and for a
    sample both hold the first row that differs. The empty list is "the diff is
    empty". ``within`` restricts the comparison to the samples that match a (segment,
    step, sample_pos) prefix given as a tuple, None standing for any."""
    wanted = expected_rows(oracle, cuts)
    got = group(reported)

    def selected(sample: Sample) -> bool:
        return within is None or all(w is None or w == s for w, s in zip(within, sample))

    findings = []
    for sample in sorted(wanted):
        if not selected(sample):
            continue
        if sample not in got:
            findings.append(f"{sample}: the join reports no row; the oracle's log has "
                            f"{len(wanted[sample])}")
        elif got[sample] != wanted[sample]:
            at = next((n for n, (a, b) in enumerate(zip(got[sample], wanted[sample]))
                       if a != b), min(len(got[sample]), len(wanted[sample])))
            findings.append(f"{sample}: row {at} differs: the join reports "
                            f"{got[sample][at:at + 1] or 'none'}, the oracle's log has "
                            f"{wanted[sample][at:at + 1] or 'none'} "
                            f"({len(got[sample])} rows against {len(wanted[sample])})")
    findings += [f"{sample}: the join reports rows the oracle's log has none of"
                 for sample in sorted(got) if sample not in wanted and selected(sample)]
    return findings


def rows_named(lines: Iterable[Mapping | str], reported: Iterable[Mapping]) -> list[str]:
    """The rows of a position the spec states (its ``log_lines``, each a row of the
    oracle's log) that the join does not report, as the sample's rows: every line must
    be among the rows of its own sample."""
    import json

    got = group(reported)
    missing = []
    for line in lines:
        row = json.loads(line) if isinstance(line, str) else line
        sample = (row["segment"], int(row["step"]), int(row["sample_pos"]))
        if span(row) not in got.get(sample, []):
            missing.append(f"{sample}: no row {span(row)} reported")
    return missing
