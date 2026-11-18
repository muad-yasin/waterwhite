"""The numbered conditions of the fixture spec's seed search (plan 5.1), each as one
small predicate over the oracle's output and the replica's manifests.

The spec's ``[search]`` table states every read position the fixture must hold as
a numbered sentence. ``c01`` to ``c23`` below check those sentences, one function
per number. A predicate reads

- the oracle's log rows of a run (docs/interfaces.md 8.3), with the record each
  row names: line i of a shard's tokenizer-input manifest is document i of the
  shard, so a row's shard position and ordinal name one record;
- the oracle's summary of a segment (the dataset's length, the launches, the
  passes each rank's sampler made) and the order digests of its run record;
- the step record of a segment;
- the replica's manifests and its sanitizer log.

It reads nothing of the tool. It returns an ``Outcome``: whether the sentence
holds, one line that says why or why not, the log lines that show the position,
and the values a person copies into the spec's ``[positions]`` tables.

The numbers the sentences state (step 7, sample position 3, line 6) are written
here as the sentences write them. ``check_spec_conditions`` compares the spec's
sentences with the ones these predicates were written against, so a reworded
sentence stops the search until its predicate has been read again.

Standard library only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from oracle import log_rows
from oracle.digests import order_digest
from oracle.errors import ConfigError
from oracle.fixture_spec import LAYOUT_ROUND_ROBIN, FixtureSpec, shard_ref

# --- What a predicate is handed -------------------------------------------------------------


@dataclass(frozen=True)
class ShardEvidence:
    """One shard of a built fixture, from the replica's files."""

    name: str  # as the spec names it: "shard-1", "E-rr/shard-1"
    layout: str
    file: str
    records: tuple[str, ...]  # the record id of each tokenizer-input line; line i is document i
    manifest: tuple[dict, ...]  # the manifest rows of the tokenizer input
    skipped: tuple[int, ...]  # places (from 1) of the lines the sanitizer skipped in its input

    def input_line(self, ordinal: int) -> int:
        """The line (from 1) a document had in the sanitizer's input: its place
        among all lines the sanitizer read, the skipped ones counted."""
        total = len(self.records) + len(self.skipped)
        surviving = [place for place in range(1, total + 1) if place not in self.skipped]
        if len(surviving) != len(self.records):
            raise ConfigError(f"shard {self.name!r}: the sanitizer log names a place past its input")
        return surviving[ordinal]


@dataclass(frozen=True)
class BuildEvidence:
    """What the replica wrote for one build, for both layouts."""

    shards: dict[tuple[str, str], ShardEvidence]  # (layout, file) -> shard
    affected: frozenset[str]  # the record ids "withdraw A" puts in scope

    def by_name(self, name: str) -> ShardEvidence:
        ref = shard_ref(name, "a shard name")
        return self.shard(ref.layout, ref.file)

    def shard(self, layout: str, file: str) -> ShardEvidence:
        if (layout, file) not in self.shards:
            raise ConfigError(f"the built fixture has no shard {file!r} in the {layout} layout")
        return self.shards[(layout, file)]


@dataclass
class SegmentEvidence:
    """What the oracle wrote, or read in memory, for one run segment."""

    name: str
    shards: tuple[str, ...]  # the shard names, by shard position
    rows: list[dict]  # the log rows, in order
    summary: dict  # the oracle's summary of the segment
    shuffle_digests: list[dict]  # as the run record holds them
    step_record: dict  # as written for the segment
    shard_documents: tuple[int, ...]  # documents Megatron reports for each shard position


@dataclass
class Evidence:
    spec: FixtureSpec
    build: BuildEvidence
    runs: dict[str, list[SegmentEvidence]]  # run -> its segments, in order
    _reads: dict[str, list[dict]] = field(default_factory=dict)
    _samples: dict[str, list[dict]] = field(default_factory=dict)

    def segments(self, run: str) -> list[SegmentEvidence]:
        if run not in self.runs:
            raise ConfigError(f"no oracle output for run {run!r} was given")
        return self.runs[run]

    def segment(self, name: str) -> SegmentEvidence:
        for segments in self.runs.values():
            for segment in segments:
                if segment.name == name:
                    return segment
        raise ConfigError(f"no oracle output for segment {name!r} was given")

    def reads(self, run: str) -> list[dict]:
        """Every log row of a run, each with the shard and the record it names
        (keys ``shard`` and ``record``), in the order of the logs."""
        if run not in self._reads:
            out = []
            for segment in self.segments(run):
                shards = [self.build.by_name(name) for name in segment.shards]
                for row in segment.rows:
                    if row["segment"] != segment.name:
                        raise ConfigError(f"a row of segment {row['segment']!r} in the log of {segment.name!r}")
                    if row["shard_position"] >= len(shards):
                        raise ConfigError(
                            f"{segment.name}: a row names shard position {row['shard_position']}, "
                            f"the segment lists {len(shards)} shards"
                        )
                    shard = shards[row["shard_position"]]
                    if row["ordinal"] >= len(shard.records):
                        raise ConfigError(
                            f"{segment.name}: a row names document {row['ordinal']} of "
                            f"{shard.name!r}, whose input has {len(shard.records)} lines"
                        )
                    out.append({**row, "shard": shard.name, "record": shard.records[row["ordinal"]]})
            self._reads[run] = out
        return self._reads[run]

    def samples(self, run: str) -> list[dict]:
        """The samples of a run, in the order of the logs (``log_rows.samples_of``)."""
        if run not in self._samples:
            self._samples[run] = log_rows.samples_of(self.reads(run))
        return self._samples[run]


@dataclass
class Outcome:
    holds: bool
    detail: str  # one line: what was found
    lines: list[dict] = field(default_factory=list)  # the log lines that show the position
    facts: dict = field(default_factory=dict)  # values for the spec's [positions] tables


def line_of(row: dict) -> dict:
    """A row as the log file holds it, without the names the evidence added."""
    return {key: row[key] for key in log_rows.LOG_FIELDS}


def _lines(rows) -> list[dict]:
    return [line_of(row) for row in rows]


# --- Names and numbers the sentences state --------------------------------------------------

RUN_E, RUN_E_RR, RUN_D, RUN_Y, RUN_K = "E", "E-rr", "D", "Y", "K"
E_SEG_1, E_SEG_2 = "E/seg-1", "E/seg-2"
E_RR_SEG_1, E_RR_SEG_2 = "E-rr/seg-1", "E-rr/seg-2"
D_SEG, Y_SEG, K_SEG = "D/seg-1", "Y/seg-1", "K/seg-1"
SHARD_1, SHARD_2, SHARD_3 = "shard-1", "shard-2", "shard-3"
BATCH = 4  # the global batch size of every run of the fixture
SKIP_LINE = 6  # the place of the skipped line in the sanitizer's input of shard-2
SKIP_SHARD_DOCUMENTS = 19
LOWERED_STEP = 8  # the step whose last sample the step record lowers
LAST_SAMPLE = BATCH - 1  # the sample position of a step's last sample
T_STEP = 7  # the step of segment 2 that reads T
T_TOKENS = 22  # T's document: 21 tokens of text and the end-of-document token
UNRELATED_SOURCE = "Z"  # the source of the unrelated records
# Where the shared-token document W of condition 12 sits, and whose record it is
# (DECISIONS.md D-13 as amended by D-24): an unrelated record in shard-3.
SHARED_TOKEN_SHARD = SHARD_3
SHARED_TOKEN_STEPS = (7, 8, 9)


def _t(ev: Evidence) -> str:
    return ev.spec.handles["T"]


def _sizes(ev: Evidence, segment: str, steps: range, per_step: int, indices: range | None) -> list[str]:
    """What is wrong with a segment's log when it should hold ``per_step`` samples
    in each of ``steps`` (and, when given, exactly the global indices
    ``indices``, each once). An empty list: nothing."""
    samples = log_rows.samples_of(ev.segment(segment).rows)
    problems = []
    expected = len(steps) * per_step
    if len(samples) != expected:
        problems.append(f"{segment} holds {len(samples)} samples, not {expected}")
    by_step: dict[int, list[int]] = {}
    for sample in samples:
        by_step.setdefault(sample["step"], []).append(sample["sample_pos"])
    if sorted(by_step) != list(steps):
        problems.append(f"{segment} holds steps {sorted(by_step)}, not {steps.start} to {steps.stop - 1}")
    short = sorted(step for step, places in by_step.items() if sorted(places) != list(range(per_step)))
    if short:
        problems.append(f"{segment}: steps {short} do not hold sample positions 0 to {per_step - 1}")
    if indices is not None and sorted(s["global_index"] for s in samples) != list(indices):
        problems.append(f"{segment} does not read global indices {indices.start} to {indices.stop - 1}, each once")
    return problems


def _lowered_steps(ev: Evidence, run: str) -> list[tuple[str, int]]:
    return [
        (segment.name, step["step"])
        for segment in ev.segments(run)
        for step in segment.step_record["steps"]
        if "last_sample_tokens_read" in step
    ]


def _sample_at(ev: Evidence, run: str, segment: str, step: int, sample_pos: int) -> dict | None:
    for sample in ev.samples(run):
        if (sample["segment"], sample["step"], sample["sample_pos"]) == (segment, step, sample_pos):
            return sample
    return None


def _place(row: dict) -> dict:
    return {"segment": row["segment"], "step": row["step"], "sample_pos": row["sample_pos"]}


def _span(row: dict) -> dict:
    return {**_place(row), "token_start": row["token_start"], "token_end": row["token_end"]}


# --- Run E -------------------------------------------------------------------------------------


def c01(ev: Evidence) -> Outcome:
    """Run E reads full batches in both segments. The log of segment 1 holds 24
    samples, four in each of steps 1 to 6, with the global indices 0 to 23 each
    once; the log of segment 2 holds 24 samples, four in each of steps 7 to 12,
    with the global indices 24 to 47 each once."""
    problems = _sizes(ev, E_SEG_1, range(1, 7), BATCH, range(0, 24))
    problems += _sizes(ev, E_SEG_2, range(7, 13), BATCH, range(24, 48))
    if problems:
        return Outcome(False, "; ".join(problems))
    return Outcome(True, "24 samples in steps 1 to 6 (indices 0 to 23) and 24 in steps 7 to 12 (24 to 47)")


def c02(ev: Evidence) -> Outcome:
    """The skipped line shifts no ordinal. Shard-2 of the partitioned layout
    holds 19 documents, both by its tokenizer input's lines and by the count
    Megatron reports for the shard in every segment of run E, and the sanitizer's
    log lists one skipped line for the shard, at place 6 of its input."""
    shard = ev.build.by_name(SHARD_2)
    counts = sorted(
        {
            segment.shard_documents[segment.shards.index(SHARD_2)]
            for segment in ev.segments(RUN_E)
            if SHARD_2 in segment.shards
        }
    )
    facts = {"documents": len(shard.records), "skipped_lines_logged": list(shard.skipped)}
    problems = []
    if len(shard.records) != SKIP_SHARD_DOCUMENTS or counts != [SKIP_SHARD_DOCUMENTS]:
        problems.append(
            f"shard-2 has {len(shard.records)} input lines and Megatron reports {counts} documents, "
            f"not {SKIP_SHARD_DOCUMENTS}"
        )
    if list(shard.skipped) != [SKIP_LINE]:
        problems.append(f"the sanitizer log lists skipped places {list(shard.skipped)} for shard-2, not [{SKIP_LINE}]")
    if problems:
        return Outcome(False, "; ".join(problems), facts=facts)
    return Outcome(True, f"shard-2 holds {SKIP_SHARD_DOCUMENTS} documents and one skipped line, at place {SKIP_LINE}", facts=facts)


def c03(ev: Evidence) -> Outcome:
    """Run E reads a document that sat behind the skipped line. Some row of run E
    names a document of shard-2 whose place in the sanitizer's input is after
    place 6. The place of document i is the place of the (i + 1)-th line the
    sanitizer kept, counted over kept and skipped lines alike."""
    shard = ev.build.by_name(SHARD_2)
    for row in ev.reads(RUN_E):
        if row["shard"] == SHARD_2 and shard.input_line(row["ordinal"]) > SKIP_LINE:
            facts = {
                **_place(row),
                "ordinal": row["ordinal"],
                "record": row["record"],
                "sanitizer_input_line": shard.input_line(row["ordinal"]),
            }
            return Outcome(
                True,
                f"step {row['step']} reads document {row['ordinal']} of shard-2, place "
                f"{facts['sanitizer_input_line']} of the sanitizer's input",
                [line_of(row)],
                facts,
            )
    return Outcome(False, f"no row of run E names a document of shard-2 from behind place {SKIP_LINE}")


def _neighbour(row: dict) -> dict:
    return {
        "record": row["record"],
        "shard_position": row["shard_position"],
        "ordinal": row["ordinal"],
        "token_start": row["token_start"],
        "token_end": row["token_end"],
    }


def c04(ev: Evidence) -> Outcome:
    """One sample of step 7 holds T whole, as its second document. In segment 2
    some sample of step 7 has

    - as its first row a document of T's shard that is an unrelated record of Z
      (the end of the document before T in the shard's packed order);
    - as its second row T's document from token 0 to token 22, which is all of
      T: its 21 tokens and the end-of-document token;
    - a third row after it. Megatron's item getter reads every document of a
      sample but the last one to its end, so a row that another row follows
      ends where its document ends, and no token of T is left for another
      sample.

    T has no other row in the log of run E. And the step records of run E lower
    step 8 and no other step, so step 7 is traced whole. Every row of run E that
    names T is listed with the outcome."""
    target = _t(ev)
    lowered = _lowered_steps(ev, RUN_E)
    rows_of_t = [_span(row) for row in ev.reads(RUN_E) if row["record"] == target]
    found = None
    for sample in ev.samples(RUN_E):
        if sample["segment"] != E_SEG_2 or sample["step"] != T_STEP or len(sample["rows"]) < 2:
            continue
        if sample["rows"][1]["record"] == target:
            found = sample
            break
    if found is None:
        return Outcome(
            False,
            f"no sample of step {T_STEP} has T as its second row; run E reads T in {len(rows_of_t)} "
            "rows, the first at " + (f"step {rows_of_t[0]['step']}" if rows_of_t else "no step"),
            facts={"rows_of_T_in_run_E": rows_of_t},
        )
    first, second = found["rows"][0], found["rows"][1]
    facts = {
        "sample_pos": found["sample_pos"],
        "global_index": found["global_index"],
        "shard_position": second["shard_position"],
        "ordinal": second["ordinal"],
        "token_end": second["token_end"],
        "first_row": _neighbour(first),
        "neighbours": [_neighbour(row) for i, row in enumerate(found["rows"]) if i != 1],
        "rows_of_T_in_run_E": rows_of_t,
    }
    where = f"sample {found['sample_pos']} of step {T_STEP}"
    problems = []
    if (second["token_start"], second["token_end"]) != (0, T_TOKENS):
        problems.append(
            f"the row of T in {where} covers tokens {second['token_start']} to {second['token_end']}, "
            f"not 0 to {T_TOKENS}"
        )
    if len(found["rows"]) < 3:
        problems.append(f"no third row follows T in {where}, so T may go on in another sample")
    if first["shard"] != second["shard"] or ev.spec.source_of(first["record"]) != UNRELATED_SOURCE:
        problems.append(
            f"the first row of {where} is {first['record']} on shard position {first['shard_position']}, "
            f"not a record of {UNRELATED_SOURCE} in T's shard"
        )
    if len(rows_of_t) != 1:
        problems.append(f"run E reads T in {len(rows_of_t)} rows, not in one")
    if lowered != [(E_SEG_2, LOWERED_STEP)]:
        problems.append(f"the step records lower {lowered}, not step {LOWERED_STEP} of {E_SEG_2} alone")
    if problems:
        return Outcome(False, "; ".join(problems), _lines(found["rows"]), facts)
    return Outcome(
        True,
        f"{where} reads the end of {first['record']}, then T whole as its second row "
        f"(tokens 0 to {T_TOKENS}), then {found['rows'][2]['record']}; T has no other row in run E; "
        f"only step {LOWERED_STEP} is lowered",
        _lines(found["rows"]),
        facts,
    )


def c05(ev: Evidence) -> Outcome:
    """No exposure before step 7. No row of steps 1 to 6 of run E names shard
    position 2, and no row of those steps names a record that "withdraw A" puts
    in scope (T is the only such record in the shards of run E)."""
    # The shard positions are read off the rows as they are, before any row is
    # tied to a record: segment 1 has no third shard a row could be tied to.
    raw = [row for segment in ev.segments(RUN_E) for row in segment.rows if row["step"] <= 6]
    on_third = [row for row in raw if row["shard_position"] == 2]
    if on_third:
        return Outcome(
            False,
            f"steps 1 to 6 have {len(on_third)} rows on shard position 2, the first at step {on_third[0]['step']}",
            _lines(on_third[:1]),
            {"rows_in_steps_1_to_6": len(raw), "rows_in_steps_1_to_6_on_shard_position_2": len(on_third)},
        )
    early = [row for row in ev.reads(RUN_E) if row["step"] <= 6]
    affected = [row for row in early if row["record"] in ev.build.affected]
    facts = {"rows_in_steps_1_to_6": len(early), "rows_in_steps_1_to_6_on_shard_position_2": 0}
    if affected:
        return Outcome(
            False,
            f"steps 1 to 6 have {len(affected)} rows that name an affected record, the first "
            f"({affected[0]['record']}) at step {affected[0]['step']}",
            _lines(affected[:1]),
            facts,
        )
    return Outcome(True, f"none of the {len(early)} rows of steps 1 to 6 is on shard position 2 or names an affected record", facts=facts)


def c06(ev: Evidence) -> Outcome:
    """The half-covered sample holds no token of T. The last sample of step 8 of
    run E, the one at sample position 3, has no row that names T."""
    sample = _sample_at(ev, RUN_E, E_SEG_2, LOWERED_STEP, LAST_SAMPLE)
    if sample is None:
        return Outcome(False, f"the log holds no sample {LAST_SAMPLE} of step {LOWERED_STEP}")
    records = [row["record"] for row in sample["rows"]]
    facts = {"records_of_the_last_sample_of_step_8": records}
    if _t(ev) in records:
        return Outcome(False, "the last sample of step 8 has a row that names T", _lines(sample["rows"]), facts)
    return Outcome(True, f"the last sample of step 8 reads {records}, not T", _lines(sample["rows"]), facts)


def c07(ev: Evidence) -> Outcome:
    """The cut of step 8 falls inside a row. The step record gives the last
    sample of step 8 a number of tokens read (16). The token that follows them,
    the sample's 17th, is not the first token of one of the sample's rows, so
    the covered part ends with a row that is cut. The outcome carries the cut:
    the shard position, ordinal and document token where the uncovered range
    starts, and the rows of the covered part."""
    sample = _sample_at(ev, RUN_E, E_SEG_2, LOWERED_STEP, LAST_SAMPLE)
    if sample is None:
        return Outcome(False, f"the log holds no sample {LAST_SAMPLE} of step {LOWERED_STEP}")
    covered = [
        step.get("last_sample_tokens_read")
        for step in ev.segment(E_SEG_2).step_record["steps"]
        if step["step"] == LOWERED_STEP
    ]
    if len(covered) != 1 or covered[0] is None:
        return Outcome(False, f"the step record of {E_SEG_2} does not lower step {LOWERED_STEP}")
    cut = log_rows.cut_sample(sample, covered[0])
    facts = {
        "covered_tokens": covered[0],
        "sample_tokens": cut["sample_tokens"],
        **cut["cut"],
        "rows_of_the_covered_part": [
            {key: row[key] for key in ("shard_position", "ordinal", "token_start", "token_end")}
            for row in cut["covered_rows"]
        ],
    }
    where = f"document {cut['cut']['ordinal']} of shard position {cut['cut']['shard_position']}, token {cut['cut']['token']}"
    if cut["starts_a_row"]:
        return Outcome(False, f"token {covered[0] + 1} of the sample is the first token of a row ({where})", _lines(sample["rows"]), facts)
    return Outcome(True, f"the uncovered range starts inside a row, at {where}", _lines(sample["rows"]), facts)


def c08(ev: Evidence) -> Outcome:
    """A repeated draw. Some document of shard-1 or shard-2 has a row in a step
    from 1 to 6 and a row in a step from 7 to 12 of run E. Which pair is shown:
    first a pair whose two rows read the same token range (the same part of the
    document drawn again), then a pair whose ranges share a token, then any
    other; inside each kind the lowest shard, ordinal and steps."""
    early: dict[tuple[str, int], list[dict]] = {}
    late: dict[tuple[str, int], list[dict]] = {}
    for row in ev.reads(RUN_E):
        if row["shard"] in (SHARD_1, SHARD_2):
            (early if row["step"] <= 6 else late).setdefault((row["shard"], row["ordinal"]), []).append(row)
    pairs = []
    for key in sorted(set(early) & set(late)):
        for first in early[key]:
            for second in late[key]:
                if (first["token_start"], first["token_end"]) == (second["token_start"], second["token_end"]):
                    kind = 0
                elif first["token_start"] < second["token_end"] and second["token_start"] < first["token_end"]:
                    kind = 1
                else:
                    kind = 2
                pairs.append((kind, key, first, second))
    if not pairs:
        return Outcome(False, "no document of shard-1 or shard-2 is read in steps 1 to 6 and again in steps 7 to 12")
    pairs.sort(key=lambda pair: (pair[0], pair[1], pair[2]["step"], pair[3]["step"]))
    kind, _key, first, second = pairs[0]
    facts = {
        "shard_position": first["shard_position"],
        "ordinal": first["ordinal"],
        "record": first["record"],
        "first_read": _span(first),
        "second_read": _span(second),
        "same_range_read_again": kind == 0,
        "same_tokens_read_again": kind <= 1,
    }
    return Outcome(
        True,
        f"{first['record']} (shard position {first['shard_position']}, document {first['ordinal']}) is "
        f"read at step {first['step']} and again at step {second['step']}",
        _lines([first, second]),
        facts,
    )


def c09(ev: Evidence) -> Outcome:
    """The same ordinal in two shards. Some document number has rows on two
    different shards in the log of run E. The lowest such number is shown."""
    by_ordinal: dict[int, dict[str, dict]] = {}
    for row in ev.reads(RUN_E):
        by_ordinal.setdefault(row["ordinal"], {}).setdefault(row["shard"], row)
    for ordinal in sorted(by_ordinal):
        shards = by_ordinal[ordinal]
        if len(shards) < 2:
            continue
        rows = sorted(shards.values(), key=lambda row: row["shard_position"])[:2]
        facts = {
            "ordinal": ordinal,
            "shard_positions": [row["shard_position"] for row in rows],
            "reads": [{"shard_position": row["shard_position"], "record": row["record"], **_place(row)} for row in rows],
        }
        return Outcome(
            True,
            f"document {ordinal} is read on shard positions {facts['shard_positions']}",
            _lines(rows),
            facts,
        )
    return Outcome(False, "no ordinal is read on two shards")


def _shared_token_documents(ev: Evidence) -> list[dict]:
    """Every document that fits condition 12, as the row that reads it, in the
    order of the log. See ``c12``."""
    target = _t(ev)
    by_document: dict[tuple[str, int], list[dict]] = {}
    # The last row of each sample that holds no row of T, by the row object's identity.
    last_rows = set()
    for sample in ev.samples(RUN_E):
        if all(row["record"] != target for row in sample["rows"]):
            last_rows.add(id(sample["rows"][-1]))
        for row in sample["rows"]:
            if row["step"] <= 9:
                by_document.setdefault((row["shard"], row["ordinal"]), []).append(row)
    found = []
    for (shard, _ordinal), rows in by_document.items():
        if shard != SHARED_TOKEN_SHARD or len(rows) != 1:
            continue
        row = rows[0]
        if (
            id(row) in last_rows
            and row["step"] in SHARED_TOKEN_STEPS
            and not (row["step"] == LOWERED_STEP and row["sample_pos"] == LAST_SAMPLE)
            and (row["token_start"], row["token_end"]) == (0, 1)
            and ev.spec.source_of(row["record"]) == UNRELATED_SOURCE
        ):
            found.append(row)
    return found


def c10(ev: Evidence) -> Outcome:
    """A document split across two samples. Some document that is neither T nor
    a shared-token document of condition 12 has two rows in two different
    samples of run E, each two tokens long or more, where the second row starts
    at the last token the first one read (its start is the first row's end
    minus one)."""
    excluded = {(row["shard"], row["ordinal"]) for row in _shared_token_documents(ev)}
    target = _t(ev)
    by_document: dict[tuple[str, int], list[dict]] = {}
    for row in ev.reads(RUN_E):
        if row["record"] != target and (row["shard"], row["ordinal"]) not in excluded and log_rows.row_tokens(row) >= 2:
            by_document.setdefault((row["shard"], row["ordinal"]), []).append(row)
    for rows in by_document.values():
        for first in rows:
            for second in rows:
                if _place(first) != _place(second) and second["token_start"] == first["token_end"] - 1:
                    facts = {
                        "shard_position": first["shard_position"],
                        "ordinal": first["ordinal"],
                        "record": first["record"],
                        "rows": [_span(first), _span(second)],
                    }
                    return Outcome(
                        True,
                        f"{first['record']} is read as [{first['token_start']}:{first['token_end']}] at step "
                        f"{first['step']} and as [{second['token_start']}:{second['token_end']}] at step {second['step']}",
                        _lines([first, second]),
                        facts,
                    )
    return Outcome(False, "no document other than T and W is read in two samples that share a token")


def c11(ev: Evidence) -> Outcome:
    """Segment 1 reads both of its shards. Steps 1 to 6 of run E have a row on
    shard position 0 and a row on shard position 1."""
    first: dict[int, dict] = {}
    for row in ev.reads(RUN_E):
        if row["step"] <= 6:
            first.setdefault(row["shard_position"], row)
    missing = [position for position in (0, 1) if position not in first]
    if missing:
        return Outcome(False, f"steps 1 to 6 have no row on shard position(s) {missing}")
    return Outcome(True, "steps 1 to 6 read shard positions 0 and 1", _lines([first[0], first[1]]))


def c12(ev: Evidence) -> Outcome:
    """A document read only through a sample's shared last token (DECISIONS.md
    D-13 as amended by D-24). Some document W of shard-3, an unrelated record
    of Z, has exactly one row in steps 1 to 9 of run E, and that row is the
    last row of its sample, sits in step 7, 8 or 9, is not in the last sample
    of step 8 and not in a sample that holds a row of T, and covers W's token 0
    alone (``token_start`` 0, ``token_end`` 1). Segment 1 does not list
    shard-3, so that row is the first read of W; condition 5 checks that on the
    log. Rows of W in steps 10 to 12 are not looked at. The first such document
    in the order of the log is W."""
    found = _shared_token_documents(ev)
    if not found:
        return Outcome(
            False,
            "no unrelated record of shard-3 has, as its one row of steps 1 to 9, the last "
            "token of a sample of step 7, 8 or 9 that holds no row of T and is not the last "
            "sample of step 8",
        )
    row = found[0]
    sample = _sample_at(ev, RUN_E, row["segment"], row["step"], row["sample_pos"])
    facts = {
        "record": row["record"],
        "source": ev.spec.source_of(row["record"]),
        "shard_position": row["shard_position"],
        "ordinal": row["ordinal"],
        "step": row["step"],
        "sample_pos": row["sample_pos"],
        "documents_that_fit": len(found),
    }
    return Outcome(
        True,
        f"{row['record']} (shard position {row['shard_position']}, document {row['ordinal']}) is read "
        f"through the last token of sample {row['sample_pos']} of step {row['step']} and by no other row of steps 1 to 9",
        _lines(sample["rows"]),
        facts,
    )


# --- Runs D and Y ------------------------------------------------------------------------------


def c13(ev: Evidence) -> Outcome:
    """Run D's reads. Its log holds 12 samples, four in each of steps 1 to 3,
    with the global indices 0 to 11 each once, and every row is on shard
    position 0 or 1."""
    problems = _sizes(ev, D_SEG, range(1, 4), BATCH, range(0, 12))
    rows = ev.segment(D_SEG).rows  # as the log holds them; no record is looked up
    elsewhere = [row for row in rows if row["shard_position"] not in (0, 1)]
    if elsewhere:
        problems.append(f"{len(elsewhere)} rows are on another shard position")
    facts = {"samples": [[s["step"], s["sample_pos"], s["global_index"]] for s in log_rows.samples_of(rows)]}
    if problems:
        return Outcome(False, "; ".join(problems), facts=facts)
    return Outcome(True, "12 samples in steps 1 to 3 (indices 0 to 11), all on shard positions 0 and 1", _lines(rows), facts)


def c14(ev: Evidence) -> Outcome:
    """Run Y never reads T. Its log holds 24 samples, four in each of steps 1 to
    6, with the global indices 0 to 23 each once; one of its shards holds T; and
    no row names T's document, whatever the row's token range."""
    problems = _sizes(ev, Y_SEG, range(1, 7), BATCH, range(0, 24))
    target = _t(ev)
    segment = ev.segment(Y_SEG)
    places = [
        (position, ev.build.by_name(name).records.index(target))
        for position, name in enumerate(segment.shards)
        if target in ev.build.by_name(name).records
    ]
    rows_of_t = [row for row in ev.reads(RUN_Y) if row["record"] == target]
    facts: dict = {"rows_that_name_T": len(rows_of_t)}
    if len(places) != 1:
        problems.append(f"T sits in {len(places)} of run Y's shards, not in one")
    else:
        facts["shard_position_of_T"], facts["ordinal_of_T"] = places[0]
    if rows_of_t:
        problems.append(f"{len(rows_of_t)} rows name T, the first at step {rows_of_t[0]['step']}")
    if problems:
        return Outcome(False, "; ".join(problems), _lines(rows_of_t[:1]), facts)
    return Outcome(True, f"24 samples in steps 1 to 6 and no row on document {facts['ordinal_of_T']} of shard position {facts['shard_position_of_T']} (T)", facts=facts)


def c15(ev: Evidence) -> Outcome:
    """Run Y reads shard-3. At least one row of run Y is on shard position 2."""
    for row in ev.reads(RUN_Y):
        if row["shard_position"] == 2:
            return Outcome(True, f"step {row['step']} reads {row['record']} on shard position 2", [line_of(row)])
    return Outcome(False, "no row of run Y is on shard position 2")


# --- The round-robin variant of run E ----------------------------------------------------------


def c16(ev: Evidence) -> Outcome:
    """In the round-robin layout T sits in chunk 3 and nothing derived from A
    sits in chunks 1 and 2. Read from the replica's manifests of the three
    tokenizer inputs: the manifest of shard-3 has a line for T, and no line of
    shard-1 or shard-2 is a record that "withdraw A" puts in scope."""
    target = _t(ev)
    third = ev.build.shard(LAYOUT_ROUND_ROBIN, SHARD_3)
    in_first_two = sorted(
        (file, record)
        for file in (SHARD_1, SHARD_2)
        for record in ev.build.shard(LAYOUT_ROUND_ROBIN, file).records
        if record in ev.build.affected
    )
    facts: dict = {"affected_records_in_chunks_1_and_2": [list(pair) for pair in in_first_two]}
    if target in third.records:
        facts["shard_of_T"] = third.name
        facts["manifest_line_of_T"] = dict(third.manifest[third.records.index(target)])
    problems = []
    if target not in third.records:
        problems.append("chunk 3 does not hold T")
    if in_first_two:
        problems.append(f"chunks 1 and 2 hold affected records: {in_first_two}")
    if problems:
        return Outcome(False, "; ".join(problems), facts=facts)
    return Outcome(True, f"T is line {facts['manifest_line_of_T']['line']} of chunk 3; chunks 1 and 2 hold no affected record", facts=facts)


def c17(ev: Evidence) -> Outcome:
    """The round-robin run is read whole and traced whole. Its log holds 24
    samples, four in each of steps 1 to 6, and 24, four in each of steps 7 to
    12, and none of its step records lowers a step."""
    problems = _sizes(ev, E_RR_SEG_1, range(1, 7), BATCH, None)
    problems += _sizes(ev, E_RR_SEG_2, range(7, 13), BATCH, None)
    lowered = _lowered_steps(ev, RUN_E_RR)
    if lowered:
        problems.append(f"its step records lower {lowered}")
    if problems:
        return Outcome(False, "; ".join(problems))
    return Outcome(True, "24 samples in steps 1 to 6 and 24 in steps 7 to 12; no step is lowered",
                   facts={"samples_in_steps_1_to_6": 24, "samples_in_steps_7_to_12": 24})


# --- Run K --------------------------------------------------------------------------------------


def _k_epoch_steps(ev: Evidence) -> tuple[int, int]:
    length = ev.segment(K_SEG).summary["dataset_samples"]
    return length, length // BATCH


def c18(ev: Evidence) -> Outcome:
    """Run K's launches. Its log holds 160 samples in steps 1 to 40, in every
    step two of rank 0 and two of rank 1, and the oracle's summary says the
    samplers of steps 1 to 3 were built at a consumed-sample count of 0 and
    those of steps 4 to 40 at a count of 12."""
    problems = _sizes(ev, K_SEG, range(1, 41), BATCH, None)
    ranks: dict[int, list[int]] = {}
    for sample in ev.samples(RUN_K):
        ranks.setdefault(sample["step"], []).append(sample["rank"])
    uneven = sorted(step for step, values in ranks.items() if sorted(values) != [0, 0, 1, 1])
    if uneven:
        problems.append(f"steps {uneven[:5]} do not hold two samples of each rank")
    launches = [
        (launch["first_step"], launch["last_step"], launch["consumed_at_start"])
        for launch in ev.segment(K_SEG).summary["launches"]
    ]
    if launches != [(1, 3, 0), (4, 40, 12)]:
        problems.append(f"the launches are (first step, last step, consumed count) {launches}")
    if problems:
        return Outcome(False, "; ".join(problems))
    return Outcome(True, "160 samples in steps 1 to 40, two per rank and step; launches at counts 0 and 12")


def c19(ev: Evidence) -> Outcome:
    """Run K's epoch length. The length N the oracle's dataset reports for run
    K's blend gives an epoch of N div 4 steps, and that is from 9 to 36 steps."""
    length, epoch_steps = _k_epoch_steps(ev)
    facts = {"blend_length": length, "epoch_steps": epoch_steps}
    if not 9 <= epoch_steps <= 36:
        return Outcome(False, f"the blend holds {length} samples, an epoch of {epoch_steps} steps, outside 9 to 36", facts=facts)
    return Outcome(True, f"the blend holds {length} samples: an epoch of {epoch_steps} steps", facts=facts)


def c20(ev: Evidence) -> Outcome:
    """Run K crosses into epoch 1 in order, and its record covers every epoch.

    Three things are checked. (a) For each rank, the pass of the sampler that
    starts at step N div 4 + 1 is the one the sampler class itself named epoch 1
    (the oracle's summary lists every pass with its epoch and first step).
    (b) The run record holds one shuffle digest for every launch, epoch and rank
    the summary lists a pass for. (c) Wherever one launch read an epoch whole
    (its pass is as long as the recorded order), the global indices that rank
    read in those steps, in the order of the log, give the recorded digest. With
    40 steps and an epoch of 14 steps, as the fixture has it, (c) covers epoch 1
    for both ranks: steps 15 to 28, which launch 1 reads whole."""
    segment = ev.segment(K_SEG)
    _length, epoch_steps = _k_epoch_steps(ev)
    boundary_step = epoch_steps + 1
    passes = segment.summary["read_orders"]
    ranks = sorted({order["rank"] for order in passes})
    recorded = {(d["launch"], d["epoch"], d["rank"]): d for d in segment.shuffle_digests}
    problems = []
    for rank in ranks:
        starting = [order for order in passes if order["rank"] == rank and order["first_step"] == boundary_step]
        if [order["epoch"] for order in starting] != [1]:
            problems.append(f"rank {rank}: no pass named epoch 1 starts at step {boundary_step}")
    missing = sorted(
        {(order["launch"], order["epoch"], order["rank"]) for order in passes} - set(recorded)
    )
    if missing:
        problems.append(f"the run record holds no shuffle digest for (launch, epoch, rank) {missing}")
    if len(recorded) != len(segment.shuffle_digests):
        problems.append("the run record holds two shuffle digests for one launch, epoch and rank")
    by_rank_step: dict[tuple[int, int], list[int]] = {}
    for sample in ev.samples(RUN_K):
        by_rank_step.setdefault((sample["rank"], sample["step"]), []).append(sample["global_index"])
    compared = 0
    for order in passes:
        key = (order["launch"], order["epoch"], order["rank"])
        if key not in recorded or order["count"] != recorded[key]["count"]:
            continue
        indices = [
            index
            for step in range(order["first_step"], order["last_step"] + 1)
            for index in by_rank_step.get((order["rank"], step), [])
        ]
        compared += 1
        if order_digest(indices) != recorded[key]["digest"]:
            problems.append(f"the log's indices of launch {key[0]}, epoch {key[1]}, rank {key[2]} do not give the recorded digest")
    first_row = next((row for row in ev.reads(RUN_K) if row["step"] == boundary_step), None)
    facts = {
        "step_after_the_boundary": boundary_step,
        "epoch_of_that_step": 1,
        "shuffle_digests": len(segment.shuffle_digests),
        "whole_epochs_compared_with_the_log": compared,
    }
    lines = [line_of(first_row)] if first_row is not None else []
    if problems:
        return Outcome(False, "; ".join(problems), lines, facts)
    return Outcome(
        True,
        f"step {boundary_step} starts the pass the sampler named epoch 1; {len(recorded)} shuffle digests "
        f"cover every launch, epoch and rank; {compared} whole epochs of the log give their recorded digest",
        lines,
        facts,
    )


def c21(ev: Evidence) -> Outcome:
    """Run K reads shard-3 before its first checkpoint. Steps 1 to 3 have a row
    on shard position 0, which is shard-3 in run K's shard list."""
    if ev.segment(K_SEG).shards[0] != SHARD_3:
        return Outcome(False, f"shard position 0 of run K is {ev.segment(K_SEG).shards[0]!r}, not shard-3")
    for row in ev.reads(RUN_K):
        if row["step"] <= 3 and row["shard_position"] == 0:
            return Outcome(True, f"step {row['step']} reads {row['record']} on shard position 0", [line_of(row)])
    return Outcome(False, "steps 1 to 3 have no row on shard position 0")


def c22(ev: Evidence) -> Outcome:
    """Run K first reads T at step 7 or 8. The lowest step of run K with a row
    that names T's document, whatever the row's token range, is 7 or 8. (Run K
    packs shard-3 under its own seed, so T need not sit whole in one sample
    there.)"""
    target = _t(ev)
    rows = [row for row in ev.reads(RUN_K) if row["record"] == target]
    if not rows:
        return Outcome(False, "no row of run K names T")
    first = min(rows, key=lambda row: row["step"])
    facts = {
        "first_read_of_T": {
            "step": first["step"],
            "sample_pos": first["sample_pos"],
            "global_index": first["global_index"],
            "rank": first["rank"],
        }
    }
    sample = _sample_at(ev, RUN_K, first["segment"], first["step"], first["sample_pos"])
    if first["step"] not in (7, 8):
        return Outcome(False, f"run K first reads T at step {first['step']}", _lines(sample["rows"]), facts)
    return Outcome(True, f"run K first reads T at step {first['step']}, on rank {first['rank']}", _lines(sample["rows"]), facts)


def c23(ev: Evidence) -> Outcome:
    """Up to its second checkpoint run K reads T on rank 1 only. Every row of
    steps 1 to 8 that names T has rank 1. With no such row the sentence holds
    emptily; condition 22 then fails."""
    target = _t(ev)
    rows = [row for row in ev.reads(RUN_K) if row["record"] == target and row["step"] <= 8]
    facts = {"rows_that_name_T_in_steps_1_to_8": [{"step": row["step"], "rank": row["rank"]} for row in rows]}
    on_other = [row for row in rows if row["rank"] != 1]
    if on_other:
        return Outcome(False, f"{len(on_other)} of the {len(rows)} rows of steps 1 to 8 that name T are on rank 0", _lines(on_other[:1]), facts)
    return Outcome(True, f"the {len(rows)} rows of steps 1 to 8 that name T are all on rank 1", _lines(rows), facts)


# --- The list -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    n: int
    id: str  # the condition's id in the spec
    run: str  # the run whose oracle output the predicate reads
    check: Callable[[Evidence], Outcome]
    sentence: str  # the first 16 hex digits of the SHA-256 of the spec's sentence


CHECKS = (
    Check(1, "sizes.E", RUN_E, c01, "1ee26372daf56f96"),
    Check(2, "skipped-line.no-shift", RUN_E, c02, "d33cb3cb1d01a586"),
    Check(3, "skipped-line.read", RUN_E, c03, "1777d692c5433e84"),
    Check(4, "T.step-7", RUN_E, c04, "051bfeecc74a9484"),
    Check(5, "T.no-earlier-exposure", RUN_E, c05, "1ce7767bd1a032e9"),
    Check(6, "T.not-in-step-8-last-sample", RUN_E, c06, "89bef52a2c83d944"),
    Check(7, "step-8.cut-inside-a-row", RUN_E, c07, "829220b256ad4aa5"),
    Check(8, "repeated-draw", RUN_E, c08, "0488bec5075adb22"),
    Check(9, "shared-ordinal", RUN_E, c09, "f0d39696aaf801e9"),
    Check(10, "split-document", RUN_E, c10, "f0f2f6c3ab414719"),
    Check(11, "seg-1.reads-both-shards", RUN_E, c11, "098dd9584bdecbc7"),
    Check(12, "shared-token", RUN_E, c12, "0a591f3c07dc46dc"),
    Check(13, "D.reads", RUN_D, c13, "1274b03e06ba789c"),
    Check(14, "Y.never-reads-T", RUN_Y, c14, "aeee8178742c4a11"),
    Check(15, "Y.reads-shard-3", RUN_Y, c15, "6a14eb78e744faa8"),
    Check(16, "round-robin.T-in-chunk-3", RUN_E_RR, c16, "fed35d3d7e2da8bb"),
    Check(17, "round-robin.fully-traced", RUN_E_RR, c17, "28b22013b8ef3d90"),
    Check(18, "K.launches", RUN_K, c18, "6aabe97643405488"),
    Check(19, "K.epoch-boundary", RUN_K, c19, "d24f1fbb8434a329"),
    Check(20, "K.epoch-order", RUN_K, c20, "a279e2e75640beee"),
    Check(21, "K.reads-shard-3-early", RUN_K, c21, "1dbd0a5e88dd0499"),
    Check(22, "K.first-read-of-T", RUN_K, c22, "4e34c1ef93a43b9a"),
    Check(23, "K.T-on-rank-1", RUN_K, c23, "dab86c6bdaaf3fc9"),
)


def checks_of(run: str) -> tuple[Check, ...]:
    return tuple(check for check in CHECKS if check.run == run)


def sentence_mark(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]


def check_spec_conditions(spec: FixtureSpec, allow_reworded: tuple[int, ...] = ()) -> list[int]:
    """Compare the spec's numbered conditions with the predicates here.

    The numbers and ids must be the same on both sides. A sentence whose text
    differs from the one its predicate was written against is refused unless its
    number is in ``allow_reworded``: the predicate may no longer check what the
    spec says. Returns the numbers that were let through as reworded."""
    by_number = {condition.n: condition for condition in spec.conditions}
    if sorted(by_number) != [check.n for check in CHECKS]:
        raise ConfigError(
            f"the spec numbers its search conditions {sorted(by_number)}, the oracle has predicates "
            f"for {[check.n for check in CHECKS]}"
        )
    reworded = []
    for check in CHECKS:
        condition = by_number[check.n]
        if condition.id != check.id:
            raise ConfigError(
                f"search condition {check.n} is {condition.id!r} in the spec, the oracle's predicate "
                f"c{check.n:02d} checks {check.id!r}"
            )
        if sentence_mark(condition.sentence) != check.sentence:
            reworded.append(check.n)
    refused = sorted(set(reworded) - set(allow_reworded))
    if refused:
        raise ConfigError(
            f"the spec's sentence of search condition(s) {refused} differs from the one the "
            "predicate was written against; read the predicate in oracle/search_conditions.py "
            "against the new sentence, then update its mark or pass --allow-reworded"
        )
    return reworded


def evaluate(ev: Evidence, numbers: tuple[int, ...] | None = None) -> dict[int, Outcome]:
    """Check the conditions whose run the evidence holds (or those named)."""
    out = {}
    for check in CHECKS:
        if numbers is not None and check.n not in numbers:
            continue
        if numbers is None and check.run not in ev.runs:
            continue
        out[check.n] = check.check(ev)
    return out
