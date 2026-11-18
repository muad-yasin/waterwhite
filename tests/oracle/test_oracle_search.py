"""Checks of the seed search that need no Megatron: the spec reader, the cut of a
sample, every numbered condition's predicate, the readers of a built fixture and
of written runs, and the TOML fragment.

The logs here are written by hand (class ``World``): a set of runs in which every
condition of the spec holds, laid out so that each condition can be broken by one
small change. A predicate is tested twice: it holds on the hand-made logs, and it
fails on logs that violate its sentence.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from oracle import fixture_runs, log_rows, search_conditions, search_evidence, seed_search
from oracle.digests import order_digest
from oracle.errors import ConfigError
from oracle.fixture_spec import expand_ids, load_spec, shard_ref
from oracle.search_conditions import BuildEvidence, Evidence, SegmentEvidence, ShardEvidence, evaluate

REPO = Path(__file__).resolve().parents[2]
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_FILE)


# --- The spec reader -----------------------------------------------------------------------------


def test_the_spec_gives_every_run_with_its_segments_and_launches(spec):
    assert list(spec.runs) == ["E", "E-rr", "D", "Y", "K"]
    e, rr, k = spec.runs["E"], spec.runs["E-rr"], spec.runs["K"]
    assert [(s.name, s.folder, s.first_step, s.last_step, s.train_iters) for s in e.segments] == [
        ("E/seg-1", "seg-1", 1, 6, 6),
        ("E/seg-2", "seg-2", 7, 12, 12),
    ]
    assert e.segments[1].lowered == ((8, 16),) and e.segments[1].checkpoints == (("E@9", 9),)
    assert (e.segments[1].start_kind, e.segments[1].parent_checkpoint) == ("resume", "E@6")
    assert [(l.consumed_at_start, l.step_at_start, l.steps) for l in e.segments[1].launches] == [(24, 6, 6)]
    # the round-robin variant: run E's settings and seed, the other layout, and one
    # checkpoint, E-rr@6, that its second segment resumes from (DECISIONS.md D-24)
    assert (rr.seed_of, rr.layout, rr.sequence_length, rr.steps) == ("E", "round-robin", 32, 12)
    assert [(ref.name, ref.layout, ref.file) for ref in rr.segments[0].shards] == [
        ("E-rr/shard-1", "round-robin", "shard-1"),
        ("E-rr/shard-2", "round-robin", "shard-2"),
    ]
    assert rr.segments[0].checkpoints == (("E-rr@6", 6),) and rr.segments[1].checkpoints == ()
    assert (rr.segments[1].start_kind, rr.segments[1].parent_checkpoint) == ("resume", "E-rr@6")
    assert rr.segments[0].lowered == () and rr.segments[1].lowered == ()
    # run K: two ranks, two launches in one segment, the shard list not in name order
    assert (k.dataloader_type, k.data_parallel_size, k.world_size, k.micro_batch_size) == ("cyclic", 2, 2, 2)
    assert [ref.name for ref in k.segments[0].shards] == ["shard-3", "shard-2"]
    assert [(l.consumed_at_start, l.step_at_start, l.steps, l.data_parallel_size) for l in k.segments[0].launches] == [
        (0, 0, 3, 2),
        (12, 3, 37, 2),
    ]
    assert [name for name, _ in k.segments[0].checkpoints] == ["K@3", "K@8", "K@40"]
    assert [c.n for c in spec.conditions] == list(range(1, 24))


def test_the_records_a_withdrawal_of_a_reaches(spec):
    # A's two records, the paraphrase of one, its translation, and the summary
    assert spec.derived_from("A", REPO) == {"A-001", "A-002", "P", "T", "S"}
    assert spec.source_of("Z-077") == "Z" and spec.source_of("B-001") == "B" and spec.source_of("T") is None
    assert expand_ids(["B-012..B-013", "T"], "x") == ("B-012", "B-013", "T")
    with pytest.raises(ConfigError, match="id range"):
        expand_ids(["B-012..Z-013"], "x")
    assert shard_ref("E-rr/shard-2", "x").layout == "round-robin"
    with pytest.raises(ConfigError):
        shard_ref("E-rr/", "x")


@pytest.mark.parametrize(
    "old, new, word",
    [
        ('"E/mix-1" = ["shard-1", "shard-2"]', '"E/mix-1" = ["shard-2", "shard-1"]', "mixture"),
        ("consumed_at_start = 24 ", "consumed_at_start = 20 ", "reaches 24"),
        ('megatron_bridge = "none"', 'megatron_bridge = "v0.7.0"', "megatron_bridge"),
        ("checkpoints = [6, 9]", "checkpoints = [6]", "checkpoints"),
        ("last_step = 40 },", "last_step = 39 },", "end at step 39"),
        ("consumed_at_start = 12, step_at_start = 3, first_step = 4", "consumed_at_start = 12, step_at_start = 4, first_step = 5", "start at step 4"),
        ("{ step = 8, batch_size = 4, last_sample_tokens_read = 16 }", "{ step = 8, batch_size = 2, last_sample_tokens_read = 16 }", "batch_size"),
        ('same_settings_as = "E"', 'same_settings_as = "E"\nrandom_seed = 3', "copies the settings"),
        ('layout = "round-robin"', 'layout = "partitioned"', "layout"),
        ("[runs.D]\nsequence_length = 32", "[runs.D]\nsequence_len = 32", "unknown key"),
        ("\nn = 7\n", "\nn = 8\n", "numbered"),
        # a resume names a checkpoint the segment before it registers at its last step
        ('parent_checkpoint = "E-rr@6"', 'parent_checkpoint = "E-rr@9"', "does not register"),
        ('parent_checkpoint = "E@6"\nmixture = "E/mix-2"', 'parent_checkpoint = "E@9"\nmixture = "E/mix-2"', "does not register"),
    ],
)
def test_a_spec_that_disagrees_with_itself_is_refused(tmp_path, old, new, word):
    text = SPEC_FILE.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    changed = tmp_path / "fixture_spec.toml"
    changed.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_spec(changed)
    assert word in str(error.value)


def test_the_predicates_are_tied_to_the_specs_sentences(spec, tmp_path):
    assert search_conditions.check_spec_conditions(spec) == []
    assert [check.n for check in search_conditions.CHECKS] == list(range(1, 24))
    text = SPEC_FILE.read_text(encoding="utf-8")
    reworded = tmp_path / "fixture_spec.toml"
    reworded.write_text(text.replace("is step 7 or step 8.", "is step 8 or step 9."), encoding="utf-8")
    changed = load_spec(reworded)
    with pytest.raises(ConfigError, match=r"condition\(s\) \[22\]"):
        search_conditions.check_spec_conditions(changed)
    assert search_conditions.check_spec_conditions(changed, allow_reworded=(22,)) == [22]
    renamed = tmp_path / "renamed.toml"
    renamed.write_text(text.replace('id = "shared-token"', 'id = "shared-label"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="condition 12"):
        search_conditions.check_spec_conditions(load_spec(renamed))


# --- The cut of a sample ----------------------------------------------------------------------------


def row(segment, step, sample_pos, global_index, rank, shard_position, ordinal, token_start, token_end) -> dict:
    return dict(zip(log_rows.LOG_FIELDS, (segment, step, sample_pos, global_index, rank, shard_position, ordinal, token_start, token_end)))


def test_a_sample_is_cut_inside_a_row_or_between_two():
    rows = [row("S", 8, 3, 31, 0, 2, 9, 5, 25), row("S", 8, 3, 31, 0, 2, 4, 0, 13)]
    (sample,) = log_rows.samples_of(rows)
    assert log_rows.sample_tokens(sample) == 33
    inside = log_rows.cut_sample(sample, 16)
    # 16 tokens: tokens 5 to 21 of document 9; the uncovered range starts at its token 21
    assert inside["cut"] == {"shard_position": 2, "ordinal": 9, "token": 21}
    assert inside["starts_a_row"] is False
    assert [(r["ordinal"], r["token_start"], r["token_end"]) for r in inside["covered_rows"]] == [(9, 5, 21)]
    between = log_rows.cut_sample(sample, 20)
    assert between["cut"] == {"shard_position": 2, "ordinal": 4, "token": 0} and between["starts_a_row"] is True
    assert [(r["ordinal"], r["token_end"]) for r in between["covered_rows"]] == [(9, 25)]
    second = log_rows.cut_sample(sample, 24)
    assert second["cut"] == {"shard_position": 2, "ordinal": 4, "token": 4}
    assert [(r["ordinal"], r["token_start"], r["token_end"]) for r in second["covered_rows"]] == [(9, 5, 25), (4, 0, 4)]
    assert log_rows.cut_sample(sample, 0)["covered_rows"] == []
    for covered in (33, 40, -1, True):
        with pytest.raises(ConfigError, match="cannot cut"):
            log_rows.cut_sample(sample, covered)
    assert rows[0]["token_end"] == 25  # the log's own rows are left as they were


def test_the_cut_of_a_lowered_step_is_read_off_the_log():
    rows = [
        row("E/seg-2", 8, 2, 30, 0, 0, 1, 0, 33),
        row("E/seg-2", 8, 3, 31, 0, 2, 9, 5, 25),
        row("E/seg-2", 8, 3, 31, 0, 2, 4, 0, 13),
    ]
    cut = fixture_runs.step_cut(rows, segment="E/seg-2", step=8, sample_pos=3, covered=16)
    assert cut["uncovered_from"] == {"shard_position": 2, "ordinal": 9, "token": 21}
    assert (cut["covered_tokens"], cut["sample_tokens"], cut["cut_falls_inside_a_row"]) == (16, 33, True)
    assert cut["log_lines"] == rows[1:]
    with pytest.raises(Exception, match="no sample 3 of step 9"):
        fixture_runs.step_cut(rows, segment="E/seg-2", step=9, sample_pos=3, covered=16)


def test_rows_of_a_sample_must_be_consecutive():
    rows = [row("S", 1, 0, 0, 0, 0, 0, 0, 33), row("S", 1, 1, 1, 0, 0, 1, 0, 33), row("S", 1, 0, 0, 0, 0, 2, 0, 3)]
    with pytest.raises(ConfigError, match="not consecutive"):
        log_rows.samples_of(rows)
    with pytest.raises(ConfigError, match="two global indices"):
        log_rows.samples_of([row("S", 1, 0, 0, 0, 0, 0, 0, 20), row("S", 1, 0, 5, 0, 0, 1, 0, 13)])


# --- Hand-made runs in which every condition holds ---------------------------------------------------

SHARD_1 = [f"B-{n:03d}" for n in range(1, 7)] + [f"Z-{n:03d}" for n in range(41, 53)]  # 18 records
SHARD_2 = [f"B-{n:03d}" for n in range(7, 14)] + [f"Z-{n:03d}" for n in range(53, 65)]  # 19 records
SHARD_3 = [f"Z-{n:03d}" for n in range(65, 71)] + ["T"] + [f"Z-{n:03d}" for n in range(71, 91)]  # T is document 6
T_ORDINAL = 6
W_ORDINAL = 20  # Z-084 in shard-3: the shared-token document
T_TOKENS = 22  # T whole: 21 tokens and the end-of-document token


def k_order(epoch: int, rank: int) -> list[int]:
    """A made-up whole order of one epoch for one rank: a permutation of the
    rank's bucket of 30 indices that differs from epoch to epoch."""
    return [rank * 30 + (i * 7 + 3 * epoch) % 30 for i in range(30)]


class World:
    """Hand-written evidence. Sequence length 32: a sample's rows cover 33 tokens.

    Run E, segment 1 (steps 1 to 6, indices 0 to 23): sample g reads one whole
    row, on shard-1 (documents 1 to 6) when g is even and on shard-2 (documents 6
    to 11) when odd. Segment 2 (steps 7 to 12, indices 24 to 47) goes round the
    three shards with the same documents and documents 7 to 16 of shard-3, and
    holds the special samples:

    - step 7, sample 1: the last 7 tokens of document 8 of shard-3, then T whole
      (22 tokens), then the first 4 tokens of document 9;
    - step 8, sample 3: two rows of 20 and 13 tokens, so token 17 is inside the first;
    - step 9, sample 0: 32 tokens of document 17 of shard-3, then token 0 of W,
      a document of shard-3 that no other row of steps 1 to 9 names;
    - step 10, samples 0 and 1: document 12 of shard-2 as [0:33] and as [32:60].
    """

    def __init__(self, spec) -> None:
        self.spec = spec
        self.records = {
            ("partitioned", "shard-1"): list(SHARD_1),
            ("partitioned", "shard-2"): list(SHARD_2),
            ("partitioned", "shard-3"): list(SHARD_3),
            ("round-robin", "shard-1"): list(SHARD_1),
            ("round-robin", "shard-2"): list(SHARD_2),
            ("round-robin", "shard-3"): list(SHARD_3),
        }
        self.skipped = {("partitioned", "shard-2"): (6,)}
        self.documents = {"E/seg-1": (18, 19), "E/seg-2": (18, 19, 27)}
        self.rows: dict[str, list[dict]] = {}
        self.summary: dict[str, dict] = {}
        self.digests: dict[str, list[dict]] = {}
        self.step_record: dict[str, dict] = {}

        def sequential(segment, first_step, steps, first_index, shards, special=None):
            rows = []
            for g in range(first_index, first_index + steps * 4):
                step, pos = first_step + (g - first_index) // 4, (g - first_index) % 4
                if special and (step, pos) in special:
                    rows += [row(segment, step, pos, g, 0, *r) for r in special[(step, pos)]]
                    continue
                position = g % shards
                ordinal = (1 + g % 6, 6 + g % 6, 7 + g % 10)[position]
                rows.append(row(segment, step, pos, g, 0, position, ordinal, 0, 33))
            self.rows[segment] = rows
            self.summary[segment] = {"dataset_samples": first_index + steps * 4, "launches": [], "read_orders": []}
            self.digests[segment] = []
            self.step_record[segment] = {
                "segment": segment,
                "steps": [{"step": s, "batch_size": 4} for s in range(first_step, first_step + steps)],
                "checkpoints": [],
            }

        sequential("E/seg-1", 1, 6, 0, 2)
        sequential(
            "E/seg-2", 7, 6, 24, 3,
            special={
                (7, 1): [(2, 8, 30, 37), (2, T_ORDINAL, 0, T_TOKENS), (2, 9, 0, 4)],
                (8, 3): [(2, 9, 0, 20), (2, 10, 0, 13)],
                (9, 0): [(2, 17, 0, 32), (2, W_ORDINAL, 0, 1)],
                (10, 0): [(1, 12, 0, 33)],
                (10, 1): [(1, 12, 32, 60), (1, 13, 0, 5)],
            },
        )
        self.step_record["E/seg-2"]["steps"][1]["last_sample_tokens_read"] = 16
        sequential("E-rr/seg-1", 1, 6, 0, 2)
        sequential("E-rr/seg-2", 7, 6, 24, 3)
        sequential("D/seg-1", 1, 3, 0, 2)
        sequential("Y/seg-1", 1, 6, 0, 3)

        # Run K: 60 samples, 2 ranks, 2 samples per rank and step, an epoch of 15
        # steps. Rank 0 reads shard position 0 (shard-3), rank 1 position 1
        # (shard-2), except that sample 2 of step 7, on rank 1, reads T.
        rows = []
        for step in range(1, 41):
            epoch, place = (step - 1) // 15, (step - 1) % 15
            for rank in (0, 1):
                for i in (0, 1):
                    index = k_order(epoch, rank)[2 * place + i]
                    pos = rank * 2 + i
                    read = (0, 7 + index % 10, 0, 33) if rank == 0 else (1, index % 19, 0, 33)
                    if (step, pos) == (7, 2):
                        read = (0, T_ORDINAL, 0, T_TOKENS)
                    rows.append(row("K/seg-1", step, pos, index, rank, *read))
        self.rows["K/seg-1"] = rows
        passes = [(0, 0, 1, 3), (1, 0, 4, 15), (1, 1, 16, 30), (1, 2, 31, 40)]  # launch, epoch, first, last step
        self.summary["K/seg-1"] = {
            "dataset_samples": 60,
            "launches": [
                {"position": 0, "first_step": 1, "last_step": 3, "consumed_at_start": 0},
                {"position": 1, "first_step": 4, "last_step": 40, "consumed_at_start": 12},
            ],
            "read_orders": [
                {"launch": launch, "rank": rank, "pass": 1, "epoch": epoch, "first_step": first, "last_step": last,
                 "count": 2 * (last - first + 1), "digest": "not read by a predicate"}
                for launch, epoch, first, last in passes
                for rank in (0, 1)
            ],
        }
        self.digests["K/seg-1"] = [
            {"launch": launch, "epoch": epoch, "rank": rank, "digest": order_digest(k_order(epoch, rank)), "count": 30}
            for launch, epoch, _, _ in passes
            for rank in (0, 1)
        ]
        self.step_record["K/seg-1"] = {"segment": "K/seg-1", "steps": [], "checkpoints": []}
        self.documents["K/seg-1"] = (27, 19)

    # -- changes a test makes ---------------------------------------------------------------------

    def set_sample(self, segment: str, step: int, pos: int, reads: list[tuple]) -> None:
        """Replace the rows of one sample, keeping its place in the log."""
        rows = self.rows[segment]
        at = [i for i, r in enumerate(rows) if (r["step"], r["sample_pos"]) == (step, pos)]
        first = rows[at[0]]
        new = [row(segment, step, pos, first["global_index"], first["rank"], *read) for read in reads]
        self.rows[segment] = rows[: at[0]] + new + rows[at[-1] + 1 :]

    def drop_sample(self, segment: str, step: int, pos: int) -> None:
        self.rows[segment] = [r for r in self.rows[segment] if (r["step"], r["sample_pos"]) != (step, pos)]

    def edit(self, segment: str, change) -> None:
        """Let ``change`` alter a copy of every row of a segment, in place."""
        rows = [dict(r) for r in self.rows[segment]]
        for r in rows:
            change(r)
        self.rows[segment] = rows

    def evidence(self, runs=("E", "E-rr", "D", "Y", "K")) -> Evidence:
        shards = {}
        for (layout, file), records in self.records.items():
            name = file if layout == "partitioned" else f"E-rr/{file}"
            shards[(layout, file)] = ShardEvidence(
                name=name,
                layout=layout,
                file=file,
                records=tuple(records),
                manifest=tuple({"line": i + 1, "handle": r, "origin": [r]} for i, r in enumerate(records)),
                skipped=tuple(self.skipped.get((layout, file), ())),
            )
        build = BuildEvidence(shards=shards, affected=frozenset({"A-001", "A-002", "P", "S", "T"}))
        out = {}
        for run in runs:
            out[run] = [
                SegmentEvidence(
                    name=segment.name,
                    shards=tuple(ref.name for ref in segment.shards),
                    rows=self.rows[segment.name],
                    summary=self.summary[segment.name],
                    shuffle_digests=self.digests[segment.name],
                    step_record=self.step_record[segment.name],
                    shard_documents=self.documents.get(segment.name, ()),
                )
                for segment in self.spec.runs[run].segments
            ]
        return Evidence(spec=self.spec, build=build, runs=out)


def test_every_condition_holds_on_the_hand_made_runs(spec):
    outcomes = evaluate(World(spec).evidence())
    assert sorted(outcomes) == list(range(1, 24))
    assert {n: o.detail for n, o in outcomes.items() if not o.holds} == {}

    t = outcomes[4]
    assert (t.facts["sample_pos"], t.facts["global_index"], t.facts["shard_position"], t.facts["ordinal"], t.facts["token_end"]) == (1, 25, 2, 6, 22)
    assert t.facts["first_row"] == {"record": "Z-072", "shard_position": 2, "ordinal": 8, "token_start": 30, "token_end": 37}
    assert t.facts["neighbours"] == [
        t.facts["first_row"],
        {"record": "Z-073", "shard_position": 2, "ordinal": 9, "token_start": 0, "token_end": 4},
    ]
    assert t.facts["rows_of_T_in_run_E"] == [{"segment": "E/seg-2", "step": 7, "sample_pos": 1, "token_start": 0, "token_end": 22}]
    assert [line["ordinal"] for line in t.lines] == [8, 6, 9]  # the log lines of T's sample, as the log holds them
    assert set(t.lines[0]) == set(log_rows.LOG_FIELDS)
    assert outcomes[3].facts["sanitizer_input_line"] > 6
    cut = outcomes[7].facts
    assert (cut["shard_position"], cut["ordinal"], cut["token"], cut["covered_tokens"], cut["sample_tokens"]) == (2, 9, 16, 16, 33)
    assert cut["rows_of_the_covered_part"] == [{"shard_position": 2, "ordinal": 9, "token_start": 0, "token_end": 16}]
    # the pair shown reads the same token range twice: the same sample drawn again
    assert outcomes[8].facts["same_range_read_again"] is True and outcomes[8].facts["same_tokens_read_again"] is True
    assert outcomes[8].facts["first_read"]["segment"] == "E/seg-1" and outcomes[8].facts["second_read"]["segment"] == "E/seg-2"
    # document 9 is the lowest number read on two shards: on shard-2 and on shard-3
    assert outcomes[9].facts["ordinal"] == 9 and outcomes[9].facts["shard_positions"] == [1, 2]
    split = outcomes[10].facts
    assert (split["record"], [(r["token_start"], r["token_end"]) for r in split["rows"]]) == ("Z-058", [(0, 33), (32, 60)])
    w = outcomes[12].facts
    assert (w["record"], w["source"], w["shard_position"], w["ordinal"], w["step"], w["sample_pos"]) == ("Z-084", "Z", 2, 20, 9, 0)
    assert w["documents_that_fit"] == 1
    assert [(line["ordinal"], line["token_start"], line["token_end"]) for line in outcomes[12].lines] == [(17, 0, 32), (20, 0, 1)]
    assert outcomes[14].facts == {"rows_that_name_T": 0, "shard_position_of_T": 2, "ordinal_of_T": 6}
    assert outcomes[16].facts["manifest_line_of_T"] == {"line": 7, "handle": "T", "origin": ["T"]}
    assert outcomes[19].facts == {"blend_length": 60, "epoch_steps": 15}
    assert outcomes[20].facts["whole_epochs_compared_with_the_log"] == 2  # epoch 1, both ranks
    assert outcomes[22].facts == {"first_read_of_T": {"step": 7, "sample_pos": 2, "global_index": k_order(0, 1)[12], "rank": 1}}


# One change to the hand-made runs per case; the named condition must then fail.


def _t_as_first_row(w):
    w.set_sample("E/seg-2", 7, 1, [(2, T_ORDINAL, 0, 33)])


def _t_from_token_five(w):
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 30, 37), (2, T_ORDINAL, 5, T_TOKENS), (2, 9, 0, 9)])


def _t_cut_by_the_sample_end(w):
    # the sample ends inside T: its row is the last one and stops short of T's end
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 20, 37), (2, T_ORDINAL, 0, 16)])


def _t_ends_with_the_sample(w):
    # T's last token is the sample's 33rd, the one the next sample starts with:
    # the row covers all of T, but no third row shows that T ended inside the sample
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 26, 37), (2, T_ORDINAL, 0, T_TOKENS)])


def _t_followed_but_not_whole(w):
    # only a log that is not Megatron's can hold this; the predicate must still say no
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 30, 37), (2, T_ORDINAL, 0, 21), (2, 9, 0, 5)])


def _t_read_again_at_step_eleven(w):
    w.set_sample("E/seg-2", 11, 2, [(2, T_ORDINAL, 21, T_TOKENS), (2, 9, 0, 32)])


def _first_row_of_ts_sample_on_another_shard(w):
    # only a log that is not Megatron's can hold this; the predicate must still say no
    w.set_sample("E/seg-2", 7, 1, [(1, 8, 52, 59), (2, T_ORDINAL, 0, T_TOKENS), (2, 9, 0, 4)])


def _t_as_third_row(w):
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 30, 37), (2, 9, 0, 3), (2, T_ORDINAL, 0, T_TOKENS), (2, 10, 0, 1)])


def _t_at_step_eight(w):
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 0, 33)])
    w.set_sample("E/seg-2", 8, 0, [(2, 8, 30, 37), (2, T_ORDINAL, 0, T_TOKENS), (2, 9, 0, 4)])


def _step_seven_lowered_too(w):
    w.step_record["E/seg-2"]["steps"][0]["last_sample_tokens_read"] = 16


def _shard_3_row_in_segment_one(w):
    # only a log that is not run E's can hold this; the predicate must still say no
    w.set_sample("E/seg-1", 3, 0, [(1, 7, 0, 33)])
    w.rows["E/seg-1"][8]["shard_position"] = 2


def _t_in_the_half_covered_sample(w):
    w.set_sample("E/seg-2", 8, 3, [(2, 9, 0, 20), (2, T_ORDINAL, 0, 13)])


def _cut_between_two_rows(w):
    w.set_sample("E/seg-2", 8, 3, [(2, 9, 0, 16), (2, 10, 0, 17)])


def _no_document_read_in_both_segments(w):
    def move(r):
        if r["shard_position"] < 2:
            r["ordinal"] = 14 + r["ordinal"] % 4  # documents 14 to 17: segment 1 reads none of them

    w.edit("E/seg-2", move)


def _one_shard_only(w):
    for segment in ("E/seg-1", "E/seg-2"):
        w.rows[segment] = [r for r in w.rows[segment] if r["shard_position"] == 0]


def _no_second_part(w):
    w.set_sample("E/seg-2", 10, 1, [(1, 13, 0, 33)])


def _the_only_split_document_is_t(w):
    w.set_sample("E/seg-2", 10, 0, [(2, 8, 4, 37)])
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 20, 37), (2, T_ORDINAL, 0, 16)])  # T cut by the sample's end
    w.set_sample("E/seg-2", 10, 1, [(2, T_ORDINAL, 15, T_TOKENS), (2, 9, 0, 26)])  # and going on from that token


def _the_only_split_document_is_w(w):
    w.set_sample("E/seg-2", 10, 0, [(2, 8, 4, 37)])
    w.set_sample("E/seg-2", 10, 1, [(2, 9, 4, 37)])
    w.set_sample("E/seg-2", 11, 0, [(2, W_ORDINAL, 0, 33)])   # W's body, read after step 9
    w.set_sample("E/seg-2", 11, 1, [(2, W_ORDINAL, 32, 65)])  # and split across two samples


def _segment_one_reads_one_shard(w):
    w.edit("E/seg-1", lambda r: r.update(shard_position=0, ordinal=1 + r["ordinal"] % 6))


def _w_read_at_step_eight_too(w):
    w.set_sample("E/seg-2", 8, 0, [(2, W_ORDINAL, 0, 33)])


def _w_not_the_last_row(w):
    w.set_sample("E/seg-2", 9, 0, [(2, W_ORDINAL, 0, 1), (2, 17, 0, 32)])


def _w_in_the_last_sample_of_step_8(w):
    w.set_sample("E/seg-2", 9, 0, [(2, 17, 0, 33)])
    w.set_sample("E/seg-2", 8, 3, [(2, 17, 0, 32), (2, W_ORDINAL, 0, 1)])


def _w_two_tokens(w):
    w.set_sample("E/seg-2", 9, 0, [(2, 17, 0, 31), (2, W_ORDINAL, 0, 2)])


def _w_in_shard_1(w):
    # the one-token read falls on a record of Z in shard-1: not a document of shard-3
    w.set_sample("E/seg-2", 9, 0, [(0, 2, 0, 32), (0, 10, 0, 1)])  # document 10 of shard-1 is Z-045


def _w_in_shard_2_and_of_source_b(w):
    w.set_sample("E/seg-2", 9, 0, [(1, 12, 0, 32), (1, 0, 0, 1)])  # document 0 of shard-2 is B-007


def _w_is_t(w):
    # T read through one token only: T is no unrelated record
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 0, 33)])
    w.set_sample("E/seg-2", 9, 0, [(2, 17, 0, 32), (2, T_ORDINAL, 0, 1)])


def _w_in_the_sample_that_holds_t(w):
    w.set_sample("E/seg-2", 9, 0, [(2, 17, 0, 33)])
    w.set_sample("E/seg-2", 7, 1, [(2, 8, 27, 37), (2, T_ORDINAL, 0, T_TOKENS), (2, W_ORDINAL, 0, 1)])


def _w_at_step_ten(w):
    w.set_sample("E/seg-2", 9, 0, [(2, 17, 0, 33)])
    w.set_sample("E/seg-2", 10, 2, [(2, 17, 0, 32), (2, W_ORDINAL, 0, 1)])


def _w_from_its_second_token(w):
    w.set_sample("E/seg-2", 9, 0, [(2, 17, 0, 32), (2, W_ORDINAL, 1, 2)])


def _d_reads_shard_3(w):
    w.rows["D/seg-1"][5]["shard_position"] = 2


def _y_reads_a_part_of_t(w):
    # run Y packs shard-3 under its own seed: one token of T is a read of T
    w.set_sample("Y/seg-1", 4, 2, [(2, 8, 0, 32), (2, T_ORDINAL, 0, 1)])


def _y_reads_two_shards_only(w):
    w.edit("Y/seg-1", lambda r: r.update(shard_position=r["shard_position"] % 2, ordinal=6))


def _t_in_chunk_2(w):
    w.records[("round-robin", "shard-3")].remove("T")
    w.records[("round-robin", "shard-2")].append("T")


def _the_paraphrase_in_chunk_1(w):
    w.records[("round-robin", "shard-1")].append("P")


def _round_robin_step_lowered(w):
    w.step_record["E-rr/seg-2"]["steps"][3]["last_sample_tokens_read"] = 16


def _three_samples_of_rank_0(w):
    w.rows["K/seg-1"][2]["rank"] = 0


def _resume_keeps_the_counter_of_e6(w):
    w.summary["K/seg-1"]["launches"][1]["consumed_at_start"] = 24


def _k_epoch_one_out_of_order(w):
    rows = w.rows["K/seg-1"]
    a = next(i for i, r in enumerate(rows) if (r["step"], r["sample_pos"]) == (16, 0))
    b = next(i for i, r in enumerate(rows) if (r["step"], r["sample_pos"]) == (17, 0))
    rows[a]["global_index"], rows[b]["global_index"] = rows[b]["global_index"], rows[a]["global_index"]


def _k_boundary_pass_named_epoch_0(w):
    for order in w.summary["K/seg-1"]["read_orders"]:
        if order["first_step"] == 16:
            order["epoch"] = 0


def _k_record_lacks_a_digest(w):
    w.digests["K/seg-1"] = [d for d in w.digests["K/seg-1"] if (d["launch"], d["epoch"], d["rank"]) != (1, 2, 1)]


def _k_early_steps_on_shard_2(w):
    w.edit("K/seg-1", lambda r: r.update(shard_position=1, ordinal=3) if r["step"] <= 3 else None)


def _k_reads_t_at_step_5(w):
    w.set_sample("K/seg-1", 5, 3, [(0, T_ORDINAL, 8, T_TOKENS), (0, 9, 0, 19)])


def _k_reads_t_at_step_9_first(w):
    w.set_sample("K/seg-1", 7, 2, [(1, 4, 0, 33)])
    w.set_sample("K/seg-1", 9, 2, [(0, T_ORDINAL, 0, T_TOKENS)])


def _k_never_reads_t(w):
    w.set_sample("K/seg-1", 7, 2, [(1, 4, 0, 33)])


def _k_reads_t_on_rank_0(w):
    w.set_sample("K/seg-1", 8, 0, [(0, T_ORDINAL, 8, T_TOKENS), (0, 9, 0, 19)])


VIOLATIONS = [
    (1, lambda w: w.drop_sample("E/seg-1", 3, 2)),
    (1, lambda w: w.rows["E/seg-2"][0].update(global_index=23)),
    (2, lambda w: w.skipped.clear()),
    (2, lambda w: w.skipped.update({("partitioned", "shard-2"): (5,)})),
    (2, lambda w: w.documents.update({"E/seg-2": (18, 20, 27)})),
    (3, lambda w: [w.edit(s, lambda r: r.update(ordinal=r["ordinal"] % 5) if r["shard_position"] == 1 else None) for s in ("E/seg-1", "E/seg-2")]),
    (4, _t_as_first_row),
    (4, _t_from_token_five),
    (4, _t_cut_by_the_sample_end),
    (4, _t_ends_with_the_sample),
    (4, _t_followed_but_not_whole),
    (4, _t_read_again_at_step_eleven),
    (4, _first_row_of_ts_sample_on_another_shard),
    (4, _t_as_third_row),
    (4, _t_at_step_eight),
    (4, _step_seven_lowered_too),
    (5, _shard_3_row_in_segment_one),
    (6, _t_in_the_half_covered_sample),
    (7, _cut_between_two_rows),
    (8, _no_document_read_in_both_segments),
    (9, _one_shard_only),
    (10, _no_second_part),
    (10, _the_only_split_document_is_t),
    (10, _the_only_split_document_is_w),
    (11, _segment_one_reads_one_shard),
    (12, _w_read_at_step_eight_too),
    (12, _w_not_the_last_row),
    (12, _w_in_the_last_sample_of_step_8),
    (12, _w_two_tokens),
    (12, _w_in_shard_1),
    (12, _w_in_shard_2_and_of_source_b),
    (12, _w_is_t),
    (12, _w_in_the_sample_that_holds_t),
    (12, _w_at_step_ten),
    (12, _w_from_its_second_token),
    (13, _d_reads_shard_3),
    (13, lambda w: w.drop_sample("D/seg-1", 3, 3)),
    (14, _y_reads_a_part_of_t),
    (14, lambda w: w.drop_sample("Y/seg-1", 6, 0)),
    (14, lambda w: w.records[("partitioned", "shard-3")].remove("T")),
    (15, _y_reads_two_shards_only),
    (16, _t_in_chunk_2),
    (16, _the_paraphrase_in_chunk_1),
    (17, _round_robin_step_lowered),
    (17, lambda w: w.drop_sample("E-rr/seg-2", 12, 3)),
    (18, _three_samples_of_rank_0),
    (18, _resume_keeps_the_counter_of_e6),
    (18, lambda w: w.drop_sample("K/seg-1", 40, 3)),
    (19, lambda w: w.summary["K/seg-1"].update(dataset_samples=35)),
    (19, lambda w: w.summary["K/seg-1"].update(dataset_samples=148)),
    (20, _k_epoch_one_out_of_order),
    (20, _k_boundary_pass_named_epoch_0),
    (20, _k_record_lacks_a_digest),
    (21, _k_early_steps_on_shard_2),
    (22, _k_reads_t_at_step_5),
    (22, _k_reads_t_at_step_9_first),
    (22, _k_never_reads_t),
    (23, _k_reads_t_on_rank_0),
]


@pytest.mark.parametrize(
    "n, change", VIOLATIONS, ids=[f"c{n:02d}-{getattr(change, '__name__', 'case').strip('_')}-{i}" for i, (n, change) in enumerate(VIOLATIONS)]
)
def test_a_log_that_violates_a_sentence_fails_its_condition(spec, n, change):
    world = World(spec)
    change(world)
    outcome = evaluate(world.evidence(), (n,))[n]
    assert outcome.holds is False, outcome.detail
    assert outcome.detail


def test_every_condition_has_a_violating_case():
    assert {n for n, _ in VIOLATIONS} == set(range(1, 24))


def test_condition_12_looks_at_steps_1_to_9_only(spec):
    """W may be read again after the second checkpoint: a row at step 11 leaves
    the condition true, and the same row at step 9 breaks it."""
    world = World(spec)
    world.set_sample("E/seg-2", 11, 0, [(2, W_ORDINAL, 0, 33)])
    assert evaluate(world.evidence(), (12,))[12].holds is True
    world.set_sample("E/seg-2", 9, 2, [(2, W_ORDINAL, 0, 33)])
    assert evaluate(world.evidence(), (12,))[12].holds is False


def test_a_shared_token_in_ts_own_sample_leaves_condition_4_and_fails_12(spec):
    """The sample that holds T may end with one token of another document: T is
    still whole in it (condition 4). That document is not W: condition 12 asks
    for a sample that holds no row of T, so the two positions are shown by two
    samples."""
    world = World(spec)
    _w_in_the_sample_that_holds_t(world)
    outcomes = evaluate(world.evidence(), (4, 12))
    assert outcomes[4].holds is True and outcomes[12].holds is False


def test_the_repeated_draw_shows_the_same_range_before_an_overlap(spec):
    """Condition 8 holds for any document read in both segments. The pair it
    shows is one that reads the same token range twice when there is one."""
    world = World(spec)

    def move(r):
        # every late whole-sample read of shard-1 and shard-2 becomes another range
        if r["shard_position"] < 2 and (r["token_start"], r["token_end"]) == (0, 33):
            r.update(token_start=20, token_end=53)

    world.edit("E/seg-2", move)
    outcome = evaluate(world.evidence(), (8,))[8]
    assert outcome.holds is True
    assert (outcome.facts["same_range_read_again"], outcome.facts["same_tokens_read_again"]) == (False, True)


def test_condition_23_holds_emptily_and_22_then_fails(spec):
    world = World(spec)
    _k_never_reads_t(world)
    outcomes = evaluate(world.evidence(), (22, 23))
    assert outcomes[23].holds is True and outcomes[22].holds is False


def test_only_the_runs_given_are_checked(spec):
    outcomes = evaluate(World(spec).evidence(runs=("E",)))
    assert sorted(outcomes) == list(range(1, 13))
    with pytest.raises(ConfigError, match="no oracle output for"):
        evaluate(World(spec).evidence(runs=("E",)), (13,))


# --- Reading a built fixture and written runs --------------------------------------------------------


def write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def skip_row(layout: str, line: int, shard: str = "shard-2") -> dict:
    """One row of the sanitizer's log as the replica writes it (docs/replica.md,
    "The folder one build writes"): the sanitizer-input file and the line there."""
    return {
        "shard": shard,
        "input": f"pipeline/{layout}/sanitizer-input/{shard}.jsonl",
        "line": line,
        "id": None,
        "origin_file": None,
        "origin_line": None,
        "reason": "sanitizer skip",
        "detail": "not JSON",
    }


@pytest.fixture
def built(tmp_path):
    """The evidence side of a built fixture, written as the replica writes it:
    tokenizer inputs with their manifests and the sanitizer log. No shard files."""
    root = tmp_path / "built"
    for layout in ("partitioned", "round-robin"):
        for file, records in (("shard-1", SHARD_1), ("shard-2", SHARD_2), ("shard-3", SHARD_3)):
            target = root / "pipeline" / layout / "tokenizer-input" / f"{file}.jsonl"
            write_lines(target, [{"id": r, "text": "t"} for r in records])
            write_lines(
                target.with_name(target.name + ".manifest.jsonl"),
                [{"line": i + 1, "handle": r, "origin": [r]} for i, r in enumerate(records)],
            )
        write_lines(
            root / "pipeline" / layout / "sanitizer.log.jsonl",
            [skip_row(layout, 6)],
        )
    return root


def test_a_built_fixture_is_read_from_its_manifests_and_sanitizer_log(spec, built):
    build = search_evidence.load_build(spec, built, REPO)
    assert sorted(build.shards) == [(layout, f"shard-{n}") for layout in ("partitioned", "round-robin") for n in (1, 2, 3)]
    second = build.by_name("shard-2")
    assert (len(second.records), second.skipped) == (19, (6,))
    # document 4 was line 5 of the sanitizer's input; document 5 was line 7, behind the skipped line
    assert [second.input_line(ordinal) for ordinal in (0, 4, 5, 18)] == [1, 5, 7, 20]
    assert build.by_name("E-rr/shard-3").records[T_ORDINAL] == "T"
    assert build.affected == {"A-001", "A-002", "P", "T", "S"}
    with pytest.raises(ConfigError, match="no shard"):
        build.shard("partitioned", "shard-9")


def test_a_manifest_that_disagrees_with_its_input_is_refused(spec, built):
    manifest = built / "pipeline" / "partitioned" / "tokenizer-input" / "shard-3.jsonl.manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[6]["origin"] = ["Z-071"]  # the manifest says line 7 is not T
    write_lines(manifest, rows)
    with pytest.raises(ConfigError, match="manifest row 7"):
        search_evidence.load_build(spec, built, REPO)
    write_lines(manifest, rows[:-1])
    with pytest.raises(ConfigError, match="27 input lines, 26 manifest rows"):
        search_evidence.load_build(spec, built, REPO)


def test_a_skipped_place_past_the_input_is_refused(spec, built):
    log = built / "pipeline" / "partitioned" / "sanitizer.log.jsonl"
    write_lines(log, [skip_row("partitioned", 40)])
    with pytest.raises(ConfigError, match="do not fit"):
        search_evidence.load_build(spec, built, REPO)


def test_a_sanitizer_log_row_must_name_the_shards_own_input_and_a_line(spec, built):
    log = built / "pipeline" / "partitioned" / "sanitizer.log.jsonl"
    # the line is a place in the sanitizer's input of this shard and layout, nothing else
    write_lines(log, [{**skip_row("partitioned", 6), "input": "pipeline/round-robin/sanitizer-input/shard-2.jsonl"}])
    with pytest.raises(ConfigError, match="names the input"):
        search_evidence.load_build(spec, built, REPO)
    row = skip_row("partitioned", 6)
    del row["line"]
    write_lines(log, [{**row, "position": 6}])  # a place under another key is not read
    with pytest.raises(ConfigError, match="without a line"):
        search_evidence.load_build(spec, built, REPO)


# --- Seeds, segment configs, and the fragment ----------------------------------------------------------


def test_seeds_come_from_the_spec_or_from_an_override(spec):
    assert fixture_runs.resolve_seeds(spec, {}) == {name: run.random_seed for name, run in spec.runs.items()}
    seeds = fixture_runs.resolve_seeds(spec, fixture_runs.parse_seed_overrides(["E=41", "K=7"]))
    assert (seeds["E"], seeds["E-rr"], seeds["K"], seeds["D"]) == (41, 41, 7, spec.runs["D"].random_seed)
    with pytest.raises(ConfigError, match="has the seed of run 'E'"):
        fixture_runs.resolve_seeds(spec, {"E-rr": 3})
    with pytest.raises(ConfigError, match="no such run"):
        fixture_runs.resolve_seeds(spec, {"Q": 3})
    for bad in (["E"], ["E=x"], ["E=1", "E=2"], ["=4"]):
        with pytest.raises(ConfigError):
            fixture_runs.parse_seed_overrides(bad)
    assert seed_search.parse_seed_range("3:5") == range(3, 6)
    for bad in ("5", "5:3", "a:b"):
        with pytest.raises(ConfigError):
            seed_search.parse_seed_range(bad)


def test_segment_configs_are_made_from_the_spec_with_no_absolute_path(spec, built, tmp_path, monkeypatch):
    for layout in ("partitioned", "round-robin"):
        for n in (1, 2, 3):
            for suffix in (".bin", ".idx"):
                target = built / "pipeline" / layout / "shards" / f"shard-{n}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"stand-in")
    env = tmp_path / "env"
    snapshot = env / "hf_home" / "hub" / "models--gpt2" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "vocab.json").write_text("{}")
    (snapshot / "merges.txt").write_text("")
    monkeypatch.setenv("WATERWHITE_MEGATRON_ENV", str(env))
    tokenizer = fixture_runs.find_tokenizer(None)
    seeds = fixture_runs.resolve_seeds(spec, {"E": 9})
    out = tmp_path / "out"
    planned = fixture_runs.plan_segments(
        spec, list(spec.runs.values()), seeds, fixture=built, root=out / "runs", tokenizer=tokenizer
    )
    assert [p.folder.relative_to(out).as_posix() for p in planned] == [
        "runs/E/seg-1", "runs/E/seg-2", "runs/E-rr/seg-1", "runs/E-rr/seg-2", "runs/D/seg-1", "runs/Y/seg-1", "runs/K/seg-1",
    ]
    assert str(tmp_path) not in json.dumps([p.raw for p in planned])
    e2, rr2, k = planned[1], planned[3], planned[6]
    assert e2.raw["shards"][2] == "../../../../built/pipeline/partitioned/shards/shard-3"
    assert rr2.raw["shards"][0] == "../../../../built/pipeline/round-robin/shards/shard-1"
    assert e2.raw["tokenizer"]["vocab_file"] == "$WATERWHITE_MEGATRON_ENV/hf_home/hub/models--gpt2/snapshots/rev1/vocab.json"
    assert (e2.raw["settings"]["random_seed"], rr2.raw["settings"]["random_seed"], k.raw["settings"]["random_seed"]) == (9, 9, spec.runs["K"].random_seed)
    assert e2.raw["step_record"] == {"lowered": [{"step": 8, "last_sample_tokens_read": 16}], "checkpoints": [{"name": "E@9", "step": 9}]}
    assert rr2.raw["parent_checkpoint"] == "E-rr@6" and rr2.segment.parent_checkpoint == "E-rr@6"
    assert planned[2].raw["step_record"] == {"lowered": [], "checkpoints": [{"name": "E-rr@6", "step": 6}]}
    assert rr2.raw["step_record"] == {"lowered": [], "checkpoints": []}
    assert e2.raw["outputs"]["log"] == "oracle_log.jsonl" and e2.raw["cache_dir"] == "cache"
    assert e2.segment.data.cache_dir == out / "runs" / "E" / "seg-2" / "cache"
    assert e2.segment.data.shards[0] == built / "pipeline" / "partitioned" / "shards" / "shard-1"
    assert [(l.consumed_at_start, l.first_step, l.last_step, l.data_parallel_size) for l in k.segment.launches] == [(0, 1, 3, 2), (12, 4, 40, 2)]
    assert (k.raw["settings"]["dataloader_type"], k.raw["settings"]["micro_batch_size"], k.raw["settings"]["train_iters"]) == ("cyclic", 2, 40)
    # the shard copies inside a mixture folder, when asked for
    mixed = fixture_runs.shard_prefix(built, spec.runs["E-rr"].segments[1], spec.runs["E-rr"].segments[1].shards[2], "mixtures")
    assert mixed == built / "mixtures" / "E-rr" / "mix-2" / "shard-3"
    with pytest.raises(ConfigError, match="found 2"):
        (snapshot.parent / "rev2").mkdir()
        (snapshot.parent / "rev2" / "vocab.json").write_text("{}")
        fixture_runs.find_tokenizer(None)
    with pytest.raises(ConfigError, match="--runs"):
        fixture_runs.select_runs(spec, ["E", "Q"])
    assert [run.name for run in fixture_runs.select_runs(spec, ["K", "E"])] == ["E", "K"]


def test_the_fragment_is_toml_in_the_shape_of_the_positions_tables(spec):
    world = World(spec)
    _k_reads_t_at_step_5(world)
    outcomes = evaluate(world.evidence())
    tables = seed_search.positions_tables(spec, outcomes)
    assert list(tables) == [name for name, table in spec.raw["positions"].items() if table["conditions"]]
    assert tables["K"]["holds"]["22"] is False and tables["T"]["holds"] == {"4": True, "5": True, "6": True}
    seeds = {
        "E": {"seed": 17, "accepted": True, "note": ""},
        "K": {"seed": 4, "accepted": False, "note": "condition(s) 22 fail"},
    }
    text = seed_search.toml_fragment(spec, seeds, tables, ["Status: NOT ACCEPTED."])
    parsed = tomllib.loads(text)
    assert parsed["runs"] == {"E": {"random_seed": 17}, "K": {"random_seed": 4}}
    assert "# run E-rr has the seed of run E" in text and "NOT ACCEPTED: condition(s) 22 fail" in text
    t = parsed["positions"]["T"]
    assert (t["conditions"], t["sample_pos"], t["ordinal"], t["token_end"]) == ([4, 5, 6], 1, 6, 22)
    assert t["first_row"]["record"] == "Z-072" and [n["record"] for n in t["neighbours"]] == ["Z-072", "Z-073"]
    # the log lines are the oracle's lines, each a JSON object, in the order of the log
    lines = [json.loads(line) for line in t["log_lines"]]
    assert lines[:3] == [line for line in world.rows["E/seg-2"] if (line["step"], line["sample_pos"]) == (7, 1)]
    assert parsed["positions"]["step8_cut"]["token"] == 16
    assert parsed["positions"]["shared_token"]["record"] == "Z-084"
    assert parsed["positions"]["round_robin"]["manifest_line_of_T"] == {"line": 7, "handle": "T", "origin": ["T"]}
    assert "# condition 22 FAILS: run K first reads T at step 5" in text
    assert seed_search.toml_value({"a b": [1, True, "x\"y"]}) == '{ "a b" = [1, true, "x\\"y"] }'
    with pytest.raises(TypeError):
        seed_search.toml_value(None)
