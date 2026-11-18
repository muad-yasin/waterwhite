"""A checkpoint whose history is whole still answers "does not contain" for a source it
never read (plan 2.2(e), 3.5, 2.4; DECISIONS.md D-50).

D-50 makes a step untraced when a sample's coverage mark claims more tokens than the
sample's stored rows add up to. The rule must fire on that and on nothing else. Here
every sample of every step in the checkpoint's history has rows that add up to its mark
and to its length, so no step is untraced and the negative is built as before. The
precondition is read from the store before the question is put, so that the test cannot
pass on a history that is not whole.
"""

from __future__ import annotations

import ww_world

from waterwhite.records import StepTrace, TraceRow, TraceSample


def samples_of(world, *segments: str) -> list[tuple[str, int, int, int, int]]:
    """(segment, step, sample_tokens, covered_tokens, sum of the rows' token lengths) of
    every stored sample of the segments, read from the store file."""
    found = []
    for name in segments:
        found += [tuple(row) for row in world.store._conn.execute(
            "SELECT s.name, t.step, x.sample_tokens, x.covered_tokens, "
            "COALESCE((SELECT SUM(w.token_end - w.token_start) FROM trace_row w "
            "          WHERE w.trace_id = x.trace_id AND w.sample_pos = x.sample_pos), 0) "
            "FROM trace_sample x JOIN step_trace t ON t.trace_id = x.trace_id "
            "JOIN run_segment s ON s.segment_id = t.segment_id WHERE s.name = ? "
            "ORDER BY t.step, x.sample_pos", (name,))]
    return found


def assert_whole(found: list[tuple[str, int, int, int, int]], steps: int) -> None:
    assert len(found) >= steps > 0, found
    for segment, step, tokens, covered, rows in found:
        assert covered == tokens == rows, (segment, step, tokens, covered, rows)


def test_a_checkpoint_with_a_whole_history_answers_does_not_contain_for_a_source_it_never_read(world, ask):
    """E@3 reads three documents of R2 and Y@2 (which branches from E@3) two of R1; in
    both histories every sample's rows add up to its mark. Neither read A."""
    assert_whole(samples_of(world, "E/seg-1"), steps=3)
    assert_whole(samples_of(world, "E/seg-1", "Y/seg-1"), steps=5)
    for name in ("E@3", "Y@2"):
        answer = ask("A", name)
        assert (answer["answer"], answer["names"]) == ("does not contain", None), (name, answer)


def test_a_sample_of_two_rows_that_add_up_to_its_mark_does_not_make_the_step_untraced(world, ask):
    """A run W reads, in its one step, one sample made of two rows (2 and 3 tokens) of
    two documents of R2, both drawn from B; the mark is the sample's five tokens. The sum
    over a sample's rows is what D-50 compares the mark with, so it is the sum of both
    rows that must match, not either one. The checkpoint answers "does not contain" for
    A and for Z, which R2's third document draws from and the step never read, and
    "contains" for B."""
    mixture = world["mix-1"]
    ww_world.segment(world.store, "W", "W/seg-1", 0, mixture, None, [(1, 1, None)], None)
    world.store.write_step_traces("W/seg-1", [StepTrace(1, 1, (
        TraceSample(0, 1, 0, ww_world.SAMPLE_TOKENS, ww_world.SAMPLE_TOKENS,
                    (TraceRow(0, 0, 0, 2), TraceRow(0, 1, 0, 3))),))])
    world.store.register_checkpoint(name="W@1", segment="W/seg-1", step=1)
    found = samples_of(world, "W/seg-1")
    assert_whole(found, steps=1)
    assert world.scalar("SELECT COUNT(*) FROM trace_row WHERE trace_id = ?",
                        (world.trace_id("W/seg-1", 1),)) == 2
    for source in ("A", "Z"):
        answer = ask(source, "W@1")
        assert (answer["answer"], answer["names"]) == ("does not contain", None), (source, answer)
    assert ask("B", "W@1")["answer"] == "contains"
