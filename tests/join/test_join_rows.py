"""From a sample to its rows (plan 3.5 "step trace", Decision 18).

The expected rows are derived by hand in ``join_kit``'s text, from index arrays
written by hand. Nothing here runs a sampler.
"""

from __future__ import annotations

import join_kit as kit
import numpy as np
import pytest

from waterwhite.join._arrays import IndexArrays
from waterwhite.join._rows import SampleResolver, cut
from waterwhite.records import TraceRow


def resolver(arrays=None, shards: int = 2, lengths=kit.LENGTHS) -> SampleResolver:
    built = IndexArrays.build(kit.ARRAYS if arrays is None else arrays, shards)
    assert built is not None
    return SampleResolver(built, [np.array(x, dtype=np.int32) for x in lengths[:shards]],
                          kit.SEQUENCE_LENGTH, 1)


def resolved(resolver_: SampleResolver, index: int):
    rows, covered = resolver_.resolve(index)
    return [(r.shard_position, r.ordinal, r.token_start, r.token_end) for r in rows], covered


@pytest.mark.parametrize("index", sorted(kit.WORLD))
def test_every_sample_of_the_blend_gives_its_hand_derived_rows(index):
    # The chain: dataset index, dataset sample index, shuffle index, sample index,
    # document index. Each sample covers sequence_length + 1 = 5 tokens.
    assert resolved(resolver(), index) == (kit.WORLD[index], kit.SAMPLE_TOKENS)


@pytest.mark.parametrize("number", sorted(kit.WORLD_SHARD_0))
def test_a_segment_of_one_shard_has_no_blend(number):
    # With one shard the index asked of the dataset is the shard's sample number.
    one = resolver(shards=1)
    assert one._arrays.dataset_index is None and one._arrays.length == 4
    assert resolved(one, number) == (kit.WORLD_SHARD_0[number], kit.SAMPLE_TOKENS)


def test_the_extra_token_is_the_last_token_of_every_sample():
    # g2 is S0 of shard 0: four tokens of D2 and, as its fifth, token 0 of D0.
    # Without the extra token the row of D0 would not exist, and D0 would not
    # count as read by a step that read only g2 (Decision 18).
    rows, covered = resolved(resolver(), 2)
    assert rows[-1] == (0, 0, 0, 1) and covered == 5


def test_a_document_split_across_samples_shares_one_token():
    # D0 of shard 0: S1 (g5) reads its tokens 0 to 5, S2 (g0) reads 4 to 6.
    # Token 4 is in both: it is the last token of S1 and the first of S2.
    r = resolver()
    assert resolved(r, 5)[0] == [(0, 0, 0, 5)]
    assert resolved(r, 0)[0][0] == (0, 0, 4, 6)


def test_the_same_ordinal_in_two_shards_stays_two_documents():
    # g1 reads ordinals 1 and 0 of shard 1, g3 reads ordinals 3 and 1 of shard 0.
    r = resolver()
    assert (1, 1, 4, 5) in resolved(r, 1)[0] and (0, 1, 0, 2) in resolved(r, 3)[0]


def test_an_index_outside_the_dataset_is_not_resolved():
    assert resolved(resolver(), 6) == ([], 0)
    assert resolved(resolver(), -1) == ([], 0)


def test_a_sample_that_points_outside_a_document_is_not_resolved():
    # Shard 0 with D3 shorter than the arrays were built for, 2 tokens: S2 (g0)
    # would end with D3[0:3], past the document's end. The sample is written with
    # no row and a mark of 0, never with a guessed row.
    shorter = ([6, 3, 4, 2], [5, 5])
    assert resolved(resolver(lengths=shorter), 0) == ([], 0)
    # S1 (g5) reads D0 alone, which is unchanged: it resolves.
    assert resolved(resolver(lengths=shorter), 5) == (kit.WORLD[5], 5)


def test_a_sample_whose_rows_hold_more_tokens_than_a_sample_is_not_resolved():
    # D0 longer than the arrays were built for: S2 (g0) reads D0 from token 4 to
    # its end, now 4 tokens, then 3 of D3: 7 tokens for a sample of 5.
    longer = ([8, 3, 4, 5], [5, 5])
    assert resolved(resolver(lengths=longer), 0) == ([], 0)


def test_a_sample_whose_rows_hold_fewer_tokens_keeps_them_and_lowers_the_mark():
    # D0 one token shorter: S2 (g0) reads D0[4:5] and D3[0:3], 4 tokens of 5.
    shorter = ([5, 3, 4, 5], [5, 5])
    assert resolved(resolver(lengths=shorter), 0) == ([(0, 0, 4, 5), (0, 3, 0, 3)], 4)


def test_arrays_of_a_wrong_shape_are_not_readable():
    arrays = dict(kit.ARRAYS)
    arrays[("sample_index", 0)] = np.array([0, 1, 1, 2, 3], dtype=np.int32)
    assert IndexArrays.build(arrays, 2) is None


def row(*values: int) -> TraceRow:
    return TraceRow(*values)


def test_cut_keeps_whole_rows_and_ends_the_row_that_holds_the_next_token():
    # g3: D3[2:5] is 3 tokens, D1[0:2] is 2. With 4 tokens read the second row
    # ends after its first token. With 3 the first row is whole and nothing of the
    # second is kept. With 2 the first row ends at token 4 of D3.
    both = (row(0, 3, 2, 5), row(0, 1, 0, 2))
    assert cut(both, 4) == (row(0, 3, 2, 5), row(0, 1, 0, 1))
    assert cut(both, 3) == (row(0, 3, 2, 5),)
    assert cut(both, 2) == (row(0, 3, 2, 4),)
    assert cut(both, 0) == ()
    assert cut(both, 5) == both
    assert cut(both, 9) == both
