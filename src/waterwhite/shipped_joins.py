"""The joins that plan Check 1 verified, shipped with the tool (plan 3.5, F6, Check 1;
DECISIONS.md D-26).

A join is the triple (Megatron release tag, sampler type, sampler class) that the
exposure join of plan 3.5 was checked against Megatron's own dataloader for. The list
below is the one Check 1 passed on: the fixture's pinned release, and for it the two
sampler types a run can be read with, each with the sampler class Megatron builds for
it. A store made by ``waterwhite init`` takes this list, or with ``--verified-join`` a
subset of it: a join outside the list is refused (a usage error, nothing written), so no
store made through the command line lists a join Check 1 did not pass for. A store made
through the library (``Store.create``) lists the joins its caller gives and is outside
what the tool verified.

The values are literals of the tool, not read from the fixture's spec at run time: a
store lists what the tool was verified for, whatever fixture files lie beside it. A
release or a sampler class that Check 1 has not passed for is not in the list, and a
segment run on it registers with the cause "data-order trace unverified for this
version" or "for this sampler type" (plan F6).
"""

from __future__ import annotations

from . import vocabulary as v

RELEASE_TAG = "core_v0.20.0"

_SAMPLER_MODULE = "megatron.training.datasets.data_samplers"

SHIPPED_JOINS: tuple[tuple[str, str, str], ...] = (
    (RELEASE_TAG, v.SAMPLER_SEQUENTIAL, f"{_SAMPLER_MODULE}.MegatronPretrainingSampler"),
    (RELEASE_TAG, v.SAMPLER_CYCLIC, f"{_SAMPLER_MODULE}.MegatronPretrainingRandomSampler"),
)
