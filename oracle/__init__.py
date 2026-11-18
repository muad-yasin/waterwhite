"""The test-time oracle and the training stand-in (plan 3.5, 5.3).

Test infrastructure, not the tool. It runs Megatron's own dataset and sampler code
and writes down what each completed step reads, and it writes the run record and
the step record a training wrapper would write. It imports nothing from
``waterwhite`` (plan W5): expected answers must not come from the tool's code.

This file imports nothing, so ``oracle.positions`` and ``oracle.digests`` can be
used without the Megatron environment. Only ``oracle.megatron_data`` and the
modules that call it import Megatron and torch.

See docs/oracle.md.
"""
