"""The pipeline replica (plan 5.2): test infrastructure, not the tool.

It is this project's own implementation of the published steps of a pretraining
data pipeline: list the JSONL files, shuffle, deal round-robin into chunks, merge,
shuffle, sanitize with logged skips, tokenize each chunk with Megatron's
preprocessing script, and select mixtures by directory. Two steps are added for the
fixture: an exact dedup on normalised text and a generation stand-in that emits
fixed outputs with no model call.

In every mode the sanitizer takes its normaliser from ``waterwhite.normalise``
(DECISIONS.md D-17); with capture off that is all the replica loads of the tool.
Three modules write to the tool, through its public packages: with capture on,
``replica.capture_hooks`` calls the store, the capture wrappers, the harness, the
tokenization maps and, when asked, the confined harness; ``replica.recover`` puts
a run made without capture into a store afterwards through the tool's recovery
calls; ``replica.pipeline_links`` holds what the two share. ``replica.scratch``
builds the two scratch stores of the hand-written table with the same calls;
``replica.probe_run`` registers a mixture in a store and starts the tool's
``register-run`` for one run over a generated set.

``replica.runs`` is the run wrapper (docs/replica.md, section 13): it stands in for the
thin wrapper around a training launch, has Megatron's own dataset code write the
data-order caches in a subprocess of the Megatron environment, and writes the run
record and the step record. The replica process itself still imports neither
Megatron nor torch; it imports nothing of the oracle. ``replica.register_runs`` starts
``waterwhite register-run`` for each segment. ``replica.probe_run`` (docs/replica.md
section 14) makes one run over a generated set through the same launch path.

See docs/replica.md.
"""

__all__ = ["__version__"]

__version__ = "2"
