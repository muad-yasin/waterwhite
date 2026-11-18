"""What one recovery run reports."""

from __future__ import annotations

from dataclasses import dataclass

from .. import vocabulary as v
from ..records import StepRef

KIND_COPY_STEP = "copy step"
KIND_SANITIZER = "sanitizer"

# How far the links of a run reach (plan 3.2, 3.3). The words are the
# vocabulary's; they are named here for code that imports them from the package.
OUTCOME_RECORD_LEVEL = v.OUTCOME_RECORD_LEVEL  # every link was derived per record
OUTCOME_FILE_LEVEL = v.OUTCOME_FILE_LEVEL  # degraded: the file is tied to a member, no record is
OUTCOME_UNKNOWN = v.OUTCOME_UNKNOWN  # degraded: nothing is known about the records

OUTCOMES = v.RECOVERY_OUTCOMES


@dataclass(frozen=True)
class RecoveryResult:
    """The report of one recovery run: one step, one later file.

    Counts of the records of the later file (they add up to ``records``):

    - ``recovered``: one candidate left after the direct comparison;
    - ``candidates``: two or more left, all named by one "candidates" edge;
    - ``no_edge``: no edge. Such a record keeps no recorded origin. ``no_text``
      says how many of them hold no text at all (a blank or unreadable line), and
      ``surplus`` how many are copies of a text that the later file holds more
      often than the named inputs do: the count shows an input nobody named, so
      no copy of such a text is given a parent.

    ``collisions_rejected`` counts candidate pairs with equal fingerprints whose
    texts differ. ``drops`` counts the lines the step's log names as skipped.
    ``input_lines_unaccounted`` (a sanitizer run only) counts the lines of the
    input that no edge of the run names and the log does not name either.

    ``edges_written``, ``drops_written`` and ``links_written`` count what this run
    added to the store. A second run of the same recovery finds everything stored
    already: all three are 0 and ``wrote_nothing`` is true.

    ``outcome`` is "record-level" for a run that matched records. A sanitizer run
    whose normaliser is not available degrades (plan 3.3, F8): "file-level" when
    a file family or the step's own log ties the file to members
    (``members_linked``), "unknown" when neither does; ``reason`` then says why.
    ``files_read`` names every file the run opened: registered files by a
    registered name, the log by its path. ``dry_run`` is true for a run that was
    asked for its result only: it wrote nothing, and the three written counts
    are 0 whatever the store holds."""

    kind: str
    step: StepRef
    artifact: str
    instance_id: str
    outcome: str
    reason: str | None
    records: int
    recovered: int
    candidates: int
    no_edge: int
    no_text: int
    collisions_rejected: int
    drops: int
    input_lines_unaccounted: int | None
    edges_written: int
    drops_written: int
    links_written: int
    members_linked: tuple[str, ...]
    files_read: tuple[str, ...]
    surplus: int = 0
    dry_run: bool = False

    @property
    def wrote_nothing(self) -> bool:
        """True when the run added nothing to the store."""
        return self.edges_written == 0 and self.drops_written == 0 and self.links_written == 0

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "step": {
                "name": self.step.name,
                "version": self.step.version,
                "param_digest": self.step.param_digest,
            },
            "artifact": self.artifact,
            "instance": self.instance_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "records": self.records,
            "recovered": self.recovered,
            "candidates": self.candidates,
            "no_edge": self.no_edge,
            "no_text": self.no_text,
            "surplus": self.surplus,
            "collisions_rejected": self.collisions_rejected,
            "drops": self.drops,
            "input_lines_unaccounted": self.input_lines_unaccounted,
            "edges_written": self.edges_written,
            "drops_written": self.drops_written,
            "links_written": self.links_written,
            "wrote_nothing": self.wrote_nothing,
            "members_linked": list(self.members_linked),
            "files_read": list(self.files_read),
            "dry_run": self.dry_run,
        }
