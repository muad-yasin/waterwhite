"""Rows that run and fail, each with a diagnosis.

A row is listed here only after someone has worked out which of three it is:

- ``tool``: the tool is wrong; ``rule`` quotes the rule of the plan it breaks;
- ``row``: the row is wrong; ``rule`` quotes the rule it contradicts;
- ``seam``: a name or a signature between two parts that a revision in progress
  changes; ``rule`` says which.

Neither the row nor the tool is bent to make the test pass. The row's test is
marked as an expected failure that carries the diagnosis. An expected failure is
not a pass, and one that starts to pass shows as XPASS: its entry is then removed.

``fields`` are the expected fields that differ. A failure in any other field is a
new failure and is reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass

VERDICTS = ("tool", "row", "seam")


@dataclass(frozen=True)
class Diagnosis:
    verdict: str
    fields: tuple[str, ...]
    rule: str
    what: str

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"a diagnosis is one of {VERDICTS}, got {self.verdict!r}")

    def reason(self) -> str:
        return f"[{self.verdict}] {self.what} Rule: {self.rule}"


EXPECTED_FAILURES: dict[str, Diagnosis] = {}
