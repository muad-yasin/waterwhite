"""Every answer of Check 8 and of the cases beside it carries its basis and scope, in both
printed forms, and the real outputs are kept.

The questions are the rows of the hand-written table (``fixture/table/check08.toml``
and the Check 5 and Check 2 rows named below), asked of the stores the acceptance world
builds, through the command line: ``query withdraw`` with ``--format json`` and with
``--format text``. What an answer must carry is the rule of docs/interfaces.md section 6
and docs/coverage.md section 2 (``acceptance_basis.problems``); the expected values are
the rows' own. Each case writes the command, the JSON document and the text into its
temporary folder, so that a run's outputs can be read afterwards.

The cases:

- every answer row of Check 8: a "does not contain" with its scope, a checkpoint's
  "contains" with its basis, first exposure, level and untraced steps, a descendant's
  "contains" inherited from an ancestor, a mixture's "contains", the cyclic run's
  checkpoints, and the record read only through a sample's shared last token (before
  and after the step that read it);
- incomplete histories: a step whose trace is gone (Check 5, state f.trace-deleted),
  where a checkpoint that still shows the read answers "contains" beside the untraced
  steps and a checkpoint that does not answers "may contain"; a run with no data-order
  trace at all (Check 2, state c2.ii);
- a negative whose proof passes generation edges (Check 5, withdraw C), which says what
  was not assessed.
"""

from __future__ import annotations

import json

import acceptance_basis as basis
import acceptance_probe
import acceptance_runs
import pytest
import table
from acceptance_world import NEED_MEGATRON_ENV

# The checkpoint's own run segment, from the fixture spec (runs.*.segments).
OWN_SEGMENT = {"E@6": "E/seg-1", "E@9": "E/seg-2", "D@3": "D/seg-1", "Y@6": "Y/seg-1",
               "K@3": "K/seg-1", "K@8": "K/seg-1", "K@40": "K/seg-1"}

BESIDE = (
    "c05.f.trace-deleted.E@6", "c05.f.trace-deleted.E@9", "c05.f.trace-deleted.D@3",
    "c02.ii.E@9", "c05.withdraw-C.unmutated.E/shard-3",
)


def _rows() -> dict[str, table.Row]:
    rows = {row.id: row for row in table.load_check(8)
            if table.interpret(row).kind == table.KIND_ANSWER}
    for check in (2, 5):
        wanted = {name for name in BESIDE if name.startswith(f"c{check:02d}.")}
        rows |= {row.id: row for row in table.load_check(check) if row.id in wanted}
    return rows


ROWS = _rows()


def test_the_cases_are_the_rows_the_docstring_names():
    assert set(BESIDE) <= set(ROWS)
    assert len([name for name in ROWS if name.startswith("c08.")]) == 11


@pytest.mark.parametrize("name", list(ROWS))
def test_the_answer_carries_its_basis_and_scope(name, world, tmp_path):
    question = table.interpret(ROWS[name])
    absent = world.unmet(question)
    if NEED_MEGATRON_ENV in absent:
        pytest.skip("not run: its store is tokenized only where the Megatron environment is set")
    if absent:
        pytest.skip("not built yet: " + ", ".join(absent))
    if acceptance_probe.rests_on_bound(question) and acceptance_probe.sandbox_missing():
        pytest.skip("not run: it rests on a bound and the sandbox cannot be made")
    expect, empty = table.fill_pending(dict(question.expect), acceptance_runs.spec())
    if empty:
        pytest.skip("not checked: the fixture spec holds no value for " + ", ".join(empty))

    store = world.store_of(question)
    arguments = world.query_arguments(question)
    printed = {form: world.cli(store.store, *arguments, "--format", form)
               for form in ("json", "text")}
    stem = name.replace("/", "_")
    (tmp_path / f"{stem}.command").write_text("waterwhite " + " ".join(arguments) + "\n")
    for form, outcome in printed.items():
        suffix = "json" if form == "json" else "txt"
        (tmp_path / f"{stem}.{suffix}").write_text(outcome.stdout)
        (tmp_path / f"{stem}.{suffix}.exit").write_text(f"{outcome.exit_code}\n")
        assert outcome.exit_code == 0, (form, outcome.stderr[:300])
        assert outcome.stderr == ""
        assert str(world.base) not in outcome.stdout, "an output holds a local path"

    document = json.loads(printed["json"].stdout)
    found = basis.problems(document, printed["text"].stdout.rstrip("\n"), expect,
                           OWN_SEGMENT.get(question.artifact))
    assert found == [], found
