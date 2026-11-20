"""The six subjects driven by the runner in one pass (plan 5.5 Check 10; table README, section
"Check 10": the six subjects, "What a run hands a subject, and its conditions", "A failed attempt,
the two routes after it, and when Check 10 is complete"; docs/baselines.md, the section on the
subjects).

Nothing of a tool is started here. Every command goes to a stand-in starter that records it, every
subject is a stand-in that reads hand-written values (the one real class in a test is
``baselines.combo_originblame.OriginBlameTrace`` over a stand-in of its tool half, in the tests of
what it is served), and every file lives under ``tmp_path``. What these tests expect comes from the
table, the README and the documents, never from running the code under test. A module that does
not exist yet fails a test on an assertion, not on an import: each test looks a name up in its own
body.

The seam is the one of the runner's own tests: ``run_pass(RunConfig(...), starter)`` with a factory
for each subject name, called with a ``SubjectContext``. What this file asks of the runner:

- ONE pass, ONE sealed record. ``config.subjects`` may name any of the six subjects of the table, in
  any order. The drivers are built and run in one fixed order, whatever that order is: plain
  ``dolma``, plain ``originblame``, then ``manifest``, ``trace``, ``dolma+trace``,
  ``originblame+trace``. The gates of the two tools open in that order too.
- A combination named without its plain half (``dolma+trace`` without ``dolma``,
  ``originblame+trace`` without ``originblame``) is refused before any command starts: a
  ``PreRunStop`` whose ``stage`` is ``subjects``.
- ``python -m baselines`` no longer refuses a subject of the table by name.
- The plain subjects are read the same whether two or six subjects are driven: their lines (none
  lines and step ids included), their kept steps with their ids, and ``attempts.jsonl`` (equal
  apart from the question ids the combinations add). A step a later driver logs under
  ``log.shared(target)`` is a step of that driver's lines and of no plain line.
- A combination reuses the runs of its plain half: its context starts with the plain half's
  ``ran_on``, ``failed_runs``, ``second_runs_on`` and ``failed_second_runs`` (the same failed
  ``Attempt`` objects), so that its none lines stand on the same attempts.
- A pass that stops (any error that leaves ``run_pass``) leaves a record that names the stop: the
  file ``stopped.json`` of the run record (listed in ``sha256sums.txt``) holds ``error`` (its text),
  ``subject`` (the driver the pass was in) and ``question`` (the id of the question being answered,
  None where the error came while a driver was being built).
- ``context.keep_fact(kind, target, outcome, attempt_id=None)`` (kind ``export``, ``half`` or
  ``shard``) keeps one fact of the subject the context belongs to; the pass writes them as the lines
  of ``subject-facts.jsonl`` (``subject``, ``kind``, ``target`` as a list, ``outcome``,
  ``attempt_id``), sealed, and they are written when the pass stops too.
- ``baselines.subjects.twin_id(question)`` gives, for a question of ``originblame+trace`` in one of
  the twelve groups the combination reads from the tool half, the id of the question of the plain
  ``originblame`` that serves it (the same listed row and size) where the plain row has a record,
  else None. ``OriginBlameTrace(..., served=function)`` takes ``served(question)`` (the plain
  ``Reading`` or failed ``Attempt`` of the twin, or None) and, for such a question with a twin,
  answers from it and calls its tool half for nothing; the line is graded as a question of the
  combination and cites the steps of the twin's line.
- ``baselines.runner._AttributingRunner._kept(argv)()`` keeps the ``driver_meta.json`` of the store
  a step names with ``--store``, with the entry files.
- ``baselines.subjects.trace_factory(documents, *, place_of)``: ``documents`` maps the key
  ``<kind>-<name>`` of a store or set (``store-main``, ``set-small``) to its trace document, or to
  ``{"none": reason}`` for a store that has no run registered; the driver's factory marks the
  subject as having run on each target it holds.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
import tomllib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"

COMMIT = "0123456789abcdef0123456789abcdef01234567"
LOCK = ["lock-prefix"]

MAIN = ("store", "main")
SCRATCH = ("store", "scratch-500")
SWAP = ("store", "swap")

SIX = ("manifest", "trace", "dolma", "originblame", "dolma+trace", "originblame+trace")
ORDERED = ("dolma", "originblame", "manifest", "trace", "dolma+trace", "originblame+trace")
SCRAMBLES = (
    ("originblame+trace", "manifest", "dolma+trace", "originblame", "trace", "dolma"),
    ("trace", "originblame", "dolma+trace", "dolma", "originblame+trace", "manifest"),
)
PLAIN_OF = {"dolma+trace": "dolma", "originblame+trace": "originblame"}


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    found = getattr(need(module_name), attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def _fn(name):
    return need_attr("baselines.runner", name)


# --- The table, as the runner's own tests read it ----------------------------------------------------


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _read_toml(TABLE_DIR / "check10.toml")["row"]}


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, tuple[str, int]]:
    found = {}
    for group in _read_toml(TABLE_DIR / "check10.toml")["group"]:
        for listed in group["rows"]:
            found[listed] = (group["id"], group["check"])
    return found


@lru_cache(maxsize=1)
def _group_rows() -> dict[str, list[str]]:
    return {group["id"]: list(group["rows"]) for group in _read_toml(TABLE_DIR / "check10.toml")["group"]}


def _question_id(listed_row: str, subject: str, size=None) -> str:
    base = f"{listed_row}|{subject}"
    return base if size is None else f"{base}@{size}"


def _make_question(listed_row: str, subject: str, size=None):
    """The question that puts ``listed_row`` to ``subject`` at ``size``, its fields taken from
    the subject's row of the group in ``check10.toml``."""
    question_type = need_attr("baselines.interfaces", "Question")
    group, check = _group_of_listed()[listed_row]
    row = _check10_rows()[f"c10.{group}.{subject}"]
    ordinal = row.get("ordinal")
    return question_type(
        id=_question_id(listed_row, subject, size),
        group=group,
        row_id=row["id"],
        listed_row=listed_row,
        subject=subject,
        check=check,
        size=size,
        excluded_from_tally=row["excluded_from_tally"],
        route=row.get("route"),
        missing=row.get("missing"),
        needs_ordinal=row["needs_ordinal"],
        marks=tuple(row["marks"]),
        tie_mark=ordinal["shown"]["mark"] if ordinal is not None else None,
    )


def _target_of(question):
    return tuple(need_attr("baselines.record", "attempt_target")(question))


def _reading(question, raw):
    return need_attr("baselines.interfaces", "Reading")(question_id=question.id, raw=raw, route=question.route)


def _grade(status, evidence=None):
    return need_attr("baselines.interfaces", "Grade")(status=status, evidence=evidence)


def _failed(attempt_id):
    return need_attr("baselines.interfaces", "Attempt")(
        id=attempt_id, command=("tool", "run"), exit_code=1, output="the run wrote no companion row")


# --- Stand-ins: a starter, the folders a pre-run reads ----------------------------------------------------


def _result(code, output="", seconds=1.0):
    return SimpleNamespace(exit_code=code, output=output, seconds=seconds)


def _is_probe(argv) -> bool:
    return "-I" in argv and "-c" in argv


class Starter:
    """Starts nothing: it records every argv and answers from a script. The probe (the copy's python
    alone, with ``-I -c``) writes the two lines of its record file, which is the last argument."""

    def __init__(self, *, probe=None):
        self.argvs = []
        self.probe = probe

    def run(self, argv, timeout=None):
        self.argvs.append(list(argv))
        if _is_probe(argv) and self.probe is not None:
            Path(argv[-1]).write_text(f"{self.probe[0]}\n{self.probe[1]}\n", encoding="utf-8")
            return _result(0, "", 0.25)
        return _result(0, "", 1.0)

    def start(self, argv):
        self.argvs.append(list(argv))
        raise AssertionError("no stand-in subject starts a writer")


def _world(tmp_path, *, name="dolma"):
    """The folders a pre-run reads: a work folder with the copy of an environment (its python a link
    to a binary outside the work folder), the setup's record folder (``compare.json`` and
    ``identity.json``) and a run record folder."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    work = tmp_path / "work"
    work.mkdir()
    uv = tmp_path / "uv" / "cpython-3.12.13" / "bin"
    uv.mkdir(parents=True)
    binary = uv / "python"
    binary.write_bytes(b"stand-in interpreter")
    env = work / f"env-{name}"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(binary)
    (env / "pyvenv.cfg").write_text(f"home = {uv}\nversion_info = 3.12.13.final.0\n", encoding="utf-8")
    sha = hashlib.sha256(b"stand-in interpreter").hexdigest()
    record = tmp_path / "setup-record" / name
    record.mkdir(parents=True)
    (record / "compare.json").write_text(json.dumps({"equal": True, "differences": []}, indent=2) + "\n",
                                         encoding="utf-8")
    identity_file = record / "identity.json"
    identity_file.write_text(json.dumps({"env": f"venv-{name}", "interpreter_sha256": sha, "files_checked": 3},
                                        indent=2) + "\n", encoding="utf-8")
    run_record = tmp_path / "run-record"
    run_record.mkdir()
    return SimpleNamespace(
        work=work, env=env, record=record, identity_sha=hashlib.sha256(identity_file.read_bytes()).hexdigest(),
        probe_path=str(binary), probe_sha=sha, run_record=run_record, name=name)


def _setup(w):
    return _fn("ToolSetup")(env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha)


def _add_tool(w, name, *, program=False):
    """The setup of a second tool inside the same work folder: a copy of the environment, the record
    of the copy, and for OriginBlame a program the identity file names by its sha256."""
    env = w.work / f"env-{name}"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(Path(w.probe_path))
    (env / "pyvenv.cfg").write_text((w.env / "pyvenv.cfg").read_text(encoding="utf-8"), encoding="utf-8")
    record = w.record.parent / name
    record.mkdir(parents=True)
    (record / "compare.json").write_text(json.dumps({"equal": True, "differences": []}, indent=2) + "\n",
                                         encoding="utf-8")
    fields = {"env": f"venv-{name}", "interpreter_sha256": w.probe_sha, "files_checked": 3}
    program_file = None
    if program:
        program_file = w.work.parent / "ob-program"
        program_file.write_bytes(b"stand-in program")
        fields["ob_binary_sha256"] = hashlib.sha256(b"stand-in program").hexdigest()
    identity_file = record / "identity.json"
    identity_file.write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")
    return _fn("ToolSetup")(env_dir=env, env_record=record,
                            identity_sha256=hashlib.sha256(identity_file.read_bytes()).hexdigest(),
                            ob_binary=program_file)


# --- Stand-in subjects --------------------------------------------------------------------------------------


class PlainStandIn:
    """A plain subject: a step of its own for each reading, the failed attempt of its context for a
    target whose run failed."""

    def __init__(self, name, context):
        self.name = name
        self.context = context

    def read(self, question):
        failed = self.context.failed_runs.get(_target_of(question))
        if failed is not None:
            return failed
        self.context.runner.run([f"{self.name}-read", question.listed_row])
        return _reading(question, {"answer": f"{self.name} read {question.listed_row}"})


class ComboStandIn:
    """A combination: the failed attempt of its plain half (the same id) for a target that failed,
    else a step of its own and a reading."""

    def __init__(self, name, context, failing):
        self.name = name
        self.context = context
        self.failing = set(failing)

    def read(self, question):
        target = _target_of(question)
        if target in self.failing:
            return _failed(f"{PLAIN_OF[self.name]}-run-{'-'.join(target)}")
        self.context.runner.run([f"{self.name}-read", question.listed_row])
        return _reading(question, {"answer": f"{self.name} read {question.listed_row}"})


class DocStandIn:
    """A subject of documents (manifest, trace): it reads one value, or raises for one question."""

    def __init__(self, name, raises_on=None):
        self.name = name
        self.raises_on = raises_on

    def read(self, question):
        if question.id == self.raises_on:
            raise ValueError("the stand-in reader failed")
        return _reading(question, {"answer": f"{self.name} read {question.listed_row}"})


def _keep(context, fact):
    """One fact (kind, target, outcome, attempt id) kept through the context of the subject."""
    keep = getattr(context, "keep_fact", None)
    assert keep is not None, "SubjectContext has no keep_fact"
    keep(fact[0], fact[1], fact[2], attempt_id=fact[3])


def _plain_factory(name, built, *, failing=(), facts=()):
    def factory(context):
        built.append(name)
        for target in (MAIN, SCRATCH):
            with context.log.shared(target):
                context.runner.run([f"{name}-run", "-".join(target)])
            if target in failing:
                context.failed_runs[target] = _failed(f"{name}-run-{'-'.join(target)}")
            else:
                context.ran_on.add(target)
        for fact in facts:
            _keep(context, fact)
        return PlainStandIn(name, context)

    return factory


def _combo_factory(name, built, *, failing=(), facts=()):
    def factory(context):
        built.append(name)
        with context.log.shared(MAIN):
            context.runner.run(["waterwhite", "show", "E/seg-1", "--trace", name])
        for fact in facts:
            _keep(context, fact)
        return ComboStandIn(name, context, failing)

    return factory


def _doc_factory(name, built, *, raises_on=None, error=None, facts=()):
    def factory(context):
        built.append(name)
        if error is not None:
            raise error
        # A subject of documents runs no tool: its gate is never opened, so it starts no step.
        for target in (MAIN, SCRATCH):
            context.ran_on.add(target)
        for fact in facts:
            _keep(context, fact)
        return DocStandIn(name, raises_on)

    return factory


def _factories(built, *, failing=None, raises_on=None, errors=None, facts=None):
    failing = failing or {}
    errors = errors or {}
    facts = facts or {}
    return {
        "dolma": _plain_factory("dolma", built, failing=failing.get("dolma", ()), facts=facts.get("dolma", ())),
        "originblame": _plain_factory("originblame", built, failing=failing.get("originblame", ()),
                                      facts=facts.get("originblame", ())),
        "manifest": _doc_factory("manifest", built, raises_on=raises_on, error=errors.get("manifest"),
                                 facts=facts.get("manifest", ())),
        "trace": _doc_factory("trace", built, raises_on=raises_on, error=errors.get("trace"),
                              facts=facts.get("trace", ())),
        "dolma+trace": _combo_factory("dolma+trace", built, failing=failing.get("dolma", ()),
                                      facts=facts.get("dolma+trace", ())),
        "originblame+trace": _combo_factory("originblame+trace", built, failing=failing.get("originblame", ()),
                                            facts=facts.get("originblame+trace", ())),
    }


IDENTITY_ROWS = ("c05.unmutated.E/shard-1", "c04.count-499", "c04.declare-499.write", "c07.report.R1.X")


def _questions(rows=IDENTITY_ROWS):
    return [_make_question(row, subject) for row in rows for subject in SIX]


def _prepare(tmp_path, subjects, questions, factories, **changes):
    """The config of one pass over the folders of ``tmp_path``, and what its tests read."""
    w = _world(tmp_path, name="dolma")
    setups = {"dolma": _setup(w), "originblame": _add_tool(w, "originblame", program=True)}
    arguments = dict(
        commit=COMMIT, work=w.work, run_record=w.run_record, lock_prefix=LOCK, subjects=tuple(subjects),
        step_cap_seconds=600.0, setups=setups, factories=factories, questions=list(questions),
        grade_of=lambda question, reading: _grade("right"), fault_putter=lambda copy, fault_id: {},
    )
    arguments.update(changes)
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    return _fn("RunConfig")(**arguments), starter, w


def _drive(tmp_path, subjects, questions, factories, **changes):
    config, starter, w = _prepare(tmp_path, subjects, questions, factories, **changes)
    result = _fn("run_pass")(config, starter)
    return SimpleNamespace(result=result, folder=w.run_record, starter=starter, w=w)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --- The registry and the fixed order -----------------------------------------------------------------------


def test_the_drivers_are_built_in_one_fixed_order_whatever_order_the_option_names(tmp_path):
    orders = []
    for number, subjects in enumerate(SCRAMBLES):
        built = []
        _drive(tmp_path / f"scramble-{number}", subjects, _questions(), _factories(built))
        assert sorted(built) == sorted(SIX), built
        # Plain Dolma, plain OriginBlame, then the rest.
        assert built[:2] == ["dolma", "originblame"], built
        orders.append(built)
    # The order is one list, whatever order the option names.
    assert orders[0] == orders[1], orders


@pytest.mark.parametrize(
    "subjects",
    [("dolma+trace",), ("originblame+trace",), ("originblame", "dolma+trace"), ("trace", "dolma", "originblame+trace")],
)
def test_a_combination_named_without_its_plain_half_stops_before_any_command_starts(tmp_path, subjects):
    built = []
    config, starter, w = _prepare(tmp_path, subjects, _questions(), _factories(built))
    with pytest.raises(_fn("PreRunStop")) as caught:
        _fn("run_pass")(config, starter)
    assert caught.value.stage == "subjects" and caught.value.reason
    # Not even a probe was started, and nothing was built.
    assert starter.argvs == []
    assert built == []


@pytest.mark.parametrize("name", ["manifest", "trace", "dolma+trace", "originblame+trace"])
def test_the_command_no_longer_refuses_a_subject_of_the_table_by_name(tmp_path, name):
    main = _fn("main")
    argv = ["--subjects", name, "--step-cap", "60", "--work", str(tmp_path / "work"),
            "--run-record", str(tmp_path / "record"), "--lock-prefix", "lock-prefix",
            "--world", str(tmp_path / "no-such-world.json")]
    with pytest.raises(BaseException) as caught:
        main(argv)
    assert "cannot be run by it yet" not in str(caught.value), str(caught.value)


# --- The plain subjects read the same whatever else is driven ------------------------------------------------


def _plain_steps(folder):
    return {step["id"]: step for step in _jsonl(Path(folder) / "steps.jsonl")
            if step["command"] and str(step["command"][0]).startswith(("dolma-", "originblame-"))}


def test_the_plain_subjects_read_the_same_whether_two_or_six_subjects_are_driven(tmp_path):
    questions = _questions()
    failing = {"dolma": {SCRATCH}}
    built_two, built_six = [], []
    two = _drive(tmp_path / "two", ("originblame", "dolma"), questions, _factories(built_two, failing=failing))
    six = _drive(tmp_path / "six", SCRAMBLES[0], questions, _factories(built_six, failing=failing))
    for subject in ("dolma", "originblame"):
        mine = [line for line in two.result.lines if line["subject"] == subject]
        assert mine, subject
        assert [line for line in six.result.lines if line["subject"] == subject] == mine
    # The kept steps of the plain subjects: the same ids, the same commands, exits and outputs.
    steps_two, steps_six = _plain_steps(two.folder), _plain_steps(six.folder)
    assert steps_two and steps_two == steps_six
    # The failed attempts: equal apart from the question ids the combinations add.
    attempts_two = {line["id"]: line for line in _jsonl(two.folder / "attempts.jsonl")}
    attempts_six = {line["id"]: line for line in _jsonl(six.folder / "attempts.jsonl")}
    assert attempts_two and set(attempts_two) <= set(attempts_six)
    for attempt_id, line in attempts_two.items():
        other = dict(attempts_six[attempt_id])
        added = other.pop("question_ids")
        assert set(line["question_ids"]) <= set(added)
        assert {key: value for key, value in line.items() if key != "question_ids"} == other


def test_a_step_a_later_driver_logs_for_a_store_is_a_step_of_its_own_lines_and_of_no_plain_line(tmp_path):
    questions = _questions()
    built = []
    six = _drive(tmp_path, ORDERED, questions, _factories(built))
    by_id = {line["id"]: line for line in six.result.lines}
    exports = {step["id"] for step in _jsonl(six.folder / "steps.jsonl") if step["command"][:2] == ["waterwhite", "show"]}
    assert exports
    plain_none = by_id[_question_id("c07.report.R1.X", "dolma")]
    assert not exports & set(plain_none["step_ids"]), plain_none["step_ids"]
    assert not exports & set(by_id[_question_id("c07.report.R1.X", "originblame")]["step_ids"])
    # The later driver's own line does cite the export it rests on.
    combined = by_id[_question_id("c05.unmutated.E/shard-1", "dolma+trace")]
    assert exports & set(combined["step_ids"]), combined["step_ids"]


# --- A failed tool run on one store ---------------------------------------------------------------------------


def test_a_tool_run_that_fails_on_one_store_leaves_the_plain_line_and_the_combination_line_complete(tmp_path):
    record = need("baselines.record")
    summary = need("baselines.summary")
    questions = _questions()
    failing = {"dolma": {SCRATCH}, "originblame": {SCRATCH}}
    built = []
    six = _drive(tmp_path, ORDERED, questions, _factories(built, failing=failing))
    # Neither a plain line nor a combination line is refused: no failed-attempt line stands without an
    # attempt of its subject on its store.
    done = record.completeness(questions, six.result.lines, six.result.store_attempts)
    wanted = {"dolma": "dolma-run-store-scratch-500", "dolma+trace": "dolma-run-store-scratch-500",
              "originblame": "originblame-run-store-scratch-500",
              "originblame+trace": "originblame-run-store-scratch-500"}
    prefix = need_attr("baselines.interfaces", "NO_ANSWER_PREFIX")
    for question in questions:
        if question.subject in wanted and _target_of(question) == SCRATCH:
            assert done.statuses[question.id] == prefix + wanted[question.subject], question.id
    # The subjects that ran on the store keep their grades and their cannot answer.
    for question in questions:
        if question.subject in ("manifest", "trace"):
            assert done.statuses[question.id] in ("right", "cannot answer"), question.id
    # summary.py reads the same record: each of the four counts its two questions of the store as failed.
    counted = summary.summarize(questions, six.result.lines, six.result.store_attempts)
    for subject in wanted:
        assert counted.count(subject, need_attr("baselines.record", "LABEL_ONCE"), "failed attempt") == 2, subject
    # The attempts file holds the one failed run once, citing the questions of both subjects of each tool.
    attempts = {line["id"]: line for line in _jsonl(six.folder / "attempts.jsonl")}
    cited = set(attempts["dolma-run-store-scratch-500"]["question_ids"])
    assert {_question_id("c04.count-499", "dolma"), _question_id("c04.count-499", "dolma+trace"),
            _question_id("c04.declare-499.write", "dolma+trace")} <= cited


# --- A pass that stops names the stop ------------------------------------------------------------------------


def _stop_of(folder):
    path = Path(folder) / "stopped.json"
    assert path.is_file(), "the sealed record of a stopped pass holds no stopped.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_pass_stopped_by_a_reader_seals_a_record_that_names_the_error_the_driver_and_the_question(tmp_path):
    questions = _questions()
    target = _question_id("c05.unmutated.E/shard-1", "manifest")
    built = []
    config, starter, w = _prepare(tmp_path, ORDERED, questions, _factories(built, raises_on=target))
    with pytest.raises(ValueError):
        _fn("run_pass")(config, starter)
    stop = _stop_of(w.run_record)
    assert "the stand-in reader failed" in str(stop["error"])
    assert stop["subject"] == "manifest" and stop["question"] == target
    sums = (w.run_record / "sha256sums.txt").read_text(encoding="utf-8")
    assert "  stopped.json" in sums and "  answers.jsonl" in sums
    # What was answered before the stop is in the sealed answers, and the question that stopped it is not.
    ids = {line["id"] for line in _jsonl(w.run_record / "answers.jsonl")}
    assert _question_id("c05.unmutated.E/shard-1", "dolma") in ids and target not in ids


def test_a_pass_stopped_while_a_driver_is_built_names_that_driver_and_no_question(tmp_path):
    questions = _questions()
    built = []
    error = ValueError("the stand-in export failed")
    config, starter, w = _prepare(tmp_path, ORDERED, questions, _factories(built, errors={"trace": error}))
    with pytest.raises(ValueError):
        _fn("run_pass")(config, starter)
    stop = _stop_of(w.run_record)
    assert "the stand-in export failed" in str(stop["error"])
    # The question the manifest answered last is not the one the pass was in.
    assert stop["subject"] == "trace" and stop["question"] is None


def test_the_facts_of_the_subjects_are_kept_and_sealed_and_survive_a_stop(tmp_path):
    questions = _questions()
    facts = {
        "trace": [("export", MAIN, "document", None), ("export", SWAP, "refusal", None),
                  ("export", SCRATCH, "failed", "trace-export-3")],
        "originblame": [("shard", MAIN, "failed", "originblame-track-1"),
                        ("shard", MAIN, "failed", "originblame-track-2")],
        "dolma+trace": [("half", SCRATCH, "failed", "dolma-run-store-scratch-500")],
    }
    built = []
    done = _drive(tmp_path / "whole", ORDERED, questions, _factories(built, facts=facts))
    held = {(line["subject"], line["kind"], tuple(line["target"]), line["outcome"], line["attempt_id"])
            for line in _jsonl(done.folder / "subject-facts.jsonl")}
    assert held == {
        ("trace", "export", MAIN, "document", None),
        ("trace", "export", SWAP, "refusal", None),
        ("trace", "export", SCRATCH, "failed", "trace-export-3"),
        ("originblame", "shard", MAIN, "failed", "originblame-track-1"),
        ("originblame", "shard", MAIN, "failed", "originblame-track-2"),
        ("dolma+trace", "half", SCRATCH, "failed", "dolma-run-store-scratch-500"),
    }
    assert "  subject-facts.jsonl" in (done.folder / "sha256sums.txt").read_text(encoding="utf-8")
    # The same facts are in the record of a pass that stops after them: the manifest raises for its
    # first question, after the plain subjects have kept theirs.
    manifest_target = _question_id("c05.unmutated.E/shard-1", "manifest")
    config, starter, w = _prepare(tmp_path / "stopped", ORDERED, questions,
                                  _factories([], facts=facts, raises_on=manifest_target))
    with pytest.raises(ValueError):
        _fn("run_pass")(config, starter)
    kept = {(line["subject"], line["kind"]) for line in _jsonl(w.run_record / "subject-facts.jsonl")}
    assert ("originblame", "shard") in kept
    assert "  subject-facts.jsonl" in (w.run_record / "sha256sums.txt").read_text(encoding="utf-8")


# --- A store with no run, for the trace -------------------------------------------------------------------------


def test_a_store_the_trace_holds_no_run_for_reads_as_the_table_says_for_a_none_line(tmp_path):
    trace_factory = need_attr("baselines.subjects", "trace_factory")
    question = _make_question("c07.swap.report.counts", "trace")
    assert question.route is None and _target_of(question) == SWAP
    reason = "the store holds no registered run"
    factory = trace_factory({"store-swap": {"none": reason}}, place_of=lambda listed: [{}])
    w = _world(tmp_path)
    config = _fn("RunConfig")(
        commit=COMMIT, work=w.work, run_record=w.run_record, lock_prefix=LOCK, subjects=("trace",),
        step_cap_seconds=600.0, setups={}, factories={"trace": factory}, questions=[question],
        grade_of=lambda q, r: _grade("right"))
    result = _fn("run_pass")(config, Starter())
    # The line is the table's own: cannot answer with the row's missing text, from the one attempt.
    (line,) = result.lines
    assert line["status"] == "cannot answer" and line["evidence"] == question.missing and line["raw"] is None
    (attempt,) = [a for a in result.store_attempts if a.subject == "trace"]
    assert tuple(attempt.target) == SWAP and attempt.attempt_id is None


# --- originblame+trace is served from the plain line --------------------------------------------------------------

# The groups the combination reads from the tool half, and the four fault groups whose shard rows it
# reads the same way. The rows named are listed rows of the table whose given names no layout and no set.
SERVED_GROUPS = (
    "c03.spans", "c04.interrupted-write", "c04.count-499", "c05.unmutated.shards", "c05.withdraw-C.unmutated",
    "c06.downstream", "c09.artifact-actions", "c09.successors",
    "c05.f.trace-deleted", "c05.f.trace-half", "c05.f.map-deleted", "c05.f.digest",
)
SERVED_ROWS = (
    "c04.count-499", "c05.unmutated.E/shard-1", "c05.withdraw-C.unmutated.E/shard-3", "c04.kill-after-100",
    "c05.f.map-deleted.E/shard-1", "c05.f.digest.E/shard-1",
)
NO_TWIN_ROWS = ("c05.f.trace-deleted.E/shard-1", "c05.f.trace-half.E/shard-1")


class CountingTool:
    """The tool half of OriginBlame as a stand-in that counts its reads. It is the plain subject too
    (the same object in a pass, as the real tool half is)."""

    name = "originblame"

    def __init__(self):
        self.calls = []
        self.readings = {}
        self.context = None

    def read(self, question):
        self.calls.append(question.id)
        self.context.runner.run(["originblame-read", question.listed_row])
        found = _reading(question, {"answer": f"read of {question.listed_row}"})
        self.readings.setdefault((question.listed_row, question.size), found)
        return found

    def served(self, question):
        return self.readings.get((question.listed_row, question.size))


def _tool_pass(tmp_path, subjects, questions, grades):
    tool = CountingTool()
    built = []
    factories = _factories(built)

    def plain(context):
        tool.context = context
        context.ran_on.add(MAIN)
        context.ran_on.add(SCRATCH)
        return tool

    def combined(context):
        cls = need_attr("baselines.combo_originblame", "OriginBlameTrace")
        extra = {"served": tool.served} if "served" in inspect.signature(cls.__init__).parameters else {}
        return cls(tool, trace_of=lambda listed: {"segments": []}, place_of=lambda listed: [{}], **extra)

    factories["originblame"] = plain
    factories["originblame+trace"] = combined

    def grade_of(question, reading):
        grades.append((question.subject, reading.question_id, question.id, reading.raw))
        return _grade("right")

    done = _drive(tmp_path, subjects, questions, factories, grade_of=grade_of)
    return tool, done


def test_the_combination_makes_no_tool_half_read_for_a_row_its_plain_half_has_a_line_for(tmp_path):
    questions = [_make_question(row, subject) for row in SERVED_ROWS for subject in ("originblame", "originblame+trace")]
    for question in questions:
        assert question.route is not None, question.id
    grades_two, grades_six = [], []
    two_tool, _ = _tool_pass(tmp_path / "two", ("dolma", "originblame"), questions, grades_two)
    six_tool, six = _tool_pass(tmp_path / "six", ORDERED, questions, grades_six)
    # A six-subject run makes no more reads of the tool half than a two-subject run.
    assert len(two_tool.calls) == len(SERVED_ROWS)
    assert len(six_tool.calls) == len(two_tool.calls), six_tool.calls
    # The raw reading of the plain line is served, and the combination's own row grades it.
    combined = [item for item in grades_six if item[0] == "originblame+trace"]
    assert len(combined) == len(SERVED_ROWS)
    for subject, reading_id, question_id, raw in combined:
        assert reading_id == question_id and question_id.endswith("|originblame+trace"), question_id
        assert raw == {"answer": f"read of {question_id.split('|')[0]}"}
    # The line cites the steps of the twin's line.
    by_id = {line["id"]: line for line in six.result.lines}
    for row in SERVED_ROWS:
        twin = by_id[_question_id(row, "originblame")]
        served = by_id[_question_id(row, "originblame+trace")]
        assert set(twin["step_ids"]) <= set(served["step_ids"]), row


def test_a_row_the_plain_half_holds_no_record_for_is_not_served_and_is_read_by_the_tool_half(tmp_path):
    questions = []
    for row in NO_TWIN_ROWS:
        assert _check10_rows()[f"c10.{_group_of_listed()[row][0]}.originblame"]["record"] == "none"
        questions += [_make_question(row, "originblame"), _make_question(row, "originblame+trace")]
    tool, done = _tool_pass(tmp_path, ORDERED, questions, [])
    assert sorted(tool.calls) == sorted(_question_id(row, "originblame+trace") for row in NO_TWIN_ROWS)


def test_the_twin_of_a_combination_question_is_worked_out_from_the_table():
    twin_id = need_attr("baselines.subjects", "twin_id")
    table10 = need("baselines.table10")
    every = table10.expand(table10.load_table10())
    plain = {(q.listed_row, q.size): q for q in every if q.subject == "originblame"}
    served = 0
    for question in every:
        if question.subject == "originblame+trace" and question.group in SERVED_GROUPS:
            twin = plain[(question.listed_row, question.size)]
            wanted = twin.id if twin.route is not None else None
            if question.group.startswith("c05.f.") and "/shard-" not in question.listed_row:
                # Only the shard rows of the four fault groups are read from the tool half.
                wanted = None
            assert twin_id(question) == wanted, question.id
            served += wanted is not None
        else:
            # A walk, a trace-only row and a question that is not the combination's have no twin.
            assert twin_id(question) is None, question.id
    assert served > 0


# --- What a step on a store keeps -----------------------------------------------------------------------------------


def test_a_step_on_a_store_keeps_the_section_hashes_of_the_store_with_its_entry_files(tmp_path):
    attributing = need_attr("baselines.runner", "_AttributingRunner")
    store = tmp_path / "tracked-0"
    folder = store / ".ob" / "token-index.gpt2"
    folder.mkdir(parents=True)
    (folder / "0").write_text("", encoding="utf-8")
    meta = store / "driver_meta.json"
    meta.write_text(json.dumps({"sections": {"gen/T.jsonl": "stand-in hash"}}), encoding="utf-8")
    held = attributing._kept(["driver", "revoke", "--store", str(store)])()
    paths = {Path(path) for path in held.values()}
    assert folder / "0" in paths
    assert meta in paths, "the section hashes a reading of gen/T is made from are not kept"


def test_a_writer_file_beside_the_entry_folder_is_not_an_entry_of_the_store(tmp_path):
    read_store_entries = need_attr("baselines.originblame", "read_store_entries")
    store = tmp_path / "store"
    (store / ".ob" / "token-index.gpt2").mkdir(parents=True)
    (store / ".ob" / "token-index.gpt2.4242").write_text("not an entry\n", encoding="utf-8")
    assert read_store_entries(store) == []


# --- The documents of the part ---------------------------------------------------------------------------------------

BANNED_WORDS: tuple[str, ...] = ()
BANNED_PATTERNS = (
    re.compile(chr(0x2014)),   # the em dash
    re.compile("/ho" + "me/"),   # a local path
    re.compile("Proj" + "ects/"),
)


def _banned_in(text: str) -> list[str]:
    text = text.replace(need_attr("baselines.interfaces", "MARK_TIE"), "")
    found = [word for word in BANNED_WORDS if word in text]
    found += [pattern.pattern for pattern in BANNED_PATTERNS if pattern.search(text)]
    return found


def test_the_temporary_decisions_and_the_temporary_section_are_written():
    text = (REPO / "DECISIONS.md").read_text(encoding="utf-8")
    entries = re.split(r"(?m)^## (?=D-SUBJ-\d+ )", text)[1:]
    assert len(entries) >= 5, "DECISIONS.md holds fewer than five D-SUBJ entries"
    for body in entries:
        for label in ("Built", "Other option", "Result of the other option", "What would change it"):
            assert label in body, body.splitlines()[0]
        assert "traced" in body.lower() or "inferred" in body.lower(), body.splitlines()[0]
        assert _banned_in(body) == [], (body.splitlines()[0], _banned_in(body))
    document = (REPO / "docs" / "baselines.md").read_text(encoding="utf-8")
    assert "(temporary section SUBJECTS)" in document
    assert _banned_in(document) == []
