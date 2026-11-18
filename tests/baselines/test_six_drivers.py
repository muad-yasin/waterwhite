"""The six drivers of Check 10 on a world built by the real builder (plan 5.5 Check 10; table
README, section "Check 10": the six subjects, "What a run hands a subject, and its conditions" and
"A failed attempt, the two routes after it, and when Check 10 is complete"; docs/baselines.md, the
sections on the world builder and on the subjects).

One world is built for the whole file by the real command (``python -m baselines.world``, the
smallest size, two workers: the machine holds no more) and then the runner is driven over it
through its own ``main`` with the default ``--subjects``, which names all six. The seam is the
one of ``test_runner.py``: ``main`` builds the run configuration, the real registry of the six
factories included, and calls ``run_pass(config, SubprocessStarter())`` with the module-level
name; the test replaces ``SubprocessStarter`` with a stand-in starter and hands the setups of the
two tools through the real ``--tool`` arguments. The test builds no registry, no driver and no
factory of its own. Only two things wrap the real ones, so that the test can look at them: the
function ``run_pass`` (it hands the configuration on unchanged) and each factory of the registry
(it calls the real factory and notes the context it was given and what it built).

The stand-in starter answers the probe of the copied environment with the two lines the real probe
writes and refuses every command of a tool (exit 1, a hand-written line of output): nothing of
Dolma or OriginBlame is started, and every tool run of the pass is a failed attempt. So what the
test shows is that the paths execute on a world the builder wrote (the reads of ``world.json``, the
plans of the shards, the four drivers that read the world's own files, the registry, the record
writers) and that the record is whole; it asserts no right or wrong of any line, and no line of a
group whose reading the table leaves open. What a successful tool run leads to (the copies, the
revokes, the walks, the stopped writer) is not executed here.

Three passes are made, in this order, on the one world: the pass over all six subjects, a pass
over the two plain subjects alone (what the plain subjects do on their own, to set beside the
six), and a pass whose stand-in starter raises when the first OriginBlame command is started,
which is a raise inside a factory that the pass was building after the plain Dolma subject had
been read.

The values these tests expect come from the table, the README, the fixture spec and the documents,
and from files this file reads itself; never from running the code under test. A module that does
not exist fails a test on an assertion: each test looks a name up in its own body.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tomllib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"
MEGATRON_VARIABLE = "WATERWHITE_MEGATRON_ENV"

COMMIT = "0123456789abcdef0123456789abcdef01234567"
JOBS = 2
BUILD_SECONDS = 2400.0
TOKENIZER_BYTES = b'{"stand_in": "a tokenizer file the world hashes and copies"}\n'
INTERPRETER_BYTES = b"stand-in interpreter"
PROGRAM_BYTES = b"stand-in program"

# The one order the drivers are built and run in (the table README's plain tools first).
ORDER = ("dolma", "originblame", "manifest", "trace", "dolma+trace", "originblame+trace")
CHECK_2_STATES = ("c2.i", "c2.ii", "c2.iii", "c2.iv")

# The three groups the trace reads, and the eight groups the combination of OriginBlame and the
# trace reads from its tool half alone (the plain line of the same listed row serves them).
TRACE_GROUPS = ("c01.positions", "c01.step-facts", "c05.unregistered-checkpoint")
SERVED_GROUPS = ("c03.spans", "c04.interrupted-write", "c04.count-499", "c05.unmutated.shards",
                 "c05.withdraw-C.unmutated", "c06.downstream", "c09.artifact-actions", "c09.successors")
PLACE_KEYS = ("segment", "step", "sample_pos", "shard_position", "ordinal")
POSITIONS_PREFIX = "pending:positions."

SUCCESSOR_ROW = "c09.fixture.successor.E/shard-3"
SUCCESSOR_NAME = "successor of E/shard-3"

NO_ANSWER_PREFIX = "no answer: failed attempt "
GRADES = ("right", "wrong", "cannot answer")

# What stays out of a tracked file, written in pieces so that this file holds none of it.
BANNED_WORDS = (
    "R2" + "A", "ROUND" + "2", "rul" + "ing", "amend" + "ed", "uns" + "ure", "BA" + "RS", "spec " + "1",
    "spec " + "section", "C" + "&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net", "advi" + "sor",
    "/ho" + "me/", "Proj" + "ects/",
)
BANNED_PATTERNS = (
    re.compile(r"\b[GQ]\d\b"),
    re.compile(r"\bitem \d"),
    re.compile(r"\bround \d", re.IGNORECASE),
    re.compile(r"\b(work" + r"flow|ag" + r"ent|br" + r"ief)\b", re.IGNORECASE),
    re.compile(chr(0x2014)),
)


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


# --- The table and the spec, as these tests read them ---------------------------------------------


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _core_rows() -> dict:
    """Every core row of Checks 1 to 9, by id, read from the table files."""
    found = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                found[row["id"]] = row
    return found


@lru_cache(maxsize=1)
def _groups() -> dict[str, list[str]]:
    """The listed rows of each group of Check 10, by group id."""
    return {group["id"]: list(group["rows"]) for group in _read_toml(TABLE_DIR / "check10.toml")["group"]}


@lru_cache(maxsize=1)
def _spec() -> dict:
    return _read_toml(SPEC_FILE)


@lru_cache(maxsize=1)
def _questions():
    table10 = need("baselines.table10")
    return table10.expand(table10.load_table10())


def _target(question) -> tuple[str, str]:
    return tuple(need_attr("baselines.record", "attempt_target")(question))


def _key_target(key: str) -> tuple[str, str]:
    """The target of a key ``store-NAME`` or ``set-SIZE`` of the world's subject files."""
    kind, _, name = key.partition("-")
    return (kind, name)


def _plain_twin(question):
    """The routed question of the plain OriginBlame with the listed row and size of ``question``."""
    for other in _questions():
        if (other.subject == "originblame" and other.listed_row == question.listed_row
                and other.size == question.size and other.route is not None):
            return other
    return None


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _banned_in(text: str) -> list[str]:
    found = [word for word in BANNED_WORDS if word in text]
    found += [pattern.pattern for pattern in BANNED_PATTERNS if pattern.search(text)]
    return found


# --- Checks of a record, each a function that can find a fault in a record it is handed -----------------


def _status_problems(questions, lines, attempts) -> list[str]:
    """What is wrong with ``lines`` as the one status of each question: a question with two lines,
    a line of no question or of another subject, a status that is neither a grade nor a failed
    attempt, and a failed-attempt line that cites no attempt of the record that names its question."""
    by_id = {question.id: question for question in questions}
    asked = {attempt["id"]: set(attempt["question_ids"]) for attempt in attempts}
    problems = [f"two lines for {found}" for found, count in Counter(line["id"] for line in lines).items()
                if count > 1]
    for line in lines:
        question = by_id.get(line["id"])
        if question is None:
            problems.append(f"a line for no question: {line['id']}")
            continue
        if line["subject"] != question.subject:
            problems.append(f"{line['id']}: a line of {line['subject']} for a question of {question.subject}")
        status = line["status"]
        if status.startswith(NO_ANSWER_PREFIX):
            cited = status[len(NO_ANSWER_PREFIX):]
            if line.get("attempt_id") != cited or line["id"] not in asked.get(cited, ()):
                problems.append(f"{line['id']}: {status} cites no attempt of the record that names the question")
        elif status not in GRADES:
            problems.append(f"{line['id']}: the status {status!r} is neither a grade nor a failed attempt")
    return problems


def _trace_line_problems(questions, lines, reached, target_of):
    """How many routed questions of the trace groups are put to the trace and its two halves on a
    store or set each reaches (``reached``: subject to targets), and what is wrong with their lines: no
    line, or a failed attempt where a document was read."""
    asked, problems = 0, []
    for question in questions:
        if (question.subject not in reached or question.route is None or question.group not in TRACE_GROUPS
                or target_of(question) not in reached[question.subject]):
            continue
        asked += 1
        line = lines.get(question.id)
        if line is None:
            problems.append(f"{question.id}: no line")
        elif line["status"] not in GRADES:
            problems.append(f"{question.id}: {line['status']}")
    return asked, problems


def _seal_problems(folder: Path) -> list[str]:
    """What is wrong with the seal of a run record: no ``sha256sums.txt``, a file below the folder
    that it does not list, a listed file that is not there, and a listed hash that is not the
    file's."""
    seal = Path(folder) / "sha256sums.txt"
    if not seal.is_file():
        return ["no sha256sums.txt"]
    listed: dict[str, str] = {}
    problems = []
    for line in seal.read_text(encoding="utf-8").splitlines():
        found = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if found is None:
            problems.append(f"a line of the seal is no hash and name: {line!r}")
        elif found.group(2) in listed:
            problems.append(f"listed twice: {found.group(2)}")
        else:
            listed[found.group(2)] = found.group(1)
    files = {path.relative_to(folder).as_posix() for path in Path(folder).rglob("*")
             if path.is_file() and path != seal}
    problems += [f"not listed: {name}" for name in sorted(files - set(listed))]
    for name, digest in sorted(listed.items()):
        if name not in files:
            problems.append(f"listed and missing: {name}")
        elif _sha256_of(Path(folder) / name) != digest:
            problems.append(f"changed: {name}")
    return problems


def _stop_problems(folder: Path) -> list[str]:
    """What is wrong with a sealed record as the record of a pass that stopped: no ``stopped.json``,
    one without its three keys, or one the seal does not list."""
    path = Path(folder) / "stopped.json"
    if not path.is_file():
        return ["no stopped.json"]
    held = json.loads(path.read_text(encoding="utf-8"))
    problems = [f"stopped.json holds no {key}" for key in ("error", "subject", "question") if key not in held]
    seal = Path(folder) / "sha256sums.txt"
    if not seal.is_file() or "  stopped.json" not in seal.read_text(encoding="utf-8"):
        problems.append("the seal does not list stopped.json")
    return problems


def _start_problems(start: str, subjects, list_sha: str) -> list[str]:
    """What is wrong with a start line as the line of a pass over ``subjects`` and a world whose
    ``WORLD.sha256`` has the sha256 ``list_sha``."""
    problems = []
    if "subjects " + ",".join(subjects) + ";" not in start:
        problems.append(f"the start line does not name the subjects {','.join(subjects)}")
    if "WORLD.sha256" not in start or list_sha not in start:
        problems.append(f"the start line does not name the sha256 of WORLD.sha256 ({list_sha})")
    return problems


def _summary_counts(text: str) -> Counter:
    """The questions of each subject the text of a summary counts, over every label and word (the
    lines of the tally are left out)."""
    found: Counter = Counter()
    pattern = re.compile(r"(\S+) (once|small|10k|100k) (right|wrong|cannot answer|failed attempt|expected value not derived|not attempted): (\d+)")
    for line in text.splitlines():
        parsed = pattern.fullmatch(line)
        if parsed is not None:
            found[parsed.group(1)] += int(parsed.group(4))
    return found


# --- The stand-ins: a starter, and the setups of two tools -------------------------------------------------


def _is_probe(argv) -> bool:
    return "-I" in argv and "-c" in argv


class _RefusedWriter:
    """What a writer that refuses gives: a pid, one line of output, an exit 1 that has already come."""

    pid = 4243

    def __init__(self):
        self.stdout = io.StringIO("the stand-in tool refuses to run\n")

    def poll(self):
        return 1

    def wait(self):
        return 1


class StandInStarter:
    """Starts nothing. The probe of a copied environment (its python alone, ``-I -c``) gets the two
    lines of its record file, which is the last argument; every other command is a command of a
    tool and is refused with exit 1 and a hand-written line, or, where ``raises_on`` says so, makes
    the starter raise. Every command is recorded."""

    def __init__(self, *, probe, raises_on=None):
        self.probe = probe
        self.raises_on = raises_on
        self.probes: list[list[str]] = []
        self.commands: list[list[str]] = []

    def run(self, argv, timeout=None, **keywords):
        words = [str(word) for word in argv]
        if _is_probe(words):
            self.probes.append(words)
            Path(words[-1]).write_text(f"{self.probe[0]}\n{self.probe[1]}\n", encoding="utf-8")
            return SimpleNamespace(exit_code=0, output="", seconds=0.25)
        self.commands.append(words)
        if self.raises_on is not None and self.raises_on(words):
            raise RuntimeError("the stand-in starter fails on this command")
        return SimpleNamespace(exit_code=1, output="the stand-in tool refuses to run", seconds=0.5)

    def start(self, argv, **keywords):
        """A writer that refuses: its first line is the refusal and it ends with exit 1."""
        words = [str(word) for word in argv]
        self.commands.append(words)
        if self.raises_on is not None and self.raises_on(words):
            raise RuntimeError("the stand-in starter fails on this command")
        return need_attr("baselines.runner", "ProcessHandle")(
            _RefusedWriter(), kill=lambda pid, number: None)


def _tool_setups(base: Path, work: Path) -> SimpleNamespace:
    """The folders a pre-run reads for the two tools: a copy of an environment inside the work
    folder (its python a link to a binary outside it), the record of that copy (``compare.json``
    and ``identity.json``), and for OriginBlame a program the identity file names by its sha256.
    Gives the ``--tool`` arguments."""
    uv = base / "uv" / "cpython-3.12.13" / "bin"
    uv.mkdir(parents=True)
    binary = uv / "python"
    binary.write_bytes(INTERPRETER_BYTES)
    interpreter_sha = hashlib.sha256(INTERPRETER_BYTES).hexdigest()
    program = base / "ob-program"
    program.write_bytes(PROGRAM_BYTES)
    program_sha = hashlib.sha256(PROGRAM_BYTES).hexdigest()
    tools: list[str] = []
    for name in ("dolma", "originblame"):
        env = work / f"env-{name}"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python").symlink_to(binary)
        (env / "pyvenv.cfg").write_text(f"home = {uv}\nversion_info = 3.12.13.final.0\n", encoding="utf-8")
        record = base / "setup-record" / name
        record.mkdir(parents=True)
        (record / "compare.json").write_text(json.dumps({"equal": True, "differences": []}, indent=2) + "\n",
                                             encoding="utf-8")
        fields = {"env": f"venv-{name}", "interpreter_sha256": interpreter_sha, "files_checked": 3}
        if name == "originblame":
            fields["ob_binary_sha256"] = program_sha
        identity = record / "identity.json"
        identity.write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")
        tools += ["--tool", name, str(env), str(record), _sha256_of(identity)]
        if name == "originblame":
            tools.append(str(program))
    return SimpleNamespace(tools=tools, program_sha=program_sha, probe=(str(binary), interpreter_sha))


# --- The world and the two passes, made once for the file ----------------------------------------------------

_STATE: dict = {}


def _environment() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO)])
    root = os.environ.get(MEGATRON_VARIABLE, "")
    if root:
        env.setdefault("HF_HOME", str(Path(root) / "hf_home"))
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _build_world(factory) -> SimpleNamespace:
    """The world, built by the real command with two workers. A build that runs over the limit is
    stopped by the id of its process group."""
    root = os.environ.get(MEGATRON_VARIABLE, "")
    if not root:
        pytest.skip(f"needs the Megatron environment: {MEGATRON_VARIABLE} is not set")
    inputs = factory.mktemp("six-inputs")
    tokenizer = inputs / "tokenizer.json"
    tokenizer.write_bytes(TOKENIZER_BYTES)
    directory = factory.mktemp("six-world") / "world"
    log = inputs / "world.log"
    argv = [sys.executable, "-m", "baselines.world", "--out", str(directory), "--megatron-env", root,
            "--tokenizer", str(tokenizer), "--jobs", str(JOBS)]
    with open(log, "wb") as handle:
        process = subprocess.Popen(argv, cwd=REPO, env=_environment(), stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=BUILD_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = "over the limit of the test"
    return SimpleNamespace(code=code, directory=directory, text=log.read_text(encoding="utf-8", errors="replace"),
                           inputs=inputs)


def _watch(key: str, factory, seen: SimpleNamespace):
    """The real factory of the registry, noted: the order they are built in, the context each was
    given and the subject each built."""
    def built(context):
        seen.built.append(key)
        seen.contexts[key] = context
        subject = factory(context)
        seen.subjects[key] = subject
        return subject

    return built


def _drive(name: str, base: Path, directory: Path, setups: SimpleNamespace, starter,
           subjects=None) -> SimpleNamespace:
    """The runner's own ``main`` over the world, its starter the stand-in, with the default
    ``--subjects`` (all six) or the subjects named. What it returned or raised, what it printed,
    the run record and what was seen."""
    runner = need("baselines.runner")
    seen = SimpleNamespace(built=[], contexts={}, subjects={}, named=None, registry=None)
    real = runner.run_pass

    def watching(config, given_starter):
        seen.named = tuple(config.subjects)
        seen.registry = tuple(config.factories)
        config.factories = {key: _watch(key, factory, seen) for key, factory in config.factories.items()}
        return real(config, given_starter)

    folder = base / f"run-record-{name}"
    arguments = [
        "--step-cap", "600", "--commit", COMMIT, "--work", directory / "work", "--run-record", folder,
        "--lock-prefix", "lock-prefix", "--world", directory / "world.json", "--program-sha256", setups.program_sha,
        *setups.tools,
    ]
    if subjects is not None:
        arguments += ["--subjects", *subjects]
    out, err = io.StringIO(), io.StringIO()
    code, error = None, None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "SubprocessStarter", lambda: starter)
        patch.setattr(runner, "run_pass", watching)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = runner.main([str(word) for word in arguments])
            except (Exception, SystemExit) as caught:
                error = caught
    return SimpleNamespace(code=code, error=error, out=out.getvalue(), err=err.getvalue(), folder=folder,
                           starter=starter, seen=seen)


def _made(factory) -> SimpleNamespace:
    """The world and the passes over it, in this order (all six subjects, the two plain subjects
    alone, a raise from the starter), once for the file."""
    if "made" in _STATE:
        return _STATE["made"]
    world = _build_world(factory)
    made = SimpleNamespace(world=world, six=None, two=None, raised=None)
    if world.code == 0:
        base = factory.mktemp("six-pass")
        setups = _tool_setups(base, world.directory / "work")
        made.six = _drive("six", base, world.directory, setups, StandInStarter(probe=setups.probe))
        made.two = _drive("two", base, world.directory, setups, StandInStarter(probe=setups.probe),
                          subjects=("dolma", "originblame"))
        made.raised = _drive("raised", base, world.directory, setups,
                             StandInStarter(probe=setups.probe, raises_on=lambda words: "track" in words))
    _STATE["made"] = made
    return made


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return _made(tmp_path_factory)


def _world_of(made) -> SimpleNamespace:
    assert made.world.code == 0, \
        f"python -m baselines.world --jobs {JOBS} ended with {made.world.code}: {made.world.text[-3000:]}"
    return made.world


def _world_json(made) -> dict:
    return json.loads((_world_of(made).directory / "world.json").read_text(encoding="utf-8"))


def _whole(made) -> SimpleNamespace:
    """The pass over the six subjects, which has to have ended by itself with the summary."""
    _world_of(made)
    six = made.six
    assert six.error is None, f"main raised {type(six.error).__name__}: {six.error}; stderr: {six.err[-1500:]}"
    assert six.code == 0, f"main returned {six.code}, not 0; stderr: {six.err[-1500:]}"
    return six


# --- The tests of the records' checks on inputs that are wrong -----------------------------------------------


def _lines_of(*pairs):
    return [{"id": question_id, "subject": subject, "status": status, "attempt_id": None}
            for question_id, subject, status in pairs]


def test_the_status_check_finds_two_lines_a_stray_line_a_wrong_word_and_a_failed_line_with_no_attempt():
    questions = [SimpleNamespace(id="q1", subject="dolma"), SimpleNamespace(id="q2", subject="manifest")]
    whole = _lines_of(("q1", "dolma", "right"), ("q2", "manifest", "cannot answer"))
    assert _status_problems(questions, whole, []) == []
    assert _status_problems(questions, whole + _lines_of(("q1", "dolma", "wrong")), [])
    assert _status_problems(questions, whole + _lines_of(("q9", "dolma", "right")), [])
    assert _status_problems(questions, _lines_of(("q1", "dolma", "maybe")), [])
    assert _status_problems(questions, _lines_of(("q1", "manifest", "right")), [])
    failed = [{"id": "q1", "subject": "dolma", "status": NO_ANSWER_PREFIX + "a-1", "attempt_id": "a-1"}]
    assert _status_problems(questions, failed, []), "a failed line with no attempt in the record was accepted"
    assert _status_problems(questions, failed, [{"id": "a-1", "question_ids": ["q2"]}])
    assert _status_problems(questions, failed, [{"id": "a-1", "question_ids": ["q1"]}]) == []


def test_the_seal_check_finds_a_changed_byte_an_unlisted_file_and_a_listed_file_that_is_gone(tmp_path):
    folder = tmp_path / "record"
    folder.mkdir()
    (folder / "steps.jsonl").write_text('{"id": "step-1"}\n', encoding="utf-8")
    (folder / "answers.jsonl").write_text('{"id": "q1"}\n', encoding="utf-8")
    (folder / "sha256sums.txt").write_text(
        "".join(f"{_sha256_of(folder / name)}  {name}\n" for name in ("answers.jsonl", "steps.jsonl")),
        encoding="utf-8")
    assert _seal_problems(folder) == []
    (folder / "answers.jsonl").write_text('{"id": "q2"}\n', encoding="utf-8")
    assert _seal_problems(folder) == ["changed: answers.jsonl"]
    (folder / "answers.jsonl").write_text('{"id": "q1"}\n', encoding="utf-8")
    (folder / "summary.txt").write_text("Check 10: open\n", encoding="utf-8")
    assert _seal_problems(folder) == ["not listed: summary.txt"]
    (folder / "summary.txt").unlink()
    (folder / "steps.jsonl").unlink()
    assert _seal_problems(folder) == ["listed and missing: steps.jsonl"]
    (folder / "sha256sums.txt").unlink()
    assert _seal_problems(folder) == ["no sha256sums.txt"]


def test_the_stop_check_finds_a_record_with_no_stopped_file_or_one_the_seal_does_not_list(tmp_path):
    folder = tmp_path / "record"
    folder.mkdir()
    assert _stop_problems(folder) == ["no stopped.json"]
    (folder / "stopped.json").write_text(json.dumps({"error": "e", "subject": "dolma"}), encoding="utf-8")
    (folder / "sha256sums.txt").write_text(f"{'0' * 64}  answers.jsonl\n", encoding="utf-8")
    assert _stop_problems(folder) == ["stopped.json holds no question", "the seal does not list stopped.json"]
    (folder / "stopped.json").write_text(json.dumps({"error": "e", "subject": "dolma", "question": None}),
                                         encoding="utf-8")
    (folder / "sha256sums.txt").write_text(f"{'0' * 64}  stopped.json\n", encoding="utf-8")
    assert _stop_problems(folder) == []


def test_the_start_check_finds_another_order_of_subjects_and_another_sha_of_the_list():
    listed = hashlib.sha256(b"a list of hashes").hexdigest()
    line = f"start: commit {COMMIT}; subjects {','.join(ORDER)}; step cap 600.0 seconds; WORLD.sha256 sha256 {listed}"
    assert _start_problems(line, ORDER, listed) == []
    assert _start_problems(line, tuple(reversed(ORDER)), listed)
    assert _start_problems(line, ORDER, hashlib.sha256(b"another list").hexdigest())
    assert _start_problems(line.replace("WORLD.sha256 sha256", "list sha256"), ORDER, listed)


def test_the_summary_check_counts_the_questions_of_each_subject_and_leaves_the_tally_out():
    text = ("Check 10: open\nonce: open\nsmall: open\n10k: open\n100k: open\nplan sizes: open\n"
            "dolma once right: 2\ndolma small not attempted: 3\ndolma+trace once failed attempt: 4\n"
            "tally dolma once right: 2\nmanifest once cannot answer: 5\n")
    assert _summary_counts(text) == Counter({"dolma": 5, "dolma+trace": 4, "manifest": 5})


# --- The world the builder wrote, read by the loaders of the drivers ---------------------------------------------


def _loaded(world: dict, directory: Path) -> dict:
    subjects = need("baselines.subjects")
    try:
        return subjects.load_subject_inputs(world, directory)
    except Exception as error:
        raise AssertionError(f"load_subject_inputs refused the world file: {type(error).__name__}: {error}") from None


def _documented(world: dict) -> dict[str, dict]:
    return {key: value for key, value in world["subject_inputs"]["trace"].items() if "document" in value}


def test_the_inputs_the_four_other_drivers_read_load_from_the_world_the_real_builder_wrote(made):
    directory = _world_of(made).directory
    world = _world_json(made)
    loaded = _loaded(world, directory)
    held = world["subject_inputs"]
    assert set(loaded["traces"]) == set(held["trace"])
    documented = _documented(world)
    assert documented, "no store or set of the world holds an exported trace document"
    for key, value in held["trace"].items():
        if "document" in value:
            wanted = json.loads((directory / value["document"]["path"]).read_text(encoding="utf-8"))
            assert loaded["traces"][key] == wanted, key
        else:
            assert loaded["traces"][key] == {"none": value["none"]}, key
        assert key in loaded["state_traces"] if "state_traces" in value else key not in loaded["state_traces"], key
        assert set(loaded["failed_exports"].get(key, {})) == set(value.get("failed") or {}), key
    with_states = {key: value for key, value in documented.items() if "state_traces" in value}
    assert with_states, "no trace of the world holds the traces of the starting states of Check 2"
    for key, value in with_states.items():
        assert tuple(sorted(value["state_traces"])) == CHECK_2_STATES, key
        assert set(loaded["state_traces"][key]) == set(CHECK_2_STATES), key
    plain = {key for key in held["manifest"] if not key.startswith("repaired-")}
    assert set(loaded["manifests"]) == plain
    for key in plain:
        value = held["manifest"][key]
        if "built_from" in value:
            text = (directory / value["built_from"]["path"]).read_text(encoding="utf-8")
            assert len(loaded["manifests"][key]) == len([row for row in text.split("\n") if row]), key
        else:
            assert loaded["manifests"][key] is None, key
    assert set(loaded["repaired"]) == {f"store-{key[len('repaired-'):]}" for key in held["manifest"]
                                       if key.startswith("repaired-")}


def test_a_trace_document_of_the_world_with_another_hash_is_refused_by_the_loader(made):
    directory = _world_of(made).directory
    world = _world_json(made)
    subjects = need("baselines.subjects")
    planted = json.loads(json.dumps(world))
    key = sorted(_documented(planted))[0]
    planted["subject_inputs"]["trace"][key]["document"]["sha256"] = "0" * 64
    with pytest.raises(subjects.SubjectInputsError):
        subjects.load_subject_inputs(planted, directory)
    assert _loaded(world, directory)["traces"], "the world as the builder wrote it is refused too"


def test_the_places_the_trace_reads_come_from_the_table_and_the_fixture_spec():
    place_of = need_attr("baselines.subjects", "table_place_of")()
    checked = 0
    for group in TRACE_GROUPS:
        for listed in _groups()[group]:
            row = _core_rows()[listed]
            if group == "c01.positions":
                named = row.get("expect", {}).get("rows")
                if isinstance(named, str) and named.startswith(POSITIONS_PREFIX):
                    lines = _spec()["positions"][named[len(POSITIONS_PREFIX):]]["log_lines"]
                    wanted = [{key: json.loads(text)[key] for key in PLACE_KEYS} for text in lines]
                else:
                    wanted = [{"run": row["given"]["run"]}]
            else:
                wanted = [dict(row["given"])]
            assert place_of(listed) == wanted, listed
            checked += 1
    assert checked >= len(TRACE_GROUPS)


# --- The pass over the six subjects ------------------------------------------------------------------------------


def test_the_pass_over_the_world_the_real_builder_wrote_ends_by_itself_with_a_summary_and_no_stop(made):
    six = _whole(made)
    assert (six.folder / "summary.txt").is_file(), "the pass left no summary"
    assert not (six.folder / "stopped.json").exists(), "a pass that ended by itself holds a stopped.json"
    assert six.out.startswith("start: commit "), six.out[:300]


def test_the_real_registry_builds_the_six_drivers_in_the_fixed_order(made):
    six = _whole(made)
    seen = six.seen
    assert set(seen.registry) == set(ORDER), f"the registry main built holds {sorted(seen.registry)}"
    assert seen.named == ORDER, f"the pass was handed the subjects {seen.named}"
    assert seen.built == list(ORDER), f"the drivers were built in the order {seen.built}"


def test_originblame_plus_trace_is_served_from_the_plain_line_by_the_registry_not_by_the_test(made):
    six = _whole(made)
    seen = six.seen
    context = seen.contexts["originblame+trace"]
    assert callable(context.served), "the context the registry gave originblame+trace has no served function"
    assert context.plain_subject is seen.subjects["originblame"], "its plain half is not the plain subject"
    assert seen.contexts["dolma+trace"].plain_subject is seen.subjects["dolma"]
    assert seen.contexts["dolma+trace"].served is None, "dolma+trace is served from nothing"
    lines = {line["id"]: line for line in _jsonl(six.folder / "answers.jsonl")}
    served = []
    for question in _questions():
        if question.subject != "originblame+trace" or question.route is None:
            continue
        if question.group in TRACE_GROUPS:
            assert context.served(question) is None, f"{question.id} is read from the trace and is not served"
            continue
        if question.group not in SERVED_GROUPS:
            continue
        twin = _plain_twin(question)
        plain = None if twin is None else lines.get(twin.id)
        if plain is None or not plain["status"].startswith(NO_ANSWER_PREFIX):
            continue
        got = context.served(question)
        if got is not None:
            assert got.id == plain["attempt_id"], f"{question.id} is served the attempt {got.id}, not its twin's"
            served.append(question.id)
    assert served, "no question of the eight groups was served the failed attempt of its plain twin"


def test_no_combination_starts_a_tool_of_its_own_so_every_command_is_a_run_of_a_plain_subject(made):
    six = _whole(made)
    world = _world_json(made)
    assert len(six.starter.probes) == 2, "each of the two tools has one probe and no other"
    # One run for each store or set of Dolma and one track for each shard of OriginBlame; the
    # writers the plain OriginBlame starts for its interrupted-write rows are on stores of their own.
    own = [words for words in six.starter.commands if not any("/ob-stores/write-" in word for word in words)]
    assert len(own) == len(world["stores"]) + len(world["shards"]), (
        f"the tools were asked {len(own)} commands, one is wanted for each of the "
        f"{len(world['stores'])} stores and sets of Dolma and the {len(world['shards'])} shards of OriginBlame")
    # The pass over the two plain subjects alone, on the same stand-in tools, makes the same commands:
    # a combination makes no read of the tool half that the plain line does not make.
    two = made.two
    assert two.error is None and two.code == 0, f"the pass over two subjects: {two.error!r} {two.err[-800:]}"
    assert len(six.starter.commands) == len(two.starter.commands), (
        f"six subjects made {len(six.starter.commands)} tool commands, two made {len(two.starter.commands)}")
    assert six.starter.commands == two.starter.commands, "the commands of the two passes are not the same"


def test_the_plain_subjects_have_the_same_lines_steps_and_attempts_whether_two_or_six_are_driven(made):
    six, two = _whole(made), made.two
    assert two.error is None and two.code == 0, f"the pass over two subjects: {two.error!r} {two.err[-800:]}"
    plain = ("dolma", "originblame")
    held = {name: [line for line in _jsonl(run.folder / "answers.jsonl") if line["subject"] in plain]
            for name, run in (("six", six), ("two", two))}
    assert held["six"], "the plain subjects have no line"
    assert held["six"] == held["two"], "the lines of the plain subjects differ between the two passes"
    # The seconds of a writer are measured, not given by the stand-in: they are left out.
    steps = {name: [{key: value for key, value in step.items() if key != "seconds"}
                    for step in _jsonl(run.folder / "steps.jsonl")] for name, run in (("six", six), ("two", two))}
    assert steps["six"] and steps["six"] == steps["two"], "the kept steps differ between the two passes"
    attempts = {name: {a["id"]: a for a in _jsonl(run.folder / "attempts.jsonl")} for name, run in
                (("six", six), ("two", two))}
    assert set(attempts["two"]) <= set(attempts["six"]), "an attempt of the two-subject pass is not in the six"
    for attempt_id, small in attempts["two"].items():
        big = attempts["six"][attempt_id]
        for key in ("command", "exit_code", "output", "step_ids", "commit"):
            assert big[key] == small[key], f"{attempt_id}: {key} differs"
        assert set(small["question_ids"]) <= set(big["question_ids"]), attempt_id


def test_the_pass_over_the_built_world_passes_the_gate_and_its_start_line_names_the_sha_of_the_list(made):
    six = _whole(made)
    directory = _world_of(made).directory
    start = (six.folder / "start.txt").read_text(encoding="utf-8")
    assert _start_problems(start, ORDER, _sha256_of(directory / "WORLD.sha256")) == [], start
    assert "world-hash" not in six.err, six.err


def test_every_question_gets_exactly_one_status_and_the_lines_of_the_pass_are_one_record(made):
    six = _whole(made)
    questions = _questions()
    lines = _jsonl(six.folder / "answers.jsonl")
    attempts = _jsonl(six.folder / "attempts.jsonl")
    assert _status_problems(questions, lines, attempts) == []
    counts = _summary_counts((six.folder / "summary.txt").read_text(encoding="utf-8"))
    asked = Counter(question.subject for question in questions)
    for subject in ORDER:
        assert counts[subject] == asked[subject], (
            f"the summary counts {counts[subject]} questions of {subject}, the table has {asked[subject]}")
    has_route = {question.id for question in questions if question.route is not None}
    routed = Counter(line["subject"] for line in lines if line["id"] in has_route)
    for subject in ORDER:
        assert routed[subject] >= 1, f"{subject} has no line of a question with a route"


def test_the_sealed_record_names_the_six_subjects_and_holds_the_steps_and_the_facts(made):
    six = _whole(made)
    for name in ("start.txt", "steps.jsonl", "subject-facts.jsonl", "answers.jsonl", "attempts.jsonl", "summary.txt"):
        path = six.folder / name
        assert path.is_file() and path.stat().st_size > 0, f"the record holds no {name}, or an empty one"
    assert _seal_problems(six.folder) == []
    start = (six.folder / "start.txt").read_text(encoding="utf-8")
    assert "subjects " + ",".join(ORDER) + ";" in start, start


def test_the_manifest_and_the_trace_keep_an_export_fact_for_each_input_the_world_names(made):
    six = _whole(made)
    held = _world_json(made)["subject_inputs"]
    facts = _jsonl(six.folder / "subject-facts.jsonl")
    for subject, keys in (("manifest", [key for key in held["manifest"] if not key.startswith("repaired-")]),
                          ("trace", list(held["trace"]))):
        exported = {tuple(fact["target"]) for fact in facts if fact["subject"] == subject and fact["kind"] == "export"}
        assert exported == {_key_target(key) for key in keys}, subject
    for fact in facts:
        if fact["subject"] != "trace" or fact["kind"] != "export":
            continue
        value = held["trace"]["-".join(fact["target"])]
        if "document" in value:
            assert not fact["outcome"].startswith("none") and fact["attempt_id"] is None, fact
        elif not fact["outcome"].startswith("segment"):
            assert fact["outcome"].startswith("none"), fact


def test_a_routed_question_of_the_trace_on_a_store_with_a_document_has_a_line_in_the_trace_and_its_halves(made):
    six = _whole(made)
    world = _world_json(made)
    documented = {_key_target(key) for key in _documented(world)}
    # A question is put to a combination only on a store or set its tool half ran on.
    reached = {
        "trace": documented,
        "dolma+trace": documented & {tuple(entry["target"]) for entry in world["stores"]},
        "originblame+trace": documented & {tuple(entry["target"]) for entry in world["shards"]},
    }
    lines = {line["id"]: line for line in _jsonl(six.folder / "answers.jsonl")}
    asked, problems = _trace_line_problems(_questions(), lines, reached, _target)
    assert asked, "the table asks no routed question of the trace groups on a store with a document"
    assert problems == [], f"{len(problems)} of {asked} questions: {problems[:6]}"


def test_the_trace_line_check_finds_a_missing_line_and_a_failed_attempt_where_a_document_was_read():
    def target_of(question):
        return ("store", "main")

    asked_questions = [
        SimpleNamespace(id="a", subject="trace", route="r", group="c01.positions"),
        SimpleNamespace(id="b", subject="dolma+trace", route="r", group="c05.unregistered-checkpoint"),
        SimpleNamespace(id="c", subject="trace", route="r", group="c03.spans"),
        SimpleNamespace(id="d", subject="trace", route=None, group="c01.step-facts"),
    ]
    reached = {"trace": {("store", "main")}, "dolma+trace": {("store", "main")}}
    whole = {"a": {"status": "wrong"}, "b": {"status": "cannot answer"}}
    assert _trace_line_problems(asked_questions, whole, reached, target_of) == (2, [])
    assert _trace_line_problems(asked_questions, {"a": whole["a"]}, reached, target_of) == (2, ["b: no line"])
    failed = {**whole, "b": {"status": NO_ANSWER_PREFIX + "x-1"}}
    assert _trace_line_problems(asked_questions, failed, reached, target_of)[1] == [f"b: {NO_ANSWER_PREFIX}x-1"]
    assert _trace_line_problems(asked_questions, whole, {"trace": set(), "dolma+trace": set()}, target_of) == (0, [])


def test_the_routed_successor_row_gets_a_line_for_originblame_from_the_last_shard_of_the_world(made):
    six = _whole(made)
    world = _world_json(made)
    last = len(world["shards"]) - 1
    assert world["shards"][last]["name"] == SUCCESSOR_NAME and world["shards"][last]["successor"] is True
    asked = [question for question in _questions()
             if question.subject == "originblame" and question.listed_row == SUCCESSOR_ROW]
    assert asked and all(question.route is not None for question in asked), "the table routes no such question"
    lines = {line["id"]: line for line in _jsonl(six.folder / "answers.jsonl")}
    attempts = {attempt["id"]: attempt for attempt in _jsonl(six.folder / "attempts.jsonl")}
    assert _successor_problems([question.id for question in asked], lines, attempts, last) == []


def _successor_problems(question_ids, lines, attempts, last: int) -> list[str]:
    """What is wrong with the lines of the successor row: no line; a line that is not the failed
    attempt of the shard the row reads (the tools refuse every run); an attempt the record does
    not hold; an attempt that is not of the tracked store of the last shard of the world (the
    number of a tracked store is the place of its shard in the list of shards)."""
    problems = []
    for question_id in question_ids:
        line = lines.get(question_id)
        if line is None:
            problems.append(f"{question_id} has no line")
        elif not line["status"].startswith(NO_ANSWER_PREFIX):
            problems.append(f"{question_id}: {line['status']}")
        elif line["attempt_id"] not in attempts:
            problems.append(f"{question_id} cites an attempt the record does not hold")
        elif not re.search(rf"tracked-{last}(?!\d)", " ".join(attempts[line["attempt_id"]]["command"])):
            problems.append(f"the attempt of {question_id} is not of tracked-{last}")
    return problems


def test_the_successor_check_finds_no_line_another_shard_and_an_attempt_the_record_lacks():
    attempts = {"t-3": {"command": ["x", "--store", "<work>/ob-stores/tracked-3"]},
                "t-13": {"command": ["x", "--store", "<work>/ob-stores/tracked-13"]}}
    line = {"status": NO_ANSWER_PREFIX + "t-3", "attempt_id": "t-3"}
    assert _successor_problems(["q"], {"q": line}, attempts, 3) == []
    assert _successor_problems(["q"], {}, attempts, 3) == ["q has no line"]
    assert _successor_problems(["q"], {"q": {"status": "right", "attempt_id": None}}, attempts, 3) == ["q: right"]
    assert _successor_problems(["q"], {"q": line}, {}, 3) == ["q cites an attempt the record does not hold"]
    assert _successor_problems(["q"], {"q": line}, attempts, 1) == ["the attempt of q is not of tracked-1"]
    other = {"status": NO_ANSWER_PREFIX + "t-13", "attempt_id": "t-13"}
    assert _successor_problems(["q"], {"q": other}, attempts, 3) == ["the attempt of q is not of tracked-3"]


# --- A raise inside a factory ------------------------------------------------------------------------------------


def test_a_raise_inside_a_factory_leaves_a_sealed_record_that_names_the_stop_and_grades_nothing(made):
    _world_of(made)
    raised = made.raised
    assert isinstance(raised.error, RuntimeError), f"main gave {raised.code!r} and {raised.error!r}, not the raise"
    assert "the stand-in starter fails" in str(raised.error)
    assert _stop_problems(raised.folder) == []
    stop = json.loads((raised.folder / "stopped.json").read_text(encoding="utf-8"))
    assert "the stand-in starter fails" in stop["error"]
    assert stop["subject"] == "originblame" and stop["question"] is None, stop
    assert _seal_problems(raised.folder) == []
    assert not (raised.folder / "summary.txt").exists(), "a stopped pass was summarised, so it could be graded"
    # What was answered before the stop is kept: the plain Dolma subject was read, and nothing after it.
    lines = _jsonl(raised.folder / "answers.jsonl")
    assert lines and {line["subject"] for line in lines} == {"dolma"}, sorted({line["subject"] for line in lines})
    has_route = {question.id for question in _questions() if question.route is not None}
    assert all(line["id"] in has_route for line in lines), "a none line was written, and it is made after the loop"
    assert raised.seen.built == ["dolma", "originblame"], raised.seen.built


def test_the_record_of_a_pass_that_ended_by_itself_is_no_record_of_a_stop(made):
    six = _whole(made)
    assert _stop_problems(six.folder) == ["no stopped.json"]


# --- The text of this file -----------------------------------------------------------------------------------------


def test_this_file_holds_no_banned_word_and_no_local_path():
    assert _banned_in(Path(__file__).read_text(encoding="utf-8")) == []
