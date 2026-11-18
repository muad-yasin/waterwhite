"""The world gate (plan 5.5 Check 10; docs/baselines.md, "Before a tool starts" and "The world
builder"): a run refuses, before any tool command is started, a world whose ``WORLD.sha256`` does
not say what the folder holds.

The world command always writes ``WORLD.sha256`` beside ``world.json``: one line, the sha256 and
the path relative to the folder, for every file ``world.json`` names and for ``world.json``
itself. The run reads that list before it starts a tool and stops (a pre-run stop of its own,
named ``world-hash``, exit status 3, a sealed record that names the stop, nothing started) when:

- a file the list names is changed (its sha256 is not the listed one);
- a file the list names is missing;
- a path is listed twice;
- there is no ``WORLD.sha256`` (a folder without one is not a built world);
- a path ``world.json`` names has no row in the list (the paths are those the world builder's own
  listing function names: the tokenizer, the tie record, the counting documents, the index files,
  the chunk files, the idx and bin files of every shard, the record files of every build, the
  request files, the files ``subject_inputs`` names);
- ``world.json`` itself has no row.

The start line (``start.txt``, and the first line the command prints) names the sha256 of
``WORLD.sha256``.

These tests build no world. Each puts the run through ``baselines.runner.main`` on a miniature
world folder of hand-written files, with a stand-in starter that records every command it is
asked to start, and a tool setup whose record holds no ``compare.json``: a world the gate lets
through meets the next check of the run (the stop ``compare``), a world the gate refuses is
stopped first. So a gate that does not exist fails each test at its assertion, and a gate that
refuses a clean world fails the first test. Where the gate is raised (the start of ``run_pass``
from the configuration, or ``main`` before it with the same catch) is the code's choice; both
leave what these tests read.

The expected values are the ones the text above sets down and the hashes of hand-written bytes
that this file computes itself; nothing is read from the code under test.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]

COMMIT = "0123456789abcdef0123456789abcdef01234567"
ANY_SHA256 = "0" * 64
WORLD_FILE = "world.json"
HASH_FILE = "WORLD.sha256"
STAGE = "world-hash"
SUCCESSOR_KEY = "successor of E/shard-3"

# The files of the miniature world, relative to the world folder.
MAIN = "work/stores/main/build"
UNSEEDED = "work/stores/no-capture/build"
TOKENIZER = "work/inputs/tokenizer.json"
TIE = "work/tie.json"
COUNTED = "work/counted/small.json"
TRACE = "work/traces/store-main/trace.json"
CHUNK = "pipeline/partitioned/tokenizer-input/shard-1.jsonl"
SUCCESSOR_CHUNK = "successor/tokenizer-input/shard-3.jsonl"
SHARD_PREFIX = "pipeline/partitioned/shards/shard-1"
SUCCESSOR_PREFIX = "successor/shards/shard-3"

# The kinds of file the world names, each by one path of the miniature world.
NAMED = {
    "the tokenizer": TOKENIZER,
    "the tie record": TIE,
    "a record file of sources/": f"{MAIN}/sources/A.jsonl",
    "a record file of gen/": f"{MAIN}/gen/P.jsonl",
    "the built-from file of a build": f"{MAIN}/built-from.jsonl",
    "a chunk file": f"{MAIN}/{CHUNK}",
    "the chunk file of the successor": f"{MAIN}/{SUCCESSOR_CHUNK}",
    "a chunk file of the build made without capture": f"{UNSEEDED}/{CHUNK}",
    "the index file of a shard": f"{MAIN}/{SHARD_PREFIX}.idx",
    "the packed file of a shard": f"{MAIN}/{SHARD_PREFIX}.bin",
    "the index file of the successor": f"{MAIN}/{SUCCESSOR_PREFIX}.idx",
    "the packed file of the successor": f"{MAIN}/{SUCCESSOR_PREFIX}.bin",
    "an exported trace document of subject_inputs": TRACE,
}
# The run parses the counting document before its pass, so a changed or missing one is a reading
# error of its own: the counting document is in the cases of a missing row only.
NAMED_WITH_ROW = {**NAMED, "a counting document": COUNTED}


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flipped(data: bytes) -> bytes:
    """The bytes with one bit of the byte in the middle changed."""
    middle = len(data) // 2
    return data[:middle] + bytes([data[middle] ^ 1]) + data[middle + 1:]


# --- The miniature world ------------------------------------------------------------------------------


def _contents() -> dict[str, bytes]:
    """The hand-written bytes of every file of the world, by path relative to the folder."""
    return {
        TOKENIZER: b'{"stand_in": "a tokenizer file"}\n',
        TIE: b'{"chunk_files": {}, "note": "a stand-in tie record"}\n',
        COUNTED: b"{}\n",
        f"{MAIN}/sources/A.jsonl": b'{"id": "a-1", "text": "a first record"}\n{"id": "a-2", "text": "a second record"}\n',
        f"{MAIN}/gen/P.jsonl": b'{"id": "p-1", "text": "a generated record"}\n',
        f"{MAIN}/built-from.jsonl": b'{"artifact": "src/A", "built_from": []}\n',
        f"{MAIN}/{CHUNK}": b'{"id": "a-1", "text": "a first record"}\n{"id": "p-1", "text": "a generated record"}\n',
        f"{MAIN}/{SUCCESSOR_CHUNK}": b'{"id": "p-1", "text": "a generated record"}\n',
        f"{MAIN}/{SHARD_PREFIX}.idx": b"the index of the shard, hand-written bytes",
        f"{MAIN}/{SHARD_PREFIX}.bin": b"the packed file of the shard, hand-written bytes",
        f"{MAIN}/{SUCCESSOR_PREFIX}.idx": b"the index of the successor, hand-written bytes",
        f"{MAIN}/{SUCCESSOR_PREFIX}.bin": b"the packed file of the successor, hand-written bytes",
        f"{UNSEEDED}/{CHUNK}": b'{"id": "b-1", "text": "a record of the build made without capture"}\n',
        TRACE: b'{"segments": []}\n',
    }


def _miniature(folder: Path) -> Path:
    """A world folder of hand-written files and its ``WORLD.sha256``, which lists every file of the
    folder (each one is named by ``world.json``) and ``world.json`` itself, in path order. Gives the
    folder (resolved)."""
    folder.mkdir(parents=True)
    root = folder.resolve()
    for relative, data in _contents().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    main, unseeded = root / MAIN, root / UNSEEDED
    contents = _contents()
    world = {
        "tokenizer": str(root / TOKENIZER),
        "tie_record": str(root / TIE),
        "requests": {"withdraw A": {"files": ["sources/A.jsonl"]}},
        "stores": [
            {"target": ["store", "main"], "build": str(main),
             "chunk_files": {"E/shard-1": CHUNK, SUCCESSOR_KEY: SUCCESSOR_CHUNK}},
            {"target": ["store", "no-capture"], "build": str(unseeded), "chunk_files": {"E/shard-1": CHUNK}},
        ],
        "shards": [
            {"name": "E/shard-1", "target": ["store", "main"], "layout": "partitioned", "run": "E",
             "successor": False, "build": str(main), "chunk_file": CHUNK, "idx_prefix": str(main / SHARD_PREFIX)},
            {"name": SUCCESSOR_KEY, "target": ["store", "main"], "layout": None, "run": None,
             "successor": True, "build": str(main), "chunk_file": SUCCESSOR_CHUNK,
             "idx_prefix": str(main / SUCCESSOR_PREFIX)},
        ],
        "counted": {"small": str(root / COUNTED)},
        "subject_inputs": {
            "manifest": {"store-main": {"built_from": {
                "path": f"{MAIN}/built-from.jsonl", "sha256": _sha256(contents[f"{MAIN}/built-from.jsonl"])}}},
            "trace": {
                "store-main": {"document": {"path": TRACE, "sha256": _sha256(contents[TRACE])}},
                "store-no-capture": {"none": "the table registers no run in this store"}},
            "repository": [{"path": "fixture/table/check01.toml", "repository": True,
                            "sha256": _sha256((REPO / "fixture" / "table" / "check01.toml").read_bytes())}],
            "records": {"store-main": ["gen/P.jsonl", "sources/A.jsonl"]},
            "world_keys": ["tie_record", "requests", "stores", "shards", "index", "counted"],
            "tracked_order": ["E/shard-1", SUCCESSOR_KEY],
        },
    }
    (root / WORLD_FILE).write_text(json.dumps(world, indent=1) + "\n", encoding="utf-8")
    _write_list(root, _rows_of(root))
    return root


def _rows_of(root: Path) -> list[tuple[str, str]]:
    """The honest rows of a folder: every file but the list, as (path, sha256) in path order."""
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                   if path.is_file() and path.name != HASH_FILE)
    return [(name, _sha256((root / name).read_bytes())) for name in names]


def _write_list(root: Path, rows) -> None:
    (root / HASH_FILE).write_text("".join(f"{digest}  {path}\n" for path, digest in rows), encoding="utf-8")


# --- Driving the run ----------------------------------------------------------------------------------


@pytest.fixture
def drive(tmp_path, capsys):
    """``drive(root)`` runs ``baselines.runner.main`` on the world folder ``root`` with a stand-in
    starter and a tool setup whose record holds no ``compare.json``, and gives what happened."""
    counter = {"runs": 0}

    def run(root: Path):
        runner = need("baselines.runner")
        counter["runs"] += 1
        label = counter["runs"]
        env, env_record = tmp_path / f"env-{label}", tmp_path / f"env-record-{label}"
        env.mkdir()
        env_record.mkdir()
        record = tmp_path / f"run-record-{label}"
        started: list = []

        class Recording:
            """Every command it is asked to start is written down and none is started."""

            def run(self, argv, timeout=None, **keywords):
                started.append(list(argv))
                return SimpleNamespace(exit_code=0, output="", seconds=0.0)

            def start(self, argv, **keywords):
                started.append(list(argv))
                raise AssertionError("a tool command was started")

        arguments = [
            "--subjects", "dolma", "--step-cap", "600", "--commit", COMMIT,
            "--work", root / "work", "--run-record", record, "--lock-prefix", "a-prefix",
            "--world", root / WORLD_FILE, "--tool", "dolma", env, env_record, ANY_SHA256,
        ]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runner, "SubprocessStarter", Recording)
            try:
                code = runner.main([str(word) for word in arguments])
            except SystemExit as leaving:
                code = ("exit", leaving.code)
        shown = capsys.readouterr()
        return SimpleNamespace(code=code, out=shown.out, err=shown.err, record=record, started=started)

    return run


def _refused(result, *, names, what):
    """The run was stopped by the gate: exit 3, the stop named ``world-hash`` and each of
    ``names`` (the path of the file the gate found), nothing started, a record sealed with a
    ``stopped.json`` that names the stop and its file, and no answer line. The words of the reason
    are not pinned. A local path in ``stopped.json`` is cut to its last name, so that file is read
    for the last name of each path."""
    assert result.code == 3, f"{what}: the run ended with {result.code!r}, not the exit 3 of a stop; {result.err[-600:]}"
    assert STAGE in result.err, f"{what}: the stop does not name the stop {STAGE}: {result.err[-600:]}"
    for name in names:
        assert name in result.err, f"{what}: the stop does not name {name}: {result.err[-600:]}"
    assert result.started == [], f"{what}: a tool command was started: {result.started}"
    seal = result.record / "sha256sums.txt"
    assert seal.is_file(), f"{what}: the record is not sealed"
    stopped = result.record / "stopped.json"
    assert stopped.is_file(), f"{what}: the sealed record holds no stopped.json"
    assert "  stopped.json" in seal.read_text(encoding="utf-8"), f"{what}: stopped.json is not in the seal"
    text = stopped.read_text(encoding="utf-8")
    body = json.loads(text)
    assert isinstance(body, dict) and "error" in body, f"{what}: stopped.json holds no error: {text}"
    assert STAGE in str(body["error"]), f"{what}: stopped.json does not name the stop: {text}"
    for name in names:
        assert Path(name).name in str(body["error"]), f"{what}: stopped.json does not name {Path(name).name}: {text}"
    assert body.get("question") is None, f"{what}: the stop names a question: {text}"
    answers = result.record / "answers.jsonl"
    assert not answers.exists() or answers.read_text(encoding="utf-8").strip() == "", \
        f"{what}: an answer line was written"


# --- The stop and the clean world ---------------------------------------------------------------------


def test_the_gate_is_a_pre_run_stop_of_its_own():
    stop = need_attr("baselines.runner", "PreRunStop")
    assert STAGE in stop.STAGES, f"{STAGE} is not one of the stops {stop.STAGES}"
    made = stop(STAGE, "a world file is changed")
    assert made.stage == STAGE and made.reason == "a world file is changed"


def test_a_clean_world_passes_the_gate_and_the_start_line_names_the_sha_of_its_list(tmp_path, drive):
    root = _miniature(tmp_path / "world")
    result = drive(root)
    # The gate lets the world through: the run meets the next check, the record of the copy of the
    # tool, which holds no compare.json, and no tool command is started.
    assert result.code == 3, f"the run ended with {result.code!r}: {result.err[-600:]}"
    assert STAGE not in result.err, f"a clean world was refused: {result.err[-600:]}"
    assert "compare" in result.err, result.err[-600:]
    assert result.started == []
    listed = _sha256((root / HASH_FILE).read_bytes())
    start = (result.record / "start.txt").read_text(encoding="utf-8")
    assert listed in start, f"the start line does not name the sha256 of {HASH_FILE} ({listed}): {start}"
    assert HASH_FILE in start, f"the start line does not say which file the sha256 is of: {start}"
    assert listed in result.out, "the line the command prints does not name the sha256 of the list"
    assert "\n" not in start.strip(), "the start line is one line"


def test_the_start_line_names_the_sha_of_the_list_a_refused_world_came_with(tmp_path, drive):
    root = _miniature(tmp_path / "world")
    path = root / NAMED["a chunk file"]
    path.write_bytes(_flipped(path.read_bytes()))
    result = drive(root)
    _refused(result, names=[NAMED["a chunk file"]], what="a changed chunk file")
    listed = _sha256((root / HASH_FILE).read_bytes())
    assert listed in (result.record / "start.txt").read_text(encoding="utf-8")


# --- The refusals -------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(NAMED))
def test_a_changed_byte_in_a_file_the_world_names_is_refused_before_any_tool(tmp_path, drive, kind):
    root = _miniature(tmp_path / "world")
    path = root / NAMED[kind]
    original = path.read_bytes()
    assert original, f"{NAMED[kind]} is empty"
    path.write_bytes(_flipped(original))
    assert path.read_bytes() != original
    result = drive(root)
    _refused(result, names=[NAMED[kind]], what=f"a changed byte in {kind}")


def test_a_changed_world_json_is_refused_before_any_tool(tmp_path, drive):
    root = _miniature(tmp_path / "world")
    path = root / WORLD_FILE
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    assert json.loads(path.read_text(encoding="utf-8")), "the changed world.json is still JSON"
    result = drive(root)
    _refused(result, names=[WORLD_FILE], what="a changed world.json")


@pytest.mark.parametrize("kind", sorted(NAMED))
def test_a_listed_file_that_is_missing_is_refused_before_any_tool(tmp_path, drive, kind):
    root = _miniature(tmp_path / "world")
    away = tmp_path / "moved-away"
    away.mkdir()
    (root / NAMED[kind]).rename(away / "the-file")
    result = drive(root)
    _refused(result, names=[NAMED[kind]], what=f"a missing file: {kind}")


@pytest.mark.parametrize("same_hash", [True, False], ids=["the same row twice", "two rows with two hashes"])
def test_a_path_listed_twice_is_refused_before_any_tool(tmp_path, drive, same_hash):
    root = _miniature(tmp_path / "world")
    rows = _rows_of(root)
    path, digest = rows[len(rows) // 2]
    other = digest if same_hash else _sha256(b"another file")
    _write_list(root, rows + [(path, other)])
    result = drive(root)
    _refused(result, names=[path], what="a path listed twice")


def test_a_world_folder_with_no_hash_list_is_refused_before_any_tool(tmp_path, drive):
    root = _miniature(tmp_path / "world")
    away = tmp_path / "moved-away"
    away.mkdir()
    (root / HASH_FILE).rename(away / HASH_FILE)
    result = drive(root)
    _refused(result, names=[HASH_FILE], what="a folder with no WORLD.sha256")


@pytest.mark.parametrize("kind", sorted(NAMED_WITH_ROW))
def test_a_path_the_world_names_with_no_row_in_the_list_is_refused_before_any_tool(tmp_path, drive, kind):
    root = _miniature(tmp_path / "world")
    wanted = NAMED_WITH_ROW[kind]
    rows = _rows_of(root)
    assert wanted in [path for path, _ in rows], f"{wanted} is not a file of the miniature world"
    _write_list(root, [(path, digest) for path, digest in rows if path != wanted])
    assert (root / wanted).is_file(), "the file is there; only its row is not"
    result = drive(root)
    _refused(result, names=[wanted], what=f"no row for {kind}")


@pytest.mark.parametrize("empty", [False, True], ids=["every other row", "an empty list"])
def test_a_world_json_with_no_row_in_the_list_is_refused_before_any_tool(tmp_path, drive, empty):
    root = _miniature(tmp_path / "world")
    rows = [] if empty else [(path, digest) for path, digest in _rows_of(root) if path != WORLD_FILE]
    _write_list(root, rows)
    result = drive(root)
    _refused(result, names=[WORLD_FILE], what="world.json with no row")


def test_the_gate_stops_the_run_before_the_other_checks_of_the_tools(tmp_path, drive):
    # The setup of the tool is bad as well (no compare.json): the first stop is the gate's.
    root = _miniature(tmp_path / "world")
    path = root / NAMED["a record file of sources/"]
    path.write_bytes(path.read_bytes() + b'{"id": "a-3", "text": "a record added after the build"}\n')
    result = drive(root)
    _refused(result, names=[NAMED["a record file of sources/"]], what="a record file with a line added")
    assert f"tool command: {STAGE}: " in result.err, f"the first stop is not the gate's: {result.err[-600:]}"


# --- The documents ------------------------------------------------------------------------------------


def _section(text: str, heading: str) -> str:
    found = re.split(rf"(?m)^#+ {re.escape(heading)}$", text)
    assert len(found) == 2, f"the document has no heading {heading!r}"
    return " ".join(re.split(r"(?m)^## ", found[1])[0].split())


def _banned(text: str) -> list[str]:
    words = ("R2" + "A", "ROUND" + "2", "rul" + "ing", "amend" + "ed", "uns" + "ure", "BA" + "RS", "spec " + "1",
             "C" + "&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net", "advi" + "sor",
             "/ho" + "me/", "Proj" + "ects/")
    found = [word for word in words if word in text]
    for pattern in (r"\b[GQ]\d\b", r"\bitem \d", r"\bround \d", r"\b(work" + r"flow|ag" + r"ent|br" + r"ief)\b",
                    chr(0x2014)):
        if re.search(pattern, text, re.IGNORECASE if pattern.startswith(r"\bround") else 0):
            found.append(pattern)
    return found


def test_the_document_names_the_gate_the_start_line_and_the_successor_shard():
    text = (REPO / "docs" / "baselines.md").read_text(encoding="utf-8")
    before = _section(text, "Before a tool starts")
    assert f"`{STAGE}`" in before and HASH_FILE in before, "the order before a tool starts does not name the gate"
    for word in ("changed", "missing", "twice"):
        assert word in before, f"the order before a tool starts does not say what is refused: {word}"
    assert re.search(r"no\s+`?WORLD\.sha256`?", before), "it does not say a folder with no WORLD.sha256 is refused"
    assert "world.json" in before, "it does not say world.json is a row of the list"
    command = re.search(r"The first line printed is the start line:.*?(?=\n\n)", text, re.DOTALL)
    assert command is not None, "the command's section has no sentence on the start line"
    assert HASH_FILE in " ".join(command.group(0).split()), "the start line's sentence does not say it names the sha256 of WORLD.sha256"
    builder = _section(text, "The world builder (temporary section WORLD)")
    assert STAGE in builder and "refus" in builder and "start line" in builder, \
        "the section of the world builder does not name the gate and the start line"
    sentences = re.split(r"(?<=[.;])\s+", builder)
    assert any(SUCCESSOR_KEY in sentence and "OriginBlame" in sentence and re.search(r"last|fourth", sentence)
               for sentence in sentences), "the section does not say the successor is the last shard of OriginBlame"
    for part in (before, builder):
        assert _banned(part) == [], _banned(part)


def _decisions() -> dict[str, str]:
    text = (REPO / "DECISIONS.md").read_text(encoding="utf-8")
    found = {}
    for block in re.split(r"(?m)^## (?=D-(?:WORLD|GATE)-\d+ )", text)[1:]:
        label = re.match(r"(D-(?:WORLD|GATE)-\d+) ", block).group(1)
        found[label] = " ".join(re.split(r"(?m)^## ", block)[0].split())
    return found


def test_the_decisions_say_where_the_sha_of_each_kind_of_file_lives():
    body = _decisions().get("D-WORLD-5")
    assert body is not None, "DECISIONS.md holds no entry D-WORLD-5"
    built = body.split(" - Built:")[1].split(" - Other option:")[0] if " - Built:" in body else ""
    assert built, "D-WORLD-5 has no Built part"
    lowered = built.lower()
    # Three statements: where the sha of each kind of file lives.
    assert "every file the world names" in lowered and HASH_FILE.lower() in lowered, \
        "D-WORLD-5 does not say the files of the builds are in WORLD.sha256"
    assert "no-capture" in lowered and re.search(r"order[- ]free", lowered), \
        "D-WORLD-5 does not keep the order-free compare of the unseeded files"
    assert "subject_inputs" in lowered and "digests" in lowered and "world.json" in lowered, \
        "D-WORLD-5 does not say the digests of the subject_inputs files are inside world.json"
    assert "read back" in lowered and "gate" in lowered, "D-WORLD-5 does not say WORLD.sha256 itself is read back by the gate"
    assert _banned(body) == [], _banned(body)


def test_the_decisions_hold_the_successor_shard_and_the_gate():
    found = _decisions()
    for label, topic in (("D-WORLD-9", "successor"), ("D-GATE-1", "gate")):
        body = found.get(label)
        assert body is not None, f"DECISIONS.md holds no entry {label}"
        assert topic in body.split(" - Built:")[0].lower(), f"the title of {label} does not say {topic}"
        for part in ("Built", "Other option", "Result of the other option", "What would change it"):
            assert f"{part}:" in body, f"{label} has no {part!r}"
        assert "traced" in body.lower() or "inferred" in body.lower(), f"{label} says neither traced nor inferred"
        assert _banned(body) == [], (label, _banned(body))
    successor = found["D-WORLD-9"]
    for word in ("idx", "last", "OriginBlame", SUCCESSOR_KEY):
        assert word in successor, f"D-WORLD-9 does not mention {word!r}"
    gate = found["D-GATE-1"]
    for word in (HASH_FILE, "world.json", "start line", STAGE):
        assert word in gate, f"D-GATE-1 does not mention {word!r}"
    for word in ("changed", "missing", "twice"):
        assert word in gate, f"D-GATE-1 does not say what is refused: {word}"
    assert re.search(r"no\s+`?WORLD\.sha256`?", gate), "D-GATE-1 does not say a folder with no WORLD.sha256 is refused"
