"""The worked example of docs/quickstart.md runs as the page shows it.

The page holds three kinds of block: a ``console`` block whose lines that start with ``$ `` are
commands (the lines after a command, up to the next command, are the output the page states for
it), a ``python`` block whose first line is ``# build_evidence.py`` (the script the page tells the
reader to run), and ``sh`` blocks (the install line, which is not run here).

Nothing of the example is held in this file. The test reads the commands, the script and the stated
output from the page, writes the script into a temporary folder, runs every command in that folder
in the order the page shows them, through the real command line (``waterwhite`` is run as
``python -m waterwhite.cli``, the form docs/cli.md gives for the same program; ``python`` is the
interpreter that runs the tests), and compares each standard output with the page. The stdout,
stderr and exit code of every command are written to ``runs/`` beside the working folder, so a
failure can be read in full.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "quickstart.md"
SRC = str(ROOT / "src")

# The commands of docs/cli.md, one heading each.
COMMANDS = {"init", "query", "show", "check", "migrate", "register-run", "replay", "recover", "report",
            "verify"}
SCRIPT_NAME = "build_evidence.py"


def blocks(text: str) -> list[tuple[str, str]]:
    return [(lang, body) for lang, body in re.findall(r"^```(\w*)\n(.*?)^```$", text, flags=re.S | re.M)]


def steps(text: str) -> list[tuple[str, list[str]]]:
    """Every command of every console block, with the lines the page states under it."""
    found: list[tuple[str, list[str]]] = []
    for lang, body in blocks(text):
        if lang != "console":
            continue
        for line in body.splitlines():
            if line.startswith("$ "):
                found.append((line[2:], []))
            else:
                assert found and found[-1][0], f"output before any command in a console block: {line!r}"
                found[-1][1].append(line)
    return found


def script(text: str) -> str:
    (body,) = [b for lang, b in blocks(text) if lang == "python" and b.startswith(f"# {SCRIPT_NAME}\n")]
    return body


def argv_of(command: str) -> list[str]:
    words = shlex.split(command)
    if words[0] == "waterwhite":
        return [sys.executable, "-m", "waterwhite.cli", *words[1:]]
    if words[0] == "python":
        return [sys.executable, *words[1:]]
    raise AssertionError(f"the page runs a program the test does not know: {command!r}")


def test_the_page_shows_what_the_test_needs_to_run():
    text = PAGE.read_text(encoding="utf-8")
    shown = steps(text)
    commands = [c for c, _ in shown]
    assert commands[0].startswith("waterwhite init ")
    assert f"python {SCRIPT_NAME}" in commands
    assert any(c.startswith("waterwhite check ") for c in commands)
    queries = [(c, out) for c, out in shown if c.startswith("waterwhite query withdraw ")]
    assert len(queries) == 2
    assert all(out for _, out in queries)  # a query's result is stated on the page
    assert script(text).count("\n") > 10
    # every command the page names (in a console block or in running text) is a command of docs/cli.md
    named = {shlex.split(c)[1] for c in commands if c.startswith("waterwhite ")}
    named |= set(re.findall(r"`waterwhite ([a-z][a-z-]*)", text))
    assert {"init", "check", "query"} <= named
    assert named <= COMMANDS, named - COMMANDS


def test_the_commands_of_the_page_give_the_results_the_page_states(tmp_path):
    text = PAGE.read_text(encoding="utf-8")
    work = tmp_path / "work"
    runs = tmp_path / "runs"
    work.mkdir()
    runs.mkdir()
    (work / SCRIPT_NAME).write_text(script(text), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=SRC)

    compared = 0
    for number, (command, stated) in enumerate(steps(text), start=1):
        done = subprocess.run(argv_of(command), cwd=work, env=env, capture_output=True, text=True,
                              timeout=120)
        (runs / f"{number}.txt").write_text(
            f"$ {command}\nexit {done.returncode}\n--- stdout\n{done.stdout}--- stderr\n{done.stderr}",
            encoding="utf-8")
        assert done.stderr == "", f"{command}: {done.stderr}"
        assert done.returncode == 0, f"{command}: exit {done.returncode}"
        if stated:
            assert done.stdout.splitlines() == stated, command
            compared += 1
    assert compared == 3  # check and the two queries
    assert (work / "evidence.db").is_file()
