"""The driver of the OriginBlame subject (plan 5.5 Check 10; table README, section
"Check 10": what OriginBlame is handed, the conditions "one process in packed order" and
"one revoke per fresh store").

This script runs INSIDE the sandbox, through the copied environment's own python
(``python -m ob_driver VERB --store FOLDER ...``). It holds the tool's calls, one verb
each, so that the host side builds one command per step and reads what the verb printed
or wrote. At module level it imports the standard library only: the tool's package is the
parameter ``ob`` of ``main`` (and its native module the parameter ``native``) and is
imported by ``load_tool`` when none is given.

The calls are those of the installed package and of the baseline run's report (sections
3.1 and 5): the public ``register_section`` (it resolves an author name, so each author is
added first through the native ``author_add``), and the native ``init``, ``track`` and
``token_index_revoke_by_sources``. The tool has no call that exports its entries, so none is
made: they are the files under ``.ob/token-index.gpt2`` of the store, which the host reads.
``clean``, ``status`` and ``generate-set`` are commands of the tool's own command-line
program, which the driver starts in the folder of the store; the program's path is the
variable the package itself reads (``PROGRAM_ENV``).

Verbs (``--store`` is the folder of the store):

- ``track --plan FILE``: ``FILE`` is a JSON file ``{"files": [..], "documents": [{"section",
  "tokens", "record", "file"}]}``. The store is made once: ``init``, one author per file
  stem, one public ``register_section`` call per file of ``files`` (every file under
  ``sources/`` and ``gen/``, not only the ones a document is tracked under); the hash each
  returned is kept in ``driver_meta.json`` in the store. Then one native ``track`` call is
  made per document, in ordinal order, in this one process, with the hash of its file.
  ``tracked K`` is printed (and flushed) after the Kth call returns, and one line ``run
  {"processes": 1, "order": "packed"}`` at the end. Run again on a store that has its
  ``driver_meta.json``, only the ``track`` calls are made.
- ``revoke FILE...``: the native by-sources call with the SET of the section hashes of the
  files, ONCE per store. A second call on the same store is refused before the tool is
  called, because a second revoke of one target un-revokes it.
- ``clean``, ``status``, ``generate-set --out FILE``: the program's ``clean``, ``status
  --tokenizer gpt2`` and ``generate-set --tokenizer gpt2 -o FILE``; what it printed is
  printed.

How an end is told apart. The native module maps every error of its own to a RuntimeError
that carries the text of the error behind it (an io error's text ends in "(os error N)"); the
library's own class (``ob.ODError``) is another class, which does not derive from
RuntimeError. A call on the changed record (the native ``track``, the by-sources call) ends
the driver with a non-zero code after one line that starts with ``TOOL_ERROR_MARKER``, the
only line the host takes for an error of the tool, when it raises
- a RuntimeError (exactly that class) with a text that is no OS error, or the library's class
  with a text;
- on a store with a planted fault, which only the host knows and tells the steps that read it
  (``revoke`` and ``generate-set`` take the flag ``--faulted``): a RuntimeError of any text,
  an OS error and an empty text included, and the library's class with or without a text; and
  the program's non-zero exit (not a signal) at ``generate-set``.
Every other end is a stop of the driver's own, one line that starts with ``FAILURE_MARKER``
and holds the raw text: a command line it does not take, a second revoke, a call the tool
lacks, an error of any other class (a subclass of RuntimeError included), an error about the
call itself (an import, a number or a type of arguments, a path the driver reads), an empty
text or an OS error where the store is not faulted, a signal, a program that ended with an
error where the store is not faulted. A panic of the native module is no Exception: it ends
the process with its traceback. The host reads such an end as a failed attempt.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MODULE_NAME = "ob_driver"
TOOL_MODULE = "ob"
NATIVE_NAME = "_ob_native"
PROGRAM_ENV = "OB_RUST_BIN"
META_FILE = "driver_meta.json"
TOKENIZER = "gpt2"

# What the baseline run registered a source file with (the author named for the file, the
# licence and the year); the table gives none for the files of a replica build.
AUTHOR_PREFIX = "author"
AUTHOR_EMAIL_SUFFIX = "@example.invalid"
LICENSE = "CC-BY-4.0"
YEAR = "2026"

OPTIONS = ("--store", "--plan", "--out")
# The flag of the two steps that read a store with a planted fault (it takes no value).
FAULTED_FLAG = "--faulted"
FAULTED_VERBS = ("revoke", "generate-set")
# The text of an io error carries this at its end.
OS_ERROR = re.compile(r"\(os error \d+\)")
REVOKED_SUFFIX = ".revoked"

EXIT_USAGE = 2
EXIT_SECOND_REVOKE = 3
EXIT_NO_TOOL = 4
EXIT_TOOL_ERROR = 5
EXIT_CALL_ERROR = 6
EXIT_PROGRAM = 7

# The line the driver prints when it stops for a reason of its own or for an end it cannot
# take as an error of the tool (see the head of this file), so that the host never reads such
# an end as an error of the tool.
FAILURE_MARKER = "ob_driver failure:"

# The line the driver prints when a call on the changed record raises the tool's own class
# of errors with a text. It is the only evidence the host takes that an end with an error is
# the tool's: a command that ends with a non-zero code and neither line (a lock that timed
# out, a sandbox that failed before the driver started, a signal) says nothing about the tool.
TOOL_ERROR_MARKER = "ob_driver tool error:"

VERBS = ("track", "revoke", "clean", "status", "generate-set")

# The calls each verb makes, looked up before any of them is made. A name that starts with
# the native module's is a call of that module.
CALLS = {
    "track": (f"{NATIVE_NAME}.init", f"{NATIVE_NAME}.author_add", "register_section",
              f"{NATIVE_NAME}.track"),
    "revoke": (f"{NATIVE_NAME}.token_index_revoke_by_sources",),
    "clean": (),
    "status": (),
    "generate-set": (),
}


class UsageError(Exception):
    """The command line names no verb the driver has, or leaves an option out."""


class ToolCallError(Exception):
    """A call on the changed record raised the tool's own class of errors, with a text."""


class CallError(Exception):
    """Any other end of a call: the text names the call and the class and text it raised."""


def _is_tool_error(error: BaseException, tool_class, native: bool, faulted: bool) -> bool:
    """Whether what a call on the changed record raised is the tool's own end (see the head of
    this file). ``tool_class`` is the library's class of errors; ``native`` says the call is
    one of the native module, which raises exactly RuntimeError, with the text of the error
    behind it. ``faulted``: the store holds a planted fault, as the host tells the step."""
    text = str(error).strip()
    if isinstance(tool_class, type) and isinstance(error, tool_class):
        return faulted or bool(text)
    if native and type(error) is RuntimeError:
        return faulted or (bool(text) and OS_ERROR.search(text) is None)
    return False


def _call(name: str, function, arguments=(), keywords=None, *, tool_class=None, native=False,
          faulted=False):
    """One call of the tool. ``tool_class`` is the library's class of errors and ``native``
    says the call is one of the native module's, for a call on the changed record; for every
    other call both are left out. What the call raises is the tool's error only as
    ``_is_tool_error`` says; every other end is a ``CallError`` that keeps the class and the
    raw text. An exception that is no ``Exception`` (a panic of the native module) is not
    caught."""
    try:
        return function(*arguments, **(keywords or {}))
    except SystemExit as error:
        raise CallError(f"{name} ended the process with the exit code {error.code!r}") from error
    except Exception as error:
        text = f"{name} raised {type(error).__name__}: {error}"
        if _is_tool_error(error, tool_class, native, faulted):
            raise ToolCallError(text) from error
        raise CallError(text) from error


def load_tool():
    """The tool's package and its native module, imported by name inside the sandbox."""
    tool = importlib.import_module(TOOL_MODULE)
    importlib.import_module(f"{TOOL_MODULE}.{NATIVE_NAME}")
    return tool


def _lookup(tool, native, dotted: str):
    owner, _, name = dotted.rpartition(".")
    base = native if owner == NATIVE_NAME else tool
    return getattr(base, name, None)


def _parse(argv):
    if not argv:
        raise UsageError("no verb")
    verb, rest = argv[0], list(argv[1:])
    options: dict[str, str] = {}
    positional: list[str] = []
    at = 0
    while at < len(rest):
        word = rest[at]
        if word == FAULTED_FLAG:
            options["faulted"] = "yes"
            at += 1
        elif word in OPTIONS:
            if at + 1 >= len(rest):
                raise UsageError(f"{word} has no value")
            options[word[2:]] = rest[at + 1]
            at += 2
        elif word.startswith("--"):
            raise UsageError(f"unknown option {word}")
        else:
            positional.append(word)
            at += 1
    return verb, options, positional


def _need(options, name: str, verb: str) -> str:
    if name not in options:
        raise UsageError(f"{verb} needs --{name}")
    return options[name]


def _print(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _fail(reason: str, code: int) -> int:
    """End for a reason of the driver's own: the marker line, then the code."""
    _print(f"{FAILURE_MARKER} {' '.join(reason.split())}")
    sys.stderr.write(f"{MODULE_NAME}: {reason}\n")
    return code


def _read_json(path, what: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CallError(f"{what} could not be read: {type(error).__name__}: {error}") from error


def _read_plan(plan_file: str):
    plan = _read_json(plan_file, "the plan")
    try:
        files = [str(name) for name in plan["files"]]
        documents = [{"section": str(item["section"]), "tokens": int(item["tokens"]),
                      "record": item["record"], "file": str(item["file"])}
                     for item in plan["documents"]]
    except (KeyError, TypeError, ValueError) as error:
        raise CallError(f"the plan is not in the shape the driver reads: {type(error).__name__}: {error}") from error
    return files, documents


def _read_sections(store: str) -> dict[str, str]:
    meta = _read_json(Path(store) / META_FILE, "the file of section hashes")
    sections = meta.get("sections") if isinstance(meta, dict) else None
    if not isinstance(sections, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in sections.items()):
        raise CallError("the file of section hashes holds no mapping of file to hash")
    return sections


def _author_of(path: str) -> str:
    return AUTHOR_PREFIX + Path(path).name.split(".")[0]


def _register(tool, native, store: str, files: list[str]) -> dict[str, str]:
    """The setup of a store, once: ``init``, the authors, one registered section per file."""
    os.makedirs(store, exist_ok=True)
    _call("init", native.init, (store,))
    for name in sorted({_author_of(path) for path in files}):
        _call("author_add", native.author_add, (name, name + AUTHOR_EMAIL_SUFFIX, store))
    sections: dict[str, str] = {}
    for path in files:
        found = _call("register_section", tool.register_section,
                      (path, [_author_of(path)], LICENSE, YEAR), {"ob_dir": store})
        if not isinstance(found, str):
            raise CallError(f"register_section gave no hash for {path}")
        sections[path] = found
    try:
        (Path(store) / META_FILE).write_text(
            json.dumps({"phase": "track", "sections": sections}, sort_keys=True), encoding="utf-8")
    except OSError as error:
        raise CallError(f"the file of section hashes could not be written: {error}") from error
    return sections


def _track(tool, native, store: str, plan_file: str) -> int:
    files, documents = _read_plan(plan_file)
    if (Path(store) / META_FILE).exists():
        sections = _read_sections(store)
    else:
        sections = _register(tool, native, store, files)
    unknown = [item["section"] for item in documents if item["section"] not in sections]
    if unknown:
        raise CallError(f"no section was registered for {unknown[0]}")
    tool_class = getattr(tool, "ODError", None)
    for number, item in enumerate(documents, start=1):
        _call("track", native.track,
              (item["record"], item["file"], [sections[item["section"]]], store, item["tokens"], TOKENIZER),
              tool_class=tool_class, native=True)
        _print(f"tracked {number}")
    _print("run " + json.dumps({"processes": 1, "order": "packed"}, sort_keys=True))
    return 0


def _revoke_guard(store: str) -> int:
    marker = Path(store).with_name(Path(store).name + REVOKED_SUFFIX)
    try:
        with open(marker, "x", encoding="utf-8") as handle:
            handle.write("revoked once\n")
    except FileExistsError:
        return _fail("this store was revoked once already; a second revoke of one target "
                     "un-revokes it", EXIT_SECOND_REVOKE)
    return 0


def _revoke(tool, native, store: str, files: list[str], faulted: bool) -> int:
    # A store revoked already is refused before any call of the tool is made.
    refused = _revoke_guard(store)
    if refused:
        return refused
    sections = _read_sections(store)
    missing = [path for path in files if path not in sections]
    if missing:
        raise CallError(f"no section was registered for {missing[0]}")
    hashes = {sections[path] for path in files}
    found = _call("token_index_revoke_by_sources", native.token_index_revoke_by_sources,
                  (store, TOKENIZER, hashes), tool_class=getattr(tool, "ODError", None), native=True,
                  faulted=faulted)
    _print(f"revoked {found}")
    return 0


def _run_program(arguments, cwd):
    """The tool's command-line program, started in ``cwd``: its exit code and its output."""
    path = os.environ.get(PROGRAM_ENV)
    if not path:
        raise CallError(f"{PROGRAM_ENV} is not set: the path of the tool's program is not known")
    try:
        done = subprocess.run([path, *arguments], cwd=cwd, capture_output=True, text=True)
    except OSError as error:
        raise CallError(f"the program could not be started: {type(error).__name__}: {error}") from error
    return done.returncode, done.stdout + done.stderr


def _program(verb: str, store: str, out, program, faulted: bool) -> int:
    arguments = {"clean": ["clean"], "status": ["status", "--tokenizer", TOKENIZER],
                 "generate-set": ["generate-set", "--tokenizer", TOKENIZER, "-o", out]}[verb]
    code, output = program(arguments, store)
    if output:
        sys.stdout.write(output if output.endswith("\n") else output + "\n")
        sys.stdout.flush()
    if code != 0:
        text = f"the program ended with the exit code {code}: {' '.join(str(output).split())}"
        if faulted and isinstance(code, int) and code > 0:
            # On a store with a planted fault the program's own error end is the tool's stop; a
            # signal (a negative code) is not.
            _print(f"{TOOL_ERROR_MARKER} {text}")
            return EXIT_TOOL_ERROR
        return _fail(text, EXIT_PROGRAM)
    return 0


def main(argv, *, ob=None, native=None, program=None) -> int:
    """Run one verb; the return value is the exit code. ``ob`` is the tool's package,
    ``native`` its native module and ``program(arguments, cwd)`` the start of its
    command-line program (the exit code and the output); each is loaded where not given."""
    try:
        verb, options, positional = _parse(list(argv))
        if verb not in VERBS:
            raise UsageError(f"unknown verb {verb}")
        store = _need(options, "store", verb)
        plan = _need(options, "plan", verb) if verb == "track" else None
        out = _need(options, "out", verb) if verb == "generate-set" else None
        if verb == "revoke" and not positional:
            raise UsageError("revoke needs the files of the request")
        faulted = "faulted" in options
        if faulted and verb not in FAULTED_VERBS:
            raise UsageError(f"{FAULTED_FLAG} is for {' and '.join(FAULTED_VERBS)}, not for {verb}")
    except UsageError as error:
        return _fail(str(error), EXIT_USAGE)
    try:
        if verb in ("clean", "status", "generate-set"):
            return _program(verb, store, out, program or _run_program, faulted)
        try:
            tool = load_tool() if ob is None else ob
        except ImportError as error:
            return _fail(f"the tool's package cannot be imported: {error}", EXIT_NO_TOOL)
        chosen = native if native is not None else getattr(tool, NATIVE_NAME, None)
        missing = [name for name in CALLS[verb] if not callable(_lookup(tool, chosen, name))]
        if missing:
            return _fail(f"the tool has no call {', '.join(missing)}", EXIT_NO_TOOL)
        if verb == "track":
            return _track(tool, chosen, store, plan)
        return _revoke(tool, chosen, store, positional, faulted)
    except ToolCallError as error:
        _print(f"{TOOL_ERROR_MARKER} {' '.join(str(error).split())}")
        sys.stderr.write(f"{MODULE_NAME}: {error}\n")
        return EXIT_TOOL_ERROR
    except CallError as error:
        return _fail(str(error), EXIT_CALL_ERROR)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
