"""The sandbox command line of the baselines (plan 5.5 Check 10; table README, section
"Check 10", condition ``sandbox``): argv construction only. Nothing is started and
nothing is written but where a test says so.

What a sandbox is, in the words of the table: an existing tool runs only inside a
bubblewrap sandbox with no network, a home folder that is an empty folder inside the
work folder, only the work folder writable, a cleared environment, a fresh device
folder (no GPU nodes), the system folders read-only; the copied environments are
bound read-only; the tool starts in the build folder and is handed relative paths.
Every step runs under the CPU lock of the machine's shared budget; the lock command is
handed to the module as a prefix, so that the module holds neither the lock file nor
the name of the machine's own unit.

The module holds no location: every path is an argument. The tests pass paths from
``tmp_path`` and check properties of the argv, not its whole text, so that an
argument a working sandbox needs and the table does not name (the process folder,
the links of the library folders) is allowed.

The names this file asks of ``baselines.sandbox``:

- ``bwrap_argv(*, work, home, chdir, command, ro_binds=(), env=None, bwrap="bwrap")``
  gives the argv, a list of strings: the program ``bwrap``, its options, then the
  command last and unchanged. ``work``, ``home``, ``chdir`` and every entry of
  ``ro_binds`` are absolute paths or the call raises ``ValueError``; ``home`` lies
  inside ``work``; ``chdir`` lies inside ``work`` or inside one of ``ro_binds`` (a
  tool may start in a build folder that is bound read-only); a ``ro_binds`` entry is neither
  ``work`` nor a folder that holds it (that would make the work folder read-only
  or hide it) or the call raises ``ValueError``; ``env`` is a mapping of the
  variables the tool is given, and ``HOME`` among them raises ``ValueError``.
  The options are ``--unshare-all``, ``--die-with-parent``, ``--ro-bind /usr /usr``,
  ``--ro-bind /etc /etc``, ``--dev /dev``, ``--bind WORK WORK`` (the only writable
  bind), one ``--ro-bind P P`` for each of ``ro_binds`` placed AFTER the work bind,
  ``--clearenv`` before every ``--setenv``, ``--setenv HOME HOME_FOLDER``, one
  ``--setenv`` for each variable of ``env``, and ``--chdir CHDIR``.
- ``locked(argv, *, lock_prefix)`` is ``[*lock_prefix, *argv]``: the argv of a step
  behind the lock command the caller hands in. An empty ``lock_prefix`` raises
  ``ValueError``, so that no step can be built unlocked.
- ``sandboxed_step(*, lock_prefix, **bwrap_arguments)`` is
  ``locked(bwrap_argv(**bwrap_arguments), lock_prefix=lock_prefix)``: the lock is
  outside, the sandbox is inside, so that every step is both.
- ``prepare_home(work, name="home")`` makes the empty home folder ``work/name``
  and returns its path; it raises ``ValueError`` for a ``name`` that leaves ``work``
  and for a folder that exists and is not empty.
- ``tool_command(env_dir, module, arguments=())`` is the command of a tool run through
  its copied environment: ``[ENV_DIR/bin/python, "-m", MODULE, *ARGUMENTS]``, the
  interpreter named by its path in the copy (the link as it stands, not the binary it
  resolves to: an interpreter finds its environment from the path it was started
  by) and never a launcher of the copy, whose first line names the original. A
  relative ``env_dir`` and a ``module`` that is not a dotted name raise ``ValueError``.
- ``interpreter_binds(env_dir)`` is a list of absolute folders to hand to
  ``bwrap_argv`` as ``ro_binds``, so that the copy's interpreter runs inside the
  sandbox: every link on the way from ``bin/python`` to the binary, the binary, and the
  folder that holds the binary's ``bin`` and ``lib`` folders are visible inside, read
  only, as a whole or through a bind whose source holds them. Which folders it
  names is its own choice. It reads links and starts nothing.
- ``recording_command(env_dir, module, arguments=(), *, work, record_file)`` is the
  command of a tool that first records, from inside the sandbox, the path that the
  copy's ``bin/python`` resolves to and the sha256 of that binary, into ``record_file``
  (a file in the work folder), and then starts the tool: a shell (``[SHELL, "-c",
  SCRIPT, ..., *tool_command]``) whose script reads the link and hashes the binary
  before it starts the tool, the record file given as an argument of the shell, and the
  tool command, unchanged, as the end of the list. ``record_file`` must be absolute,
  inside ``work`` and not ``work`` itself, and not inside the copy (which is bound
  read-only), or the call raises ``ValueError``; so does whatever ``tool_command``
  refuses. Nothing is read, made or started.
- ``interpreter_probe_command(env_dir, record_file, *, work)`` is the first start in a
  copy, its python alone: ``[ENV_DIR/bin/python, "-I", "-c", TEXT, RECORD_FILE]``. The
  fixed text writes the resolved path of ``sys.executable`` and the sha256 of that binary
  into the file it is given as its argument, so that a caller can read both and stop
  before any tool command is built. No tool command is in it. ``record_file`` is
  checked as for ``recording_command``, and a relative ``env_dir`` raises ``ValueError``.
  Nothing is read, made or started.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace
import pytest

# Helpers that read a module's source and a folder's bytes. Nothing here imports the
# ``baselines`` package. The word lists are written here, never taken from the
# code under test.


def _make_sbx_kit() -> SimpleNamespace:
    def tree_digest(folder: Path) -> dict[str, str]:
        """Every entry of a folder: relative path -> a digest of its bytes, or the link
        text of a link, or "dir". Equal digests mean nothing was added, removed or
        changed."""
        folder = Path(folder)
        out: dict[str, str] = {}
        for current, dirs, files in os.walk(folder, followlinks=False):
            for name in dirs + files:
                path = Path(current) / name
                rel = str(path.relative_to(folder))
                if path.is_symlink():
                    out[rel] = "link:" + os.readlink(path)
                elif path.is_dir():
                    out[rel] = "dir"
                else:
                    out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out


    # --- Reading a module's source ---------------------------------------------------------


    def source_text(module) -> str:
        return Path(module.__file__).read_text(encoding="utf-8")


    def string_constants(module) -> list[str]:
        """Every string constant of a module's source, except the pieces of an f-string
        (``f"{folder}/RECORD"`` has the piece ``"/RECORD"``, which is no location)."""
        tree = ast.parse(source_text(module))
        pieces = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                  for part in ast.walk(node) if isinstance(part, ast.Constant)}
        return [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in pieces]


    SYSTEM_FOLDERS = ("usr", "etc", "dev", "proc", "tmp", "lib", "lib64", "bin", "sbin")
    # Words of a location of this machine, written in two pieces so that this file holds
    # none of them whole.
    LOCAL_WORDS = ("/ho" + "me/", "/ro" + "ot/", "Proj" + "ects/", ".wo" + "rk/")


    def local_path_constants(module) -> list[str]:
        """The string constants of a module that look like a location of this machine:
        an absolute path (one word with no space, starting with a slash, more than a bare
        separator) whose first folder is no system folder, or any text naming a home
        folder or a working folder of this repository."""
        bad = []
        for text in string_constants(module):
            is_path = text.startswith("/") and text.strip("/") != "" and not any(c.isspace() for c in text)
            if is_path and text.lstrip("/").split("/", 1)[0] not in SYSTEM_FOLDERS:
                bad.append(text)
            elif any(word in text for word in LOCAL_WORDS):
                bad.append(text)
        return bad

    return SimpleNamespace(tree_digest=tree_digest, source_text=source_text, string_constants=string_constants, SYSTEM_FOLDERS=SYSTEM_FOLDERS, LOCAL_WORDS=LOCAL_WORDS, local_path_constants=local_path_constants)


sbx_kit = _make_sbx_kit()

WRITABLE_BINDS = ("--bind", "--bind-try", "--dev-bind", "--dev-bind-try")
# A tool runs through the interpreter of its copied environment with ``-m``, never through
# a launcher of that environment (a launcher's first line names the original).
COMMAND = ["/work/venv-dolma/bin/python", "-m", "dolma", "tokens", "--documents",
           "pipeline/round-robin/tokenizer-input/shard-1.jsonl"]


@pytest.fixture
def places(tmp_path):
    """Paths of one sandboxed step. None of them exists, except where a test makes it."""
    work = tmp_path / "work"
    return {
        "work": work,
        "home": work / "home",
        "chdir": work / "build",
        "inside_env": work / "venv-a",
        "outside_env": tmp_path / "earlier" / "venv-b",
        "lock_prefix": ["lock-command", str(tmp_path / "lock-file"), "--", "low-priority"],
    }


def _argv(sandbox, places, **changes):
    arguments = dict(work=places["work"], home=places["home"], chdir=places["chdir"],
                     command=list(COMMAND),
                     ro_binds=[places["inside_env"], places["outside_env"]],
                     env={"PATH": "/usr/bin", "LANG": "C.UTF-8"})
    arguments.update(changes)
    return sandbox.bwrap_argv(**arguments)


def _options(argv, command=COMMAND):
    """The options before the command."""
    return argv[: len(argv) - len(command)]


def _triples(options, flag):
    """The (source, target) pairs that follow each ``flag``."""
    return [(options[i + 1], options[i + 2]) for i, item in enumerate(options)
            if item == flag and i + 2 < len(options)]


def _sandbox(need_module):
    return need_module("baselines.sandbox")


def _named(module, name):
    """A name of the module, or a failed assertion that says it is not there."""
    found = getattr(module, name, None)
    assert found is not None, f"{module.__name__} has no {name}"
    return found


def test_names_of_the_sandbox_module(need_attribute):
    for name in ("bwrap_argv", "locked", "sandboxed_step", "prepare_home"):
        assert callable(need_attribute("baselines.sandbox", name))


def test_the_argv_is_a_list_of_strings_that_ends_with_the_command(need_module, places):
    argv = _argv(_sandbox(need_module), places)
    assert isinstance(argv, list) and all(isinstance(item, str) for item in argv)
    assert argv[0] == "bwrap"
    assert argv[-len(COMMAND):] == COMMAND


def test_the_program_name_is_an_argument(need_module, places):
    argv = _argv(_sandbox(need_module), places, bwrap="bwrap-other")
    assert argv[0] == "bwrap-other"


def test_every_namespace_is_unshared_and_the_sandbox_dies_with_its_parent(need_module, places):
    options = _options(_argv(_sandbox(need_module), places))
    assert "--unshare-all" in options
    assert "--die-with-parent" in options
    assert "--share-net" not in options


def test_usr_and_etc_are_read_only_and_dev_is_fresh(need_module, places):
    options = _options(_argv(_sandbox(need_module), places))
    assert ("/usr", "/usr") in _triples(options, "--ro-bind")
    assert ("/etc", "/etc") in _triples(options, "--ro-bind")
    assert "--dev" in options and options[options.index("--dev") + 1] == "/dev"
    assert "--dev-bind" not in options and "--dev-bind-try" not in options
    writable = [pair for flag in WRITABLE_BINDS for pair in _triples(options, flag)]
    assert ("/usr", "/usr") not in writable and ("/etc", "/etc") not in writable
    assert ("/dev", "/dev") not in writable


def test_the_work_folder_is_the_only_writable_bind(need_module, places):
    options = _options(_argv(_sandbox(need_module), places))
    writable = [(flag, pair) for flag in WRITABLE_BINDS for pair in _triples(options, flag)]
    work = str(places["work"])
    assert writable == [("--bind", (work, work))]


def test_each_environment_is_bound_read_only_after_the_work_folder(need_module, places):
    options = _options(_argv(_sandbox(need_module), places))
    work_at = next(i for i, item in enumerate(options)
                   if item == "--bind" and options[i + 1] == str(places["work"]))
    for key in ("inside_env", "outside_env"):
        path = str(places[key])
        at = [i for i, item in enumerate(options)
              if item == "--ro-bind" and options[i + 1] == path and options[i + 2] == path]
        assert len(at) == 1, f"{path} is bound read-only once"
        assert at[0] > work_at, "a bind that comes later wins: the read-only bind comes after the work folder"
        assert path not in [p for flag in WRITABLE_BINDS for pair in _triples(options, flag) for p in pair]


def test_a_read_only_bind_of_the_work_folder_or_of_a_parent_is_refused(need_module, places):
    sandbox = _sandbox(need_module)
    with pytest.raises(ValueError):
        _argv(sandbox, places, ro_binds=[places["work"]])
    with pytest.raises(ValueError):
        _argv(sandbox, places, ro_binds=[places["work"].parent])


def test_the_environment_is_cleared_then_given_home_and_the_named_variables(need_module, places):
    options = _options(_argv(_sandbox(need_module), places))
    assert "--clearenv" in options
    clear_at = options.index("--clearenv")
    setenv_at = [i for i, item in enumerate(options) if item == "--setenv"]
    assert setenv_at and all(i > clear_at for i in setenv_at), "--clearenv comes before every --setenv"
    given = {options[i + 1]: options[i + 2] for i in setenv_at}
    assert given == {"HOME": str(places["home"]), "PATH": "/usr/bin", "LANG": "C.UTF-8"}


def test_the_home_variable_is_never_given_through_env(need_module, places):
    with pytest.raises(ValueError):
        _argv(_sandbox(need_module), places, env={"HOME": "/elsewhere"})


def test_home_and_chdir_must_lie_inside_the_work_folder(need_module, places, tmp_path):
    sandbox = _sandbox(need_module)
    with pytest.raises(ValueError):
        _argv(sandbox, places, home=tmp_path / "other-home")
    with pytest.raises(ValueError):
        _argv(sandbox, places, chdir=tmp_path / "elsewhere")


def test_a_relative_location_is_refused(need_module, places):
    sandbox = _sandbox(need_module)
    for name in ("work", "home", "chdir"):
        with pytest.raises(ValueError):
            _argv(sandbox, places, **{name: "relative/folder"})
    with pytest.raises(ValueError):
        _argv(sandbox, places, ro_binds=["relative/env"])


def test_a_tool_may_start_in_a_folder_of_a_read_only_bind(need_module, places):
    build = places["outside_env"] / "build"
    options = _options(_argv(_sandbox(need_module), places, chdir=build))
    assert options[options.index("--chdir") + 1] == str(build)


def test_the_tool_starts_in_the_build_folder(need_module, places):
    options = _options(_argv(_sandbox(need_module), places))
    assert "--chdir" in options
    assert options[options.index("--chdir") + 1] == str(places["chdir"])


def test_building_the_argv_makes_no_file_and_no_folder(need_module, places, tmp_path):
    before = sbx_kit.tree_digest(tmp_path)
    _argv(_sandbox(need_module), places)
    assert sbx_kit.tree_digest(tmp_path) == before


def test_a_step_is_its_argv_behind_the_lock_prefix_it_is_given(need_module, places):
    sandbox = _sandbox(need_module)
    prefix = places["lock_prefix"]
    assert sandbox.locked(["echo", "one"], lock_prefix=prefix) == [*prefix, "echo", "one"]
    assert sandbox.locked(["echo", "one"], lock_prefix=("other-lock",)) == ["other-lock", "echo", "one"]


def test_a_step_with_no_lock_prefix_is_refused(need_module, places):
    sandbox = _sandbox(need_module)
    for empty in ([], ()):
        with pytest.raises(ValueError):
            sandbox.locked(["echo", "one"], lock_prefix=empty)
        with pytest.raises(ValueError):
            sandbox.sandboxed_step(
                lock_prefix=empty, work=places["work"], home=places["home"], chdir=places["chdir"],
                command=list(COMMAND))


def test_every_step_is_both_locked_and_sandboxed(need_module, places):
    sandbox = _sandbox(need_module)
    arguments = dict(work=places["work"], home=places["home"], chdir=places["chdir"],
                     command=list(COMMAND), ro_binds=[places["outside_env"]],
                     env={"PATH": "/usr/bin"})
    prefix = places["lock_prefix"]
    step = sandbox.sandboxed_step(lock_prefix=prefix, **arguments)
    inner = sandbox.bwrap_argv(**arguments)
    assert step == [*prefix, *inner]
    assert step[: len(prefix)] == prefix and step[len(prefix)] == "bwrap"


def test_prepare_home_makes_an_empty_folder_inside_the_work_folder(need_module, places):
    sandbox = _sandbox(need_module)
    places["work"].mkdir()
    made = sandbox.prepare_home(places["work"])
    assert made == places["home"]
    assert made.is_dir() and list(made.iterdir()) == []


def test_prepare_home_refuses_a_name_that_leaves_the_work_folder(need_module, places, tmp_path):
    sandbox = _sandbox(need_module)
    places["work"].mkdir()
    with pytest.raises(ValueError):
        sandbox.prepare_home(places["work"], "../escaped-home")
    assert not (tmp_path / "escaped-home").exists()


def test_prepare_home_refuses_a_home_that_holds_something(need_module, places):
    sandbox = _sandbox(need_module)
    (places["work"] / "home").mkdir(parents=True)
    (places["work"] / "home" / "credentials").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        sandbox.prepare_home(places["work"])


def test_the_module_holds_no_location_of_this_machine(need_module):
    sandbox = _sandbox(need_module)
    assert sbx_kit.local_path_constants(sandbox) == []
    text = sbx_kit.source_text(sandbox)
    assert "fd-cpu" not in text and "systemd" not in text, "the lock command is an argument"


# --- A tool runs through the interpreter of its copied environment --------------------

INTERPRETER_FOLDER = "cpython-3.12.13-linux-x86_64-gnu"
INTERPRETER_LINK = "cpython-3.12-linux-x86_64-gnu"
READ_ONLY_BINDS = ("--ro-bind", "--ro-bind-try")


@pytest.fixture
def tool_copy(places, tmp_path):
    """A copied environment inside the work folder, shaped like the real ones: its
    ``bin/python`` is a link to a binary that lies in an interpreter folder OUTSIDE the
    work folder, reached through a second link that carries the minor version only;
    ``bin/dolma`` is a launcher. The binary is a plain file: nothing here starts it."""
    interpreters = tmp_path / "uv-python"
    real = interpreters / INTERPRETER_FOLDER
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "python3.12").write_bytes(b"a stand-in for an interpreter binary\n")
    (real / "lib" / "python3.12").mkdir(parents=True)
    (real / "lib" / "python3.12" / "os.py").write_bytes(b"# the standard library\n")
    os.symlink(INTERPRETER_FOLDER, interpreters / INTERPRETER_LINK)
    copy = places["work"] / "venv-dolma"
    (copy / "bin").mkdir(parents=True)
    os.symlink(str(interpreters / INTERPRETER_LINK / "bin" / "python3.12"), copy / "bin" / "python")
    (copy / "bin" / "dolma").write_bytes(b"#!/original/venv/bin/python\n")
    return SimpleNamespace(copy=copy, interpreters=interpreters, real=real,
                           binary=real / "bin" / "python3.12")


def _links_on_the_way(link):
    """The paths a chain of links leads through, as the links write them."""
    hops = []
    current = Path(link)
    while current.is_symlink() and len(hops) < 20:
        current = Path(os.path.normpath(os.path.join(current.parent, os.readlink(current))))
        hops.append(current)
    return hops


def _visible(path, options, work):
    """True when ``path`` is there inside the sandbox of ``options``: below the work
    folder, or below the target of a read-only bind whose source holds it."""
    path = str(path)
    if _inside(path, str(work)):
        return True
    for source, target in [pair for flag in READ_ONLY_BINDS for pair in _triples(options, flag)]:
        if _inside(path, target):
            if os.path.lexists(os.path.join(source, os.path.relpath(path, target))):
                return True
    return False


def _inside(path, folder):
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def test_names_of_the_tool_route(need_attribute):
    for name in ("tool_command", "interpreter_binds"):
        assert callable(need_attribute("baselines.sandbox", name))


def test_a_tool_runs_through_the_interpreter_of_its_copy_with_m(need_module, tool_copy):
    sandbox = _sandbox(need_module)
    command = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", ["tokens", "--documents", "shard-1.jsonl"])
    assert command == [str(tool_copy.copy / "bin" / "python"), "-m", "dolma", "tokens",
                       "--documents", "shard-1.jsonl"]
    assert command[0] != str(tool_copy.binary), "the interpreter is named by its path in the copy"
    assert not any(part.endswith("/dolma") for part in command), "the launcher of the copy is not the route"
    assert _named(sandbox, "tool_command")(tool_copy.copy, "originblame") == [
        str(tool_copy.copy / "bin" / "python"), "-m", "originblame"]


def test_the_command_of_the_argv_tests_is_the_one_a_tool_is_given(need_module):
    # COMMAND, the command the argv tests above run, is a tool's command: the interpreter
    # of its copy, then ``-m``, then the module and its arguments.
    sandbox = _sandbox(need_module)
    assert _named(sandbox, "tool_command")(Path("/work/venv-dolma"), "dolma", COMMAND[3:]) == COMMAND


def test_a_tool_command_refuses_a_relative_environment_and_a_module_that_is_no_name(need_module, tool_copy):
    sandbox = _sandbox(need_module)
    with pytest.raises(ValueError):
        _named(sandbox, "tool_command")("venv-dolma", "dolma")
    for module in ("", "bin/dolma", "../dolma", "-c", "dolma tokens", "dolma;ls"):
        with pytest.raises(ValueError):
            _named(sandbox, "tool_command")(tool_copy.copy, module)


def test_the_interpreter_of_the_copy_is_visible_inside_the_sandbox_and_read_only(
        need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    binds = [Path(bind) for bind in _named(sandbox, "interpreter_binds")(tool_copy.copy)]
    assert binds and all(bind.is_absolute() for bind in binds)
    command = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", ["tokens"])
    argv = sandbox.bwrap_argv(work=places["work"], home=places["home"], chdir=places["chdir"],
                              command=command, ro_binds=[*binds], env={"PATH": "/usr/bin"})
    options = _options(argv, command)
    assert argv[-len(command):] == command
    # Every link from bin/python to the binary, the binary, the folder that holds its bin
    # and lib folders and a file of the standard library are there inside.
    on_the_way = _links_on_the_way(tool_copy.copy / "bin" / "python")
    assert on_the_way, "bin/python of the copy is a link"
    needed = [tool_copy.copy / "bin" / "python", *on_the_way, tool_copy.binary, tool_copy.real,
              tool_copy.real / "lib" / "python3.12" / "os.py"]
    missing = [str(path) for path in needed if not _visible(path, options, places["work"])]
    assert missing == []
    # Read only: the work folder stays the only writable bind.
    writable = [(flag, pair) for flag in WRITABLE_BINDS for pair in _triples(options, flag)]
    assert writable == [("--bind", (str(places["work"]), str(places["work"])))]


def test_the_step_of_a_tool_is_locked_and_sandboxed_and_starts_in_the_copys_interpreter(
        need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    command = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", ["tokens"])
    step = sandbox.sandboxed_step(
        lock_prefix=places["lock_prefix"], work=places["work"], home=places["home"],
        chdir=places["chdir"], command=command, ro_binds=_named(sandbox, "interpreter_binds")(tool_copy.copy))
    prefix = places["lock_prefix"]
    assert step[: len(prefix)] == prefix and step[len(prefix)] == "bwrap"
    assert step[-len(command):] == command
    assert step[-len(command)] == str(tool_copy.copy / "bin" / "python")
    assert "--unshare-all" in step and "--clearenv" in step


def test_the_binds_of_the_interpreter_are_found_by_reading_links_and_make_nothing(need_module, tool_copy, tmp_path):
    sandbox = _sandbox(need_module)
    before = sbx_kit.tree_digest(tmp_path)
    _named(sandbox, "interpreter_binds")(tool_copy.copy)
    _named(sandbox, "tool_command")(tool_copy.copy, "dolma")
    assert sbx_kit.tree_digest(tmp_path) == before


# --- The interpreter of the copy is recorded from inside the sandbox ----------------------
#
# Before a tool runs, the sandboxed step writes, from inside the sandbox, two facts into a
# file in the work folder: the path that ``bin/python`` of the copy resolves to and the
# sha256 of that binary. ``recording_command`` builds the command that does so and then
# starts the tool. These tests read the argument list only: nothing is started (not the
# shell part either), and the stand-in binary is a plain file.

SHELLS = ("sh", "bash", "dash")
READS_THE_PATH = re.compile(r"readlink|realpath")
HASHES_THE_BINARY = re.compile(r"sha256sum|shasum")
STARTS_THE_TOOL = re.compile(r'"\$@"|"\$\{@\}"')
TOOL_ARGUMENTS = ["tokens", "--documents", "pipeline/round-robin/tokenizer-input/shard-1.jsonl"]


def _recording(sandbox, tool_copy, places, arguments=TOOL_ARGUMENTS, record_file=None, module="dolma"):
    record = places["work"] / "interpreter-record.txt" if record_file is None else record_file
    return _named(sandbox, "recording_command")(
        tool_copy.copy, module, list(arguments), work=places["work"], record_file=record)


def test_names_of_the_recording(need_attribute):
    assert callable(need_attribute("baselines.sandbox", "recording_command"))


def test_the_tool_command_is_the_unchanged_end_of_the_recording_command(need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    command = _recording(sandbox, tool_copy, places)
    tool = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", TOOL_ARGUMENTS)
    assert isinstance(command, list) and all(isinstance(part, str) for part in command)
    assert command[-len(tool):] == tool
    assert len(command) > len(tool), "something comes before the tool"


def test_the_recording_runs_in_a_shell_that_is_given_the_record_file_as_an_argument(
        need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    record = places["work"] / "interpreter-record.txt"
    command = _recording(sandbox, tool_copy, places, record_file=record)
    tool = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", TOOL_ARGUMENTS)
    assert os.path.isabs(command[0]) and os.path.basename(command[0]) in SHELLS
    assert command[1] == "-c" and isinstance(command[2], str) and command[2].strip() != ""
    between = command[3: len(command) - len(tool)]
    assert str(record) in between, "the file is an argument of the shell, one element"
    assert str(record) not in command[2], "the script holds no path: every path is an argument"


def test_the_script_reads_the_path_and_hashes_the_binary_before_it_starts_the_tool(
        need_module, places, tool_copy):
    script = _recording(_sandbox(need_module), tool_copy, places)[2]
    read = READS_THE_PATH.search(script)
    hashed = HASHES_THE_BINARY.search(script)
    assert read, "the script resolves the path of the interpreter with a program that reads links"
    assert hashed, "the script takes the sha256 of the binary"
    starts = [match.start() for match in STARTS_THE_TOOL.finditer(script)]
    assert starts, "the script starts the tool from its own arguments"
    assert starts[-1] > read.start() and starts[-1] > hashed.start(), "the record comes first"


def test_the_script_never_holds_the_text_of_an_argument_of_the_tool(need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    hostile = ["a b;touch x", "$(id)", "`id`", "it's"]
    command = _recording(sandbox, tool_copy, places, arguments=hostile)
    tool = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", hostile)
    assert command[-len(tool):] == tool
    for text in hostile:
        assert text not in command[2]
    spaced = places["work"] / "a record.txt"
    command = _recording(sandbox, tool_copy, places, record_file=spaced)
    plain = _named(sandbox, "tool_command")(tool_copy.copy, "dolma", TOOL_ARGUMENTS)
    assert str(spaced) in command[3: len(command) - len(plain)]
    assert str(spaced) not in command[2]


def test_a_record_file_that_cannot_be_written_inside_the_work_folder_is_refused(
        need_module, places, tool_copy, tmp_path):
    sandbox = _sandbox(need_module)
    refused = {
        "relative": Path("interpreter-record.txt"),
        "outside the work folder": tmp_path / "elsewhere" / "interpreter-record.txt",
        "the work folder itself": places["work"],
        "inside the copy, which is bound read-only": tool_copy.copy / "interpreter-record.txt",
    }
    for record in refused.values():
        with pytest.raises(ValueError, match=r"(?s).+"):
            _recording(sandbox, tool_copy, places, record_file=record)
    assert _recording(sandbox, tool_copy, places, record_file=places["work"] / "records" / "one.txt")


def test_the_recording_command_refuses_what_a_tool_command_refuses(need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    with pytest.raises(ValueError):
        _named(sandbox, "recording_command")(
            "venv-dolma", "dolma", [], work=places["work"], record_file=places["work"] / "r.txt")
    for module in ("", "bin/dolma", "dolma;ls"):
        with pytest.raises(ValueError):
            _recording(sandbox, tool_copy, places, module=module)


def test_the_step_of_a_recorded_tool_sees_the_binary_and_writes_only_in_the_work_folder(
        need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    record = places["work"] / "interpreter-record.txt"
    command = _recording(sandbox, tool_copy, places, record_file=record)
    step = sandbox.sandboxed_step(
        lock_prefix=places["lock_prefix"], work=places["work"], home=places["home"],
        chdir=places["chdir"], command=command, ro_binds=_named(sandbox, "interpreter_binds")(tool_copy.copy))
    prefix = places["lock_prefix"]
    assert step[: len(prefix)] == prefix and step[len(prefix)] == "bwrap"
    assert step[-len(command):] == command
    options = step[len(prefix): len(step) - len(command)]
    # The binary whose sha256 is taken is there inside, the record file lies in the only
    # writable bind, and nothing else is writable.
    assert _visible(tool_copy.binary, options, places["work"])
    assert _inside(str(record), str(places["work"]))
    writable = [(flag, pair) for flag in WRITABLE_BINDS for pair in _triples(options, flag)]
    assert writable == [("--bind", (str(places["work"]), str(places["work"])))]


def test_building_the_recording_makes_no_file_and_starts_nothing(need_module, places, tool_copy, tmp_path):
    sandbox = _sandbox(need_module)
    before = sbx_kit.tree_digest(tmp_path)
    _recording(sandbox, tool_copy, places)
    assert sbx_kit.tree_digest(tmp_path) == before
    assert not (places["work"] / "interpreter-record.txt").exists()


# --- The python of a copy alone records itself, with no tool in the command ---------------
#
# The first start in a copy is its python ALONE, inside the sandbox, as its own command and
# apart from any tool command: it writes the path its link resolves to and the sha256 of
# that binary into a file of the work folder. A caller reads both and stops before it
# builds a tool command. These tests read the argument list only: nothing is started.

PROBE_RECORD = "interpreter-probe.txt"


def _probe(sandbox, tool_copy, places, record_file=None, env_dir=None):
    record = places["work"] / PROBE_RECORD if record_file is None else record_file
    return _named(sandbox, "interpreter_probe_command")(
        tool_copy.copy if env_dir is None else env_dir, record, work=places["work"])


def test_names_of_the_interpreter_probe(need_attribute):
    assert callable(need_attribute("baselines.sandbox", "interpreter_probe_command"))


def test_the_probe_is_the_copys_python_alone_with_isolated_mode_a_fixed_text_and_the_file(
        need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    record = places["work"] / PROBE_RECORD
    command = _probe(sandbox, tool_copy, places, record_file=record)
    assert isinstance(command, list) and all(isinstance(part, str) for part in command)
    assert len(command) == 5
    assert command[0] == str(tool_copy.copy / "bin" / "python")
    assert command[0] == _named(sandbox, "tool_command")(tool_copy.copy, "dolma")[0]
    assert command[0] != str(tool_copy.binary), "the interpreter is named by its path in the copy"
    assert command[1] == "-I" and command[2] == "-c"
    assert command[3].strip() != ""
    assert command[4] == str(record), "the file is its own element"


def test_the_probe_holds_no_tool_command(need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    command = _probe(sandbox, tool_copy, places)
    assert "-m" not in command
    assert not any(part.endswith("/python") for part in command[1:]), "one interpreter only"
    for module in ("dolma", "originblame"):
        tool = _named(sandbox, "tool_command")(tool_copy.copy, module)
        assert command[: len(tool)] != tool
        assert all(part != module for part in command)
    assert not any(part.endswith("/dolma") for part in command), "no launcher of the copy"


def test_the_text_of_the_probe_is_fixed_and_never_holds_the_file(need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    other_copy = places["work"] / "venv-originblame"
    texts = set()
    for record in (places["work"] / PROBE_RECORD, places["work"] / "a record $(id).txt",
                   places["work"] / "deeper" / "one.txt"):
        for env_dir in (tool_copy.copy, other_copy):
            command = _probe(sandbox, tool_copy, places, record_file=record, env_dir=env_dir)
            texts.add(command[3])
            assert command[4] == str(record)
            assert str(record) not in command[3]
            assert str(env_dir) not in command[3]
    assert len(texts) == 1, "the same text whatever the copy and the file"


def test_the_text_of_the_probe_resolves_the_interpreter_hashes_it_and_writes_the_given_file(
        need_module, places, tool_copy):
    text = _probe(_sandbox(need_module), tool_copy, places)[3]
    ast.parse(text)
    assert re.search(r"sys\.executable", text), "it starts from the python that runs it"
    assert re.search(r"realpath|resolve", text), "it resolves the links to the binary"
    assert re.search(r"sha256", text), "it takes the sha256 of the binary"
    assert re.search(r"sys\.argv\[1\]", text), "it writes to the file it is given as its argument"


def test_a_probe_file_that_cannot_be_written_inside_the_work_folder_is_refused(
        need_module, places, tool_copy, tmp_path):
    sandbox = _sandbox(need_module)
    refused = {
        "relative": Path(PROBE_RECORD),
        "outside the work folder": tmp_path / "elsewhere" / PROBE_RECORD,
        "the work folder itself": places["work"],
        "inside the copy, which is bound read-only": tool_copy.copy / PROBE_RECORD,
    }
    for record in refused.values():
        with pytest.raises(ValueError, match=r"(?s).+"):
            _probe(sandbox, tool_copy, places, record_file=record)
    with pytest.raises(ValueError, match=r"(?s).+"):
        _probe(sandbox, tool_copy, places, env_dir=Path("venv-dolma"))
    assert _probe(sandbox, tool_copy, places, record_file=places["work"] / "records" / "one.txt")


def test_the_step_of_the_probe_sees_the_binary_and_writes_only_in_the_work_folder(
        need_module, places, tool_copy):
    sandbox = _sandbox(need_module)
    record = places["work"] / PROBE_RECORD
    command = _probe(sandbox, tool_copy, places, record_file=record)
    step = sandbox.sandboxed_step(
        lock_prefix=places["lock_prefix"], work=places["work"], home=places["home"],
        chdir=places["chdir"], command=command, ro_binds=_named(sandbox, "interpreter_binds")(tool_copy.copy))
    prefix = places["lock_prefix"]
    assert step[: len(prefix)] == prefix and step[len(prefix)] == "bwrap"
    assert step[-len(command):] == command
    assert "--unshare-all" in step and "--clearenv" in step
    options = step[len(prefix): len(step) - len(command)]
    assert _visible(tool_copy.binary, options, places["work"])
    assert _inside(str(record), str(places["work"]))
    writable = [(flag, pair) for flag in WRITABLE_BINDS for pair in _triples(options, flag)]
    assert writable == [("--bind", (str(places["work"]), str(places["work"])))]


def test_building_the_probe_makes_no_file_and_starts_nothing(need_module, places, tool_copy, tmp_path):
    sandbox = _sandbox(need_module)
    before = sbx_kit.tree_digest(tmp_path)
    _probe(sandbox, tool_copy, places)
    assert sbx_kit.tree_digest(tmp_path) == before
    assert not (places["work"] / PROBE_RECORD).exists()
