"""The sandbox command line of the baselines (plan 5.5 Check 10): argv construction.

An existing tool runs only inside a bubblewrap sandbox: no network (every namespace
unshared), the system folders read-only, a fresh device folder, a cleared
environment, a home folder that is an empty folder inside the work folder, only the
work folder writable, and the copied tool environments bound read-only. The tool
starts in the build folder and is handed relative paths.

A tool runs through its copied environment: its command is the copy's own interpreter
with ``-m`` (``tool_command``), never a launcher script of the copy, whose first line
names the original environment. The interpreter of a copy is a link to a binary that
lies outside the work folder; ``interpreter_binds`` names the folders that make it
visible inside the sandbox, read-only.

Before a tool starts, the interpreter of its copy is recorded from inside the sandbox:
the path its ``bin/python`` resolves to and the sha256 of that binary go to a file in
the work folder. ``interpreter_probe_command`` is the copy's python alone doing that
(no tool command is part of it, so a caller can read the record and stop before it
builds one); ``recording_command`` is a shell that does the same and then starts the
tool, failing closed: the tool does not start when the record cannot be made.

This module builds argument lists and starts nothing. It holds no location: every
path is an argument, and so is the command that sits in front of a step to serialise
it behind the machine's shared budget (a "lock prefix"), which is required: a step
cannot be built without one. A tool that needs a scratch folder is handed one
inside the work folder through ``env`` (for example a temporary-folder variable);
nothing outside the work folder is writable.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def _absolute(value, what: str) -> str:
    """The lexically normalised absolute path of an argument, or ``ValueError``."""
    text = os.fspath(value)
    if not isinstance(text, str) or not os.path.isabs(text):
        raise ValueError(f"{what} must be an absolute path, got {text!r}")
    return os.path.normpath(text)


def _inside(path: str, folder: str) -> bool:
    """True when ``path`` is ``folder`` or lies below it (lexically)."""
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def bwrap_argv(
    *,
    work,
    home,
    chdir,
    command: Sequence[str],
    ro_binds: Iterable = (),
    env: Mapping[str, str] | None = None,
    bwrap: str = "bwrap",
) -> list[str]:
    """The argv of one sandboxed step: the program, its options, then the command.

    ``work`` is the only writable bind. ``home`` lies strictly inside ``work`` and
    outside every read-only bind. ``chdir`` lies inside ``work`` or inside one of
    ``ro_binds`` (a tool may start in a build folder that is bound read-only). Each
    of ``ro_binds`` is bound read-only AFTER the work folder, so that it wins where
    it lies inside it; one that is the work folder, or a folder that holds it, is
    refused. ``env`` holds the variables the tool is given; ``HOME`` is set from
    ``home`` and is refused there."""
    work_path = _absolute(work, "work")
    home_path = _absolute(home, "home")
    chdir_path = _absolute(chdir, "chdir")
    binds = [_absolute(entry, "a read-only bind") for entry in ro_binds]
    variables = dict(env or {})

    if home_path == work_path or not _inside(home_path, work_path):
        raise ValueError("home must lie inside the work folder")
    if not (_inside(chdir_path, work_path) or any(_inside(chdir_path, bind) for bind in binds)):
        raise ValueError("chdir must lie inside the work folder or inside a read-only bind")
    for bind in binds:
        if _inside(work_path, bind):
            raise ValueError("a read-only bind must be neither the work folder nor a folder that holds it")
        if _inside(home_path, bind):
            raise ValueError("the home folder must not lie inside a read-only bind")
    if "HOME" in variables:
        raise ValueError("HOME is set from the home folder and is not an env variable")
    for name, value in variables.items():
        if not isinstance(name, str) or not isinstance(value, str) or not name or "=" in name:
            raise ValueError(f"an env entry must be a name and a text, got {name!r}")

    argv = [
        str(bwrap),
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/etc", "/etc",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--dev", "/dev",
        "--proc", "/proc",
        "--bind", work_path, work_path,
    ]
    for bind in binds:
        argv += ["--ro-bind", bind, bind]
    argv += ["--clearenv", "--setenv", "HOME", home_path]
    for name, value in variables.items():
        argv += ["--setenv", name, value]
    argv += ["--chdir", chdir_path, "--"]
    argv += [str(part) for part in command]
    return argv


def _prefix(lock_prefix) -> list[str]:
    if isinstance(lock_prefix, str) or not lock_prefix:
        raise ValueError("a step needs a lock prefix: a non-empty list of arguments")
    return [str(part) for part in lock_prefix]


def locked(argv: Sequence[str], *, lock_prefix) -> list[str]:
    """``argv`` behind the lock command the caller hands in."""
    return [*_prefix(lock_prefix), *argv]


def sandboxed_step(*, lock_prefix, **bwrap_arguments) -> list[str]:
    """One step: the lock command outside, the sandbox inside."""
    prefix = _prefix(lock_prefix)
    return locked(bwrap_argv(**bwrap_arguments), lock_prefix=prefix)


def prepare_home(work, name: str = "home") -> Path:
    """Make the empty home folder ``work/name`` and return its path.

    Refused (``ValueError``): a work folder that does not exist, a name that is not
    a folder inside the work folder, and a folder that exists and holds something
    (nothing is made or removed then)."""
    work_path = Path(_absolute(work, "work"))
    if not work_path.is_dir():
        raise ValueError(f"the work folder {work_path} does not exist")
    target = Path(os.path.normpath(work_path / name))
    if target == work_path or not _inside(str(target), str(work_path)):
        raise ValueError(f"the home name {name!r} is not a folder inside the work folder")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise ValueError(f"{target} exists and is not an empty folder")
        return target
    target.mkdir()
    return target


def tool_command(env_dir, module: str, arguments: Iterable = ()) -> list[str]:
    """The command of a tool run through its copied environment.

    ``[ENV_DIR/bin/python, "-m", MODULE, *ARGUMENTS]``: the interpreter named by its
    path in the copy (the link as it stands, not the binary it resolves to, because an
    interpreter finds its environment from the path it was started by), never a
    launcher script of the copy. ``env_dir`` must be absolute and ``module`` a dotted
    name; anything else raises ``ValueError``. Nothing is read or started."""
    env_path = _absolute(env_dir, "env_dir")
    if not isinstance(module, str) or MODULE_NAME.fullmatch(module) is None:
        raise ValueError(f"a tool module must be a dotted name, got {module!r}")
    return [os.path.join(env_path, "bin", "python"), "-m", module, *[str(part) for part in arguments]]


def _links_on_the_way(link: str) -> list[str]:
    """The paths a chain of links leads through, from the link's own target to the end,
    each as the link writes it (made absolute and normalised, not resolved)."""
    hops: list[str] = []
    current = link
    while os.path.islink(current):
        if len(hops) >= 40:
            raise ValueError(f"{link} leads through too many links")
        current = os.path.normpath(os.path.join(os.path.dirname(current), os.readlink(current)))
        hops.append(current)
    return hops


def interpreter_binds(env_dir) -> list[str]:
    """The folders to hand to ``bwrap_argv`` as ``ro_binds`` so that the interpreter of a
    copied environment runs inside the sandbox.

    An interpreter folder holds ``bin`` and ``lib`` and sits beside a link that names
    it by the minor version only; ``bin/python`` of the copy and the ``home`` of its
    ``pyvenv.cfg`` lead through that link. The folder that holds the interpreter folders
    (the real one and the one a link on the way leads through) is bound at its own
    path, read-only, so that every link on the way reads the same inside as outside;
    this exposes every interpreter that folder holds, read-only. The list holds each
    such folder once, sorted. It leaves out what is already there: ``/usr`` and
    ``/etc`` (a system interpreter), and anything inside the copy itself. The copy lies
    inside the work folder, which is writable: the caller passes the copy itself as a
    read-only bind too (``ro_binds`` may lie inside the work folder), together with
    the folders of this list.
    It reads links and ``pyvenv.cfg`` and starts nothing; a ``bin/python`` that leads
    to no file raises ``ValueError``."""
    env_path = _absolute(env_dir, "env_dir")
    link = os.path.join(env_path, "bin", "python")
    binary = os.path.realpath(link)
    if not os.path.isfile(binary):
        raise ValueError(f"{link} does not lead to an existing file")
    interpreter_files = [binary, *_links_on_the_way(link)]
    config = os.path.join(env_path, "pyvenv.cfg")
    if os.path.isfile(config):
        with open(config, encoding="utf-8") as handle:
            for line in handle:
                key, separator, value = line.partition("=")
                if separator and key.strip() == "home" and os.path.isabs(value.strip()):
                    interpreter_files.append(os.path.join(os.path.normpath(value.strip()), "python"))
    folders: set[str] = set()
    for path in interpreter_files:
        root = os.path.dirname(os.path.dirname(path))
        if root == os.sep or _inside(root, "/usr") or _inside(root, "/etc") or _inside(root, env_path):
            continue
        parent = os.path.dirname(root)
        folder = parent if parent != os.sep else root
        if os.path.isdir(folder):
            folders.add(folder)
    kept = [folder for folder in sorted(folders)
            if not any(other != folder and _inside(folder, other) for other in folders)]
    return kept


# The two lines of a record file: the resolved path of the interpreter, then the sha256 of
# the binary it leads to (hex).
_PROBE_TEXT = (
    "import hashlib, os, sys\n"
    "path = os.path.realpath(sys.executable)\n"
    "with open(path, 'rb') as binary:\n"
    "    digest = hashlib.sha256(binary.read()).hexdigest()\n"
    "with open(sys.argv[1], 'x', encoding='utf-8') as out:\n"
    "    out.write(path + '\\n' + digest + '\\n')\n"
)

# A shell script whose first argument is the record file and whose other arguments are
# the tool command, whose first word is the copy's python. Every path is an argument,
# none is written here. It stops on any failure before the tool starts, and an existing
# record file is not overwritten.
_RECORD_SCRIPT = (
    "set -eu -C\n"
    'record=$1\n'
    "shift\n"
    'resolved=$(readlink -f -- "$1")\n'
    'test -f "$resolved"\n'
    'digest=$(sha256sum -- "$resolved")\n'
    'printf \'%s\\n%s\\n\' "$resolved" "${digest%% *}" > "$record"\n'
    'exec "$@"\n'
)

_SHELL = "/bin/sh"


def _record_file(record_file, work, env_dir) -> str:
    """The absolute path of a record file: inside ``work`` (not ``work`` itself) and not
    inside the copy, which is bound read-only, or ``ValueError``."""
    record = _absolute(record_file, "record_file")
    work_path = _absolute(work, "work")
    env_path = _absolute(env_dir, "env_dir")
    if record == work_path or not _inside(record, work_path):
        raise ValueError("the record file must lie inside the work folder")
    if _inside(record, env_path):
        raise ValueError("the record file must not lie inside the copied environment")
    return record


def interpreter_probe_command(env_dir, record_file, *, work) -> list[str]:
    """The command that records the interpreter of a copy: its python alone.

    ``[ENV_DIR/bin/python, "-I", "-c", TEXT, RECORD_FILE]``: the text resolves
    ``sys.executable`` and writes two lines into the NEW file ``record_file``, the
    resolved path and the sha256 (hex) of the binary it leads to. The file is an
    argument, never part of the text. No tool command is in it. The call raises
    ``ValueError`` for a record file that is not inside ``work`` (and not ``work``), or
    is inside the copy, and for a relative path. Nothing is read, made or started."""
    record = _record_file(record_file, work, env_dir)
    env_path = _absolute(env_dir, "env_dir")
    return [os.path.join(env_path, "bin", "python"), "-I", "-c", _PROBE_TEXT, record]


def recording_command(env_dir, module: str, arguments: Iterable = (), *, work, record_file) -> list[str]:
    """The command of a tool that first records its copy's interpreter, then starts.

    ``[SHELL, "-c", SCRIPT, "sh", RECORD_FILE, *tool_command]``: the script reads the link
    ``bin/python`` of the copy resolves through, takes the sha256 of the binary, writes
    both into ``record_file`` and only then starts the tool, which is ``tool_command``
    unchanged at the end of the list. A failure of any step stops the shell before the
    tool starts. The call raises ``ValueError`` for what ``tool_command`` refuses and for
    a record file that is relative, outside ``work``, ``work`` itself, or inside the
    copy. Nothing is read, made or started."""
    tool = tool_command(env_dir, module, arguments)
    record = _record_file(record_file, work, env_dir)
    return [_SHELL, "-c", _RECORD_SCRIPT, "sh", record, *tool]
