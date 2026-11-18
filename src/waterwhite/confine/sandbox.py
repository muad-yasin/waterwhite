"""The sandbox a confined job runs in: bubblewrap on Linux (plan 5.8 B-1, F10).

This module finds the ``bwrap`` command on PATH, checks that it can build the
sandbox on this machine, and writes the command line that starts a program in
it. It starts nothing itself except the check. ``runner`` starts the job.

What the command line builds, in the words of docs/confine.md:

- every namespace is new and each is required (user, mount, pid, ipc, uts,
  cgroup, network). The network namespace holds a loopback device of its own
  and no other device, so the program reaches nothing outside the sandbox;
- the file system is an empty folder into which are mounted, read-only, the
  system's library folders, the program's own code, and the job-side client;
  nothing else of the machine is there: no home folder, no data folder, no
  ``/etc``, no ``/proc``, no ``/sys``, no ``/tmp``, no ``/run``; the root itself
  is read-only;
- one folder can be written, ``/scratch``, a memory file system of a fixed size
  that exists only inside the sandbox and is gone when the program ends;
- the environment is emptied and then set to a few fixed variables;
- the program cannot make a user namespace of its own.

There is no fallback. When ``bwrap`` is missing or cannot build this, the named
error "sandbox unavailable" is raised and no job runs.
"""

from __future__ import annotations

import os
import posixpath
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import InvalidRecord
from .errors import SandboxUnavailable

__all__ = [
    "CLIENT_FOLDER",
    "CodeMount",
    "DEFAULT_SCRATCH_BYTES",
    "ENV_FDS",
    "JOB_FOLDER",
    "Program",
    "SANDBOX_COMMAND",
    "SCRATCH_FOLDER",
    "Sandbox",
    "find_sandbox",
    "python_program",
]

SANDBOX_COMMAND = "bwrap"

# Paths inside the sandbox.
CLIENT_FOLDER = "/waterwhite/client"  # the job-side client, read-only
JOB_FOLDER = "/waterwhite/job"  # where python_program puts the job's script
SCRATCH_FOLDER = "/scratch"  # the one writable folder; thrown away
_PROBE_PATH = "/waterwhite/probe"

# The variable that tells the job which two inherited file descriptors are its
# pipes. waterwhite_job reads the same name; it cannot import this module.
ENV_FDS = "WATERWHITE_JOB_FDS"

DEFAULT_SCRATCH_BYTES = 64 * 1024 * 1024

# The folders the dynamic loader and the C library live in. Each is mounted
# read-only when it is a folder, and made as the same symbolic link when it is one
# (on a merged-usr system /lib and /lib64 point into /usr).
_SYSTEM_LIBRARY_FOLDERS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64")

_PROBE_TIMEOUT_SECONDS = 20

_CLIENT_FILE = Path(__file__).with_name("waterwhite_job.py")


@dataclass(frozen=True)
class CodeMount:
    """A file or folder of code, mounted read-only into the sandbox.

    ``source`` is its place on this machine, ``target`` its absolute path inside
    the sandbox. The program can read everything under a mount, and no such read
    passes the harness or reaches its log. So a mount may hold code and must hold
    no record: the harness cannot check that, and a mount that holds records
    breaks the bound of every edge the run gives (docs/confine.md)."""

    source: str
    target: str


@dataclass(frozen=True)
class Program:
    """What runs inside the sandbox: a command line as it reads inside, and the
    read-only code it needs beyond the system's library folders."""

    argv: tuple[str, ...]
    mounts: tuple[CodeMount, ...] = ()


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _system_library_args() -> list[str]:
    args: list[str] = []
    for folder in _SYSTEM_LIBRARY_FOLDERS:
        if os.path.islink(folder):
            args += ["--symlink", os.readlink(folder), folder]
        elif os.path.isdir(folder):
            args += ["--ro-bind", folder, folder]
    return args


def _mounted_library_folders() -> tuple[str, ...]:
    return tuple(
        folder
        for folder in _SYSTEM_LIBRARY_FOLDERS
        if os.path.isdir(folder) and not os.path.islink(folder)
    )


def _check_program(program: Program) -> None:
    if (
        not isinstance(program, Program)
        or not isinstance(program.argv, tuple)
        or not program.argv
        or not all(isinstance(part, str) and part for part in program.argv)
    ):
        raise InvalidRecord("a confined job is a Program with a command line of non-empty texts")
    reserved = (SCRATCH_FOLDER, "/dev", CLIENT_FOLDER)
    seen: set[str] = set()
    for mount in program.mounts:
        if (
            not isinstance(mount, CodeMount)
            or not isinstance(mount.source, str)
            or not isinstance(mount.target, str)
        ):
            raise InvalidRecord(f"a mount is a CodeMount(source, target): {mount!r}")
        target = mount.target
        if not target.startswith("/") or posixpath.normpath(target) != target or target == "/":
            raise InvalidRecord(
                f"a mount's target is an absolute, normalised path inside the sandbox "
                f"and not its root: {target!r}"
            )
        if any(_inside(target, folder) or _inside(folder, target) for folder in reserved):
            raise InvalidRecord(
                f"a mount's target may not lie in or over one of {', '.join(reserved)}: "
                f"{target!r}"
            )
        if target in seen:
            raise InvalidRecord(f"two mounts have the target {target!r}")
        seen.add(target)
        if not os.path.exists(mount.source):
            raise InvalidRecord(f"a mount's source does not exist: {mount.source!r}")


@dataclass(frozen=True)
class Sandbox:
    """The sandbox command, found on PATH and checked on this machine."""

    command: str

    def _base_args(self, scratch_bytes: int) -> list[str]:
        return [
            self.command,
            # Each namespace is named and required. The option that unshares all
            # of them at once only tries the user and the cgroup namespace and
            # goes on without them.
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-uts",
            "--unshare-cgroup",
            "--disable-userns",
            "--die-with-parent",
            "--new-session",
            "--hostname",
            "waterwhite-job",
            "--clearenv",
            *_system_library_args(),
            "--dev",
            "/dev",
            "--size",
            str(scratch_bytes),
            "--tmpfs",
            SCRATCH_FOLDER,
        ]

    def argv(
        self,
        program: Program,
        *,
        request_fd: int,
        reply_fd: int,
        scratch_bytes: int = DEFAULT_SCRATCH_BYTES,
    ) -> list[str]:
        """The command line that starts ``program`` in the sandbox. ``request_fd``
        and ``reply_fd`` are the job's ends of the two pipes, which the process
        that runs this command line must hand down."""
        _check_program(program)
        args = self._base_args(scratch_bytes)
        for name, value in (
            ("LANG", "C.UTF-8"),
            ("HOME", SCRATCH_FOLDER),
            ("TMPDIR", SCRATCH_FOLDER),
            ("PYTHONPATH", CLIENT_FOLDER),
            ("PYTHONDONTWRITEBYTECODE", "1"),
            (ENV_FDS, f"{request_fd},{reply_fd}"),
        ):
            args += ["--setenv", name, value]
        for mount in program.mounts:
            args += ["--ro-bind", mount.source, mount.target]
        args += ["--ro-bind", str(_CLIENT_FILE), f"{CLIENT_FOLDER}/{_CLIENT_FILE.name}"]
        args += ["--remount-ro", "/", "--chdir", SCRATCH_FOLDER, "--", *program.argv]
        return args

    def _probe_argv(self) -> list[str]:
        # The sandbox command itself is the program of the check: it needs only
        # the system's library folders, and it prints its version and exits.
        return [
            *self._base_args(DEFAULT_SCRATCH_BYTES),
            "--ro-bind",
            self.command,
            _PROBE_PATH,
            "--remount-ro",
            "/",
            "--chdir",
            SCRATCH_FOLDER,
            "--",
            _PROBE_PATH,
            "--version",
        ]


def find_sandbox() -> Sandbox:
    """Find the sandbox command on PATH and check that it builds the sandbox here.

    Raises ``SandboxUnavailable`` when the command is not on PATH (an unset PATH
    is an empty one: no default folders are searched), or when the check run
    fails, which is what happens where user namespaces are switched off for
    unprivileged users. The check starts the sandbox with the same namespaces and
    mounts a job gets and runs the sandbox command's own version print in it."""
    command = shutil.which(SANDBOX_COMMAND, path=os.environ.get("PATH", ""))
    if command is None:
        raise SandboxUnavailable(
            f"the command {SANDBOX_COMMAND!r} (bubblewrap) is not on PATH; a confined job "
            f"is not run without its sandbox",
            command=SANDBOX_COMMAND,
        )
    sandbox = Sandbox(command=os.path.realpath(command))
    try:
        check = subprocess.run(
            sandbox._probe_argv(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={},
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SandboxUnavailable(
            f"{sandbox.command} could not be run to check the sandbox: {error}",
            command=sandbox.command,
        ) from None
    if check.returncode != 0:
        said = check.stderr.decode("utf-8", "replace").strip()
        raise SandboxUnavailable(
            f"{sandbox.command} cannot build the sandbox on this machine (exit status "
            f"{check.returncode}): {said}",
            command=sandbox.command,
        )
    return sandbox


def python_program(
    script: str | os.PathLike,
    args: Sequence[str] = (),
    *,
    code: Sequence[CodeMount] = (),
) -> Program:
    """A job written in Python, run by the interpreter this process runs on.

    The job gets the interpreter and its standard library and nothing installed
    beside them: when this process runs in a virtual environment, the base
    interpreter is used and the environment's packages are not mounted. ``script``
    is the job's file; it is mounted alone as ``/waterwhite/job/<its name>``.
    ``args`` are its arguments. ``code`` names further read-only code the job
    imports (see ``CodeMount`` for what such a mount must not hold)."""
    source = os.path.realpath(os.fspath(script))
    if not os.path.isfile(source):
        raise InvalidRecord(f"the job's script is not a file: {os.fspath(script)!r}")
    executable = os.path.realpath(getattr(sys, "_base_executable", "") or sys.executable)
    mounts: list[CodeMount] = []
    library_folders = _mounted_library_folders()
    for prefix in dict.fromkeys(
        os.path.realpath(p) for p in (sys.base_prefix, sys.base_exec_prefix)
    ):
        if prefix in ("/", "/usr"):
            # A system interpreter: its library lies in the system's library
            # folders, which every sandbox mounts.
            continue
        mounts.append(CodeMount(prefix, prefix))
    covered = (*library_folders, *(mount.source for mount in mounts))
    if not any(_inside(executable, folder) for folder in covered):
        mounts.append(CodeMount(executable, executable))
    mounts.append(CodeMount(source, f"{JOB_FOLDER}/{os.path.basename(source)}"))
    mounts.extend(code)
    return Program(
        argv=(
            executable,
            "-s",  # no user site folder
            "-B",  # no bytecode files
            f"{JOB_FOLDER}/{os.path.basename(source)}",
            *args,
        ),
        mounts=tuple(mounts),
    )
