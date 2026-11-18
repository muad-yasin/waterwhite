"""The check of a copied tool environment and the script that makes the copies (plan
5.5 Check 10; table README, section "Check 10", condition ``sandbox``).

The setup downloads nothing and starts nothing of an environment. It copies the two
environments ``venv-dolma`` and ``venv-ob`` out of an earlier install folder
(read-only: nothing is written there) into the work folder. Before any tool runs in a
copy, every file of the copy is compared with the original's, including what the
``RECORD`` check leaves out (launcher scripts, files in no ``RECORD``,
``pyvenv.cfg``, links); each difference is listed with its reason, and a difference
STOPS that tool. Then each copy is checked as it will be used: the sha256 of every
package file against the row of the environment's own ``RECORD`` files (launcher
scripts excluded: they carry the old location of the environment in their first
line), and the interpreter the copy binds to, read from files only: the version from
``pyvenv.cfg`` and from the name of the interpreter folder ``bin/python`` resolves
into, and the sha256 of the interpreter binary. A mismatch, a missing file, or a copy
that differs from its original STOPS that tool: a named message and a nonzero exit;
nothing is repaired and nothing is installed.

These tests run the check on a miniature environment made by the ``sbx_kit`` helper
below (a fake distribution with ``RECORD`` rows and a few files; its ``bin/python``
leads to a plain file that is not executable, shaped like the uv interpreter folders
of the real environments). No real environment is copied or run, and every in-process
call of the check runs with every way of starting a process made to raise.

The names this file asks of ``baselines.envcheck``:

- ``EnvCheckError``: raised by the check, its text names the distribution folder and
  the file (the path of the ``RECORD`` row) or the interpreter at fault;
- ``check_environment(env_dir)``: reads every top-level
  ``<name>-<version>.dist-info/RECORD`` under ``lib/python3.12/site-packages``; for
  each row with a ``sha256=`` hash whose path does not lead to the environment's own
  ``bin`` folder, the file's sha256 (URL-safe base64, no padding) must equal the row.
  It starts nothing. It returns a report with ``env`` (the environment folder's
  name), ``interpreter_home`` (the ``home`` of ``pyvenv.cfg``), ``interpreter_folder``
  (the NAME of the folder that ``bin/python`` resolves into, two levels above the
  binary), ``version_info`` (the ``version_info`` of ``pyvenv.cfg``),
  ``interpreter_sha256`` (the hex sha256 of the interpreter binary that ``bin/python``
  resolves to), ``files_checked`` (the number of rows compared) and ``as_dict()``
  with those six names and no other. A ``bin/python`` that does not resolve to an
  existing file raises. A ``version_info`` that the version in the interpreter folder's
  name does not begin with (read as numbers: 3.12 begins 3.12.13, 3.11 and 3.1 do not)
  raises, and the text names the environment and both versions. Nothing in the
  environment is written.
- ``main(argv=None)``: ``ENV_DIR [--out REPORT.json]``; returns 0 and writes the
  report's ``as_dict()`` as JSON to ``--out`` (a new file) when the check passes;
  returns a nonzero number, prints the error's text to standard error and writes no
  report when it does not. With ``--original`` and ``--record`` a copy that differs
  returns 1, prints each difference to standard error and leaves ``compare.json``
  alone in the new record folder. The module runs it under ``python -m baselines.envcheck``.
- ``write_record(env_dir, record_dir)``: writes, into the new folder ``record_dir``
  (``FileExistsError`` or ``EnvCheckError`` if it exists), the record of what the
  environment uses, ``rows.tsv``, ``unlisted.tsv`` and ``identity.json`` (which holds
  ``rows_sha256``, the sha256 of the rows file), in the form given in the tests below. It does not refuse a mismatch or a missing file: it records them.
- ``compare_environments(original, copy)``: a list with one item for each file, link
  or folder that differs between the two environment folders (each item has ``path``,
  relative to the environment folder, and ``reason``, a non-empty text); an empty
  list for a copy equal to its original. It starts
  nothing and writes nothing; a folder that does not exist raises ``EnvCheckError``.

The names this file asks of ``tools/setup_baselines_env.sh``:

- ``bash tools/setup_baselines_env.sh EARLIER WORK PYTHON``: ``EARLIER`` is the
  earlier install folder (holding ``venv-dolma`` and ``venv-ob``), ``WORK`` the work
  folder, ``PYTHON`` the interpreter that runs ``baselines.envcheck``. It copies
  each environment to ``WORK/<name>`` keeping links as they are (with ``cp``, found
  on the command path: the tests that need a difference between the original and
  its copy put a ``cp`` of their own first on the path), compares the copy with the
  original, runs the check on the copy, and writes ``WORK/env-check-<name>.json``
  and the folder ``WORK/env-record-<name>`` (``write_record``) for a copy that
  passed; each record folder also holds ``compare.json``, the result of the compare of
  that copy with its original (``{"equal": true, "differences": []}`` for an equal
  copy; ``main`` writes it into the folder given by ``--record`` when ``--original`` is
  given). A copy that differs from its original gets a record folder that holds
  ``compare.json`` alone, with ``equal`` false and every difference as a ``path`` and a
  ``reason``. It handles both environments even when the first fails; it exits nonzero
  when either fails or is missing, and its output names the environment
  (``venv-dolma``) and the file or the reason. It writes nothing under ``EARLIER``.
  No location is in the file; it holds no command that fetches or installs and none
  that starts anything of either environment.

A module or a script that does not exist yet fails a test on an assertion.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import os
import shutil
from dataclasses import dataclass, field
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The miniature environment follows the layout of an installed environment as the
# installer writes it (read once, in place, from an environment that was made earlier;
# nothing was copied or run):
#
# - ``pyvenv.cfg`` holds ``home`` (the ``bin`` folder of the interpreter the environment
#   binds to) and ``version_info`` (major and minor only; there is no ``version`` key);
# - ``bin/python`` is a link whose chain ends at the interpreter binary inside a folder
#   named for the full version (``cpython-3.12.13-linux-x86_64-gnu``); ``home`` names a
#   second link to that folder that carries the minor version only
#   (``cpython-3.12-linux-x86_64-gnu``), so the full version is read only from the
#   resolved folder;
# - each distribution has ``<name>-<version>.dist-info/RECORD`` beside its package
#   folders in ``lib/python3.12/site-packages``; a row is ``path,hash,size``, the path
#   relative to ``site-packages``, the hash ``sha256=`` and the URL-safe base64 of the
#   digest with no padding;
# - a launcher script is listed as ``../../../bin/<name>`` and its bytes name the
#   environment's own old path in the first line;
# - the RECORD file lists itself with an empty hash and size, and a compiled file may
#   carry an empty hash too;
# - a package may carry a vendored distribution whose own RECORD sits deeper in the
#   tree; the files of it are rows of the outer RECORD;
# - ``site-packages`` holds two files that no RECORD lists (a startup hook and its
#   module) and a compiled file under ``__pycache__``.
#
# The interpreter binary of the miniature is a plain file that is NOT executable: a test
# that started it would fail. The expected hashes and the expected rows here are worked
# out with ``hashlib`` from the bytes the test wrote, by the rule of the record (plan 5.5
# Check 10, and the table README's sandbox condition), never taken from the code under
# test. Nothing here imports the ``baselines`` package.


def _make_sbx_kit() -> SimpleNamespace:
    SITE = Path("lib") / "python3.12" / "site-packages"

    DIST = "fakepkg-1.2"
    OTHER_DIST = "otherpkg-0.4"
    LAUNCHER = "faketool"

    FILES = {
        "fakepkg/__init__.py": b"VERSION = '1.2'\n",
        "fakepkg/core.py": b"def answer():\n    return 42\n",
        f"{DIST}.dist-info/METADATA": b"Name: fakepkg\nVersion: 1.2\n",
    }
    OTHER_FILES = {
        "otherpkg/__init__.py": b"VALUE = 1\n",
        "otherpkg/util.py": b"def twice(x):\n    return 2 * x\n",
        f"{OTHER_DIST}.dist-info/METADATA": b"Name: otherpkg\nVersion: 0.4\n",
    }
    LAUNCHER_BYTES = b"#!/old/place/venv/bin/python\nprint('tool')\n"
    PYC_ROW = "fakepkg/__pycache__/core.cpython-312.pyc"

    VENDOR_MODULE = "fakepkg/_vendor/vend/mod.py"
    VENDOR_MODULE_BYTES = b"VENDORED = True\n"
    VENDOR_RECORD = "fakepkg/_vendor/vend-0.1.dist-info/RECORD"

    INTERPRETER_FOLDER = "cpython-3.12.13-linux-x86_64-gnu"
    INTERPRETER_LINK = "cpython-3.12-linux-x86_64-gnu"
    INTERPRETER_BYTES = b"a stand-in for an interpreter binary: a plain file, not executable\n"
    UNLISTED = {
        "_virtualenv.pth": b"import _virtualenv\n",
        "_virtualenv.py": b"# a startup hook that no RECORD lists\n",
    }
    PRUNED = "__pycache__/_virtualenv.cpython-312.pyc"

    def record_hash(data: bytes) -> str:
        """The hash field of a RECORD row."""
        digest = hashlib.sha256(data).digest()
        return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def hex_sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @dataclass
    class Miniature:
        """A made environment: where it is, what its RECORD files list, and the rows
        the record of it must hold."""

        root: Path
        home: str
        version_info: str
        interpreters: Path
        # RECORD rows that are checked: path (relative to site-packages) -> bytes.
        checked: dict[str, bytes] = field(default_factory=dict)
        # The rows of the record, in file order: (package, path, sha256 hex, verdict).
        rows: list[tuple[str, str, str, str]] = field(default_factory=list)
        # The files in no RECORD (a compiled file under __pycache__ is left out):
        # (path relative to site-packages, sha256 hex), sorted by path.
        unlisted: list[tuple[str, str]] = field(default_factory=list)

        @property
        def site(self) -> Path:
            return self.root / SITE

        @property
        def python(self) -> Path:
            return self.root / "bin" / "python"

        @property
        def binary(self) -> Path:
            return self.interpreters / INTERPRETER_FOLDER / "bin" / "python3.12"

        @property
        def interpreter_sha256(self) -> str:
            return hex_sha256(INTERPRETER_BYTES)

        @property
        def checked_count(self) -> int:
            """How many rows a check reads: the rows with a hash that are no launcher.
            The RECORD rows (empty hash) and the compiled-file row (empty hash) are not
            among them."""
            return len(self.checked)

        def file(self, relative: str) -> Path:
            return self.site / relative

        def rows_text(self, rows=None) -> str:
            """The text of the rows file of the record: a header, then one row per
            line, the four words separated by tabs."""
            rows = self.rows if rows is None else rows
            return "package\tpath\tsha256\tverdict\n" + "".join("\t".join(r) + "\n" for r in rows)

        def unlisted_text(self) -> str:
            return "path\tsha256\n" + "".join(f"{p}\t{h}\n" for p, h in self.unlisted)

    def _record_text(rows: dict[str, bytes], extra: list[str], dist: str) -> str:
        lines = [f"{path},{record_hash(data)},{len(data)}" for path, data in rows.items()]
        lines += extra
        lines.append(f"{dist}.dist-info/RECORD,,")
        return "\n".join(lines) + "\n"

    def make_environment(
        root: Path,
        *,
        home: str | None = None,
        version_info: str | None = None,
        interpreter_missing: bool = False,
        second_distribution: bool = False,
        vendored: bool = False,
        interpreters: Path | None = None,
    ) -> Miniature:
        """Make a small environment under ``root`` (a new folder).

        The interpreter folders are made under ``interpreters`` (default: a folder
        ``uv-python`` beside ``root``; one folder may serve several environments):
        the binary, a plain file, in ``cpython-3.12.13-linux-x86_64-gnu/bin``, and a
        link ``cpython-3.12-linux-x86_64-gnu`` to that folder. ``bin/python`` leads to
        the binary through the link and ``home`` (unless a text is given) names the
        link's ``bin`` folder. With ``interpreter_missing`` True, ``bin/python`` is a
        link to a file that does not exist."""
        root = Path(root)
        (root / "bin").mkdir(parents=True)
        site = root / SITE
        site.mkdir(parents=True)

        interpreters = Path(interpreters) if interpreters is not None else root.parent / "uv-python"
        real_folder = interpreters / INTERPRETER_FOLDER
        link = interpreters / INTERPRETER_LINK
        if interpreter_missing:
            target = interpreters / "cpython-3.12.99-linux-x86_64-gnu" / "bin" / "python3.12"
        else:
            binary = real_folder / "bin" / "python3.12"
            if not binary.exists():
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(INTERPRETER_BYTES)
                os.chmod(binary, 0o644)
            if not link.is_symlink():
                os.symlink(INTERPRETER_FOLDER, link)
            target = link / "bin" / "python3.12"
        home = home or str(link / "bin")
        version_info = version_info or "3.12"
        os.symlink(str(target), root / "bin" / "python")
        (root / "pyvenv.cfg").write_text(
            f"home = {home}\nimplementation = CPython\nuv = 0.11.27\n"
            f"version_info = {version_info}\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )

        checked: dict[str, bytes] = {}
        rows: list[tuple[str, str, str, str]] = []
        for dist, files in ((DIST, FILES), (OTHER_DIST, OTHER_FILES)):
            if dist == OTHER_DIST and not second_distribution:
                continue
            files = dict(files)
            if dist == DIST and vendored:
                files[VENDOR_MODULE] = VENDOR_MODULE_BYTES
                files[VENDOR_RECORD] = (
                    f"vend/mod.py,{record_hash(VENDOR_MODULE_BYTES)},{len(VENDOR_MODULE_BYTES)}\n"
                    "vend-0.1.dist-info/RECORD,,\n").encode("utf-8")
            package = f"{dist}.dist-info"
            for relative, data in files.items():
                path = site / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                rows.append((package, relative, hex_sha256(data), "match"))
            checked.update(files)
            extra = []
            if dist == DIST:
                (root / "bin" / LAUNCHER).write_bytes(LAUNCHER_BYTES)
                launcher_row = f"../../../bin/{LAUNCHER}"
                extra.append(f"{launcher_row},{record_hash(LAUNCHER_BYTES)},{len(LAUNCHER_BYTES)}")
                extra.append(f"{PYC_ROW},,")  # a compiled file: empty hash, and not on disk
                rows.append((package, launcher_row, "", "launcher"))
                rows.append((package, PYC_ROW, "", "MISSING"))
            record = site / package / "RECORD"
            text = _record_text(files, extra, dist)
            record.write_text(text, encoding="utf-8")
            rows.append((package, f"{package}/RECORD", hex_sha256(text.encode("utf-8")), "no-hash"))

        unlisted = []
        for name, data in UNLISTED.items():
            (site / name).write_bytes(data)
            unlisted.append((name, hex_sha256(data)))
        (site / PRUNED).parent.mkdir(parents=True, exist_ok=True)
        (site / PRUNED).write_bytes(b"compiled bytes")
        return Miniature(root=root, home=home, version_info=version_info, interpreters=interpreters,
                         checked=checked, rows=rows, unlisted=sorted(unlisted))

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

    return SimpleNamespace(
        SITE=SITE, DIST=DIST, OTHER_DIST=OTHER_DIST, LAUNCHER=LAUNCHER, FILES=FILES,
        OTHER_FILES=OTHER_FILES, LAUNCHER_BYTES=LAUNCHER_BYTES, PYC_ROW=PYC_ROW,
        VENDOR_MODULE=VENDOR_MODULE, VENDOR_RECORD=VENDOR_RECORD,
        INTERPRETER_FOLDER=INTERPRETER_FOLDER, INTERPRETER_LINK=INTERPRETER_LINK,
        INTERPRETER_BYTES=INTERPRETER_BYTES, UNLISTED=UNLISTED, PRUNED=PRUNED,
        record_hash=record_hash, hex_sha256=hex_sha256, Miniature=Miniature,
        _record_text=_record_text, make_environment=make_environment, tree_digest=tree_digest,
        source_text=source_text, string_constants=string_constants,
        SYSTEM_FOLDERS=SYSTEM_FOLDERS, LOCAL_WORDS=LOCAL_WORDS,
        local_path_constants=local_path_constants)


sbx_kit = _make_sbx_kit()

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "setup_baselines_env.sh"
TOOLS = ("venv-dolma", "venv-ob")

# Every way a test process can start another one. A check of an environment starts none:
# it reads files.
PROCESS_STARTS = (
    (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
    (os, ("system", "popen", "posix_spawn", "posix_spawnp", "execv", "execve", "execvp",
          "execvpe", "execl", "execle", "execlp", "execlpe", "spawnl", "spawnle", "spawnlp",
          "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe")),
)


@contextlib.contextmanager
def _no_process_started():
    """Inside the block every way of starting a process raises ``AssertionError`` and is
    noted in the list this yields. On a normal exit the list must be empty (a start that
    the code under test caught and swallowed shows there)."""
    started: list[str] = []

    def refuse(name):
        def raiser(*args, **kwargs):
            started.append(name)
            raise AssertionError(f"{name} was called: nothing of an environment is started")

        return raiser

    with pytest.MonkeyPatch.context() as patch:
        for owner, names in PROCESS_STARTS:
            for name in names:
                if hasattr(owner, name):
                    patch.setattr(owner, name, refuse(f"{owner.__name__}.{name}"))
        yield started
    assert started == [], f"a process was started: {started}"


def _named(module, name):
    """A name of the module, or a failed assertion that says it is not there."""
    found = getattr(module, name, None)
    assert found is not None, f"{module.__name__} has no {name}"
    return found


def _check(need_module, env_dir):
    module = need_module("baselines.envcheck")
    with _no_process_started():
        return module.check_environment(env_dir)


def _refused(need_module, env_dir):
    module = need_module("baselines.envcheck")
    with _no_process_started():
        with pytest.raises(module.EnvCheckError) as raised:
            module.check_environment(env_dir)
    return str(raised.value)


def test_names_of_the_check(need_attribute):
    assert callable(need_attribute("baselines.envcheck", "check_environment"))
    assert callable(need_attribute("baselines.envcheck", "main"))
    assert issubclass(need_attribute("baselines.envcheck", "EnvCheckError"), Exception)


def test_names_of_the_record_and_the_compare(need_attribute):
    assert callable(need_attribute("baselines.envcheck", "write_record"))
    assert callable(need_attribute("baselines.envcheck", "compare_environments"))


def test_an_environment_whose_files_match_its_record_passes(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    report = _check(need_module, env.root)
    assert report.files_checked == env.checked_count == 3
    assert report.env == "venv-x"


def test_the_interpreter_path_and_version_are_recorded(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", home="/some/interpreter/bin", version_info="3.12")
    report = _check(need_module, env.root)
    assert str(report.interpreter_home) == "/some/interpreter/bin"
    assert report.version_info == "3.12"
    assert report.interpreter_folder == "cpython-3.12.13-linux-x86_64-gnu"
    assert report.interpreter_sha256 == hashlib.sha256(sbx_kit.INTERPRETER_BYTES).hexdigest()
    assert set(report.as_dict()) == {"env", "interpreter_home", "interpreter_folder", "version_info",
                                     "interpreter_sha256", "files_checked"}
    assert not hasattr(report, "python_version")


def test_the_folder_name_is_that_of_the_resolved_interpreter_and_not_of_the_uv_link(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    assert Path(env.home).parent.name == "cpython-3.12-linux-x86_64-gnu", "home names the minor-version link"
    report = _check(need_module, env.root)
    assert report.interpreter_folder == "cpython-3.12.13-linux-x86_64-gnu"
    assert report.version_info == "3.12"


def test_one_file_whose_bytes_differ_is_refused_and_named(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    env.file("fakepkg/core.py").write_bytes(b"def answer():\n    return 43\n")
    text = _refused(need_module, env.root)
    assert "fakepkg/core.py" in text
    assert sbx_kit.DIST in text


def test_a_difference_of_one_byte_in_the_second_distribution_is_refused(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True)
    assert _check(need_module, env.root).files_checked == env.checked_count == 6
    path = env.file("otherpkg/util.py")
    path.write_bytes(path.read_bytes() + b" ")
    text = _refused(need_module, env.root)
    assert "otherpkg/util.py" in text and sbx_kit.OTHER_DIST in text


def test_a_listed_file_that_is_missing_is_refused_and_named(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    env.file("fakepkg/__init__.py").unlink()
    assert "fakepkg/__init__.py" in _refused(need_module, env.root)


def test_a_launcher_row_is_not_compared(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    (env.root / "bin" / sbx_kit.LAUNCHER).write_bytes(b"#!/new/place/venv/bin/python\nprint('tool')\n")
    report = _check(need_module, env.root)
    assert report.files_checked == env.checked_count, "the launcher is not among the rows compared"


def test_the_row_of_the_record_file_and_a_row_without_a_hash_are_not_compared(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    # The RECORD rows hold an empty hash for the file itself and for a compiled file that
    # is not on disk: a clean environment passes, so neither is read as a mismatch.
    record = (env.site / f"{sbx_kit.DIST}.dist-info" / "RECORD").read_text(encoding="utf-8")
    assert f"{sbx_kit.DIST}.dist-info/RECORD,," in record and f"{sbx_kit.PYC_ROW},," in record
    assert not env.file(sbx_kit.PYC_ROW).exists()
    assert _check(need_module, env.root).files_checked == env.checked_count


def test_a_vendored_record_is_not_read_and_its_files_stay_rows_of_the_outer_record(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", vendored=True)
    assert _check(need_module, env.root).files_checked == env.checked_count == 5
    env.file(sbx_kit.VENDOR_MODULE).write_bytes(b"VENDORED = False\n")
    text = _refused(need_module, env.root)
    assert sbx_kit.VENDOR_MODULE in text and sbx_kit.DIST in text


def test_an_interpreter_link_to_nothing_is_refused(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", interpreter_missing=True)
    assert env.python.is_symlink() and not env.python.exists()
    text = _refused(need_module, env.root)
    assert "python" in text


def test_the_interpreter_is_read_from_the_copys_own_link_and_not_from_its_home(need_module, tmp_path):
    # pyvenv.cfg names the bin folder of ANOTHER interpreter, which exists and holds
    # other bytes: the report still describes the one bin/python of the copy resolves to.
    other = tmp_path / "uv-python" / "cpython-3.11.9-linux-x86_64-gnu" / "bin"
    other.mkdir(parents=True)
    (other / "python3.11").write_bytes(b"another stand-in\n")
    env = sbx_kit.make_environment(tmp_path / "venv-x", home=str(other))
    report = _check(need_module, env.root)
    assert report.interpreter_home == str(other)
    assert report.interpreter_folder == "cpython-3.12.13-linux-x86_64-gnu"
    assert report.interpreter_sha256 == hashlib.sha256(sbx_kit.INTERPRETER_BYTES).hexdigest()
    assert report.interpreter_sha256 != hashlib.sha256(b"another stand-in\n").hexdigest()


def test_nothing_is_repaired_or_written_by_a_check(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    before = sbx_kit.tree_digest(env.root)
    _check(need_module, env.root)
    assert sbx_kit.tree_digest(env.root) == before
    env.file("fakepkg/core.py").write_bytes(b"changed\n")
    changed = sbx_kit.tree_digest(env.root)
    _refused(need_module, env.root)
    assert sbx_kit.tree_digest(env.root) == changed


def test_main_writes_the_report_of_a_passing_environment(need_module, tmp_path):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x", home="/some/interpreter/bin",
                                   version_info="3.12")
    out = tmp_path / "report.json"
    with _no_process_started():
        assert module.main([str(env.root), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "env": "venv-x", "interpreter_home": "/some/interpreter/bin",
        "interpreter_folder": "cpython-3.12.13-linux-x86_64-gnu", "version_info": "3.12",
        "interpreter_sha256": env.interpreter_sha256, "files_checked": 3}


def test_main_refuses_a_changed_file_with_a_named_message_and_no_report(need_module, tmp_path, capsys):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    env.file("fakepkg/core.py").write_bytes(b"changed\n")
    out = tmp_path / "report.json"
    with _no_process_started():
        code = module.main([str(env.root), "--out", str(out)])
    assert code != 0
    assert "fakepkg/core.py" in capsys.readouterr().err
    assert not out.exists()


def test_the_module_fetches_nothing_and_holds_no_location(need_module):
    module = need_module("baselines.envcheck")
    text = sbx_kit.source_text(module)
    for word in ("urllib", "socket", "requests", "http.client", "ensurepip", "pip install"):
        assert word not in text
    assert sbx_kit.local_path_constants(module) == []


def test_the_module_holds_no_way_to_start_a_process(need_module):
    module = need_module("baselines.envcheck")
    text = sbx_kit.source_text(module)
    for word in ("subprocess", "Popen", "os.system", "os.popen", "posix_spawn", "os.exec"):
        assert word not in text


# --- The record of what is used ---------------------------------------------------------


def _record(need_module, env_dir, record_dir):
    module = need_module("baselines.envcheck")
    with _no_process_started():
        _named(module, "write_record")(env_dir, record_dir)


def test_the_rows_file_holds_one_row_per_record_entry_in_the_form_of_the_pinned_checker(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True, vendored=True)
    out = tmp_path / "record"
    _record(need_module, env.root, out)
    text = (out / "rows.tsv").read_bytes()
    assert text == env.rows_text().encode("utf-8")
    lines = text.decode("utf-8").splitlines()
    assert lines[0] == "package\tpath\tsha256\tverdict"
    assert {line.split("\t")[3] for line in lines[1:]} == {"match", "launcher", "MISSING", "no-hash"}
    # The RECORD files read are the top-level ones, in sorted order of their paths; the
    # vendored RECORD has no rows of its own (its files are rows of the outer RECORD).
    assert [line.split("\t")[0] for line in lines[1:]] == [row[0] for row in env.rows]
    assert not any(line.startswith("vend-0.1.dist-info\t") for line in lines)
    assert any(line.startswith(f"fakepkg-1.2.dist-info\t{sbx_kit.VENDOR_MODULE}\t") for line in lines)


def test_a_launcher_and_a_row_without_a_hash_are_recorded_with_their_own_words(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    out = tmp_path / "record"
    _record(need_module, env.root, out)
    rows = [line.split("\t") for line in (out / "rows.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    by_path = {row[1]: row for row in rows}
    launcher = by_path[f"../../../bin/{sbx_kit.LAUNCHER}"]
    assert launcher[2:] == ["", "launcher"], "a path that leaves site-packages has no hash and the word launcher"
    record = by_path[f"{sbx_kit.DIST}.dist-info/RECORD"]
    record_bytes = (env.site / f"{sbx_kit.DIST}.dist-info" / "RECORD").read_bytes()
    assert record[2:] == [hashlib.sha256(record_bytes).hexdigest(), "no-hash"]
    assert by_path[sbx_kit.PYC_ROW][2:] == ["", "MISSING"], "a listed file that is not on disk is MISSING"


def test_a_changed_file_and_a_missing_file_are_recorded_with_their_words_and_not_refused(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    changed = b"def answer():\n    return 43\n"
    env.file("fakepkg/core.py").write_bytes(changed)
    env.file("fakepkg/__init__.py").unlink()
    expected = []
    for package, path, sha, verdict in env.rows:
        if path == "fakepkg/core.py":
            sha, verdict = hashlib.sha256(changed).hexdigest(), "MISMATCH"
        elif path == "fakepkg/__init__.py":
            sha, verdict = "", "MISSING"
        expected.append((package, path, sha, verdict))
    out = tmp_path / "record"
    _record(need_module, env.root, out)
    assert (out / "rows.tsv").read_bytes() == env.rows_text(expected).encode("utf-8")


def test_the_files_in_no_record_are_listed_with_their_hashes_and_pycache_is_pruned(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True)
    out = tmp_path / "record"
    _record(need_module, env.root, out)
    text = (out / "unlisted.tsv").read_bytes()
    assert text == env.unlisted_text().encode("utf-8")
    assert [path for path, _ in env.unlisted] == sorted(sbx_kit.UNLISTED)
    assert b"__pycache__" not in text


def test_the_record_folder_must_be_new_and_the_environment_is_never_written(need_module, tmp_path):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    before = sbx_kit.tree_digest(env.root)
    out = tmp_path / "record"
    _record(need_module, env.root, out)
    assert sbx_kit.tree_digest(env.root) == before
    written = sbx_kit.tree_digest(out)
    with _no_process_started():
        with pytest.raises((FileExistsError, module.EnvCheckError)):
            _named(module, "write_record")(env.root, out)
    assert sbx_kit.tree_digest(out) == written, "a record that exists is not overwritten"


def test_the_identity_file_holds_the_sha256_of_the_rows_file_exactly_as_written(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True, vendored=True)
    out = tmp_path / "record"
    _record(need_module, env.root, out)
    identity = json.loads((out / "identity.json").read_text(encoding="utf-8"))
    written = hashlib.sha256((out / "rows.tsv").read_bytes()).hexdigest()
    assert identity.get("rows_sha256") == written
    # The same figure from the rows the test worked out itself, not from the file the
    # code wrote.
    assert identity.get("rows_sha256") == hashlib.sha256(env.rows_text().encode("utf-8")).hexdigest()


# --- The compare of a copy with its original -------------------------------------------


def _copy_of(env, tmp_path):
    copy = tmp_path / "copy" / env.root.name
    copy.parent.mkdir()
    shutil.copytree(env.root, copy, symlinks=True)
    return copy


def _compare(need_module, original, copy):
    module = need_module("baselines.envcheck")
    with _no_process_started():
        return _named(module, "compare_environments")(original, copy)


def _change_an_unlisted_file(copy):
    (copy / sbx_kit.SITE / "_virtualenv.py").write_bytes(b"# changed\n")


def _change_a_launcher(copy):
    (copy / "bin" / sbx_kit.LAUNCHER).write_bytes(b"#!/other/place/venv/bin/python\n")


def _change_the_cfg(copy):
    path = copy / "pyvenv.cfg"
    path.write_text(path.read_text(encoding="utf-8") + "extra = 1\n", encoding="utf-8")


def _retarget_a_link(copy):
    link = copy / "bin" / "python"
    link.unlink()
    os.symlink(str(copy / "bin" / sbx_kit.LAUNCHER), link)


def _drop_a_file(copy):
    (copy / sbx_kit.SITE / "fakepkg" / "core.py").unlink()


def _add_a_file(copy):
    (copy / sbx_kit.SITE / "extra.py").write_bytes(b"x = 1\n")


def _a_file_where_a_link_was(copy):
    link = copy / "bin" / "python"
    link.unlink()
    link.write_bytes(b"not a link\n")


def _a_link_where_a_file_was(copy):
    path = copy / "bin" / sbx_kit.LAUNCHER
    path.unlink()
    os.symlink(str(copy / "pyvenv.cfg"), path)


SITE_TEXT = sbx_kit.SITE.as_posix()
DIFFERENCES = {
    "the bytes of a file in no RECORD": (_change_an_unlisted_file, f"{SITE_TEXT}/_virtualenv.py"),
    "the bytes of a launcher": (_change_a_launcher, f"bin/{sbx_kit.LAUNCHER}"),
    "pyvenv.cfg": (_change_the_cfg, "pyvenv.cfg"),
    "the target of a link": (_retarget_a_link, "bin/python"),
    "a file gone from the copy": (_drop_a_file, f"{SITE_TEXT}/fakepkg/core.py"),
    "a file only in the copy": (_add_a_file, f"{SITE_TEXT}/extra.py"),
    "a file where the original has a link": (_a_file_where_a_link_was, "bin/python"),
    "a link where the original has a file": (_a_link_where_a_file_was, f"bin/{sbx_kit.LAUNCHER}"),
}


def test_a_copy_equal_to_its_original_has_no_difference_and_nothing_is_written(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True, vendored=True)
    copy = _copy_of(env, tmp_path)
    before = (sbx_kit.tree_digest(env.root), sbx_kit.tree_digest(copy))
    assert list(_compare(need_module, env.root, copy)) == []
    assert (sbx_kit.tree_digest(env.root), sbx_kit.tree_digest(copy)) == before


@pytest.mark.parametrize("name", list(DIFFERENCES))
def test_one_difference_between_a_copy_and_its_original_is_listed_with_its_path_and_a_reason(
        need_module, tmp_path, name):
    alter, path = DIFFERENCES[name]
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    copy = _copy_of(env, tmp_path)
    alter(copy)
    found = list(_compare(need_module, env.root, copy))
    assert [item.path for item in found] == [path]
    assert isinstance(found[0].reason, str) and found[0].reason.strip() != ""


def test_a_difference_is_listed_the_same_way_whichever_way_the_two_folders_are_given(need_module, tmp_path):
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    copy = _copy_of(env, tmp_path)
    _change_the_cfg(copy)
    _add_a_file(copy)
    one = sorted(item.path for item in _compare(need_module, env.root, copy))
    other = sorted(item.path for item in _compare(need_module, copy, env.root))
    assert one == other == sorted(["pyvenv.cfg", f"{SITE_TEXT}/extra.py"])


def test_a_folder_that_does_not_exist_is_refused_by_the_compare(need_module, tmp_path):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    with _no_process_started():
        with pytest.raises(module.EnvCheckError):
            _named(module, "compare_environments")(env.root, tmp_path / "no-such-copy")
        with pytest.raises(module.EnvCheckError):
            _named(module, "compare_environments")(tmp_path / "no-such-original", env.root)


# --- The script ----------------------------------------------------------------------


def _script_lines():
    assert SCRIPT.is_file(), f"{SCRIPT.name} does not exist"
    return SCRIPT.read_text(encoding="utf-8").splitlines()


def _commands():
    """The script's lines that are not comments."""
    return [line for line in _script_lines() if line.strip() and not line.lstrip().startswith("#")]


def test_the_script_fetches_and_installs_nothing():
    text = "\n".join(_commands())
    banned = [r"\bpip3?\b", r"\buv\b", r"\bcurl\b", r"\bwget\b", r"\bgit\s+(clone|fetch|pull)\b",
              r"\bcargo\b", r"\brustup\b", r"\bnpm\b", r"\bsudo\b", r"\bapt(-get)?\b", r"\bdnf\b",
              r"https?:", r"\bsetup\.sh\b", r"\bmake\s+setup\b", r"\bensurepip\b"]
    found = [pattern for pattern in banned if re.search(pattern, text)]
    assert found == []


def test_the_script_holds_no_location_and_every_location_is_an_argument():
    text = "\n".join(_commands())
    # An absolute path starts after a space, a quote, a bracket, a colon or an equals sign;
    # "$2/tmp" is a folder under an argument and is no location of the script's own.
    folders = "|".join(("ho" + "me", "ro" + "ot", "op" + "t", "mn" + "t", "tm" + "p", "va" + "r",
                        "us" + "r"))
    absolute = re.compile(r"(?:^|[\s=\"'(:])/(?:" + folders + r")\b", re.MULTILINE)
    assert absolute.findall(text) == []
    assert "Proj" + "ects" not in text
    for number in ("1", "2", "3"):
        assert re.search(r"\$\{?" + number + r"\b", text), f"the script reads its argument {number}"


def test_the_script_runs_the_check_on_each_copy_and_names_both_environments():
    text = "\n".join(_commands())
    assert "baselines.envcheck" in text
    assert "venv-dolma" in text and "venv-ob" in text


def _run_script(tmp_path, earlier, work, path_front=None):
    assert SCRIPT.is_file(), f"{SCRIPT.name} does not exist"
    environment = dict(os.environ)
    if path_front is not None:
        environment["PATH"] = f"{path_front}{os.pathsep}{environment['PATH']}"
    return subprocess.run(["bash", str(SCRIPT), str(earlier), str(work), sys.executable],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120, env=environment)


def _earlier_install(tmp_path):
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    interpreters = tmp_path / "uv-python"
    made = {name: sbx_kit.make_environment(earlier / name, interpreters=interpreters) for name in TOOLS}
    return earlier, made


def test_the_script_copies_both_environments_checks_them_and_writes_nothing_to_the_source(tmp_path):
    earlier, made = _earlier_install(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    source_before = sbx_kit.tree_digest(earlier)
    done = _run_script(tmp_path, earlier, work)
    assert done.returncode == 0, done.stdout + done.stderr
    assert sbx_kit.tree_digest(earlier) == source_before, "nothing is written in the earlier install"
    for name in TOOLS:
        assert sbx_kit.tree_digest(work / name) == sbx_kit.tree_digest(earlier / name)
        report = json.loads((work / f"env-check-{name}.json").read_text(encoding="utf-8"))
        assert report == {
            "env": name, "interpreter_home": made[name].home,
            "interpreter_folder": "cpython-3.12.13-linux-x86_64-gnu",
            "version_info": made[name].version_info,
            "interpreter_sha256": hashlib.sha256(sbx_kit.INTERPRETER_BYTES).hexdigest(),
            "files_checked": made[name].checked_count}


def test_the_script_writes_the_record_of_what_each_copy_uses(tmp_path):
    earlier, made = _earlier_install(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    done = _run_script(tmp_path, earlier, work)
    assert done.returncode == 0, done.stdout + done.stderr
    for name in TOOLS:
        record = work / f"env-record-{name}"
        assert (record / "rows.tsv").read_bytes() == made[name].rows_text().encode("utf-8")
        assert (record / "unlisted.tsv").read_bytes() == made[name].unlisted_text().encode("utf-8")


def test_the_script_stops_the_tool_whose_file_differs_and_still_checks_the_other(tmp_path):
    earlier, made = _earlier_install(tmp_path)
    changed = b"def answer():\n    return 43\n"
    made["venv-dolma"].file("fakepkg/core.py").write_bytes(changed)
    work = tmp_path / "work"
    work.mkdir()
    source_before = sbx_kit.tree_digest(earlier)
    done = _run_script(tmp_path, earlier, work)
    output = done.stdout + done.stderr
    assert done.returncode != 0
    assert "venv-dolma" in output and "fakepkg/core.py" in output
    assert not (work / "env-check-venv-dolma.json").exists()
    assert (work / "env-check-venv-ob.json").is_file()
    assert sbx_kit.tree_digest(earlier) == source_before
    copied = work / "venv-dolma" / sbx_kit.SITE / "fakepkg" / "core.py"
    assert copied.read_bytes() == changed, "the copy is left as it is, not repaired"


def test_the_script_stops_for_an_environment_that_is_not_in_the_earlier_install(tmp_path):
    earlier, made = _earlier_install(tmp_path)
    for path in sorted((earlier / "venv-ob").rglob("*"), reverse=True):
        path.unlink() if path.is_symlink() or path.is_file() else path.rmdir()
    (earlier / "venv-ob").rmdir()
    work = tmp_path / "work"
    work.mkdir()
    done = _run_script(tmp_path, earlier, work)
    assert done.returncode != 0
    assert "venv-ob" in done.stdout + done.stderr
    assert (work / "env-check-venv-dolma.json").is_file()
    assert not (work / "env-check-venv-ob.json").exists()


# A copy that is made by the script differs from its original only if the copy command
# makes it so. These tests put a ``cp`` of their own first on the command path: it makes
# the copy with the real one, then changes ONE thing in the copy of ``venv-dolma``. Each
# change is one that the check of the RECORD rows does not see (a file in no RECORD, the
# configuration, a launcher, a link that still leads to an existing file), so that only
# the compare of the copy with its original can stop the tool.
CHANGES_BY_THE_COPY_COMMAND = {
    "a file in no RECORD": (f'printf x >> "$dest/{SITE_TEXT}/_virtualenv.py"', "_virtualenv.py"),
    "pyvenv.cfg": ('printf "extra = 1\\n" >> "$dest/pyvenv.cfg"', "pyvenv.cfg"),
    "a launcher": ('printf "#!/other/place/venv/bin/python\\n" > "$dest/bin/faketool"', "bin/faketool"),
    "a link": ('ln -sfn "$dest/bin/faketool" "$dest/bin/python"', "bin/python"),
}


def _copy_command_that_changes_one_thing(tmp_path, command):
    folder = tmp_path / "shadow"
    folder.mkdir()
    real = shutil.which("cp")
    assert real, "there is a cp on the command path"
    script = folder / "cp"
    script.write_text(
        "#!/bin/bash\n"
        f'"{real}" "$@" || exit "$?"\n'
        'dest="${@: -1}"\n'
        'case "$dest" in\n'
        f"  */venv-dolma)\n    {command}\n    ;;\n"
        "esac\n", encoding="utf-8")
    os.chmod(script, 0o755)
    return folder


@pytest.mark.parametrize("name", list(CHANGES_BY_THE_COPY_COMMAND))
def test_the_script_stops_the_tool_whose_copy_differs_from_its_original_and_still_handles_the_other(
        tmp_path, name):
    command, path = CHANGES_BY_THE_COPY_COMMAND[name]
    earlier, made = _earlier_install(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    source_before = sbx_kit.tree_digest(earlier)
    shadow = _copy_command_that_changes_one_thing(tmp_path, command)
    done = _run_script(tmp_path, earlier, work, path_front=shadow)
    output = done.stdout + done.stderr
    assert done.returncode != 0, output
    assert "venv-dolma" in output and path in output, output
    assert not (work / "env-check-venv-dolma.json").exists()
    stopped = work / "env-record-venv-dolma"
    assert stopped.is_dir() and sorted(os.listdir(stopped)) == ["compare.json"], \
        "a stopped copy leaves its compare result and nothing else in its record folder"
    result = json.loads((stopped / "compare.json").read_text(encoding="utf-8"))
    assert result["equal"] is False
    listed = [item["path"] for item in result["differences"]]
    assert len(listed) == 1 and listed[0].endswith(path)
    assert sbx_kit.tree_digest(work / "venv-dolma") != sbx_kit.tree_digest(earlier / "venv-dolma"), \
        "the copy is left as it is, not repaired"
    assert (work / "env-check-venv-ob.json").is_file()
    assert (work / "env-record-venv-ob" / "rows.tsv").is_file()
    assert sbx_kit.tree_digest(earlier) == source_before


# --- The version of pyvenv.cfg against the interpreter folder ---------------------------
#
# ``version_info`` of ``pyvenv.cfg`` and the version in the name of the interpreter folder
# that ``bin/python`` resolves into (``cpython-3.12.13-linux-x86_64-gnu`` reads 3.12.13)
# must agree, or the check refuses with a named message. They agree when the folder's
# version, read as its numbers, begins with the numbers of ``version_info``: the pair of
# the real environments, 3.12 beside 3.12.13, agrees (the tests above run on it); 3.11
# beside 3.12.13 does not. The compare is by numbers, not by letters: 3.1 is not the start
# of 3.12.13.

DISAGREEING_VERSIONS = ("3.11", "3.13", "3.1", "3.12.1", "2.12", "3.12.14")


@pytest.mark.parametrize("version_info", DISAGREEING_VERSIONS)
def test_a_version_info_that_the_interpreter_folders_version_does_not_begin_with_is_refused_and_named(
        need_module, tmp_path, version_info):
    env = sbx_kit.make_environment(tmp_path / "venv-x", version_info=version_info)
    text = _refused(need_module, env.root)
    assert "venv-x" in text
    assert version_info in text, "the message names the version pyvenv.cfg gives"
    assert "3.12.13" in text, "the message names the version of the interpreter folder"


def test_the_whole_version_of_the_folder_agrees_and_one_number_more_does_not(need_module, tmp_path):
    agreeing = sbx_kit.make_environment(tmp_path / "venv-a", version_info="3.12.13")
    assert _check(need_module, agreeing.root).version_info == "3.12.13"
    longer = sbx_kit.make_environment(tmp_path / "venv-b", version_info="3.12.13.1")
    assert "3.12.13.1" in _refused(need_module, longer.root)


def test_main_refuses_a_version_that_disagrees_with_the_folder_and_writes_no_report(
        need_module, tmp_path, capsys):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x", version_info="3.11")
    out = tmp_path / "report.json"
    with _no_process_started():
        code = module.main([str(env.root), "--out", str(out)])
    assert code != 0
    error = capsys.readouterr().err
    assert "venv-x" in error and "3.11" in error and "3.12.13" in error
    assert not out.exists()


def test_the_script_stops_the_tool_whose_version_disagrees_with_its_interpreter_folder_and_handles_the_other(
        tmp_path):
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    interpreters = tmp_path / "uv-python"
    sbx_kit.make_environment(earlier / "venv-dolma", interpreters=interpreters, version_info="3.11")
    sbx_kit.make_environment(earlier / "venv-ob", interpreters=interpreters)
    work = tmp_path / "work"
    work.mkdir()
    source_before = sbx_kit.tree_digest(earlier)
    done = _run_script(tmp_path, earlier, work)
    output = done.stdout + done.stderr
    assert done.returncode != 0, output
    assert "venv-dolma" in output and "3.11" in output and "3.12.13" in output, output
    assert not (work / "env-check-venv-dolma.json").exists()
    assert (work / "env-check-venv-ob.json").is_file()
    assert (work / "env-record-venv-ob" / "rows.tsv").is_file()
    assert sbx_kit.tree_digest(earlier) == source_before


# --- The result of the copy compare is a file of the record -----------------------------
#
# The compare of a copy with its original is part of the record of what the copy uses, and
# so is its result when the two are equal: the record folder holds ``compare.json``, a
# JSON object with ``equal`` (true or false) and ``differences`` (a list, empty for an
# equal copy). A line on standard output is not that file.


def test_main_writes_the_result_of_an_equal_compare_as_a_file_of_the_record_folder(
        need_module, tmp_path, capsys):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True)
    copy = _copy_of(env, tmp_path)
    out = tmp_path / "report.json"
    record = tmp_path / "record"
    with _no_process_started():
        code = module.main([str(copy), "--original", str(env.root), "--out", str(out),
                            "--record", str(record)])
    assert code == 0, capsys.readouterr().err
    written = record / "compare.json"
    assert written.is_file(), "the equal result is a file of the record, not a line of output"
    assert json.loads(written.read_text(encoding="utf-8")) == {"equal": True, "differences": []}
    assert (record / "rows.tsv").is_file(), "the record of what the copy uses is still written"


def test_the_script_puts_the_result_of_the_compare_into_the_record_of_each_copy(tmp_path):
    earlier, made = _earlier_install(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    done = _run_script(tmp_path, earlier, work)
    assert done.returncode == 0, done.stdout + done.stderr
    for name in TOOLS:
        written = work / f"env-record-{name}" / "compare.json"
        assert written.is_file(), f"{name}: the result of the compare is a file of its record"
        assert json.loads(written.read_text(encoding="utf-8")) == {"equal": True, "differences": []}
        assert (work / f"env-record-{name}" / "rows.tsv").read_bytes() == made[name].rows_text().encode("utf-8")


# --- A copy that differs leaves its compare result as a file of the record --------------
#
# The same file holds the result of a copy that differs: ``equal`` false and one entry
# for each difference, with its ``path`` and its ``reason``. Nothing else is written into
# the record folder (the run stops before the check), the differences are still printed
# and the exit code stays 1. A record folder that exists stays a refusal with code 2.


@pytest.mark.parametrize("name", list(DIFFERENCES))
def test_main_writes_the_differences_of_a_copy_as_a_file_of_the_record_folder_and_stops(
        need_module, tmp_path, capsys, name):
    module = need_module("baselines.envcheck")
    alter, path = DIFFERENCES[name]
    env = sbx_kit.make_environment(tmp_path / "venv-x", second_distribution=True)
    copy = _copy_of(env, tmp_path)
    alter(copy)
    out = tmp_path / "report.json"
    record = tmp_path / "record"
    with _no_process_started():
        code = module.main([str(copy), "--original", str(env.root), "--out", str(out),
                            "--record", str(record)])
    captured = capsys.readouterr()
    assert code == 1, captured.err
    assert record.is_dir() and sorted(os.listdir(record)) == ["compare.json"]
    result = json.loads((record / "compare.json").read_text(encoding="utf-8"))
    assert result["equal"] is False
    assert [item["path"] for item in result["differences"]] == [path]
    for item in result["differences"]:
        assert set(item) == {"path", "reason"}
        assert isinstance(item["reason"], str) and item["reason"].strip() != ""
    assert path in captured.err, "the difference is still printed"
    assert not out.exists(), "no report is written for a copy that differs"


def test_main_lists_every_difference_of_a_copy_in_the_file(need_module, tmp_path, capsys):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    copy = _copy_of(env, tmp_path)
    _change_the_cfg(copy)
    _add_a_file(copy)
    record = tmp_path / "record"
    with _no_process_started():
        code = module.main([str(copy), "--original", str(env.root), "--record", str(record)])
    assert code == 1, capsys.readouterr().err
    assert (record / "compare.json").is_file()
    result = json.loads((record / "compare.json").read_text(encoding="utf-8"))
    assert result["equal"] is False
    assert sorted(item["path"] for item in result["differences"]) == sorted(
        ["pyvenv.cfg", f"{SITE_TEXT}/extra.py"])


def test_main_refuses_an_existing_record_folder_for_a_copy_that_differs(need_module, tmp_path, capsys):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-x")
    copy = _copy_of(env, tmp_path)
    _change_the_cfg(copy)
    record = tmp_path / "record"
    record.mkdir()
    with _no_process_started():
        code = module.main([str(copy), "--original", str(env.root), "--record", str(record)])
    assert code == 2, capsys.readouterr().err
    assert os.listdir(record) == [], "a record folder that exists is not written into"


# --- The identity writer names the program of the tool ---------------------------------
#
# ``main`` takes ``--program FILE`` beside ``--record``: ``identity.json`` then holds one key
# more, ``ob_binary_sha256``, the sha256 (hex) of that file, and nothing else of the record
# changes. The script takes the program as an optional fourth argument and passes it for
# ``venv-ob`` only. The runner reads the same key at its stage ``binary``. The program of
# these tests is a few bytes that are never started.

PROGRAM_BYTES = b"a stand-in for the program of the tool: bytes only, never started\n"
PROGRAM_KEY = "ob_binary_sha256"


def _main_code(module, arguments):
    """The exit code of ``main`` for these arguments; an option it does not know makes the
    argument parser leave with ``SystemExit``, and that code is returned."""
    try:
        with _no_process_started():
            return module.main([str(word) for word in arguments])
    except SystemExit as leaving:
        return leaving.code


def test_the_writer_puts_the_sha256_of_the_program_into_identity_json_and_changes_nothing_else(
        need_module, tmp_path):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-ob", second_distribution=True)
    program = tmp_path / "ob-program"
    program.write_bytes(PROGRAM_BYTES)
    plain, named = tmp_path / "record-plain", tmp_path / "record-named"
    report = tmp_path / "report.json"
    assert _main_code(module, [env.root, "--record", plain]) == 0
    assert _main_code(module, [env.root, "--record", named, "--program", program, "--out", report]) == 0
    before = json.loads((plain / "identity.json").read_text(encoding="utf-8"))
    after = json.loads((named / "identity.json").read_text(encoding="utf-8"))
    assert PROGRAM_KEY not in before, "a record made with no program names none"
    assert after.get(PROGRAM_KEY) == hashlib.sha256(PROGRAM_BYTES).hexdigest()
    assert {key: value for key, value in after.items() if key != PROGRAM_KEY} == before, \
        "the one key is the only difference between the two identity files"
    for name in ("rows.tsv", "unlisted.tsv"):
        assert (named / name).read_bytes() == (plain / name).read_bytes()
    assert PROGRAM_KEY not in json.loads(report.read_text(encoding="utf-8")), "the report keeps its six names"
    assert program.read_bytes() == PROGRAM_BYTES, "the program is read and never written"


@pytest.mark.parametrize("kind", ["missing", "folder"])
def test_a_program_that_is_no_file_is_refused_by_name_and_no_identity_file_is_written(
        need_module, tmp_path, capsys, kind):
    module = need_module("baselines.envcheck")
    env = sbx_kit.make_environment(tmp_path / "venv-ob")
    program = tmp_path / "ob-program"
    if kind == "folder":
        program.mkdir()
    record = tmp_path / "record"
    code = _main_code(module, [env.root, "--record", record, "--program", program])
    error = capsys.readouterr().err
    assert code == 1, error
    assert "ob-program" in error, "the message names the program"
    assert not (record / "identity.json").exists()


def _run_script_with_program(tmp_path, earlier, work, program):
    assert SCRIPT.is_file(), f"{SCRIPT.name} does not exist"
    return subprocess.run(["bash", str(SCRIPT), str(earlier), str(work), sys.executable, str(program)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120, env=dict(os.environ))


def test_the_script_passes_the_program_for_the_environment_of_the_program_only(tmp_path):
    earlier, made = _earlier_install(tmp_path)
    program = tmp_path / "ob-program"
    program.write_bytes(PROGRAM_BYTES)
    work = tmp_path / "work"
    work.mkdir()
    done = _run_script_with_program(tmp_path, earlier, work, program)
    assert done.returncode == 0, done.stdout + done.stderr
    ob = json.loads((work / "env-record-venv-ob" / "identity.json").read_text(encoding="utf-8"))
    dolma = json.loads((work / "env-record-venv-dolma" / "identity.json").read_text(encoding="utf-8"))
    assert ob.get(PROGRAM_KEY) == hashlib.sha256(PROGRAM_BYTES).hexdigest()
    assert PROGRAM_KEY not in dolma, "the other environment names no program"
    assert set(ob) == set(dolma) | {PROGRAM_KEY}, "the one key is the only difference in what identity.json holds"
    assert program.read_bytes() == PROGRAM_BYTES


def test_an_identity_file_made_by_the_real_writer_passes_the_stage_binary_of_the_runner(need_module, tmp_path):
    envcheck = need_module("baselines.envcheck")
    runner = need_module("baselines.runner")
    original = sbx_kit.make_environment(tmp_path / "earlier" / "venv-ob", interpreters=tmp_path / "uv-python")
    work = tmp_path / "work"
    work.mkdir()
    copy = work / "venv-ob"
    shutil.copytree(original.root, copy, symlinks=True)
    program = tmp_path / "ob-program"
    program.write_bytes(PROGRAM_BYTES)
    record = work / "env-record-venv-ob"
    code = _main_code(envcheck, [copy, "--original", original.root, "--record", record, "--program", program])
    assert code == 0, "the writer gives a record with the program's sha256"
    identity_sha256 = hashlib.sha256((record / "identity.json").read_bytes()).hexdigest()

    class Probe:
        """Stands in for the sandboxed start of the copy's python alone: writes its record."""

        def __init__(self):
            self.argvs = []

        def run(self, argv, timeout=None, **keywords):
            self.argvs.append(list(argv))
            Path(argv[-1]).write_text(f"{original.binary}\n{original.interpreter_sha256}\n", encoding="utf-8")
            return SimpleNamespace(exit_code=0, output="", seconds=0.1)

        def start(self, argv):
            raise AssertionError("no tool command is started")

    def pre_run(run_record):
        probe = Probe()
        gated = runner.GatedStarter(probe)
        run_record.mkdir()
        try:
            found = runner.pre_run(
                "originblame", env_dir=copy, env_record=record, identity_sha256=identity_sha256, work=work,
                lock_prefix=["a-lock-prefix"], starter=gated, run_record=run_record, ob_binary=program)
        except runner.PreRunStop as stop:
            return None, stop, gated
        return found, None, gated

    found, stop, gated = pre_run(tmp_path / "run-record-1")
    assert stop is None, f"the runner stopped at its stage {stop.stage}: {stop.reason}"
    assert found.ob_binary_sha256 == hashlib.sha256(PROGRAM_BYTES).hexdigest()
    assert found.interpreter_sha256 == original.interpreter_sha256
    assert gated.released is True
    # The program changed after the setup: the same record stops the run at the same stage.
    program.write_bytes(PROGRAM_BYTES + b"changed\n")
    found, stop, gated = pre_run(tmp_path / "run-record-2")
    assert found is None and stop is not None and stop.stage == "binary" and stop.reason
    assert gated.released is False
