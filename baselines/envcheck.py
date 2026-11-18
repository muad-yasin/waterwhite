"""The check, the record and the compare of a copied tool environment (plan 5.5 Check 10).

Nothing of an environment is started here, and nothing is written in it: every
function reads files. A copied environment is checked as it will be used: the sha256
of every package file against the row of the environment's own RECORD files, and the
interpreter the copy binds to, read from files only. The version comes from the
``version_info`` of ``pyvenv.cfg`` and from the name of the interpreter folder that
``bin/python`` resolves into; the interpreter binary's sha256 is recorded. Launcher
scripts (rows that lead out of ``site-packages``, into the environment's own ``bin``
folder) are left out of the check: they carry the old location of the environment in
their first line. A row with no hash (the RECORD file itself, a compiled file) is not
compared.

``write_record`` writes the record of what the environment uses: a rows file with one
row per RECORD entry, the files that no RECORD lists, and the interpreter's identity.
It records a mismatch or a missing file, it does not refuse one.

``compare_environments`` lists every difference between a copy and its original, for
every file, link and folder, including what the RECORD check leaves out (launcher
scripts, files in no RECORD, ``pyvenv.cfg``, links). When ``main`` is given both the
original and a record folder, the result of the compare is a file of that folder,
``compare.json`` (``{"equal": true, "differences": []}`` for an equal copy). A copy that
differs is stopped before the check: its differences are printed with their reasons, the
exit code is 1, and the record folder is created and holds ``compare.json`` alone, with
``"equal": false`` and one ``{"path": ..., "reason": ...}`` entry for each difference
(no rows file and no identity file, because the check did not run). A record folder that
already exists is refused with exit code 2 in this case too.

The ``version_info`` of ``pyvenv.cfg`` and the version in the name of the interpreter
folder must agree: the folder's version, read as numbers, begins with the numbers of
``version_info``. Otherwise the check refuses and names both.

A mismatch, a missing file, a copy with no RECORD, or an interpreter link that leads
to no file raises ``EnvCheckError`` with a message that names the distribution
folder and the file or the interpreter at fault. Nothing is repaired, installed or
written in the environment. Run it as ``python -m baselines.envcheck ENV_DIR
[--out REPORT] [--original ORIGINAL] [--record RECORD_DIR] [--program FILE]``.

``--program FILE`` (with ``--record``) adds the one key ``ob_binary_sha256`` to ``identity.json``:
the sha256 of the program of the tool, so that the runner compares the program it is handed
with what the setup saw. A program that is no file is refused by name (exit 1) before any
record folder is made.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


COMPARE_FILE = "compare.json"


class EnvCheckError(Exception):
    """An environment that must not be used, with the reason."""


@dataclass(frozen=True)
class EnvReport:
    env: str
    interpreter_home: str
    interpreter_folder: str
    version_info: str
    interpreter_sha256: str
    files_checked: int

    def as_dict(self) -> dict:
        return {"env": self.env, "interpreter_home": self.interpreter_home,
                "interpreter_folder": self.interpreter_folder,
                "version_info": self.version_info,
                "interpreter_sha256": self.interpreter_sha256,
                "files_checked": self.files_checked}


@dataclass(frozen=True)
class Difference:
    """One entry of a copy that is not the original's: its path (relative to the
    environment folder) and how it differs."""

    path: str
    reason: str


def _config(env_dir: Path) -> dict[str, str]:
    """The ``key = value`` lines of the environment's ``pyvenv.cfg``."""
    path = env_dir / "pyvenv.cfg"
    if not path.is_file():
        raise EnvCheckError(f"{env_dir.name}: pyvenv.cfg is missing")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _hex_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _digest(path: Path) -> str:
    raw = bytes.fromhex(_hex_sha256(path))
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _check_files(env_dir: Path) -> int:
    name = env_dir.name
    env_path = os.path.normpath(env_dir)
    bin_path = os.path.join(env_path, "bin")
    sites = sorted((env_dir / "lib").glob("python*/site-packages"))
    records = [record for site in sites for record in sorted(site.glob("*.dist-info/RECORD"))]
    if not records:
        raise EnvCheckError(f"{name}: no dist-info RECORD file found under lib/python*/site-packages")
    checked = 0
    for record in records:
        site, dist = record.parent.parent, record.parent.name
        with open(record, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows:
            if not row or len(row) < 2 or not row[1]:
                continue
            relative, recorded = row[0], row[1]
            target = os.path.normpath(os.path.join(site, relative))
            if _inside(target, bin_path):
                continue
            if not _inside(target, env_path):
                raise EnvCheckError(f"{name}: {dist}: {relative}: the row leads out of the environment")
            if not recorded.startswith("sha256="):
                raise EnvCheckError(f"{name}: {dist}: {relative}: the row's hash is not a sha256")
            if not os.path.isfile(target):
                raise EnvCheckError(f"{name}: {dist}: {relative}: the listed file is missing")
            if _digest(Path(target)) != recorded:
                raise EnvCheckError(f"{name}: {dist}: {relative}: the file's sha256 differs from its RECORD row")
            checked += 1
    return checked


def _interpreter(env_dir: Path) -> tuple[Path, str]:
    """The binary that ``bin/python`` of the copy resolves to, and the name of the
    folder that holds its ``bin`` folder. Reads links; starts nothing."""
    name = env_dir.name
    link = env_dir / "bin" / "python"
    binary = link.resolve()
    if not binary.is_file():
        raise EnvCheckError(f"{name}: bin/python does not lead to an existing file")
    return binary, binary.parent.parent.name


def _identity(env_dir, files_checked: int = 0) -> EnvReport:
    """The report of an environment, the interpreter read from files (the ``home`` and
    the version of ``pyvenv.cfg``, the folder and the binary ``bin/python`` resolves
    to)."""
    env_dir = Path(env_dir)
    config = _config(env_dir)
    if "home" not in config:
        raise EnvCheckError(f"{env_dir.name}: pyvenv.cfg names no interpreter home")
    binary, folder = _interpreter(env_dir)
    return EnvReport(
        env=env_dir.name,
        interpreter_home=config["home"],
        interpreter_folder=folder,
        version_info=config.get("version_info", config.get("version", "")),
        interpreter_sha256=_hex_sha256(binary),
        files_checked=files_checked,
    )


def _numbers(text: str) -> tuple[int, ...] | None:
    """The numbers of a dotted version (``3.12`` reads ``(3, 12)``), or ``None``."""
    if re.fullmatch(r"\d+(\.\d+)*", text) is None:
        return None
    return tuple(int(part) for part in text.split("."))


def _require_agreeing_versions(report: EnvReport) -> None:
    """Refuse an environment whose ``version_info`` of ``pyvenv.cfg`` and whose
    interpreter folder disagree. They agree when the version in the folder's name, read
    as numbers, begins with the numbers of ``version_info`` (3.12 agrees with 3.12.13;
    3.11, 3.1 and 3.12.14 do not). An empty or non-numeric ``version_info``, or a folder
    name with no version in it, is a disagreement: nothing can be compared."""
    wanted = _numbers(report.version_info)
    found = re.search(r"\d+(?:\.\d+)+", report.interpreter_folder)
    actual = _numbers(found.group(0)) if found else None
    if wanted is None:
        raise EnvCheckError(
            f"{report.env}: the version_info of pyvenv.cfg ({report.version_info!r}) is not a version")
    if actual is None:
        raise EnvCheckError(
            f"{report.env}: the interpreter folder {report.interpreter_folder} names no version "
            f"to compare with version_info {report.version_info} of pyvenv.cfg")
    if actual[: len(wanted)] != wanted:
        raise EnvCheckError(
            f"{report.env}: version_info {report.version_info} of pyvenv.cfg disagrees with the "
            f"interpreter folder {report.interpreter_folder}, which is version {found.group(0)}")


def check_environment(env_dir) -> EnvReport:
    """Check one copied environment; return its report or raise ``EnvCheckError``.

    Refused as well: a ``version_info`` of ``pyvenv.cfg`` that the version of the
    interpreter folder does not begin with (see ``_require_agreeing_versions``)."""
    env_dir = Path(env_dir)
    if not env_dir.is_dir():
        raise EnvCheckError(f"{env_dir.name}: the environment folder does not exist")
    report = _identity(env_dir, _check_files(env_dir))
    _require_agreeing_versions(report)
    return report


PROGRAM_KEY = "ob_binary_sha256"


def program_sha256(program) -> str:
    """The sha256 (hex) of the program file a record names. ``EnvCheckError`` naming the
    file where it is no file or cannot be read. The file is only read."""
    path = Path(program)
    if not path.is_file():
        raise EnvCheckError(f"{path.name}: the program is not a file")
    try:
        return _hex_sha256(path)
    except OSError as error:
        raise EnvCheckError(f"{path.name}: the program cannot be read: {error.strerror or error}") from error


def write_record(env_dir, record_dir, program=None) -> None:
    """Write the record of what the environment uses into the NEW folder ``record_dir``.

    ``rows.tsv``: a header ``package, path, sha256, verdict`` (tab separated), then one
    row per entry of every top-level ``dist-info/RECORD`` under the one
    ``site-packages``, RECORD files in the order of their sorted paths, entries in file
    order. ``sha256`` is the hex digest computed from the file, empty for a launcher
    and for a missing file. The verdict is ``launcher`` (the entry leads outside
    ``site-packages``), ``MISSING`` (no such file), ``no-hash`` (the RECORD gives no
    hash), ``match`` or ``MISMATCH``. ``unlisted.tsv``: ``path`` and ``sha256`` of each
    file under ``site-packages`` that no RECORD lists (``__pycache__`` left out),
    sorted by path. ``identity.json``: the interpreter's identity.

    ``identity.json`` also holds ``rows_sha256``, the sha256 of the rows file exactly
    as written. Given a ``program`` (a file), it holds one key more, ``ob_binary_sha256``,
    the sha256 of that file, hashed before the record folder is made; nothing else of the
    record changes.

    A mismatch or a missing file is recorded, not refused. The environment is only
    read; ``record_dir`` must not exist (``FileExistsError``)."""
    env = Path(env_dir).resolve()
    if not env.is_dir():
        raise EnvCheckError(f"{env.name}: the environment folder does not exist")
    sites = sorted(env.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        raise EnvCheckError(f"{env.name}: expected one site-packages folder, found {len(sites)}")
    site = sites[0].resolve()
    records = sorted(site.glob("*.dist-info/RECORD"))
    if not records:
        raise EnvCheckError(f"{env.name}: no dist-info RECORD file found under site-packages")
    report = _identity(env)
    named = program_sha256(program) if program is not None else None
    Path(record_dir).mkdir(parents=False, exist_ok=False)
    listed, rows = set(), []
    for record in records:
        package = record.parent.name
        with record.open(newline="", encoding="utf-8") as handle:
            for entry in csv.reader(handle):
                if not entry:
                    continue
                relative = entry[0]
                recorded = entry[1] if len(entry) > 1 else ""
                target = (site / relative).resolve()
                if not target.is_relative_to(site):
                    verdict, hexsum = "launcher", ""
                elif not target.is_file():
                    verdict, hexsum = "MISSING", ""
                else:
                    listed.add(target)
                    hexsum = _hex_sha256(target)
                    if not recorded:
                        verdict = "no-hash"
                    else:
                        algorithm, _, value = recorded.partition("=")
                        wanted = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
                        verdict = "match" if algorithm == "sha256" and wanted.hex() == hexsum else "MISMATCH"
                rows.append((package, relative, hexsum, verdict))
    unlisted = sorted(
        (str(path.relative_to(site)), path)
        for path in site.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(site).parts
        and path.resolve() not in listed)
    with open(Path(record_dir) / "rows.tsv", "w", encoding="utf-8") as handle:
        handle.write("package\tpath\tsha256\tverdict\n")
        handle.writelines("\t".join(row) + "\n" for row in rows)
    with open(Path(record_dir) / "unlisted.tsv", "w", encoding="utf-8") as handle:
        handle.write("path\tsha256\n")
        handle.writelines(f"{relative}\t{_hex_sha256(path)}\n" for relative, path in unlisted)
    identity = dict(report.as_dict())
    del identity["files_checked"]
    identity["verdicts"] = dict(sorted(Counter(row[3] for row in rows).items()))
    identity["rows_sha256"] = _hex_sha256(Path(record_dir) / "rows.tsv")
    if named is not None:
        identity[PROGRAM_KEY] = named
    with open(Path(record_dir) / "identity.json", "w", encoding="utf-8") as handle:
        json.dump(identity, handle, indent=2)
        handle.write("\n")


def _entries(root: Path) -> dict[str, tuple]:
    """Every entry below ``root``, links not followed: relative path -> (kind, detail,
    permission bits). A file's detail is its sha256, a link's the text it holds."""
    found: dict[str, tuple] = {}

    def walk(folder: Path, prefix: str) -> None:
        for entry in sorted(os.scandir(folder), key=lambda item: item.name):
            relative = prefix + entry.name
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                found[relative] = ("link", os.readlink(entry.path), None)
            elif stat.S_ISDIR(mode):
                found[relative] = ("folder", "", stat.S_IMODE(mode))
                walk(Path(entry.path), relative + "/")
            elif stat.S_ISREG(mode):
                found[relative] = ("file", _hex_sha256(Path(entry.path)), stat.S_IMODE(mode))
            else:
                found[relative] = ("other", "", stat.S_IMODE(mode))

    walk(root, "")
    return found


def compare_environments(original, copy) -> list[Difference]:
    """Every difference between a copy and its original, one item for each file, link
    or folder that differs; an empty list for an equal copy.

    Compared: whether the entry is there, what kind it is (file, link, folder), a
    file's bytes, a link's text, and the permission bits of a file or folder. Times
    are not compared. A folder that is only in one of the two lists its contents too.
    Nothing is started, nothing is written."""
    original, copy = Path(original), Path(copy)
    for folder in (original, copy):
        if not folder.is_dir():
            raise EnvCheckError(f"{folder.name}: the environment folder does not exist")
    before, after = _entries(original), _entries(copy)
    found: list[Difference] = []
    for relative in sorted(set(before) | set(after)):
        if relative not in after:
            found.append(Difference(relative, "only in the original, gone from the copy"))
        elif relative not in before:
            found.append(Difference(relative, "only in the copy, not in the original"))
        else:
            one, other = before[relative], after[relative]
            if one[0] != other[0]:
                found.append(Difference(
                    relative, f"a {other[0]} in the copy where the original has a {one[0]}"))
            elif one[0] == "link" and one[1] != other[1]:
                found.append(Difference(
                    relative, f"the link leads to {other[1]} in the copy and to {one[1]} in the original"))
            elif one[0] == "file" and one[1] != other[1]:
                found.append(Difference(relative, "the bytes differ"))
            elif one[2] != other[2]:
                found.append(Difference(relative, "the permission bits differ"))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="baselines.envcheck", description=__doc__.split("\n")[0])
    parser.add_argument("env_dir", type=Path)
    parser.add_argument("--out", type=Path, help="a new file for the report (JSON)")
    parser.add_argument("--original", type=Path,
                        help="the environment the folder was copied from: every difference is "
                             "listed and stops the run before the check")
    parser.add_argument("--record", type=Path,
                        help="a new folder for the record of what the environment uses")
    parser.add_argument("--program", type=Path,
                        help="the program of the tool: its sha256 goes into identity.json of the record "
                             "as ob_binary_sha256 (the record folder is needed)")
    arguments = parser.parse_args(argv)
    name = arguments.env_dir.name
    if arguments.program is not None:
        if arguments.record is None:
            print(f"{name}: --program names a file for identity.json, which needs --record", file=sys.stderr)
            return 2
        try:
            program_sha256(arguments.program)
        except EnvCheckError as refusal:
            print(str(refusal), file=sys.stderr)
            return 1
    try:
        if arguments.original is not None:
            differences = compare_environments(arguments.original, arguments.env_dir)
            if differences:
                for difference in differences:
                    print(f"{name}: {difference.path}: {difference.reason}", file=sys.stderr)
                if arguments.record is not None:
                    try:
                        arguments.record.mkdir(parents=False, exist_ok=False)
                        with open(arguments.record / COMPARE_FILE, "x", encoding="utf-8") as handle:
                            json.dump({"equal": False,
                                       "differences": [{"path": difference.path, "reason": difference.reason}
                                                       for difference in differences]},
                                      handle, indent=2)
                            handle.write("\n")
                    except FileExistsError:
                        print(f"{name}: the record folder {arguments.record.name} already exists",
                              file=sys.stderr)
                        return 2
                return 1
        report = check_environment(arguments.env_dir)
    except EnvCheckError as refusal:
        print(str(refusal), file=sys.stderr)
        return 1
    if arguments.out is not None:
        try:
            with open(arguments.out, "x", encoding="utf-8") as handle:
                json.dump(report.as_dict(), handle, indent=2)
                handle.write("\n")
        except FileExistsError:
            print(f"{name}: the report file {arguments.out.name} already exists", file=sys.stderr)
            return 2
    if arguments.record is not None:
        try:
            write_record(arguments.env_dir, arguments.record, program=arguments.program)
            if arguments.original is not None:
                # Reaching here means the compare found nothing: the result of it is a
                # file of the record, an equal result included.
                with open(arguments.record / COMPARE_FILE, "x", encoding="utf-8") as handle:
                    json.dump({"equal": True, "differences": []}, handle, indent=2)
                    handle.write("\n")
        except FileExistsError:
            print(f"{name}: the record folder {arguments.record.name} already exists", file=sys.stderr)
            return 2
        except EnvCheckError as refusal:
            print(str(refusal), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
