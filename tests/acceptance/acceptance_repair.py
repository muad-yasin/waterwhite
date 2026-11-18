"""Driving a repair through the documented command lines, and the helpers that need
no tool (Checks 6, 7 and 9).

A repair is three commands, in this order (docs/interfaces.md 7.1, 9.5, 7.3):

1. ``waterwhite report withdraw A --store HISTORICAL``, its output kept as a file;
2. ``python -m replica repair --report FILE --historical DIR --out DIR --store COPY ...``,
   run on a COPY of the store's folder, so that the stores the other rows read never
   hold a successor (the listings of Check 5 must not meet one);
3. ``waterwhite verify --store COPY --report FILE --sides FILE``.

Nothing here imports a module of the tool or reads the replica's Python names: the
replica is run as a command, and what it printed is read as JSON. A wrong report (the
states ``r.descendant-kept`` and ``r.unrelated-paraphrase-retired``) is the report file
with one entry's ``action`` changed before step 2 (``edited_report``).

The expected values never come from here: the helpers name where a record is, never
what its action must be.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import acceptance_runs
import acceptance_sets
from table import Mismatch, TableError

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
REPLICA_TIMEOUT = 3600.0  # seconds, for one replica command
CLI_TIMEOUT = 3600.0      # seconds, for one command of the tool over a set


class RepairError(Exception):
    """A step of a repair did not run: the infrastructure's error, never a skip and never a
    pass."""


@dataclass(frozen=True)
class Repaired:
    """One repaired side: the copy of the store the successors are registered in, the report
    the replica was driven by (a file), the sides file the replica wrote, the folder it built
    into and what it printed."""

    store: Path
    report: Path
    sides: Path
    out: Path
    summary: Mapping


def replica_command(arguments: Sequence[str | Path], timeout: float = REPLICA_TIMEOUT) -> dict:
    """Run ``python -m replica ...`` and read the JSON summary it printed."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(REPO)])}
    root = acceptance_sets.megatron_root()
    if root is not None:
        env.setdefault("HF_HOME", str(root / "hf_home"))
        env.setdefault("HF_HUB_OFFLINE", "1")
    done = subprocess.run([sys.executable, "-m", "replica", *map(str, arguments)], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RepairError(f"replica {arguments[0]} ended with exit code {done.returncode}: "
                          f"{done.stderr[-600:]}")
    try:
        return json.loads(done.stdout)
    except ValueError:
        raise RepairError(f"replica {arguments[0]} printed no JSON summary: "
                          f"{done.stdout[:300]!r}") from None


def tool_command(store: Path, arguments: Sequence[str | Path], timeout: float = CLI_TIMEOUT):
    """One command of the tool in a subprocess, ``--store`` last: (exit code, stdout, stderr)."""
    done = subprocess.run(
        [sys.executable, "-m", "waterwhite.cli", *map(str, arguments), "--store", str(store)],
        capture_output=True, text=True, timeout=timeout, cwd=store.parent,
        env={**os.environ, "PYTHONPATH": str(SRC)})
    return done.returncode, done.stdout, done.stderr


# --- Which records a row names ---------------------------------------------------------------

_RANGE = re.compile(r"(?P<prefix>[A-Za-z-]*?)(?P<first>\d+)\.\.(?P=prefix)(?P<last>\d+)")


def handles_of(given: Mapping) -> list[str]:
    """The handles of a row: ``record`` is one, ``records`` a list or a range written
    ``U-001..U-020`` (the table's README section 5)."""
    if "record" in given:
        return [given["record"]]
    named = given.get("records")
    if isinstance(named, list):
        return list(named)
    if isinstance(named, str):
        match = _RANGE.fullmatch(named)
        if match is None:
            return [named]
        width = len(match["first"])
        return [f"{match['prefix']}{n:0{width}d}"
                for n in range(int(match["first"]), int(match["last"]) + 1)]
    raise TableError("a row that names records names one record or a list or a range")


# --- A wrong report --------------------------------------------------------------------------


def edited_report(report: Mapping, artifact: str, record: str, was: str, now: str) -> dict:
    """The report with the ``action`` of ONE record entry changed from ``was`` to ``now`` (the
    table's README section 7: the first planted translation from retire to keep, the first
    unrelated paraphrase from keep to retire). The entry must exist and say ``was``: a report
    that does not hold it is an error of the run, so that a wrong report that was never made
    cannot pass as one."""
    edited = copy.deepcopy(dict(report))
    found = [e for e in edited.get("records", [])
             if e.get("artifact") == artifact and e.get("record") == record]
    if len(found) != 1 or found[0].get("action") != was:
        raise RepairError(f"the report holds {len(found)} entries for {artifact}:{record}"
                          f"{'' if len(found) != 1 else ' with the action ' + repr(found[0].get('action'))}"
                          f"; the state needs exactly one, whose action is {was!r}")
    found[0]["action"] = now
    return edited


def manifest_line(folder: Path, relative: str, handle: str) -> int:
    """The line of the record whose manifest handle is ``handle`` in the record file
    ``relative`` under a build folder (the replica's manifest, never the tool's)."""
    path = folder / f"{relative}.manifest.jsonl"
    lines = [row["line"] for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
             if row.get("handle") == handle]
    if len(lines) != 1:
        raise RepairError(f"the manifest {path} holds {handle!r} on {len(lines)} lines")
    return lines[0]


# --- What the sides hold, read from their files ----------------------------------------------


def sides_files(sides: Mapping, base: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """(repaired, reference): artifact name -> the path its side names, relative to the folder
    of the sides file."""
    def take(key: str) -> dict[str, Path]:
        held = sides.get(key)
        if not isinstance(held, Mapping):
            raise TableError(f"the sides file holds no {key!r} object")
        return {name: (Path(path) if Path(path).is_absolute() else base / path)
                for name, path in held.items()}

    return take("repaired"), take("reference")


def jsonl_texts(path: Path, field: str = "text") -> list[str]:
    """The text of every line of a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[field] for line in f if line.strip()]


def shard_digest(prefix: Path) -> str:
    """The SHA-256 over a shard's two files (``.bin`` then ``.idx``): this folder's own
    reading of "the same bytes", never the tool's digest."""
    digest = hashlib.sha256()
    for suffix in (".bin", ".idx"):
        with open(f"{prefix}{suffix}", "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def is_jsonl(path: Path) -> bool:
    return path.suffix == ".jsonl"


NO_TEXT = "no text"   # the one key of the lines that hold no text; a fingerprint is 64 hex digits


def own_fingerprints(path: Path) -> Counter:
    """How many lines of a JSONL file hold each fingerprint, read by this folder alone: the
    SHA-256 of a line's text as UTF-8 (never the tool's normaliser). A non-blank line that
    holds no text (it is not JSON, it is JSON but not an object, or it is an object whose
    ``text`` is not a string) is counted under ``NO_TEXT``, all such lines together, so that a
    difference in their number is a difference of the sides. Such a line never raises and is
    never read as an empty text; a blank line is skipped."""
    held: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            text = row.get("text") if isinstance(row, dict) else None
            held[hashlib.sha256(text.encode("utf-8")).hexdigest()
                 if isinstance(text, str) else NO_TEXT] += 1
    return held


def own_multiset_gap(repaired: Mapping[str, Path], reference: Mapping[str, Path]) -> tuple[int, int]:
    """(files compared, records the two sides differ by) over the JSONL files named on both
    sides, by this folder's own fingerprint (``own_fingerprints``). The multiplicity of every
    fingerprint is compared, and the number of the lines that hold no text with them."""
    compared = gap = 0
    for name in sorted(set(repaired) & set(reference)):
        if not (is_jsonl(repaired[name]) and is_jsonl(reference[name])):
            continue
        compared += 1
        left, right = own_fingerprints(repaired[name]), own_fingerprints(reference[name])
        gap += sum(abs(left[k] - right[k]) for k in left.keys() | right.keys())
    return compared, gap


def own_shard_gap(repaired: Mapping[str, Path], reference: Mapping[str, Path]) -> tuple[int, list[str]]:
    """(shards compared, the shards whose bytes differ) over the shards named on both sides."""
    names = [n for n in sorted(set(repaired) & set(reference)) if not is_jsonl(repaired[n])]
    differ = [n for n in names if shard_digest(repaired[n]) != shard_digest(reference[n])]
    return len(names), differ


def shards_folder(out: Path, layout: str) -> Path:
    """Where a build folder holds the shard files of a layout."""
    return Path(out) / "pipeline" / layout / "shards"


def shard_stems(folder: Path) -> list[str]:
    """The shard names the files of a shards folder belong to (``shard-B`` for ``shard-B.bin``);
    none where the folder does not exist."""
    if not folder.is_dir():
        return []
    return sorted({p.name.split(".", 1)[0] for p in folder.iterdir() if p.is_file()})


def files_of_shard(folder: Path, shard: str) -> list[str]:
    """The files named ``<shard>`` or ``<shard>.<suffix>`` in a shards folder."""
    return sorted(p.name for p in folder.glob(f"{shard}*") if p.name.split(".", 1)[0] == shard)


def reference_folder(folder: Path, state: str) -> Path:
    """Where the reference side of a repair is built, beside the repaired side's folder."""
    return folder / f"reference-{state}"


def own_folder_shard_gap(repaired: Path, reference: Path, stems: Iterable[str]) -> tuple[int, list[str]]:
    """(shards compared, the shard names that differ) for shards found in the shards folders of
    the two builds. A shard that both builds hold whole (``.bin`` and ``.idx``) is compared by
    ``shard_digest``. One that a build lacks, or holds in part, is a difference: it is never an
    error and never skipped."""
    compared = 0
    differ: list[str] = []
    for stem in stems:
        whole = [all((folder / f"{stem}{suffix}").is_file() for suffix in (".bin", ".idx"))
                 for folder in (repaired, reference)]
        if not all(whole):
            differ.append(stem)
            continue
        compared += 1
        if shard_digest(repaired / stem) != shard_digest(reference / stem):
            differ.append(stem)
    return compared, differ


def shard_prefix(names: Iterable[str]) -> str:
    """The name prefix a layout's shards share (``set-rr/``), for ``report --layout NAME=PREFIX``."""
    folders = {n.rsplit("/", 1)[0] for n in names if "/" in n}
    if len(folders) != 1:
        raise RepairError(f"the shards of one layout share one folder name, these have {sorted(folders)}")
    return f"{folders.pop()}/"


# --- Where the partitioned layout places the records ------------------------------------------
#
# A placement is the file ``report --placement NAME=FILE`` reads (docs/cli.md "report"): the
# shard each source's copies land in, and the shard for the rest. For the partitioned layout
# of a generated set the documented rule (docs/replica.md, the table "What set.toml does not
# say") gives it: one shard ``shard-<NAME>`` per source file, and ``shard-gen`` for the
# generation outputs. The file is written from that rule and the set's own ``set.toml``,
# never from a report.

PLACEMENT_FORM = "waterwhite.placement/1"
PARTITIONED = "partitioned"
PLACEMENT_FILE = "placement.json"


def placement_for(set_folder, prefix: str) -> dict:
    """The placement document of the partitioned layout of a set whose shards are
    registered as ``<prefix>shard-<NAME>``: ``src/<NAME>`` lands in ``<prefix>shard-<NAME>``
    for every source of ``set.toml``, and everything else in ``<prefix>shard-gen``."""
    return {
        "form": PLACEMENT_FORM,
        "by": "source",
        "shards": {f"src/{name}": f"{prefix}shard-{name}"
                   for name in acceptance_sets.source_names(set_folder)},
        "otherwise": f"{prefix}{acceptance_sets.GENERATION_SHARD}",
    }


def placement_path(folder: Path) -> Path:
    """Where the helper writes the placement file of a layout's repair folder."""
    return Path(folder) / PLACEMENT_FILE


def write_placement(folder: Path, set_folder, prefix: str) -> Path:
    path = placement_path(folder)
    path.write_text(json.dumps(placement_for(set_folder, prefix), indent=2) + "\n", encoding="utf-8")
    return path


def returning_copies(plant, releases: Iterable[str] | None = None) -> dict[str, list[str]]:
    """The records the report promotes or re-sources onto, as ``src/<NAME>:L<n>``, by the
    source file they are a line of: for each release, the copy the plant list says a
    re-sourced row takes (``resource_to``), which is also the dedup loser the report
    promotes. Read from the plant list (``fixture/plant/count.py``), never from a report."""
    resource = acceptance_sets.load_script("count.py").RE_SOURCE
    found: dict[str, set[str]] = {}
    for name in (plant.release_inputs if releases is None else releases):
        for record in plant.release_inputs[name]:
            outcome = plant.expected[name][record]
            if outcome["withdraw_A"] != resource:
                continue
            row = plant.by_id[outcome["resource_to"]]
            if row["source"] is None:
                raise RepairError(f"the copy {row['id']} is no line of a source file; the "
                                  "documented placement names a shard for a source's copies")
            found.setdefault(row["source"], set()).add(f"src/{row['source']}:L{row['line']}")
    return {source: sorted(copies) for source, copies in sorted(found.items())}


def expected_landing(plant, prefix: str, held: Iterable[str],
                     releases: Iterable[str] | None = None) -> list[dict]:
    """``groups.placements[NAME].landing`` the documented layout rule gives: for each shard
    ``<prefix>shard-<SOURCE>`` the store does not hold (``held`` are the store names of the
    shards it does), the copies that are lines of ``SOURCE``."""
    held = set(held)
    return [{"shard": f"{prefix}shard-{source}", "in_store": False, "receives": copies}
            for source, copies in returning_copies(plant, releases).items()
            if f"{prefix}shard-{source}" not in held]


def check_landing(expected: Sequence[Mapping], got: object) -> list[Mismatch]:
    """Where the returning copies land in a shard the store does not hold, held to the
    documented layout rule (docs/replica.md: a partitioned layout has one shard per source
    file) and never to the report: each shard of ``expected`` (the shards the layout names
    that the store does not hold, with the copies that are lines of its source) is named in
    the report's ``landing`` with ``in_store`` false and exactly those copies, and the report
    names no other shard there. ``expected`` must not be empty: the 10k set has a shard of
    B's file the historical build never wrote, and an empty list would test nothing."""
    if not expected:
        return [Mismatch("landing", ">= 1 shard the store does not hold", 0)]
    if not isinstance(got, list):
        return [Mismatch("landing", "a list", got)]
    held = {e.get("shard"): e for e in got if isinstance(e, Mapping)}
    found = []
    for want in expected:
        have = held.pop(want["shard"], None)
        if have is None:
            found.append(Mismatch(f"landing of {want['shard']}", want, None))
        elif have.get("in_store") is not False or sorted(have.get("receives") or []) != want["receives"]:
            found.append(Mismatch(f"landing of {want['shard']}", want, have))
    found += [Mismatch(f"landing of {name}", None, extra) for name, extra in held.items()]
    return found


def put_landing(repairs, size: str, layout: str = "partitioned") -> list[Mismatch]:
    """The landing of the generated set's partitioned run: ``repairs`` is
    ``SetRepairs`` (below)."""
    return check_landing(*repairs.landing(size, layout))


# --- The 10k set, in each layout --------------------------------------------------------------

STATE_UNMUTATED = "unmutated"
STATE_DESCENDANT_KEPT = "r.descendant-kept"
STATE_UNRELATED_RETIRED = "r.unrelated-paraphrase-retired"
FIRST_TRANSLATION = "T-000001"      # the first planted translation (README section 7)
FIRST_UNRELATED = "UP-000001"       # the first generated record that stays (README section 7)
RELEASE = "R1"
RELEASE_FILE = "releases/R1.jsonl"


@dataclass(frozen=True)
class Historical:
    """The generated set built with capture on, in one layout, and its report."""

    layout: str
    store: Path
    out: Path
    prefix: str          # the name prefix of its shards, for `report --layout`
    shards: tuple[str, ...]   # store names of its shards
    counted: Mapping     # the counting script's count of its chunk files
    report: Mapping


@dataclass(frozen=True)
class SetRepaired:
    """One repaired side of the set in one layout, for one state, and its verification."""

    historical: Historical
    repair: Repaired
    report: Mapping      # the report the replica was driven by (edited, in a wrong-report state)
    repaired_files: Mapping[str, Path]
    reference_files: Mapping[str, Path]


class SetRepairs:
    """Builds and repairs the generated sets of one run of the tests, once each.

    The historical build and the replica's repair run on the documented command lines
    (docs/interfaces.md 9.5); the tool prints the report and the verification. Nothing here
    is asked of a store the other rows read: each repair works on a COPY of the store file."""

    def __init__(self, sets: acceptance_sets.Sets) -> None:
        self.sets = sets
        self._historical: dict[tuple[str, str], Historical] = {}
        self._repaired: dict[tuple[str, str, str], SetRepaired] = {}
        self._verified: dict[tuple[str, str, str, str | None], Mapping] = {}
        self._clean: dict[str, Path] = {}
        self._plant: dict[str, list[dict]] = {}

    def plant_rows(self, size: str) -> list[dict]:
        if size not in self._plant:
            self._plant[size] = acceptance_sets.load_script("count.py").load_set(
                self.sets.set_folder(size))[1]
        return self._plant[size]

    def landing(self, size: str, layout: str = PARTITIONED) -> tuple[list[dict], object]:
        """(what the documented layout rule says lands in the shards the store does not hold,
        what the report says): the first from the plant list and the shards the historical
        build registered (never from the report), the second ``groups.placements[LAYOUT]
        .landing`` of the report the historical build was asked for."""
        hist = self.historical(size, layout)
        counting = acceptance_sets.load_script("count.py")
        settings, rows = counting.load_set(self.sets.set_folder(size))
        expected = expected_landing(counting.Plant(settings, rows), hist.prefix, hist.shards)
        groups = hist.report.get("groups")
        placements = groups.get("placements") if isinstance(groups, Mapping) else None
        held = placements.get(layout) if isinstance(placements, Mapping) else None
        return expected, (held.get("landing") if isinstance(held, Mapping) else None)

    def clean_folder(self, size: str) -> Path:
        """The set as it would be had A never existed (docs/plant.md 4.4)."""
        if size not in self._clean:
            folder = self.sets.base / size / "clean"
            acceptance_sets.load_script("generate.py").generate(
                folder, dedup=acceptance_sets.DEDUP[size], clean=True,
                **acceptance_sets.SIZES[size])
            self._clean[size] = folder
        return self._clean[size]

    def historical(self, size: str, layout: str) -> Historical:
        key = (size, layout)
        if key not in self._historical:
            acceptance_sets.need_megatron()
            folder = self.sets.base / size / f"repair-{layout}"
            store, out = folder / "store.sqlite", folder / "build"
            summary = replica_command([
                "build-set", "--set", self.sets.set_folder(size), "--out", out, "--layout", layout,
                "--tokenizer-spec", acceptance_sets.SPEC, "--capture", "--store", store,
                "--jobs", acceptance_sets.JOBS, "--generation-harness", "confined"])
            shards = tuple(acceptance_sets.layout_shards(
                summary["instances"], layout, self.sets.set_folder(size)))
            if not shards:
                raise RepairError(f"the historical build of the {size} set in {layout} registered "
                                  "no shard")
            prefix = shard_prefix(shards)
            counting = acceptance_sets.load_script("count.py")
            pipeline = out / "pipeline" / layout
            counted = counting.count(self.sets.set_folder(size), chunks=pipeline / "tokenizer-input",
                                     layout=layout, sanitizer_input=pipeline / "sanitizer-input")
            # The partitioned layout's placement is stated from the documented rule; the
            # round-robin layout places no record by its source, and states none.
            stated = [] if layout != PARTITIONED else [
                "--placement",
                f"{layout}={write_placement(folder, self.sets.set_folder(size), prefix)}"]
            code, stdout, stderr = tool_command(
                store, ["report", "withdraw", "A", "--layout", f"{layout}={prefix}", *stated])
            if code != 0:
                raise RepairError(f"report ended with exit code {code}: {stderr[:300]}")
            self._historical[key] = Historical(layout, store, out, prefix, shards, counted,
                                               json.loads(stdout))
        return self._historical[key]

    def repaired(self, size: str, layout: str, state: str = STATE_UNMUTATED) -> SetRepaired:
        key = (size, layout, state)
        if key not in self._repaired:
            hist = self.historical(size, layout)
            report = dict(hist.report)
            if state == STATE_DESCENDANT_KEPT:
                line = manifest_line(hist.out, RELEASE_FILE, FIRST_TRANSLATION)
                report = edited_report(report, RELEASE, f"{RELEASE}:L{line}", "retire", "keep")
            elif state == STATE_UNRELATED_RETIRED:
                line = manifest_line(hist.out, RELEASE_FILE, FIRST_UNRELATED)
                report = edited_report(report, RELEASE, f"{RELEASE}:L{line}", "keep", "retire")
            elif state != STATE_UNMUTATED:
                raise TableError(f"the state {state!r} is no state of a repair of a set")
            # A store's paths are relative to its folder, so the COPY the repair registers its
            # successors in stays beside the historical store (SQLite's own backup, as the run
            # world does), and what the repair builds goes into the same folder.
            folder = hist.store.parent
            report_file, sides = folder / f"report-{state}.json", folder / f"sides-{state}.json"
            report_file.write_text(json.dumps(report), encoding="utf-8")
            store = acceptance_runs.snapshot_store(hist.store, folder / f"store-{state}.sqlite")
            summary = replica_command([
                "repair", "--report", report_file, "--historical", hist.out, "--out",
                folder / f"repaired-{state}", "--store", store, "--set",
                self.sets.set_folder(size), "--layout", layout, "--reference-set",
                self.clean_folder(size), "--reference-out", reference_folder(folder, state),
                "--sides", sides, "--jobs", acceptance_sets.JOBS,
                "--tokenizer-spec", acceptance_sets.SPEC])
            document = json.loads(sides.read_text(encoding="utf-8"))
            repaired, reference = sides_files(document, sides.parent)
            self._repaired[key] = SetRepaired(
                hist, Repaired(store, report_file, sides, folder / f"repaired-{state}", summary),
                report,
                repaired, reference)
        return self._repaired[key]

    def verification(self, size: str, layout: str, state: str = STATE_UNMUTATED,
                     mode: str | None = None) -> tuple[Mapping | None, str | None]:
        """(what ``verify`` printed, None) or (None, why not)."""
        repaired = self.repaired(size, layout, state).repair
        arguments = ["verify", "--report", repaired.report, "--sides", repaired.sides]
        if mode is not None:
            arguments += ["--mode", mode]
        code, stdout, stderr = tool_command(repaired.store, arguments)
        if code != 0:
            return None, f"no verification: exit code {code}, standard error {stderr[:300]!r}"
        return json.loads(stdout), None

    def _asked(self, size: str, layout: str, artifact: str, state: str) -> Mapping:
        repaired = self.repaired(size, layout, state).repair
        code, stdout, _ = tool_command(
            repaired.store, ["query", "withdraw", "A", "--artifact", artifact])
        return json.loads(stdout) if code == 0 else {}

    def answer(self, size: str, layout: str, artifact: str,
               state: str = STATE_UNMUTATED) -> str | None:
        """``answer`` of ``query withdraw A --artifact NAME`` put to the repaired store."""
        return self._asked(size, layout, artifact, state).get("answer")

    def answer_and_reason(self, size: str, layout: str, artifact: str,
                          state: str = STATE_UNMUTATED) -> tuple[str | None, str | None]:
        """``answer`` and ``reason`` of the same question: a name the store holds no row for
        answers "not covered" and says why in ``reason``."""
        asked = self._asked(size, layout, artifact, state)
        return asked.get("answer"), asked.get("reason")

    def shard_files(self, size: str, layout: str, shard: str,
                    state: str = STATE_UNMUTATED) -> list[str]:
        """The files named ``<shard>`` or ``<shard>.<suffix>`` in the shards folder of the
        repaired build (``shard`` is the replica's own name of the shard, ``shard-A``)."""
        out = self.repaired(size, layout, state).repair.out
        return files_of_shard(shards_folder(out, layout), shard)

    def reference_shard_files(self, size: str, layout: str, shard: str,
                              state: str = STATE_UNMUTATED) -> list[str]:
        """The same files in the shards folder of the reference build, which the replica
        wrote beside the repaired one (the sides file names only the shards the report lists,
        landing shards included)."""
        out = reference_folder(self.repaired(size, layout, state).repair.out.parent, state)
        return files_of_shard(shards_folder(out, layout), shard)

    def shards_folders(self, size: str, layout: str,
                       state: str = STATE_UNMUTATED) -> tuple[Path, Path]:
        """(repaired, reference): the shards folders of the two builds."""
        repaired = self.repaired(size, layout, state).repair.out
        return (shards_folder(repaired, layout),
                shards_folder(reference_folder(repaired.parent, state), layout))

    def shard_stems(self, size: str, layout: str, state: str = STATE_UNMUTATED) -> list[str]:
        """The shard names that have a file in the repaired or the reference build."""
        repaired = self.repaired(size, layout, state).repair.out
        reference = reference_folder(repaired.parent, state)
        return sorted(set(shard_stems(shards_folder(repaired, layout)))
                      | set(shard_stems(shards_folder(reference, layout))))
