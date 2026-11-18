"""The recorder of the core result (fixture/table/README.md "Core result").

A recorder puts every core row's question the way the acceptance checks put it
(``acceptance_rows.grade`` and the check 9 test, called, never copied) and keeps, for
each row, what the tool gave: the outcome of every program the helpers started during
that row (the tool's command line with its whole standard output; any other program
with its exit code and the digest of its output), the outcome of a job run in the
confined harness, and how the row ended. It keeps the outcome the comparison receives,
not the fields the row expects: a row that passes is only known to match the fields the
table pins.

The record of a row is one line of canonical ASCII JSON; its sha256 is kept beside it.
The lines travel in the junit file of the run (one test property per row), so a run on
another machine returns them with its logs; ``merge`` reads the junit files into the one
file, refusing a duplicate row, and ``differences`` compares two such files row by row.

Nothing here imports the tool: the merge and the comparison read text, so a lane's
guard can use them with no store open.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

PROPERTY_RECORD = "core_record"
PROPERTY_SHA = "core_record_sha256"
FORMAT = 1
# Programs whose whole output is kept: the tool's command line. Any other program
# (the replica's build, a baseline tool) is kept as its exit code, the length of its
# output and the digest of it, because those outputs are logs of a build.
TOOL_MARKER = "waterwhite.cli"
KEPT_AT_MOST = 400_000


def canonical(value: object) -> str:
    """One line of ASCII JSON, keys sorted, no spaces."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ADDRESS = re.compile(r"( object at )0x[0-9a-f]+")
# The world numbers the scratch folders and the copies it makes for the rows, in the order the rows ask:
# <BASE>/scratch/001-repair-main, <BASE>/copies/scratch-500--w.retry-002. The number says how many came before.
SCRATCH_NUMBER = re.compile(r"(<BASE>/scratch/)\d{3}(-)")
COPY_NUMBER = re.compile(r"(<BASE>/copies/[^/\s\"']*?-)\d{3}(?=[/\"'\s]|$)")


class Scrub:
    """Replaces the places that differ between two runs of the same tree: the folder the
    run builds in, the repository, the interpreter. Nothing else is replaced."""

    def __init__(self, replacements: Mapping[str, str]):
        # Longest first, so a folder inside another is replaced as a whole.
        self.pairs = sorted(((k, v) for k, v in replacements.items() if k), key=lambda kv: -len(kv[0]))

    def __call__(self, text: str) -> str:
        for old, new in self.pairs:
            text = text.replace(old, new)
        # The address of a Python object in a log line (" object at 0x7b6a..") differs with every run of a program.
        text = SCRATCH_NUMBER.sub(r"\1NNN\2", text)
        text = COPY_NUMBER.sub(r"\1NNN", text)
        return ADDRESS.sub(r"\1<ADDRESS>", text)


class Calls:
    """The programs started while one row is put. ``watch`` wraps ``subprocess.run``
    for the length of the block and keeps each call."""

    def __init__(self, scrub: Scrub):
        self.scrub = scrub
        self.calls: list[dict] = []
        self.evidence: list[dict] = []

    def _keep(self, args: Sequence[str], done: subprocess.CompletedProcess) -> None:
        stdout = done.stdout if isinstance(done.stdout, str) else (done.stdout or b"").decode("utf-8", "replace")
        stderr = done.stderr if isinstance(done.stderr, str) else (done.stderr or b"").decode("utf-8", "replace")
        shown = [self.scrub(str(a)) for a in args]
        if "-m" in shown and shown[shown.index("-m") + 1:shown.index("-m") + 3] == ["replica", "build"]:
            return  # the build of a store or a set is done once, lazily, for whichever row asks first: not that row's answer
        entry: dict = {"args": shown, "exit": done.returncode}
        if TOOL_MARKER in shown:
            out, err = self.scrub(stdout), self.scrub(stderr)
            entry.update({"stdout": out[:KEPT_AT_MOST], "stdout_length": len(out), "stdout_sha256": digest(out),
                          "stderr": err[:KEPT_AT_MOST], "stderr_length": len(err), "stderr_sha256": digest(err)})
        else:
            entry["stdout_length"] = len(stdout)
            entry["stdout_sha256"] = digest(self.scrub(stdout))
            entry["stderr_length"] = len(stderr)
            entry["stderr_sha256"] = digest(self.scrub(stderr))
        self.calls.append(entry)

    @contextlib.contextmanager
    def watch(self) -> Iterator[None]:
        real = subprocess.run

        def run(args, *a, **kw):
            done = real(args, *a, **kw)
            try:
                self._keep([args] if isinstance(args, (str, bytes)) else list(args), done)
            except Exception as error:  # a record that cannot be made is said, never skipped
                self.calls.append({"args": ["<unrecordable>"], "error": f"{type(error).__name__}: {error}"})
            return done

        subprocess.run = run
        try:
            yield
        finally:
            subprocess.run = real


def canonical_scrub(scrub: "Scrub", value: object) -> object:
    """The value with the places that differ between runs replaced in every text of it."""
    return json.loads(scrub(canonical(value)))


def row_record(*, check: int, row_id: str, size: str | None, part: str, core: bool, extension: str | None,
               ended: str, reason: str | None, calls: Sequence[dict], evidence: object | None,
               mismatch: str | None, kind: str | None = None, store: str | None = None,
               unseeded: bool = False) -> dict:
    return {"format": FORMAT, "check": check, "row": row_id, "size": size, "part": part, "core": bool(core),
            "extension": extension, "kind": kind, "store": store, "unseeded": bool(unseeded), "ended": ended, "reason": reason, "calls": list(calls),
            "evidence": evidence, "mismatch": mismatch}


def key_of(record: Mapping) -> str:
    """The identity of a record: the row, the part and the size."""
    return f"{record['row']}|{record['part']}|{record['size'] or '-'}"


# --- the junit files and the one file ----------------------------------------------------------------


def read_junit(path: Path) -> list[tuple[str, dict, str]]:
    """(test name, record, stated sha256) of every record property in a junit file. The
    sha256 is re-checked against the line itself: a record that does not match is an error."""
    found = []
    for case in ET.parse(path).iter("testcase"):
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        if PROPERTY_RECORD not in props:
            continue
        line, stated = props[PROPERTY_RECORD], props.get(PROPERTY_SHA, "")
        if digest(line) != stated:
            raise ValueError(f"{path}: {case.get('name')}: the record does not match its sha256")
        found.append((case.get("name"), json.loads(line), stated))
    return found


def merge(junit_files: Iterable[Path]) -> list[dict]:
    """The records of every file, ordered by key. A key given twice, or a record with another
    format, is refused."""
    by_key: dict[str, dict] = {}
    for path in junit_files:
        for name, record, _ in read_junit(Path(path)):
            if record.get("format") != FORMAT:
                raise ValueError(f"{path}: {name}: format {record.get('format')!r}, expected {FORMAT}")
            key = key_of(record)
            if key in by_key:
                raise ValueError(f"{path}: {name}: the record {key} is given twice")
            by_key[key] = record
    return [by_key[k] for k in sorted(by_key)]


def table_ids() -> set[str]:
    """The id of every core row of Checks 1 to 9, from the TABLE (never from a junit file)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import table
    return {row.id for n in range(1, 10) for row in table.load_check(n) if row.core}


def check_ids_against_table(records: Sequence[dict], expected: set[str]) -> None:
    """Every core row of the table has a record, and no record names a row the table lacks."""
    named = {r["row"] for r in records}
    missing, unknown = sorted(expected - named), sorted(named - expected - {r["row"] for r in records if not r.get("core")})
    if missing or unknown:
        raise ValueError(f"the records and the table differ: no record for {missing}; not in the table: {unknown}")


def write_file(records: Sequence[dict], path: Path, header: Mapping[str, str] | None = None) -> str:
    """The one file: a header line (format, record count, the tree id and the machine when
    given), then one canonical line per record; returns its sha256."""
    unseeded = sorted({r["row"] for r in records if r.get("unseeded") and r.get("core")})
    extra = {"unseeded_rows_not_compared": unseeded,
             "not_compared_for_them": "the raw calls (values of ids, digests, spans, fingerprints); compared: how the row "
                                      "ended, the reason of a skip, whether a program was started, the distinct exit codes of the programs "
                                      "started (their number is not compared: it follows the shards of the build)"} if unseeded else {}
    lines = [canonical({"core_result_format": FORMAT, "records": len(records), **extra, **(header or {})})]
    lines += [canonical(r) for r in records]
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="ascii")
    return digest(text)


def read_file(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    head = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:]]
    if head.get("core_result_format") != FORMAT or head.get("records") != len(records):
        raise ValueError(f"{path}: the header does not match the file")
    return records


def guard_view(record: Mapping) -> Mapping:
    """What the comparison of two files reads of a record. A row asked of a store built unseeded
    (every shuffle drawn from the operating system) gives other ids, digests and token spans with every
    build, so the raw calls of its record are marked "not compared" and the row is compared by how it
    ended: its outcome, the reason it was skipped, whether a program was started and the distinct exit codes
    of the programs started. How many programs a row starts follows the shards the build drew (a row that
    shows every shard started 52 on one build and 50 on the next), so that number is not compared. The grader
    judged the values of the build in question when the row ended (README "Core result")."""
    if not record.get("unseeded"):
        return record
    return {"row": record["row"], "part": record["part"], "size": record["size"], "ended": record["ended"],
            "reason": record["reason"], "program_started": bool(record["calls"]),
            "exits": sorted({c.get("exit") for c in record["calls"]}, key=str), "unseeded": True}


def differences(before: Sequence[dict], after: Sequence[dict], *, core_only: bool = True) -> list[str]:
    """The keys whose record differs, one line each, naming the part that differs. A key in one
    list only is named too. With core_only the rows of an extension (core false) are left out."""
    def index(records):
        return {key_of(r): r for r in records if r.get("core") or not core_only}
    a, b = index(before), index(after)
    out = []
    for key in sorted(set(a) | set(b)):
        if key not in a:
            out.append(f"{key}: only in the second")
        elif key not in b:
            out.append(f"{key}: only in the first")
        elif guard_view(a[key]) != guard_view(b[key]):
            va, vb = guard_view(a[key]), guard_view(b[key])
            parts = [f for f in sorted(set(va) | set(vb)) if va.get(f) != vb.get(f)]
            out.append(f"{key}: differs in {', '.join(parts)}")
    return out


def no_program_rows(records: Sequence[dict]) -> list[str]:
    """The core rows that ended with no program started, by id and kind."""
    return [f"{r['row']} [{r.get('kind')}]" for r in records
            if r.get("core") and r["ended"] != "skipped" and not r["calls"] and not r.get("evidence")]


def lacking(records: Sequence[dict]) -> dict[str, int]:
    """What the file lacks, by reason: a row that was skipped has no answer; a row whose
    question never started a program has the way it ended only."""
    count: dict[str, int] = {}
    for r in records:
        if not r.get("core"):
            continue
        if r["ended"] == "skipped":
            why = "skipped: " + re.sub(r"\s+", " ", str(r.get("reason") or ""))[:110]
        elif not r["calls"] and not r.get("evidence"):
            why = f"no program started (the way it ended only): {r['ended']}"
        else:
            continue
        count[why] = count.get(why, 0) + 1
    return count


if __name__ == "__main__":  # merge OUT TREE MACHINE JUNIT...   |   diff A B   |   lacking FILE
    usage = "usage: core_record.py merge OUT HEADER-JSON JUNIT... | diff A B | lacking FILE | noprogram FILE"
    if len(sys.argv) < 3:
        raise SystemExit(usage)
    command = sys.argv[1]
    if command == "merge" and len(sys.argv) >= 5:
        merged = merge(Path(p) for p in sys.argv[4:])
        header = json.loads(sys.argv[3])
        if not header.get("partial"):  # a probe of some checks only says so in its header
            check_ids_against_table(merged, table_ids())
        out = write_file(merged, Path(sys.argv[2]), header)
        print(f"{len(merged)} records, sha256 {out}")
    elif command == "diff" and len(sys.argv) == 4:
        found = differences(read_file(Path(sys.argv[2])), read_file(Path(sys.argv[3])))
        print("\n".join(found) if found else "no difference in the core rows")
        raise SystemExit(1 if found else 0)
    elif command == "noprogram" and len(sys.argv) == 3:
        print("\n".join(no_program_rows(read_file(Path(sys.argv[2])))))
    elif command == "lacking" and len(sys.argv) == 3:
        for why, n in sorted(lacking(read_file(Path(sys.argv[2]))).items(), key=lambda kv: -kv[1]):
            print(f"{n:5d}  {why}")
    else:
        raise SystemExit(usage)
