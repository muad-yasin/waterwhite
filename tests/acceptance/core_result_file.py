"""The one file of the core result (fixture/table/README.md "Core result": "The core result is one file").

The recorder (core_record.py) records one run. A run that skips the long runs (the 10k and 100k sets) has
those rows as skipped gates. The long runs were recorded apart, once each, on the same tree; this module
makes the one file from them: the first run's records, with each skipped long-run record replaced by the
record of the run that ran it. It refuses where it would have to choose.

The file is committed gzip-compressed (the 100k records are large), written so that the same records give the
same bytes (no file name, no time in the gzip header). The sha256 of the text before compression is kept in
the header of the commit's note, not in the file. ``read_any`` reads the plain and the gzip form; ``diff`` is
the comparison of a committed file with a later run, either form (``compare``: core rows only, a core key in one
file only is a difference, the replaced gates the later run did not run are "not run here" and make the verdict PARTIAL,
the rows of extensions are listed and not compared).

Nothing here imports the tool.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the script runs under -I, which leaves its own folder off the path

import core_record as cr  # noqa: E402

GATE = "a long run: "  # the start of the reason a gated long-run row is skipped with


def is_gate(record: Mapping) -> bool:
    return bool(record.get("core")) and record["ended"] == "skipped" and str(record.get("reason") or "").startswith(GATE)


def combine(base: Sequence[dict], addenda: Mapping[str, Sequence[dict]]) -> tuple[list[dict], list[str], int]:
    """The base records with every gated long-run record replaced by the record of the addendum that ran
    it. Returns the records (ordered by key), the replaced keys, and how many addendum records were
    left out as duplicates. Refuses: a gated key that no addendum ran, or that more than one ran; an
    addendum key the base lacks; a left-out duplicate whose record differs from the one kept (as the
    comparison of two files reads it), which also keeps two addenda that hold one key equal."""
    kept = {cr.key_of(r): r for r in base}
    if len(kept) != len(base):
        raise ValueError("the base file holds a key twice")
    gates = {k for k, r in kept.items() if is_gate(r)}
    ran: dict[str, list[tuple[str, dict]]] = {}
    left_out = 0
    for name, records in addenda.items():
        for r in records:
            k = cr.key_of(r)
            if k not in kept:
                raise ValueError(f"{name}: the key {k} is not in the base file")
            if k in gates and r["ended"] != "skipped":
                ran.setdefault(k, []).append((name, r))
                continue
            if k in gates:
                left_out += 1  # an addendum that did not run a gated row says nothing about it
                continue
            if cr.guard_view(r) != cr.guard_view(kept[k]):
                raise ValueError(f"{name}: the record {k} differs from the base file's")
            left_out += 1
    for k in sorted(gates):
        got = ran.get(k, [])
        if len(got) != 1:
            raise ValueError(f"the gated record {k} was run by {len(got)} addenda ({[n for n, _ in got]}); exactly one is needed")
    replaced = sorted(ran)
    for k in replaced:
        kept[k] = ran[k][0][1]
    return [kept[k] for k in sorted(kept)], replaced, left_out


def text_of(records: Sequence[dict], header: Mapping[str, object]) -> str:
    """The header line and one canonical line per record, as ``core_record.write_file`` writes them
    (without its unseeded-rows block: the header given is the whole header)."""
    lines = [cr.canonical({"core_result_format": cr.FORMAT, "records": len(records), **header})]
    lines += [cr.canonical(r) for r in records]
    return "\n".join(lines) + "\n"


def write_gzip(records: Sequence[dict], header: Mapping[str, object], path: Path) -> str:
    """Writes the file gzip-compressed, reproducibly; returns the sha256 of the text before compression."""
    text = text_of(records, header)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(text.encode("ascii"))
    return cr.digest(text)


def read_any(path: Path) -> list[dict]:
    """The records of a core-result file, plain or gzip (by its first two bytes, not its name)."""
    raw = Path(path).read_bytes()
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("ascii")
    lines = text.splitlines()
    head = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:]]
    if head.get("core_result_format") != cr.FORMAT or head.get("records") != len(records):
        raise ValueError(f"{path}: the header does not match the file")
    return records


def header_of(path: Path) -> dict:
    raw = Path(path).read_bytes()
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("ascii")
    return json.loads(text.splitlines()[0])


def sha_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Comparison:
    """What the comparison of a committed file with a later run says. ``differences`` are the core keys that
    differ or are in one file only; ``not_run`` are the keys the later run did not run (and only those the
    committed file names as replaced gates, skipped in the run with the gate's own reason); ``extension_rows``
    are the rows of extensions (core = false) of either file, not compared. FULL means no difference and no
    row not run: the only verdict that says the later run gave the committed core result."""

    def __init__(self, differences: list[str], not_run: list[str], extension_rows: list[str], compared: int):
        self.differences, self.not_run, self.extension_rows, self.compared = differences, not_run, extension_rows, compared

    @property
    def verdict(self) -> str:
        return "DIFFERENCES" if self.differences else ("PARTIAL" if self.not_run else "FULL")

    def lines(self) -> list[str]:
        out = list(self.differences)
        if self.verdict == "DIFFERENCES":
            out.append(f"DIFFERENCES: {len(self.differences)} of the core keys")
        elif self.verdict == "PARTIAL":
            out.append(f"PARTIAL, not the full verdict: {self.compared} core rows compared, {len(self.not_run)} not run here:")
            out += [f"  not run: {k}" for k in self.not_run]
        else:
            out.append(f"FULL: no difference in the core rows ({self.compared} compared, 0 not run here)")
        if self.extension_rows:
            out.append("extension rows not compared (core = false): " + ", ".join(self.extension_rows))
        return out

    @property
    def exit_code(self) -> int:
        return {"FULL": 0, "DIFFERENCES": 1, "PARTIAL": 3}[self.verdict]


def compare(before: Sequence[dict], after: Sequence[dict], replaced: Sequence[str] = ()) -> Comparison:
    """The core rows of the two files, key by key. A core key in one file only is a difference. A key of
    ``replaced`` (the committed file's own list of replaced gates) that ran in ``before`` and is skipped in
    ``after`` with the gate's reason is "not run here"; nothing else is."""
    named = set(replaced)
    a = {cr.key_of(r): r for r in before if r.get("core")}
    b = {cr.key_of(r): r for r in after if r.get("core")}
    extension = sorted({cr.key_of(r) for r in [*before, *after] if not r.get("core")})
    differences, not_run = [], []
    for key in sorted(set(a) | set(b)):
        if key not in a:
            differences.append(f"{key}: only in the second")
        elif key not in b:
            differences.append(f"{key}: only in the first")
        elif key in named and is_gate(b[key]) and not is_gate(a[key]):
            not_run.append(key)
        elif cr.guard_view(a[key]) != cr.guard_view(b[key]):
            va, vb = cr.guard_view(a[key]), cr.guard_view(b[key])
            parts = [f for f in sorted(set(va) | set(vb)) if va.get(f) != vb.get(f)]
            differences.append(f"{key}: differs in {', '.join(parts)}")
    return Comparison(differences, not_run, extension, len(set(a) & set(b)) - len(not_run))


if __name__ == "__main__":  # combine OUT.gz HEADER-JSON BASE ADDENDUM...   |   diff A B   |   lacking FILE
    usage = "usage: core_result_file.py combine OUT HEADER-JSON BASE ADDENDUM... | diff A B | lacking FILE"
    if len(sys.argv) < 3:
        raise SystemExit(usage)
    command = sys.argv[1]
    if command == "combine" and len(sys.argv) >= 6:
        out, header, base_path, add_paths = Path(sys.argv[2]), json.loads(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]
        base = read_any(base_path)
        addenda = {Path(p).name: read_any(Path(p)) for p in add_paths}
        records, replaced, left_out = combine(base, addenda)
        sources = {Path(p).name: sha_of_file(Path(p)) for p in [str(base_path), *add_paths]}
        sha = write_gzip(records, {**header, "sources_sha256": sources, "replaced_keys": replaced, "left_out_duplicates": left_out}, out)
        print(f"{len(records)} records, {len(replaced)} replaced, {left_out} duplicates left out, text sha256 {sha}, gzip sha256 {sha_of_file(out)}")
    elif command == "diff" and len(sys.argv) == 4:
        first = Path(sys.argv[2])
        made = compare(read_any(first), read_any(Path(sys.argv[3])), header_of(first).get("replaced_keys", []))
        print("\n".join(made.lines()))
        raise SystemExit(made.exit_code)
    elif command == "lacking" and len(sys.argv) == 3:
        for why, n in sorted(cr.lacking(read_any(Path(sys.argv[2]))).items(), key=lambda kv: -kv[1]):
            print(f"{n:5d}  {why}")
    else:
        raise SystemExit(usage)
