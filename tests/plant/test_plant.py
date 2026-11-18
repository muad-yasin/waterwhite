"""Fast unit tests of the plant generator and the counting script (plan 5.1, W5).

The two scripts are loaded from their files, since ``fixture/plant`` is a folder of
scripts and not a package. Wherever a test needs chunk files it makes them itself:
it reads the set's files, applies its own dedup and its own normalisation, deals
the ids and writes one ``<shard>.jsonl`` per shard, the way a pipeline would hand
them to ``count.py``. Nothing here imports the tool.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLANT = ROOT / "fixture" / "plant"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, PLANT / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gen = _load("plant_generate", "generate.py")
cnt = _load("plant_count", "count.py")

# One small set for most tests. Worked out by hand from docs/plant.md:
#   X pairs 800 // 20 = 40, summary passages 800 // 100 = 8
#   A = 40 + 8 = 48 lines, B = 2 * 40 = 80, C = 2 * 8 + 24 = 40
#   paraphrases of unrelated records: max(2, 800 // 1000) = 2, UP-000001 and UP-000002
#   generated = 40 P + 2 UP + 40 T + 8 S = 90, Z = 800 - 48 - 80 - 40 - 90 = 542
#   Z lines 1..271 are inputs of R1, the rest inputs of R2
SMALL = dict(records=800, seed=7, length=160, pairs=24, sanitizer_cases=3, chunks=20)
N_X, N_S, PAIRS, Z_LINES, Z_R1, N_KEPT = 40, 8, 24, 542, 271, 2
KEPT = ["UP-000001", "UP-000002"]  # the generated records that do not derive from A

SAFE_CONTROLS = {chr(c) for c in (*range(1, 9), 11, 12, *range(14, 32))}


def normalise(text: str) -> str:
    """The test's own reading of "remove control characters and strip the ends"."""
    return "".join(c for c in text if not (ord(c) < 32 or 127 <= ord(c) < 160)).strip()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expand(items):
    out = []
    for item in items:
        if ".." not in item:
            out.append(item)
            continue
        first, last = item.split("..")
        prefix, a = first.rsplit("-", 1)
        b = last.rsplit("-", 1)[1]
        out += [f"{prefix}-{n:0{len(a)}d}" for n in range(int(a), int(b) + 1)]
    return out


class SetFiles:
    """A set folder as a pipeline would read it: the files, not the plant list."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.cfg = tomllib.loads((folder / "set.toml").read_text(encoding="utf-8"))
        self.raw: dict[str, str] = {}
        self.source: dict[str, str | None] = {}
        for name, entry in self.cfg["sources"].items():
            with open(folder / entry["file"], encoding="utf-8", newline="\n") as f:
                for n, line in enumerate(f, start=1):
                    self.raw[f"{name}-{n:06d}"] = line.rstrip("\n")
                    self.source[f"{name}-{n:06d}"] = name
        self.generation = []
        self.made: dict[str, dict] = {}
        with open(folder / self.cfg["generation"]["file"], encoding="utf-8", newline="\n") as f:
            for line in f:
                row = json.loads(line)
                self.generation.append(row)
                self.made[row["id"]] = row
                self.raw[row["id"]] = line.rstrip("\n")
                self.source[row["id"]] = None
        plant = folder / "plant_list.jsonl"
        self.plant = (
            [json.loads(line) for line in plant.read_text(encoding="utf-8").splitlines()]
            if plant.exists()
            else []
        )
        self.by_id = {row["id"]: row for row in self.plant}

    def text(self, record: str) -> str | None:
        """The sanitized text of a line, or None when the sanitizer skips it."""
        try:
            text = normalise(json.loads(self.raw[record])["text"])
        except ValueError:
            return None
        return text or None

    def release_rows(self, release: str) -> list[str]:
        """The rows of a release: its inputs after the test's own dedup."""
        entry = self.cfg["releases"][release]
        inputs = expand(entry["inputs"])
        if not entry["dedup"]:
            return inputs
        rank = {name: i for i, name in enumerate(entry["dedup_priority"])}
        best: dict[str, str] = {}
        for record in inputs:
            text = self.text(record)
            if text is None:
                continue
            if text not in best or rank[self.source[record]] < rank[self.source[best[text]]]:
                best[text] = record
        return [r for r in inputs if self.text(r) is None or best[self.text(r)] == r]


def comes_from(sf: SetFiles, record: str, source: str) -> bool:
    """The test's own walk over the files: is the record a line of the source, or
    was it made, through any chain of jobs, from a record that is? Every job here
    needs each record it used, so one such parent is enough. The plant list and the
    ids' prefixes of generated records are not consulted."""
    if sf.source[record] is not None:
        return sf.source[record] == source
    return any(comes_from(sf, parent, source) for parent in sf.made[record]["used"])


def deal_round_robin(sf: SetFiles, release: str, seed: int, start=lambda k: 0) -> dict[str, list[str]]:
    """Each origin file shuffled on its own, then dealt over the chunks from its own
    start chunk (the per-file reading of plan 4.5)."""
    chunks = sf.cfg["layouts"]["round-robin"]["chunks"]
    by_origin: dict[str, list[str]] = {}
    for record in sf.release_rows(release):
        by_origin.setdefault(record.split("-")[0], []).append(record)
    rng = random.Random(seed)
    shards = {f"shard-{c + 1}": [] for c in range(chunks)}
    for k, ids in enumerate(by_origin.values()):
        rng.shuffle(ids)
        for j, record in enumerate(ids):
            shards[f"shard-{(start(k) + j) % chunks + 1}"].append(record)
    return shards


def deal_partitioned(sf: SetFiles, release: str, per_shard: int = 60) -> dict[str, list[str]]:
    by_origin: dict[str, list[str]] = {}
    for record in sf.release_rows(release):
        by_origin.setdefault(record.split("-")[0], []).append(record)
    shards = {}
    for origin, ids in by_origin.items():
        for j in range(0, len(ids), per_shard):
            shards[f"{origin}-part-{j // per_shard + 1}"] = ids[j : j + per_shard]
    return shards


def write_chunks(sf: SetFiles, out: Path, shards: dict[str, list[str]]) -> Path:
    """Tokenizer-input files: surviving sanitized lines only, in order."""
    out.mkdir(parents=True)
    for shard, ids in shards.items():
        with open(out / f"{shard}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for record in ids:
                text = sf.text(record)
                if text is not None:
                    f.write(json.dumps({"id": record, "text": text}, ensure_ascii=False) + "\n")
    return out


def shard_of(shards: dict[str, list[str]], record: str) -> str:
    return next(name for name, ids in shards.items() if record in ids)


def move(shards: dict[str, list[str]], record: str, to: str) -> None:
    shards[shard_of(shards, record)].remove(record)
    shards[to].append(record)


def planted_pairs(sf: SetFiles) -> list[tuple[str, str]]:
    """(C member, Z member) of every planted pair."""
    pairs: dict[str, list[str]] = {}
    for row in sf.plant:
        if row["kind"] == "pair-member":
            pairs.setdefault(row["pair"], []).append(row["id"])
    return [tuple(sorted(ids)) for ids in pairs.values()]


@pytest.fixture(scope="module")
def small(tmp_path_factory) -> SetFiles:
    folder = tmp_path_factory.mktemp("plant") / "small"
    gen.generate(folder, **SMALL)
    return SetFiles(folder)


@pytest.fixture(scope="module")
def small_off(tmp_path_factory) -> SetFiles:
    folder = tmp_path_factory.mktemp("plant") / "small-off"
    gen.generate(folder, dedup=False, **SMALL)
    return SetFiles(folder)


# --- The generator -----------------------------------------------------------------


def test_same_call_gives_same_bytes(tmp_path, small):
    again = gen.generate(tmp_path / "again", **SMALL)
    on_disk = {}
    for rel in again["files"]:
        on_disk[rel] = hashlib.sha256((small.folder / rel).read_bytes()).hexdigest()
    assert on_disk == {rel: entry["sha256"] for rel, entry in again["files"].items()}
    other = gen.generate(tmp_path / "other", **{**SMALL, "seed": 8})
    assert other["set_digest"] != again["set_digest"]
    assert set(again["files"]) == {
        "generation.jsonl", "plant_list.jsonl", "set.toml",
        "sources/A.jsonl", "sources/B.jsonl", "sources/C.jsonl", "sources/Z.jsonl",
    }


def test_scripts_import_the_standard_library_only():
    for file in ("generate.py", "count.py"):
        tree = ast.parse((PLANT / file).read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{file} has a relative import"
                modules.add(node.module.split(".")[0])
        assert modules <= set(sys.stdlib_module_names), f"{file} imports {modules}"
        assert not modules & {"waterwhite", "replica", "oracle", "random", "time", "datetime"}


def test_output_folder_must_be_empty(tmp_path, small):
    with pytest.raises(gen.OutputFolderNotEmpty):
        gen.generate(small.folder, **SMALL)
    proc = subprocess.run(
        [sys.executable, str(PLANT / "generate.py"), "--out", str(small.folder),
         "--records", "800", "--seed", "7", "--length", "160", "--pairs", "24",
         "--sanitizer-cases", "3"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2 and proc.stdout == ""
    assert json.loads(proc.stderr)["error"] == "OutputFolderNotEmpty"
    # The same command into a new folder writes the same bytes as the function call.
    proc = subprocess.run(
        [sys.executable, str(PLANT / "generate.py"), "--out", str(tmp_path / "cli"),
         "--records", "800", "--seed", "7", "--length", "160", "--pairs", "24",
         "--sanitizer-cases", "3"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    summary = json.loads(proc.stdout)
    for rel, entry in summary["files"].items():
        assert hashlib.sha256((small.folder / rel).read_bytes()).hexdigest() == entry["sha256"]


def test_parameters_that_break_a_guarantee_are_refused(tmp_path):
    with pytest.raises(gen.ParameterError, match="fewer than the 20 chunks"):
        gen.generate(tmp_path / "a", records=300, seed=1, length=160, pairs=4, sanitizer_cases=3)
    with pytest.raises(gen.ParameterError, match="too few lines in source Z"):
        gen.generate(tmp_path / "b", records=800, seed=1, length=160, pairs=400)
    with pytest.raises(gen.ParameterError):
        gen.generate(tmp_path / "c", records=800, seed=1, derivations=3)
    with pytest.raises(gen.ParameterError):
        gen.generate(tmp_path / "d", records=800, seed=1, sources=5)
    assert not any((tmp_path / name).exists() for name in "abcd")


def test_file_formats(small):
    cfg = small.cfg
    assert cfg["format"] == "waterwhite.plant-set/1" and cfg["clean"] is False
    assert cfg["withdrawn"] == "A"
    assert {name: entry["lines"] for name, entry in cfg["sources"].items()} == {
        "A": 48, "B": 80, "C": 40, "Z": Z_LINES,
    }
    for name, entry in cfg["sources"].items():
        assert entry["member"] == f"{name}/r1" and entry["revision"] == "r1"
        assert expand(entry["records"]) == [f"{name}-{n:06d}" for n in range(1, entry["lines"] + 1)]
    unparseable = 0
    for record, raw in small.raw.items():
        if small.source[record] is None:
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            unparseable += 1
            continue
        assert list(row) == ["id", "text"] and row["id"] == record
    assert unparseable == 3
    assert cfg["generation"]["rows"] == len(small.generation) == 90
    assert cfg["generation"]["model"] == "none"
    assert cfg["generation"]["jobs"] == {
        "paraphrase": ["P-000001..P-000040", "UP-000001..UP-000002"],
        "translation": ["T-000001..T-000040"],
        "summary": ["S-000001..S-000008"],
    }
    for job, listed in cfg["generation"]["jobs"].items():
        # A job's ids are listed in the order of its rows in the file.
        assert [r["id"] for r in small.generation if r["job"] == job] == expand(listed)
    for row in small.generation:
        assert list(row) == ["id", "job", "handed", "used", "text"]
        assert row["handed"] == row["used"] == small.by_id[row["id"]]["parents"]
    assert [r["job"] for r in small.generation] == (
        ["paraphrase"] * 42 + ["translation"] * 40 + ["summary"] * 8
    )
    assert cfg["layouts"]["round-robin"]["chunks"] == 20
    assert cfg["releases"]["R1"]["dedup_priority"] == ["A", "B", "C", "Z"]
    assert cfg["releases"]["R1"]["dedup_winner"] == "A"


def test_plant_list_describes_every_line(small):
    assert len(small.plant) == 800 == len(small.raw)
    for row in small.plant:
        assert row["id"] in small.raw
        text = small.text(row["id"])
        if row["sanitizer"] and row["sanitizer"]["skip"]:
            assert text is None and row["text_sha256"] is None
        else:
            assert row["text_sha256"] == sha(text) and row["chars"] == len(text)
        if row["sanitizer"]:
            assert row["sanitizer"]["raw_sha256"] == sha(small.raw[row["id"]])


def test_texts_are_unique_except_the_planted_pairs(small):
    texts: dict[str, list[str]] = {}
    for record in small.raw:
        text = small.text(record)
        if text is not None:
            texts.setdefault(text, []).append(record)
    repeated = {tuple(sorted(ids)) for ids in texts.values() if len(ids) > 1}
    x_pairs = {(f"A-{k:06d}", f"B-{k:06d}") for k in range(1, N_X + 1)}
    assert repeated == x_pairs | set(planted_pairs(small))
    assert len(planted_pairs(small)) == PAIRS
    for c, z in planted_pairs(small):
        assert c.startswith("C-") and z.startswith("Z-")


def test_structure_repeats_the_fixture(small):
    by_id = small.by_id
    kinds = Counter(row["kind"] for row in small.plant)
    assert kinds == {
        "x-copy": 80, "passage": 16, "paraphrase": 40, "translation": 40, "summary": 8,
        "paraphrase-of-unrelated": N_KEPT,
        "pair-member": 48, "sanitizer-case": 9, "unrelated": 800 - 241 - N_KEPT,
    }
    for row in small.plant:
        if row["kind"] == "paraphrase":
            (parent,) = row["parents"]
            assert by_id[parent]["kind"] == "x-copy" and by_id[parent]["source"] == "A"
            assert "restates entry" in small.text(row["id"])
        elif row["kind"] == "translation":
            (parent,) = row["parents"]
            assert by_id[parent]["kind"] == "paraphrase"
            assert small.text(row["id"]).startswith("Übersetzung ")
            assert row["withdraw_A"]["roots"] == by_id[parent]["parents"]
        elif row["kind"] == "summary":
            a, c = row["parents"]
            assert (by_id[a]["source"], by_id[c]["source"]) == ("A", "C")
            assert by_id[a]["kind"] == by_id[c]["kind"] == "passage"
            assert row["parent_mode"] == "all" and row["withdraw_A"]["roots"] == [a]
    # Nothing was made from a record of B. The only records made from a record of Z
    # are the paraphrases of unrelated records, and "withdraw A" does not reach them.
    made_from = {p: row["id"] for row in small.plant for p in row["parents"]}
    assert not any(p.startswith("B-") for p in made_from)
    assert sorted(child for p, child in made_from.items() if p.startswith("Z-")) == KEPT
    in_scope = {row["id"] for row in small.plant if row["withdraw_A"]["in_scope"]}
    assert in_scope == {r for r in small.raw if r.split("-")[0] in ("A", "P", "T", "S")}
    r2 = expand(small.cfg["releases"]["R2"]["inputs"])
    assert {r.split("-")[0] for r in r2} == {"B", "Z"} and not in_scope & set(r2)


def test_record_lengths_are_about_the_target(small):
    for row in small.plant:
        if row["chars"] is None:
            continue
        target = 120 if row["kind"] == "summary" else 160
        assert target <= row["chars"] < target + 130


def test_sanitizer_cases(small):
    cases = [row for row in small.plant if row["kind"] == "sanitizer-case"]
    assert Counter(row["sanitizer"]["case"] for row in cases) == {
        "unparseable": 3, "empty": 1, "control-only": 1, "control-and-space": 1,
        "control-inside": 3,
    }
    assert all(row["source"] == "Z" and "R1" in row["releases"] for row in cases)
    for row in cases:
        raw = small.raw[row["id"]]
        case = row["sanitizer"]
        if case["case"] == "unparseable":
            with pytest.raises(ValueError):
                json.loads(raw)
            assert (case["skip"], case["reason"]) == (True, "sanitizer skip")
            continue
        text = json.loads(raw)["text"]
        controls = {c for c in text if ord(c) < 32 or 127 <= ord(c) < 160}
        assert controls <= SAFE_CONTROLS
        if case["case"] == "control-inside":
            assert controls and (case["skip"], case["reason"]) == (False, None)
            assert text[0] in controls and text[-1] in controls
            assert sha(normalise(text)) == row["text_sha256"]
        else:
            assert normalise(text) == "" and (case["skip"], case["reason"]) == (True, "empty line")
            assert bool(controls) == (case["case"] != "empty")
    # No other line holds a character a reading of "control character" could dispute.
    plain = set(small.raw) - {row["id"] for row in cases}
    for record in plain:
        text = json.loads(small.raw[record])["text"]
        assert text == text.strip() and all(ord(c) >= 32 and not 127 <= ord(c) < 160 for c in text)


def test_derivations_zero_and_two(tmp_path):
    none = gen.generate(tmp_path / "d0", **{**SMALL, "derivations": 0})
    # No derivations: no generation job at all, so no paraphrase of an unrelated
    # record either, and the counting script has nothing to refuse.
    assert none["files"]["generation.jsonl"]["lines"] == 0
    assert "paraphrase-of-unrelated" not in none["counts"]
    assert cnt.count(tmp_path / "d0")["generated"] == {
        "total": 0, "derived": 0, "stay": {"count": 0, "ids": [], "list": []},
    }
    assert cnt.count(tmp_path / "d0")["records"]["in_scope"] == {"total": 48, "own": 48, "derived": 0}
    gen.generate(tmp_path / "d2", **{**SMALL, "derivations": 2})
    two = SetFiles(tmp_path / "d2")
    assert Counter(r["job"] for r in two.generation) == {
        "paraphrase": 80 + N_KEPT, "translation": 80, "summary": 16,
    }
    # The paraphrases of unrelated records do not grow with the derivations.
    assert [r["id"] for r in two.generation if r["id"].startswith("U")] == KEPT
    per_parent = Counter(row["parents"][0] for row in two.plant if row["kind"] == "paraphrase")
    assert set(per_parent.values()) == {2} and len(per_parent) == N_X
    per_paraphrase = Counter(row["parents"][0] for row in two.plant if row["kind"] == "translation")
    assert set(per_paraphrase.values()) == {1} and len(per_paraphrase) == 80
    per_passage = Counter(row["parents"][0] for row in two.plant if row["kind"] == "summary")
    assert set(per_passage.values()) == {2} and len(per_passage) == N_S
    assert len(two.plant) == 800
    assert cnt.count(tmp_path / "d2")["releases"]["R1"]["actions"]["retire"] == 176


def test_ten_sources_hold_six_further_unrelated_sources(tmp_path):
    gen.generate(tmp_path / "ten", **{**SMALL, "records": 1200, "sources": 10})
    ten = SetFiles(tmp_path / "ten")
    assert list(ten.cfg["sources"]) == ["A", "B", "C", "Z", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"]
    further = [row for row in ten.plant if row["source"] in ("Z2", "Z3", "Z4", "Z5", "Z6", "Z7")]
    # 1200 records: A 72, B 120, C 48, generated 132 + 2, which leaves 826 lines;
    # each further source takes 826 // 10 = 82 and Z the remaining 334.
    assert Counter(row["source"] for row in further) == {f"Z{k}": 82 for k in range(2, 8)}
    assert ten.cfg["sources"]["Z"]["lines"] == 334
    # The paraphrases of unrelated records are made from lines of Z, not of Z2 to Z7.
    assert [ten.by_id[r]["parents"][0].split("-")[0] for r in KEPT] == ["Z", "Z"]
    assert {row["kind"] for row in further} == {"unrelated"}
    assert all(len(row["releases"]) == 1 for row in further)
    result = cnt.count(tmp_path / "ten")
    assert result["records"]["total"] == 1200
    assert result["releases"]["R2"]["successor_needed"] is False


def test_clean_set_is_the_full_set_without_the_withdrawn_content(tmp_path, small):
    summary = gen.generate(tmp_path / "clean", clean=True, **SMALL)
    assert set(summary["files"]) == {
        "generation.jsonl", "set.toml", "sources/B.jsonl", "sources/C.jsonl", "sources/Z.jsonl",
    }
    for name in "BCZ":
        rel = f"sources/{name}.jsonl"
        assert (tmp_path / "clean" / rel).read_bytes() == (small.folder / rel).read_bytes()
    # The clean sources hold the generated records that do not derive from A, each
    # with the bytes it has in the full set, and nothing else that was generated.
    full_lines = (small.folder / "generation.jsonl").read_bytes().splitlines(keepends=True)
    assert (tmp_path / "clean" / "generation.jsonl").read_bytes() == b"".join(
        full_lines[N_X : N_X + N_KEPT]
    )
    assert summary["counts"]["paraphrase-of-unrelated"] == N_KEPT
    assert not {"paraphrase", "translation", "summary"} & set(summary["counts"])
    clean = SetFiles(tmp_path / "clean")
    assert clean.cfg["clean"] is True and "A" not in clean.cfg["sources"]
    assert clean.cfg["releases"]["R1"]["dedup_priority"] == ["B", "C", "Z"]
    assert [row["id"] for row in clean.generation] == KEPT
    assert clean.cfg["generation"]["rows"] == N_KEPT
    assert clean.cfg["generation"]["jobs"] == {
        "paraphrase": ["UP-000001..UP-000002"], "translation": [], "summary": [],
    }
    clean_r1 = expand(clean.cfg["releases"]["R1"]["inputs"])
    for record in KEPT:
        assert record in clean_r1 and clean.made[record]["handed"][0] in clean_r1
        assert not comes_from(clean, record, "A")
    result = cnt.count(small.folder, expand_ids=True)
    omitted = set(result["clean_omits"]["ids"])
    assert set(small.raw) - omitted == set(clean.raw)
    for release in ("R1", "R2"):
        full_inputs = expand(small.cfg["releases"][release]["inputs"])
        assert expand(clean.cfg["releases"][release]["inputs"]) == [
            r for r in full_inputs if r not in omitted
        ]
        # A from-scratch build of the clean set has as many rows as the count expects.
        assert len(clean.release_rows(release)) == result["releases"][release]["rows_after"]
    with pytest.raises(cnt.UsageError, match="clean set"):
        cnt.count(tmp_path / "clean")


# --- The counting script -----------------------------------------------------------


def test_counts_add_up_with_dedup_on(small):
    result = cnt.count(small.folder)
    assert result["request"] == {"kind": "withdraw", "source": "A"}
    assert result["records"]["in_scope"] == {"total": 136, "own": 48, "derived": 88}
    assert (result["records"]["removed"], result["records"]["retained"]) == (136, 664)
    assert result["clean_omits"] == {
        "count": 136,
        "ids": ["A-000001..A-000048", "P-000001..P-000040", "T-000001..T-000040",
                "S-000001..S-000008"],
    }
    r1, r2 = result["releases"]["R1"], result["releases"]["R2"]
    # R1 inputs: 40 X of A + 40 X of B + 90 generated + 24 pair members of C + 271 of Z.
    # Dedup drops B's 40 copies and Z's 24 pair members. Kept: C's 24 pair members,
    # Z's 247 other lines and the 2 paraphrases of unrelated records.
    assert (r1["inputs"], r1["rows_before"], r1["dedup_losers"]) == (465, 401, 64)
    assert r1["actions"] == {
        "keep": 24 + (Z_R1 - 24) + N_KEPT, "re-source": 40, "promote": 40, "retire": 88,
        "none": 24,
    }
    assert r1["actions"]["keep"] == 273
    assert r1["retire_by_why"] == {"own": 0, "derived": 88}
    assert (r1["removed"], r1["retained"], r1["rows_after"], r1["rows_from_scratch"]) == (
        88, 313, 313, 313,
    )
    assert r1["generated"] == {
        "inputs": 90, "out_of_scope": 2,
        "actions": {"keep": 2, "re-source": 0, "promote": 0, "retire": 88, "none": 0},
        "kept_ids": ["UP-000001..UP-000002"],
    }
    assert r2["generated"]["inputs"] == 0 and r2["generated"]["kept_ids"] == []
    assert list(r1["ids"]) == ["re-source", "promote", "retire"]
    assert r1["successor_needed"] is True
    assert r1["ids"]["re-source"] == ["A-000001..A-000040"]
    assert r1["ids"]["promote"] == ["B-000001..B-000040"]
    assert r2["actions"] == {
        "keep": 80 + (Z_LINES - Z_R1), "re-source": 0, "promote": 0, "retire": 0, "none": 0,
    }
    assert r2["successor_needed"] is False
    assert all(identity["holds"] for identity in result["identities"])
    assert len(result["identities"]) == 2 + 7 * 2
    names = [identity["name"] for identity in result["identities"]]
    assert "R1: generated rows kept = generated inputs the source does not reach" in names
    assert "R1: rows after = rows of a from-scratch build without the source" in names
    assert "R1: keep + re-source + retire = rows before" in names
    assert "R1: removed = rows in scope before - re-source" in names
    assert (r1["in_scope_rows_before"], r2["in_scope_rows_before"]) == (40 + 88, 0)
    assert result["actions"]["shards"] is None and result["shards"] is None
    # The test's own dedup over the files gives the same rows as the plant list states.
    present = [r["id"] for r in small.plant if r["releases"].get("R1", {}).get("row") == "present"]
    assert sorted(small.release_rows("R1")) == sorted(present)


def test_counts_with_dedup_off(small_off):
    result = cnt.count(small_off.folder)
    r1 = result["releases"]["R1"]
    assert (r1["inputs"], r1["rows_before"], r1["dedup_losers"]) == (465, 465, 0)
    # With dedup off no copy of X is dropped, so nothing is re-sourced or promoted:
    # A's 40 X rows leave the release with the 88 generated records that derive from
    # A, and B's 40 copies, which are rows of their own, stay (DECISIONS.md D-19).
    # So do the 2 paraphrases of unrelated records (DECISIONS.md D-23).
    assert r1["actions"] == {
        "keep": 40 + 24 + Z_R1 + N_KEPT, "re-source": 0, "promote": 0, "retire": 40 + 88,
        "none": 0,
    }
    assert r1["retire_by_why"] == {"own": 40, "derived": 88}
    assert (r1["removed"], r1["retained"], r1["rows_after"], r1["rows_from_scratch"]) == (
        128, 337, 337, 337,
    )
    assert r1["generated"]["actions"] == {
        "keep": 2, "re-source": 0, "promote": 0, "retire": 88, "none": 0,
    }
    assert r1["ids"]["retire"] == [
        "A-000001..A-000040", "P-000001..P-000040", "T-000001..T-000040", "S-000001..S-000008",
    ]
    assert result["actions"]["records"]["retire"] == 128
    assert all(identity["holds"] for identity in result["identities"])


FIVE_OUTCOMES = {"keep", "re-source", "promote", "retire", "none"}


def test_no_release_row_has_the_outcome_remove(small, small_off):
    """DECISIONS.md D-19: a release row that copies A's own record and has no
    equivalent copy to take its place is retired. No outcome is called "remove"."""
    assert set(cnt.OUTCOMES) == FIVE_OUTCOMES
    assert not hasattr(gen, "ACT_REMOVE") and not hasattr(cnt, "REMOVE")
    for sf in (small, small_off):
        stated = {entry["withdraw_A"] for row in sf.plant for entry in row["releases"].values()}
        assert stated <= FIVE_OUTCOMES, stated
        result = cnt.count(sf.folder)
        for release in result["releases"].values():
            assert set(release["actions"]) == FIVE_OUTCOMES
            assert "remove" not in release["ids"]
        assert set(result["actions"]["records"]) == FIVE_OUTCOMES
    for k in range(1, N_X + 1):
        # A's copy of X with dedup off: a row of its own, and so is B's copy.
        assert small_off.by_id[f"A-{k:06d}"]["releases"]["R1"] == {
            "row": "present", "withdraw_A": "retire",
        }
        assert small_off.by_id[f"B-{k:06d}"]["releases"]["R1"] == {
            "row": "present", "withdraw_A": "keep",
        }


def test_a_plant_row_that_says_remove_is_refused(small_off, tmp_path):
    def change(by_id, rows):
        by_id["A-000003"]["releases"]["R1"]["withdraw_A"] = "remove"

    folder = _mutated(small_off, tmp_path, change)
    with pytest.raises(cnt.PlantListInconsistent, match="'remove' is not an outcome"):
        cnt.count(folder)


def test_check6_candidates_name_their_sources(small):
    found = cnt.count(small.folder)["check6"]["candidates"]
    assert len(found) == N_X + PAIRS and {c["release"] for c in found} == {"R1"}
    x = [c for c in found if c["pair"].startswith("X-")]
    assert len(x) == N_X
    for c in x:
        assert c["sources"] == c["may_contain_for"] == ["A", "B"]
        assert c["withdrawn_source_among_candidates"] is True and c["label"] == "candidates"
        assert [m["source"] for m in c["candidates"]] == ["A", "B"]
        assert c["rows"] == [c["candidates"][0]["id"]]
    for c in found:
        if c["pair"].startswith("PAIR-"):
            assert c["sources"] == ["C", "Z"] and c["withdrawn_source_among_candidates"] is False


def test_sanitizer_drops_are_listed(small):
    drops = cnt.count(small.folder)["check3"]["sanitizer_drops"]
    assert drops["R2"] == []
    assert Counter(d["reason"] for d in drops["R1"]) == {"sanitizer skip": 3, "empty line": 3}
    for d in drops["R1"]:
        assert d["source"] == "Z" and d["id"] == f"Z-{d['line']:06d}"
        assert d["raw_sha256"] == sha(small.raw[d["id"]]) and small.text(d["id"]) is None


def test_cli_prints_one_json_document(small):
    proc = subprocess.run(
        [sys.executable, str(PLANT / "count.py"), "--set", str(small.folder)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    assert json.loads(proc.stdout) == cnt.count(small.folder)


def _mutated(small: SetFiles, tmp_path: Path, change) -> Path:
    """A copy of the small set whose plant list has one row changed."""
    folder = tmp_path / "mutated"
    shutil.copytree(small.folder, folder)
    rows = [json.loads(line) for line in (folder / "plant_list.jsonl").read_text().splitlines()]
    rows = change({row["id"]: row for row in rows}, rows) or rows
    (folder / "plant_list.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return folder


def _scope_flag(by_id, rows):
    by_id["P-000003"]["withdraw_A"] = {"in_scope": False, "why": None, "roots": []}


def _kept_x(by_id, rows):
    by_id["A-000002"]["releases"]["R1"] = {"row": "present", "withdraw_A": "keep"}


def _unpromoted(by_id, rows):
    by_id["B-000002"]["releases"]["R1"] = {"row": "dedup-loser", "withdraw_A": "none"}


def _kept_summary(by_id, rows):
    by_id["S-000001"]["releases"]["R1"]["withdraw_A"] = "keep"


def _missing_parent(by_id, rows):
    by_id["T-000001"]["parents"] = ["P-999999"]


def _other_parent(by_id, rows):
    by_id["S-000002"]["parents"] = ["C-000001", "C-000002"]


def _repeated_text(by_id, rows):
    by_id["Z-000300"]["text_sha256"] = by_id["Z-000301"]["text_sha256"]


def _dropped_row(by_id, rows):
    return [row for row in rows if row["id"] != "Z-000400"]


def _wrong_release(by_id, rows):
    by_id["Z-000001"]["releases"] = {"R2": {"row": "present", "withdraw_A": "keep"}}


def _unpaired(by_id, rows):
    by_id["B-000005"]["twin"] = "A-000006"


def _unskipped(by_id, rows):
    row = next(r for r in rows if r["sanitizer"] and r["sanitizer"]["skip"])
    row["sanitizer"]["skip"] = False
    row["sanitizer"]["reason"] = None


def _kept_retired(by_id, rows):
    by_id["UP-000001"]["releases"]["R1"]["withdraw_A"] = "retire"


def _kept_in_scope(by_id, rows):
    by_id["UP-000002"]["withdraw_A"] = {"in_scope": True, "why": "derived", "roots": ["A-000001"]}


def _kept_made_from_a(by_id, rows):
    # A row that agrees with itself: made from A's record, in scope, retired. Only
    # what it is called is wrong.
    row = by_id["UP-000001"]
    row["parents"] = ["A-000001"]
    row["withdraw_A"] = {"in_scope": True, "why": "derived", "roots": ["A-000001"]}
    row["releases"]["R1"]["withdraw_A"] = "retire"


def _kept_from_a_passage_of_c(by_id, rows):
    by_id["UP-000001"]["parents"] = ["C-000001"]


def _kept_parent_in_the_other_release(by_id, rows):
    by_id["UP-000001"]["parents"] = ["Z-000500"]  # an unrelated line, an input of R2


@pytest.mark.parametrize(
    "change",
    [_scope_flag, _kept_x, _unpromoted, _kept_summary, _missing_parent, _other_parent,
     _repeated_text, _dropped_row, _wrong_release, _unpaired, _unskipped,
     _kept_retired, _kept_in_scope, _kept_made_from_a, _kept_from_a_passage_of_c,
     _kept_parent_in_the_other_release],
)
def test_an_inconsistent_plant_row_fails_loudly(small, tmp_path, change):
    folder = _mutated(small, tmp_path, change)
    with pytest.raises(cnt.PlantListInconsistent):
        cnt.count(folder)
    proc = subprocess.run(
        [sys.executable, str(PLANT / "count.py"), "--set", str(folder)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3 and proc.stdout == ""
    assert json.loads(proc.stderr)["error"] == "PlantListInconsistent"


# --- The generated records that stay (plan 5.1 "Sizes", DECISIONS.md D-23) ----------


def test_paraphrases_of_unrelated_records_are_planted_in_r1(small):
    kept = [row for row in small.plant if row["kind"] == "paraphrase-of-unrelated"]
    assert [row["id"] for row in kept] == KEPT
    r1 = expand(small.cfg["releases"]["R1"]["inputs"])
    r2 = expand(small.cfg["releases"]["R2"]["inputs"])
    parents = []
    for number, row in enumerate(kept, start=1):
        (parent,) = row["parents"]
        parents.append(parent)
        made = small.made[row["id"]]
        # Made like P: the job "paraphrase" is handed one record and uses it.
        assert (made["job"], made["handed"], made["used"]) == ("paraphrase", [parent], [parent])
        assert row == {
            "id": row["id"], "file": "generation.jsonl", "line": N_X + number,
            "source": None, "kind": "paraphrase-of-unrelated", "job": "paraphrase",
            "parents": [parent], "parent_mode": "all", "pair": None, "twin": None,
            "withdraw_A": {"in_scope": False, "why": None, "roots": []},
            "releases": {"R1": {"row": "present", "withdraw_A": "keep"}},
            "sanitizer": None, "text_sha256": sha(small.text(row["id"])),
            "chars": len(small.text(row["id"])),
        }
        # The parent: an unrelated record of Z that is itself a row of R1 and stays.
        origin = small.by_id[parent]
        assert (origin["source"], origin["kind"], origin["parents"]) == ("Z", "unrelated", [])
        assert origin["line"] <= Z_R1 and origin["sanitizer"] is None and origin["pair"] is None
        assert origin["releases"] == {"R1": {"row": "present", "withdraw_A": "keep"}}
        assert parent in r1 and row["id"] in r1 and row["id"] not in r2
        # Its first sentence names its own code word and its parent's.
        parent_word = small.text(parent).split()[1]
        first = small.text(row["id"]).split(". ")[0]
        assert first.startswith("Paraphrase ") and first.endswith(f"restates entry {parent_word}")
    assert len(set(parents)) == N_KEPT
    # No id can be taken for one of A's paraphrases: nothing else in the set starts
    # with their first letter, and they do not start with P.
    assert [record for record in small.raw if record.startswith("U")] == KEPT
    assert {small.by_id[f"P-{k:06d}"]["parents"][0] for k in range(1, N_X + 1)} == {
        f"A-{k:06d}" for k in range(1, N_X + 1)
    }


@pytest.mark.parametrize("records, derivations, want", [
    (800, 1, 2), (1999, 1, 2), (3000, 1, 3), (10_000, 1, 10), (10_000, 2, 10),
    (100_000, 1, 100), (10_000, 0, 0), (100_000, 0, 0),
])
def test_how_many_paraphrases_of_unrelated_records_a_set_holds(records, derivations, want):
    params = gen.Params(records=records, seed=1, derivations=derivations)
    assert gen.kept_paraphrases(params) == want


def test_layout_of_the_two_sets_the_checks_use():
    ten = gen.plan_layout(gen.Params(records=10_000, seed=1))
    assert ten.lines == {"A": 600, "B": 1000, "C": 500, "Z": 6790}
    assert (ten.n_p, ten.n_sum, ten.n_kept, ten.n_derived, ten.n_generated) == (
        500, 100, 10, 1100, 1110,
    )
    assert ten.z_r1 == {"Z": 3395}
    hundred = gen.plan_layout(gen.Params(records=100_000, seed=1, dedup=False))
    assert hundred.lines == {"A": 6000, "B": 10000, "C": 2300, "Z": 70600}
    assert (hundred.n_p, hundred.n_sum, hundred.n_kept, hundred.n_derived, hundred.n_generated) == (
        5000, 1000, 100, 11000, 11100,
    )
    assert hundred.z_r1 == {"Z": 35300}
    for layout, total in ((ten, 10_000), (hundred, 100_000)):
        # The new records are inside the total.
        assert sum(layout.lines.values()) + layout.n_generated == total
        # Their parents: distinct lines of Z's part of R1 that hold nothing else planted.
        chosen = layout.z_kept_parent
        assert len(set(chosen)) == len(chosen) == layout.n_kept
        assert all(1 <= line <= layout.z_r1["Z"] for line in chosen)
        assert not set(chosen) & (set(layout.z_pair_line) | set(layout.z_case_line))


ZERO = {"re-source": 0, "promote": 0, "retire": 0, "none": 0}


def test_counts_of_the_10k_set(tmp_path):
    """The set of plan Check 9: 10,000 records, middle length, 1 derivation, 4
    sources, dedup on. The numbers are those of docs/plant.md section 7.1."""
    summary = gen.generate(tmp_path / "s10k", records=10_000, seed=1)
    assert summary["parameters"] == {
        "records": 10_000, "length": 2000, "derivations": 1, "sources": 4, "seed": 1,
        "dedup": True, "chunks": 20, "pairs": 300, "sanitizer_cases": 6, "clean": False,
    }
    assert summary["files"]["generation.jsonl"]["lines"] == 1110
    assert summary["files"]["sources/Z.jsonl"]["lines"] == 6790
    result = cnt.count(tmp_path / "s10k")
    assert result["records"] == {
        "total": 10_000,
        "by_kind": {
            "pair-member": 600, "paraphrase": 500, "paraphrase-of-unrelated": 10,
            "passage": 200, "sanitizer-case": 18, "summary": 100, "translation": 500,
            "unrelated": 7072, "x-copy": 1000,
        },
        "by_source": {"A": 600, "B": 1000, "C": 500, "Z": 6790, "generated": 1110},
        "in_scope": {"total": 1700, "own": 600, "derived": 1100},
        "removed": 1700, "retained": 8300,
    }
    r1, r2 = result["releases"]["R1"], result["releases"]["R2"]
    assert (r1["inputs"], r1["rows_before"], r1["dedup_losers"]) == (5805, 5005, 800)
    assert r1["actions"] == {
        "keep": 3405, "re-source": 500, "promote": 500, "retire": 1100, "none": 300,
    }
    assert r1["retire_by_why"] == {"own": 0, "derived": 1100}
    assert (r1["removed"], r1["retained"], r1["rows_after"], r1["rows_from_scratch"]) == (
        1100, 3905, 3905, 3905,
    )
    assert r1["generated"] == {
        "inputs": 1110, "out_of_scope": 10,
        "actions": {"keep": 10, "re-source": 0, "promote": 0, "retire": 1100, "none": 0},
        "kept_ids": ["UP-000001..UP-000010"],
    }
    assert r1["ids"] == {
        "re-source": ["A-000001..A-000500"], "promote": ["B-000001..B-000500"],
        "retire": ["P-000001..P-000500", "T-000001..T-000500", "S-000001..S-000100"],
    }
    assert (r2["inputs"], r2["actions"]) == (4395, {"keep": 4395, **ZERO})
    assert result["actions"]["records"] == {
        "keep": 7800, "re-source": 500, "promote": 500, "retire": 1100, "none": 300,
    }
    generated = result["generated"]
    assert (generated["total"], generated["derived"], generated["stay"]["count"]) == (1110, 1100, 10)
    assert generated["stay"]["ids"] == ["UP-000001..UP-000010"]
    assert Counter(d["reason"] for d in result["check3"]["sanitizer_drops"]["R1"]) == {
        "sanitizer skip": 6, "empty line": 6,
    }
    assert all(identity["holds"] for identity in result["identities"])
    # The counts follow from the parameters: another seed and another record length
    # move none of them.
    gen.generate(tmp_path / "other", records=10_000, seed=5, length=120)
    other = cnt.count(tmp_path / "other")
    for part in ("records", "releases", "actions"):
        assert other[part] == result[part]
    assert other["generated"]["stay"]["ids"] == generated["stay"]["ids"]


@pytest.mark.slow
def test_counts_of_the_100k_set(tmp_path):
    """The set of plan Check 3: 100,000 records, 1 derivation, 4 sources, dedup off.
    It is written here with short records; test_counts_of_the_10k_set shows that the
    counts do not depend on the length. The numbers are those of docs/plant.md 7.1."""
    gen.generate(tmp_path / "s100k", records=100_000, seed=1, dedup=False, length=120)
    result = cnt.count(tmp_path / "s100k")
    assert result["records"] == {
        "total": 100_000,
        "by_kind": {
            "pair-member": 600, "paraphrase": 5000, "paraphrase-of-unrelated": 100,
            "passage": 2000, "sanitizer-case": 18, "summary": 1000, "translation": 5000,
            "unrelated": 76282, "x-copy": 10000,
        },
        "by_source": {"A": 6000, "B": 10000, "C": 2300, "Z": 70600, "generated": 11100},
        "in_scope": {"total": 17000, "own": 6000, "derived": 11000},
        "removed": 17000, "retained": 83000,
    }
    r1, r2 = result["releases"]["R1"], result["releases"]["R2"]
    assert (r1["inputs"], r1["rows_before"], r1["dedup_losers"]) == (56700, 56700, 0)
    assert r1["actions"] == {
        "keep": 40700, "re-source": 0, "promote": 0, "retire": 16000, "none": 0,
    }
    assert r1["retire_by_why"] == {"own": 5000, "derived": 11000}
    assert (r1["removed"], r1["retained"], r1["rows_after"], r1["rows_from_scratch"]) == (
        16000, 40700, 40700, 40700,
    )
    assert r1["generated"] == {
        "inputs": 11100, "out_of_scope": 100,
        "actions": {"keep": 100, "re-source": 0, "promote": 0, "retire": 11000, "none": 0},
        "kept_ids": ["UP-000001..UP-000100"],
    }
    assert (r2["inputs"], r2["actions"]) == (45300, {"keep": 45300, **ZERO})
    assert result["actions"]["records"] == {
        "keep": 86000, "re-source": 0, "promote": 0, "retire": 16000, "none": 0,
    }
    assert result["generated"]["stay"]["ids"] == ["UP-000001..UP-000100"]
    assert len(result["check3"]["sanitizer_drops"]["R1"]) == 12
    assert all(identity["holds"] for identity in result["identities"])


def test_a_generated_record_that_does_not_derive_from_a_is_never_counted_as_retired(
        small, small_off):
    """DECISIONS.md D-23. Which generated records come from A is worked out here from
    the files alone. The counting script must retire exactly those, and print the
    others as the generated records that stay. A script that retired every generated
    record fails here."""
    for sf, retired_rows in ((small, 88), (small_off, 40 + 88)):
        generated = [row["id"] for row in sf.generation]
        from_a = {record for record in generated if comes_from(sf, record, "A")}
        apart = [record for record in generated if record not in from_a]
        assert apart == KEPT and len(from_a) == 88

        result = cnt.count(sf.folder, expand_ids=True)
        r1 = result["releases"]["R1"]
        retired = set(r1["ids"]["retire"])
        assert not retired & set(apart), "a generated record that does not derive from A is retired"
        assert retired & set(generated) == from_a
        assert r1["actions"]["retire"] == len(retired) == retired_rows
        assert r1["generated"]["actions"] == {
            "keep": N_KEPT, "re-source": 0, "promote": 0, "retire": 88, "none": 0,
        }
        assert r1["generated"]["kept_ids"] == KEPT

        part = result["generated"]
        assert (part["total"], part["derived"]) == (90, 88)
        assert (part["stay"]["count"], part["stay"]["ids"]) == (N_KEPT, KEPT)
        assert part["stay"]["list"] == [
            {
                "id": record, "kind": "paraphrase-of-unrelated", "job": "paraphrase",
                "parents": sf.made[record]["used"], "parent_mode": "all", "in_scope": False,
                "releases": {"R1": "keep"}, "lines": None,
            }
            for record in KEPT
        ]
        # The clean sources do not omit them, and the successor holds them: its rows
        # are the test's own from-scratch build of the inputs that do not come from A.
        assert not set(result["clean_omits"]["ids"]) & set(apart)
        entry = sf.cfg["releases"]["R1"]
        survivors = [r for r in expand(entry["inputs"]) if not comes_from(sf, r, "A")]
        assert set(apart) <= set(survivors)
        rows = (
            len({sf.text(r) or f"line:{r}" for r in survivors}) if entry["dedup"] else len(survivors)
        )
        assert r1["rows_after"] == rows == (313 if entry["dedup"] else 337)


@pytest.mark.parametrize("change, message", [
    (_kept_retired, "UP-000001.*'withdraw_A': 'retire'.*structure gives.*'withdraw_A': 'keep'"),
    (_kept_in_scope, "UP-000002.*states scope.*'in_scope': True.*give.*'in_scope': False"),
    (_kept_made_from_a, "called a paraphrase of an unrelated record, and its parents lead to "
                        "source A"),
    (_kept_from_a_passage_of_c, "exactly one parent, an unrelated line"),
    (_kept_parent_in_the_other_release, "Z-000500 is not a row of release R1"),
])
def test_a_plant_row_that_misstates_a_paraphrase_of_an_unrelated_record_is_refused(
        small, tmp_path, change, message):
    with pytest.raises(cnt.PlantListInconsistent, match=message):
        cnt.count(_mutated(small, tmp_path, change))


def _recount_kinds(folder: Path, changes: dict[str, str | None]) -> None:
    """Change lines of the [counts] table of a mutated set, so that a test reaches
    the check it is about and not the count of rows per kind."""
    path = folder / "set.toml"
    text = path.read_text(encoding="utf-8")
    for old, new in changes.items():
        assert text.count(old + "\n") == 1, old
        text = text.replace(old + "\n", "" if new is None else new + "\n")
    path.write_text(text, encoding="utf-8")


def test_what_a_generated_record_is_called_must_agree_with_its_parents(small, tmp_path):
    def rename(by_id, rows):
        by_id["UP-000001"]["kind"] = "paraphrase"

    folder = _mutated(small, tmp_path, rename)
    _recount_kinds(folder, {
        '"paraphrase" = 40': '"paraphrase" = 41',
        '"paraphrase-of-unrelated" = 2': '"paraphrase-of-unrelated" = 1',
    })
    with pytest.raises(
        cnt.PlantListInconsistent,
        match="UP-000001.*called a paraphrase made from records of source A, and its parents "
              "do not lead there",
    ):
        cnt.count(folder)


def test_a_set_in_which_every_generated_record_derives_from_a_is_refused(small, tmp_path):
    """The set as it was before DECISIONS.md D-23, written as a plant list that
    agrees with itself: every generated record is made from a record of A."""
    def derive_all(by_id, rows):
        for k, record in enumerate(KEPT, start=1):
            row = by_id[record]
            row["kind"] = "paraphrase"
            row["parents"] = [f"A-{k:06d}"]
            row["withdraw_A"] = {"in_scope": True, "why": "derived", "roots": [f"A-{k:06d}"]}
            row["releases"]["R1"]["withdraw_A"] = "retire"

    folder = _mutated(small, tmp_path, derive_all)
    _recount_kinds(folder, {
        '"paraphrase" = 40': '"paraphrase" = 42',
        '"paraphrase-of-unrelated" = 2': None,
    })
    with pytest.raises(
        cnt.NoGeneratedRecordStays,
        match="release R1 retires 90 generated records and keeps none",
    ):
        cnt.count(folder)
    proc = subprocess.run(
        [sys.executable, str(PLANT / "count.py"), "--set", str(folder)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3 and proc.stdout == ""
    assert json.loads(proc.stderr)["error"] == "NoGeneratedRecordStays"


# --- Chunk files -------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("start", [lambda k: 0, lambda k: 7 * k + 3, lambda k: 19])
def test_a_reaches_every_chunk_when_files_are_dealt_one_by_one(small, tmp_path, seed, start):
    shards = deal_round_robin(small, "R1", seed, start)
    for ids in shards.values():
        assert sum(1 for record in ids if record.startswith("A-")) >= N_X // 20
    result = cnt.count(small.folder, chunks=write_chunks(small, tmp_path / "rr", shards),
                       layout="round-robin")
    part = result["shards"]
    assert part["all_rewrite"] is True and part["keep"] == []
    assert part["rewrite"] == [f"shard-{c}" for c in range(1, 21)]
    assert result["actions"]["shards"] == {"keep": 0, "rewrite": 20}
    assert part["lines"] == 401 - 6  # the release's rows less the six sanitizer drops
    # Each generated record that stays is found in the chunk files, in a chunk that
    # is rewritten for other reasons, and is itself no reason to rewrite one.
    assert sum(entry["generated_kept"] for entry in part["shards"]) == N_KEPT
    for entry in result["generated"]["stay"]["list"]:
        (line,) = entry["lines"]
        assert entry["id"] in shards[line["shard"]] and line["shard_action"] == "rewrite"
        assert line["ordinal"] == line["line"] - 1
    for entry in part["shards"]:
        ids = shards[entry["shard"]]
        assert entry["own"] == sum(1 for r in ids if r.startswith("A-"))
        assert entry["derived"] == sum(1 for r in ids if r[0] in "PTS")
        assert entry["receives"] == entry["own"]  # each X row of A brings B's copy in
        assert entry["lines"] == sum(1 for r in ids if small.text(r) is not None)
    # With dedup on, one member of each pair never reaches a shard, and that is said.
    pairs = result["check3"]["pairs"]
    assert pairs["required"] is False
    assert pairs["counts"]["pair-member"]["single"] == PAIRS
    assert pairs["counts"]["x-copy"]["single"] == N_X
    with pytest.raises(cnt.PairKindsMissing):
        cnt.count(small.folder, chunks=tmp_path / "rr", layout="round-robin",
                  require_pair_kinds="yes")


def test_round_robin_must_rewrite_every_chunk(small, tmp_path):
    shards = deal_round_robin(small, "R1", 4)
    clean_shard = "shard-9"
    for record in [r for r in shards[clean_shard] if r[0] in "APTS"]:
        move(shards, record, "shard-10")
    with pytest.raises(cnt.LayoutExpectationFailed, match="shard-9"):
        cnt.count(small.folder, chunks=write_chunks(small, tmp_path / "gap", shards),
                  layout="round-robin")
    shards["shard-20"] += shards.pop("shard-19")
    with pytest.raises(cnt.LayoutExpectationFailed, match="19 shards"):
        cnt.count(small.folder, chunks=write_chunks(small, tmp_path / "few", shards),
                  layout="round-robin")


def test_pairs_are_classified_from_the_chunk_ids(small_off, tmp_path):
    sf = small_off
    shards = deal_round_robin(sf, "R1", 11, lambda k: 5 * k)
    pairs = planted_pairs(sf)
    together, apart = pairs[0], pairs[1]
    move(shards, together[1], shard_of(shards, together[0]))
    if shard_of(shards, apart[0]) == shard_of(shards, apart[1]):
        other = next(s for s in shards if s != shard_of(shards, apart[0]))
        move(shards, apart[1], other)
    chunks = write_chunks(sf, tmp_path / "pairs", shards)
    result = cnt.count(sf.folder, chunks=chunks, layout="round-robin")
    part = result["check3"]["pairs"]
    assert part["required"] is True  # dedup is off in this set
    counts = part["counts"]["pair-member"]
    assert counts["same-shard"] >= 1 and counts["split"] >= 1
    assert counts["same-shard"] + counts["split"] == PAIRS
    assert sum(part["counts"]["x-copy"].values()) == N_X
    listed = {entry["pair"]: entry for entry in part["list"]}
    lines = {
        shard: [json.loads(line)["id"] for line in (chunks / f"{shard}.jsonl").read_text().splitlines()]
        for shard in shards
    }
    for pair, want in ((together, "same-shard"), (apart, "split")):
        entry = listed[sf.by_id[pair[0]]["pair"]]
        assert entry["placement"] == want and entry["kind"] == "pair-member"
        assert sorted(m["id"] for m in entry["members"]) == sorted(pair)
        for member in entry["members"]:
            assert lines[member["shard"]][member["line"] - 1] == member["id"]
            assert member["ordinal"] == member["line"] - 1
            assert member["expected_label"] == ("candidates" if want == "same-shard" else "recovered")
    # Every pair's placement agrees with where the test put its members.
    for entry in part["list"]:
        a, b = (m["id"] for m in entry["members"])
        same = shard_of(shards, a) == shard_of(shards, b)
        assert entry["placement"] == ("same-shard" if same else "split")

    # Take every same-shard planted pair apart: the check has nothing to stand on.
    for c, z in pairs:
        if shard_of(shards, c) == shard_of(shards, z):
            move(shards, z, next(s for s in shards if s != shard_of(shards, c)))
    with pytest.raises(cnt.PairKindsMissing, match="0 same-shard"):
        cnt.count(sf.folder, chunks=write_chunks(sf, tmp_path / "apart", shards),
                  layout="round-robin")
    quiet = cnt.count(sf.folder, chunks=tmp_path / "apart", layout="round-robin",
                      require_pair_kinds="no")
    assert quiet["check3"]["pairs"]["counts"]["pair-member"]["same-shard"] == 0


def test_partitioned_layout_gives_the_kept_shard_list(small, tmp_path):
    shards = deal_partitioned(small, "R1")
    chunks = write_chunks(small, tmp_path / "parts", shards)
    (chunks / "A-part-1.jsonl.manifest.jsonl").write_text('{"line": 1}\n')
    (chunks / "A-part-1.bin").write_bytes(b"\x00")
    result = cnt.count(small.folder, chunks=chunks, layout="partitioned")
    part = result["shards"]
    # A shard that holds nothing but paraphrases of unrelated records is kept too.
    want_keep = sorted(s for s in shards if s[0] in "CZU")
    want_rewrite = sorted(s for s in shards if s[0] in "APTS")
    assert "UP-part-1" in want_keep
    kept_shard = next(e for e in part["shards"] if e["shard"] == "UP-part-1")
    assert (kept_shard["lines"], kept_shard["generated_kept"], kept_shard["in_scope_lines"],
            kept_shard["action"]) == (N_KEPT, N_KEPT, 0, "keep")
    assert sum(e["generated_kept"] for e in part["shards"]) == N_KEPT
    for number, entry in enumerate(result["generated"]["stay"]["list"], start=1):
        # The list says where each one is, so a test can aim at one.
        assert entry["lines"] == [
            {"shard": "UP-part-1", "line": number, "ordinal": number - 1, "shard_action": "keep"}
        ]
        there = (chunks / "UP-part-1.jsonl").read_text(encoding="utf-8").splitlines()[number - 1]
        assert json.loads(there)["id"] == entry["id"] == KEPT[number - 1]
    assert sorted(part["keep"]) == want_keep and sorted(part["rewrite"]) == want_rewrite
    assert want_keep and want_rewrite and part["all_rewrite"] is False
    assert result["actions"]["shards"] == {"keep": len(want_keep), "rewrite": len(want_rewrite)}
    receiving = [e["shard"] for e in part["shards"] if e["receives"]]
    assert receiving == ["A-part-1"]  # the 40 re-sourced rows sit there

    # One affected record in every shard: nothing can be kept, and that is refused.
    spread = {name: list(ids) for name, ids in shards.items()}
    affected = [r for r in spread["P-part-1"]]
    for name, record in zip(want_keep, affected):
        move(spread, record, name)
    with pytest.raises(cnt.LayoutExpectationFailed, match="kept-shard list is empty"):
        cnt.count(small.folder, chunks=write_chunks(small, tmp_path / "spread", spread),
                  layout="partitioned")


def _rewrite_line(path: Path, index: int, row: dict | None) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if row is None:
        del lines[index]
    else:
        lines[index] = json.dumps(row, ensure_ascii=False)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_chunk_files_must_hold_exactly_the_surviving_rows(small, tmp_path):
    shards = deal_round_robin(small, "R1", 6)

    def fresh(name: str) -> Path:
        return write_chunks(small, tmp_path / name, shards)

    def fails(chunks: Path, match: str) -> None:
        with pytest.raises(cnt.ChunkFilesMismatch, match=match):
            cnt.count(small.folder, chunks=chunks, layout="round-robin")

    chunks = fresh("missing")
    _rewrite_line(chunks / "shard-3.jsonl", 0, None)
    fails(chunks, "1 missing")

    chunks = fresh("text")
    row = json.loads((chunks / "shard-3.jsonl").read_text(encoding="utf-8").splitlines()[1])
    _rewrite_line(chunks / "shard-3.jsonl", 1, {"id": row["id"], "text": row["text"] + " More."})
    fails(chunks, "not that record's sanitized text")

    chunks = fresh("unknown")
    _rewrite_line(chunks / "shard-3.jsonl", 1, {"id": "Q-000001", "text": "x"})
    fails(chunks, "no plant row")

    skipped = next(r["id"] for r in small.plant if r["sanitizer"] and r["sanitizer"]["skip"])
    chunks = fresh("skipped")
    with open(chunks / "shard-3.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": skipped, "text": ""}) + "\n")
    fails(chunks, "must skip")

    loser = "B-000001"  # dropped by dedup, so not a row of R1
    chunks = fresh("loser")
    with open(chunks / "shard-3.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": loser, "text": small.text(loser)}) + "\n")
    fails(chunks, "1 not expected")

    chunks = fresh("junk")
    with open(chunks / "shard-3.jsonl", "a", encoding="utf-8") as f:
        f.write("not a row\n")
    fails(chunks, "surviving lines only")

    with pytest.raises(cnt.UsageError):
        cnt.count(small.folder, chunks=fresh("nolayout"))
    with pytest.raises(cnt.UsageError):
        cnt.count(small.folder, releases=["R1"])


def test_chunks_of_two_releases(small, tmp_path):
    shards = deal_round_robin(small, "R1", 2)
    for j, record in enumerate(small.release_rows("R2")):
        shards[f"shard-{j % 20 + 1}"].append(record)
    result = cnt.count(small.folder, chunks=write_chunks(small, tmp_path / "both", shards),
                       layout="round-robin", releases=["R1", "R2"])
    assert result["shards"]["lines"] == 395 + 351
    # B's copies of X reach a shard through R2, so each X pair now has two lines.
    x = result["check3"]["pairs"]["counts"]["x-copy"]
    assert x["same-shard"] + x["split"] == N_X and x["single"] == 0
    with pytest.raises(cnt.ChunkFilesMismatch):
        cnt.count(small.folder, chunks=tmp_path / "both", layout="round-robin")


# --- What the rows of Checks 3 and 6 on a generated set are asked of --------------------


def write_sanitizer_input(sf: SetFiles, out: Path, shards: dict[str, list[str]]) -> Path:
    """Sanitizer-input files: every line of a chunk as the sanitizer reads it,
    byte for byte, readable or not."""
    out.mkdir(parents=True)
    for shard, ids in shards.items():
        (out / f"{shard}.jsonl").write_bytes(
            "".join(sf.raw[record] + "\n" for record in ids).encode("utf-8"))
    return out


def chunk_ids(chunks: Path, shards) -> dict[str, list[str]]:
    """The ids of the lines of each chunk file, read by the test itself."""
    return {
        shard: [json.loads(line)["id"]
                for line in (chunks / f"{shard}.jsonl").read_text(encoding="utf-8").splitlines()]
        for shard in shards
    }


def test_the_count_format_names_the_parts_checks_3_and_6_read(small):
    result = cnt.count(small.folder)
    assert result["format"] == "waterwhite.plant-count/4"
    # The parts that need the chunk files say so by being null without them.
    assert result["documents"] is None and result["check3"]["drops"] is None
    assert result["check6"]["translations"]["shards"] is None


def test_translations_are_listed_with_the_releases_and_shards_that_receive_each(small, tmp_path):
    sf = small
    made = [row["id"] for row in sf.generation if row["job"] == "translation"]
    assert len(made) == N_X
    part = cnt.count(sf.folder)["check6"]["translations"]
    assert part["count"] == N_X and expand(part["ids"]) == made
    assert part["file"] == "generation.jsonl" and part["job"] == "translation"
    in_r1, in_r2 = set(sf.release_rows("R1")), set(sf.release_rows("R2"))
    assert all(record in in_r1 and record not in in_r2 for record in made)
    assert part["releases"] == {"R1": N_X, "R2": 0}
    assert [entry["id"] for entry in part["list"]] == made
    for entry in part["list"]:
        row = sf.made[entry["id"]]
        # Made from a paraphrase, which was made from one X copy of A.
        assert entry["parents"] == row["used"] and len(row["used"]) == 1
        assert entry["roots"] == sf.made[row["used"][0]]["used"]
        assert entry["releases"] == ["R1"] and entry["lines"] is None
    assert part["shards"] is None and part["shards_without"] is None

    # With the chunk files: the shards that receive each, from the chunk-file ids.
    shards = deal_round_robin(sf, "R1", 5)
    # One chunk is left without any translation, so that both lists hold something.
    for record in [r for r in shards["shard-4"] if r.startswith("T-")]:
        move(shards, record, "shard-5")
    chunks = write_chunks(sf, tmp_path / "chunks", shards)
    part = cnt.count(sf.folder, chunks=chunks, layout="round-robin")["check6"]["translations"]
    lines = chunk_ids(chunks, shards)
    holding: Counter = Counter()
    for entry in part["list"]:
        (line,) = entry["lines"]
        assert lines[line["shard"]][line["line"] - 1] == entry["id"]
        assert line["ordinal"] == line["line"] - 1
        holding[line["shard"]] += 1
    want = {shard: sum(1 for r in ids if r.startswith("T-")) for shard, ids in lines.items()}
    assert {e["shard"]: e["translations"] for e in part["shards"]} == {
        shard: n for shard, n in want.items() if n} == dict(holding)
    assert part["shards_without"] == ["shard-4"]
    # A translation is listed once; none is missing.
    assert sum(holding.values()) == N_X


def test_documents_name_the_planted_record_of_every_chunk_line(small_off, tmp_path):
    sf = small_off
    shards = deal_round_robin(sf, "R1", 11, lambda k: 5 * k)
    together = planted_pairs(sf)[0]
    move(shards, together[1], shard_of(shards, together[0]))
    chunks = write_chunks(sf, tmp_path / "chunks", shards)
    result = cnt.count(sf.folder, chunks=chunks, layout="round-robin")
    part = result["documents"]
    lines = chunk_ids(chunks, shards)
    assert [entry["shard"] for entry in part["shards"]] == sorted(lines, key=cnt._natural)
    assert part["total"] == sum(len(ids) for ids in lines.values()) == result["shards"]["lines"]
    twins_seen = 0
    for entry in part["shards"]:
        ids = lines[entry["shard"]]
        # Line i of the chunk file is document i - 1: the ids, in order.
        assert entry["ids"] == ids and entry["documents"] == len(ids)
        # The documents with identical text in this shard, by the test's own reading
        # of the files.
        by_text: dict[str, list[int]] = {}
        for ordinal, record in enumerate(ids):
            by_text.setdefault(sf.text(record), []).append(ordinal)
        want = sorted(ordinals for ordinals in by_text.values() if len(ordinals) > 1)
        assert sorted(group["ordinals"] for group in entry["identical"]) == want
        for group in entry["identical"]:
            assert group["ids"] == [ids[o] for o in group["ordinals"]]
            assert sf.by_id[group["ids"][0]]["pair"] == group["pair"]
        twins = sum(len(ordinals) for ordinals in want)
        assert entry["labels"] == {"recovered": len(ids) - twins, "candidates": twins}
        twins_seen += twins
    # The pair the test put together is among them, and the pair list agrees.
    there = next(e for e in part["shards"] if e["shard"] == shard_of(shards, together[0]))
    assert sorted(together) in [sorted(group["ids"]) for group in there["identical"]]
    same_shard = [p for p in result["check3"]["pairs"]["list"] if p["placement"] == "same-shard"]
    assert twins_seen == 2 * len(same_shard) > 0


def test_drops_are_placed_in_their_shard_from_the_sanitizer_input(small, tmp_path):
    sf = small
    shards = deal_round_robin(sf, "R1", 6)
    chunks = write_chunks(sf, tmp_path / "chunks", shards)
    raw = write_sanitizer_input(sf, tmp_path / "raw", shards)
    result = cnt.count(sf.folder, chunks=chunks, layout="round-robin", sanitizer_input=raw)
    part = result["check3"]["drops"]
    listed = result["check3"]["sanitizer_drops"]["R1"]
    assert part["count"] == len(listed) == 6
    assert part["by_reason"] == {"sanitizer skip": 3, "empty line": 3}
    assert [entry["shard"] for entry in part["shards"]] == sorted(shards, key=cnt._natural)
    placed = []
    for entry in part["shards"]:
        ids = shards[entry["shard"]]
        skipped = [(n, record) for n, record in enumerate(ids, start=1) if sf.text(record) is None]
        assert entry["input_lines"] == len(ids)
        assert entry["surviving_lines"] == len(ids) - len(skipped)
        assert [(d["input_line"], d["id"]) for d in entry["drops"]] == skipped
        for drop in entry["drops"]:
            try:
                json.loads(sf.raw[drop["id"]])
                want = "empty line"
            except ValueError:
                want = "sanitizer skip"
            assert drop["reason"] == want and drop["raw_sha256"] == sha(sf.raw[drop["id"]])
            # The ordinal the next surviving line gets: nothing shifts.
            assert drop["ordinal_of_next_line"] == sum(
                1 for record in ids[: drop["input_line"] - 1] if sf.text(record) is not None)
        placed += [d["id"] for d in entry["drops"]]
    assert sorted(placed) == sorted(d["id"] for d in listed)

    def fails(folder: Path, match: str) -> None:
        with pytest.raises(cnt.SanitizerInputMismatch, match=match):
            cnt.count(sf.folder, chunks=chunks, layout="round-robin", sanitizer_input=folder)

    victim_shard = next(e["shard"] for e in part["shards"] if e["drops"])
    victim = next(e for e in part["shards"] if e["shard"] == victim_shard)["drops"][0]

    # A line the sanitizer must skip is not in the sanitizer's input.
    gone = {name: list(ids) for name, ids in shards.items()}
    gone[victim_shard].remove(victim["id"])
    fails(write_sanitizer_input(sf, tmp_path / "gone", gone), "1 of the lines the sanitizer must")

    # Two surviving lines change places: an ordinal would shift.
    swapped = {name: list(ids) for name, ids in shards.items()}
    kept = [i for i, record in enumerate(swapped[victim_shard]) if sf.text(record) is not None]
    a, b = kept[0], kept[1]
    swapped[victim_shard][a], swapped[victim_shard][b] = (
        swapped[victim_shard][b], swapped[victim_shard][a])
    fails(write_sanitizer_input(sf, tmp_path / "swapped", swapped), "is not the next line")

    # A line that is neither a planted skip nor a row.
    junk = write_sanitizer_input(sf, tmp_path / "junk", shards)
    with open(junk / f"{victim_shard}.jsonl", "a", encoding="utf-8") as f:
        f.write("not a row\n")
    fails(junk, "neither a line the sanitizer must skip nor a row")

    # A chunk file with no sanitizer input beside it.
    fewer = {name: ids for name, ids in shards.items() if name != victim_shard}
    fails(write_sanitizer_input(sf, tmp_path / "fewer", fewer), "no sanitizer input")

    with pytest.raises(cnt.UsageError):
        cnt.count(sf.folder, sanitizer_input=raw)
    proc = subprocess.run(
        [sys.executable, str(PLANT / "count.py"), "--set", str(sf.folder), "--chunks", str(chunks),
         "--layout", "round-robin", "--sanitizer-input", str(raw)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0 and json.loads(proc.stdout)["check3"]["drops"] == part
