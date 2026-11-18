"""What the join imports (plan 3.5: no trainer code; docs/interfaces.md 1).

The tool's one dependency beside the standard library is numpy. torch is an optional
dependency, called for a run read with the cyclic sampler and only then (the owner's
ruling on the order of a cyclic run, plan 3.5): importing the join loads no torch,
joining a sequential run loads none, and the one place that imports it is
``waterwhite/join/_shuffle.py``. Each check that loads modules runs in a fresh
interpreter, because the test process may hold other modules already.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import waterwhite

# The source tree that is imported, so a copy of src/ put first on the path is the one checked.
SRC = Path(waterwhite.__file__).resolve().parents[1]
HEAVY = ("torch", "megatron", "oracle", "replica", "triton")

PROBE = """
import sys
import waterwhite.join
from waterwhite.join import _arrays, _join, _record, _replay, _rows, _sampler, _shuffle, _store_adapter
heavy = sorted(name for name in sys.modules if name.split(".")[0] in {heavy!r})
print(",".join(heavy))
print(int("numpy" in sys.modules))
"""

# A sequential join of the hand world, and a cyclic join that stops at "cyclic order
# not recorded", each in a fresh interpreter, then the heavy modules that are loaded.
JOINS = """
import sys
from waterwhite.join import compute_join
import join_kit as kit
from pathlib import Path
folder = Path(sys.argv[1])
kit.write_cache(folder / "cache", kit.ARRAYS)
sequential = compute_join(kit.run_record(folder), verified_joins=kit.JOINS, record_dir=folder,
                          cache_dir=folder / "cache")
record = kit.cyclic_record(folder, [kit.launch(0, 2, 0, 1, 4)], kit.stand_in_orders({{0: [0, 1]}}))
record["shuffle_digests"] = []
cyclic = compute_join(record, verified_joins=kit.JOINS, record_dir=folder, cache_dir=folder / "cache")
print(sequential.route, cyclic.cause)
print(",".join(sorted(name for name in sys.modules if name.split(".")[0] in {heavy!r})))
"""


def run(code: str, *argv: str) -> list[str]:
    here = str(Path(__file__).resolve().parent)
    done = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path[:0] = [{str(SRC)!r}, {here!r}]\n{code}", *argv],
        capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_the_join_loads_numpy_and_no_trainer_code():
    heavy, numpy_loaded = run(PROBE.format(heavy=HEAVY))
    assert heavy == ""
    assert numpy_loaded == "1"


def test_joining_a_run_that_needs_no_shuffle_loads_no_torch(tmp_path):
    routes, heavy = run(JOINS.format(heavy=HEAVY), str(tmp_path))
    assert routes == "cache files cyclic order not recorded"
    assert heavy == ""


def test_the_one_place_that_imports_torch_is_the_shuffle_module():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(name.split(".")[0] == "torch" for name in names):
                found.append(path.relative_to(SRC).as_posix())
    assert found == ["waterwhite/join/_shuffle.py"]


def test_torch_is_imported_inside_the_function_that_draws_and_not_at_module_level():
    tree = ast.parse((SRC / "waterwhite" / "join" / "_shuffle.py").read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in ast.dump(node) for node in top_level)
    drawer = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "epoch_permutation")
    assert any(isinstance(node, ast.Import) and node.names[0].name == "torch"
               for node in ast.walk(drawer))
