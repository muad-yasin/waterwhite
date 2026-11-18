"""Kept data-order cache files do not count without the digests recorded at training
time: a decision of the project's owner, stricter than plan 3.5 and 4.6, which say this
of a replay only (DECISIONS.md D-41; docs/join.md section 4: "An array counts only when
its digest ... equals the digest the run record holds").

A sequential segment that kept its cache folder but whose run record holds no cache
digest has no route. Its checkpoints answer
"may contain" naming "data-order trace missing", with the reason "no recorded digest",
and never "does not contain" or "contains". Everything goes through the command line
(``register-run``, ``query withdraw``) on the hand world of ``test_register_run_command``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_register_run_command import ARRAYS, LENGTHS, kit, register, withdraw, write_run  # noqa: E402


def a_sequential_record(folder: Path) -> dict:
    return kit.run_record(folder, shards=1, arrays=ARRAYS, lengths=LENGTHS, micro=1, batch=1,
                          launches=[kit.launch(0, 1, 0, 1, 3)])


def test_kept_cache_files_do_not_count_without_a_recorded_digest(world, tmp_path):
    record = a_sequential_record(tmp_path / "run")
    record["cache_digests"] = []  # the run record holds no cache digest
    files = write_run(tmp_path / "run", record=record)
    # The cache folder is really kept and really given: its files are there, and the flag is set.
    assert "--cache" in files and any(Path(files["--cache"]).iterdir())

    code, out, error = register(world, files, "--mixture", "mix-1")
    assert (code, error) == (0, None)
    assert (out["route"], out["cause"], out["cause_detail"]) == (
        "none", "data-order trace missing", "no recorded digest")

    # Source Z is held by the mixture and read at step 3; H@3 is the checkpoint after that read.
    answer = withdraw(world, "Z", "H@3")
    assert answer["answer"] == "may contain"
    assert answer["names"] == "data-order trace missing"
    assert answer["names_detail"] == "no recorded digest"


def test_control_the_same_record_with_its_cache_digests_gives_cache_files(world, tmp_path):
    record = a_sequential_record(tmp_path / "run")
    assert record["cache_digests"]  # the digests are recorded
    files = write_run(tmp_path / "run", record=record)

    code, out, error = register(world, files, "--mixture", "mix-1")
    assert (code, error) == (0, None)
    assert (out["route"], out["cause"], out["cause_detail"]) == ("cache files", None, None)

    answer = withdraw(world, "Z", "H@3")
    assert answer["answer"] == "contains"
