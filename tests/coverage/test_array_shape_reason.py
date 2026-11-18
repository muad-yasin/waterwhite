"""A segment whose kept cache files all match the recorded digests, but one of whose
arrays has a shape no index array of its kind can have, is registered with the reason
"index array malformed" (docs/join.md section 3, question 7; DECISIONS.md D-41; plan 3.5).

The run record holds the digest of every file actually kept, so no digest is missing and
none differs: the cache-files route finds all three arrays of the one shard. The
document index is two-dimensional, which no document index is, so the arrays cannot be
read. The segment has route "none", cause "data-order trace missing" and the reason
"index array malformed", and its checkpoints answer "may contain" naming the same cause
and reason. Everything goes through the command line (``register-run``, ``query
withdraw``) on the hand world of ``test_register_run_command``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_register_run_command import ARRAYS, LENGTHS, kit, register, withdraw, write_run  # noqa: E402

from waterwhite.join._arrays import digest_of  # noqa: E402

# The same three arrays as the hand world, but the document index is a column, (3, 1).
BAD_ARRAYS = ARRAYS | {("document_index", 0): ARRAYS[("document_index", 0)].reshape(3, 1)}


def a_segment_with_a_malformed_array(folder: Path) -> dict:
    """The files of a one-shard sequential segment whose kept cache folder holds a
    two-dimensional document index and whose run record holds the digests of the files
    kept. Checks that the state is the one meant: every recorded digest is the digest
    of a file in the folder, and the bad file is among them."""
    record = kit.run_record(folder, shards=1, arrays=BAD_ARRAYS, lengths=LENGTHS, micro=1, batch=1,
                            launches=[kit.launch(0, 1, 0, 1, 3)])
    files = write_run(folder, record=record, cache=False)
    kit.write_cache(folder / "cache", BAD_ARRAYS)
    files["--cache"] = str(folder / "cache")

    kept: dict[str, list[np.ndarray]] = {}
    for path in sorted((folder / "cache").glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        kept.setdefault(path.name.rsplit("-", 1)[1].removesuffix(".npy"), []).append(array)
    bad_files = kept["document_index"]
    assert len(bad_files) == 1 and bad_files[0].ndim == 2 and bad_files[0].shape == (3, 1)
    recorded = {(entry["kind"], entry["shard"]): entry["digest"] for entry in record["cache_digests"]}
    assert set(recorded) == {("document_index", 0), ("sample_index", 0), ("shuffle_index", 0)}
    for (kind, shard), digest in recorded.items():
        assert shard == 0 and [digest_of(array) for array in kept[kind]] == [digest]
    return files


def test_a_malformed_index_array_gives_no_route_and_the_reason_index_array_malformed(world, tmp_path):
    files = a_segment_with_a_malformed_array(tmp_path / "run")

    code, out, error = register(world, files, "--mixture", "mix-1")
    assert (code, error) == (0, None)
    assert (out["route"], out["cause"], out["cause_detail"]) == (
        "none", "data-order trace missing", "index array malformed")


def test_a_checkpoint_of_that_segment_may_contain_and_names_the_malformed_array(world, tmp_path):
    files = a_segment_with_a_malformed_array(tmp_path / "run")
    code, _, error = register(world, files, "--mixture", "mix-1")
    assert (code, error) == (0, None)

    # Source Z is held by the mixture and read at step 3; H@3 is the checkpoint after that read.
    answer = withdraw(world, "Z", "H@3")
    assert (answer["answer"], answer["names"], answer["names_detail"]) == (
        "may contain", "data-order trace missing", "index array malformed")
