"""What ``show SEGMENT --trace`` and ``replay`` print, read as pure functions (plan
5.5 Checks 1, 2 and 8; docs/interfaces.md sections 8.3 and 8.4).

Nothing here starts a process: the world runs the commands and hands the printed
documents to these functions, which are tested on documents and files written by
hand (``test_trace.py``).

``show SEGMENT --trace`` prints the rows the join wrote for a registered segment:
steps, each with its samples, each with the rows (shard position, ordinal, token
start and end) in reading order. ``flatten`` turns that into the shape of the oracle's
log (one row per row read, with its step and the place of its sample), which is what
``acceptance_join.diff`` compares.

``show SEGMENT --trace`` also prints the run record's launches (``launches``) and, for a
segment read with the cyclic sampler, the epochs each step draws its samples from
(``epochs`` of a step). ``register-run`` prints the shuffle the join computed
(``shuffle``). The functions below read those forms; a form that is absent is told
apart from a form that is wrong (``None`` or ``False``, never a finding), so that a command
that does not print it yet skips the row by name and a command that prints it wrongly fails it.

``replay`` writes the index arrays it derived from a run record into a folder and
prints one entry per array. Check 2 (i) compares each array with the kept cache file of
the same kind and shard. The plan states the comparison on the ``.npy`` index arrays
(plan 3.5, F3: the file names and ``description.txt`` carry the paths of the run and
change with them), so an array is the same when its dtype, its shape and its bytes are
the same; the header of the file is not part of the question.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

# The kinds of the cache files of a segment (docs/interfaces.md 8.4): three per shard and,
# when the segment has more than one shard, two for the blend.
PER_SHARD_KINDS = ("document_index", "sample_index", "shuffle_index")
BLEND_KINDS = ("dataset_index", "dataset_sample_index")

Sample = tuple[str, int, int]


class TraceError(Exception):
    """A document that is not what the command's contract says it prints."""


def _need(document: Mapping, key: str, where: str):
    if key not in document:
        raise TraceError(f"{where} has no field {key!r}: {sorted(document)}")
    return document[key]


def flatten(trace: Mapping) -> list[dict]:
    """The rows of a printed trace in reading order, one dict per row read:
    segment, step, sample_pos, global_index, rank, shard_position, ordinal, token_start,
    token_end (the fields of the oracle's log, docs/interfaces.md 8.3)."""
    segment = _need(trace, "segment", "the trace")
    out = []
    for step in _need(trace, "steps", "the trace"):
        for sample in _need(step, "samples", "a step of the trace"):
            for row in _need(sample, "rows", "a sample of the trace"):
                out.append({
                    "segment": segment, "step": step["step"],
                    "sample_pos": sample["sample_pos"], "global_index": sample["global_index"],
                    "rank": sample["rank"], "shard_position": row["shard_position"],
                    "ordinal": row["ordinal"], "token_start": row["token_start"],
                    "token_end": row["token_end"]})
    return out


def samples(trace: Mapping) -> dict[Sample, Mapping]:
    """The samples of a printed trace by (segment, step, sample_pos), each as printed."""
    segment = _need(trace, "segment", "the trace")
    return {(segment, step["step"], sample["sample_pos"]): sample
            for step in _need(trace, "steps", "the trace") for sample in step["samples"]}


def steps(trace: Mapping) -> dict[int, Mapping]:
    return {step["step"]: step for step in _need(trace, "steps", "the trace")}


def uncovered(sample: Mapping) -> list[int] | None:
    """The uncovered range of a sample, as ``[from, to]`` in the sample's own tokens, or
    None when the sample is whole: derived from the coverage mark, as the plan says
    (plan 3.5: the uncovered range is derived from the mark whenever it is reported)."""
    covered, total = sample["covered_tokens"], sample["sample_tokens"]
    return None if covered >= total else [covered, total]


def steps_with_uncovered_range(trace: Mapping) -> list[int]:
    return [step["step"] for step in _need(trace, "steps", "the trace")
            if any(uncovered(sample) for sample in step["samples"])]


def rank_layout(trace: Mapping) -> dict[int, dict]:
    """Per step: how many samples it holds, and the sample positions of each rank."""
    out: dict[int, dict] = {}
    for step in _need(trace, "steps", "the trace"):
        by_rank: dict[int, list[int]] = {}
        for sample in step["samples"]:
            by_rank.setdefault(sample["rank"], []).append(sample["sample_pos"])
        out[step["step"]] = {"samples": len(step["samples"]),
                             "by_rank": {rank: sorted(places) for rank, places in by_rank.items()}}
    return out


# --- The launches, the epochs of the steps and the shuffle ----------------------------------


def launches(trace: Mapping) -> list[Mapping] | None:
    """The launches a printed trace holds, or None where the trace prints none of the
    field at all (a command that does not have the form yet). An empty list is a record
    that holds no launch."""
    if "launches" not in trace:
        return None
    held = trace["launches"]
    if not isinstance(held, list) or not all(isinstance(each, Mapping) for each in held):
        raise TraceError(f"the launches of a trace are a list of objects: {held!r}")
    return held


def launch_at(held: Sequence[Mapping], position: int) -> Mapping | None:
    """The launch of a run record at a position, or None."""
    found = [each for each in held if each.get("position") == position]
    return found[0] if len(found) == 1 else None


def epochs_printed(trace: Mapping) -> bool:
    """Whether a trace prints ``epochs`` at all: some step of it holds the key. A trace with
    no step says nothing about the form (the rows that read the steps find that)."""
    return any("epochs" in step for step in _need(trace, "steps", "the trace"))


def epoch_boundary(trace: Mapping, first_epoch: int, epoch_after: int
                   ) -> tuple[int | None, list[str]]:
    """The step after which the epoch changes, read from the ``epochs`` of the steps, and
    what is wrong with the epochs on the way.

    Each step must print a list of integers; a step reads one epoch (a step whose rounds
    lie on both sides of a boundary prints two, and the table's boundary lies between
    steps); the epoch of the first step is ``first_epoch``; the epoch never goes down and
    rises by one at a time; and the first step of the new epoch is read in ``epoch_after``.
    The boundary is the last step whose epoch is the first one (None where the epoch never
    changes)."""
    findings: list[str] = []
    boundary: int | None = None
    seen: int | None = None
    for step in sorted(_need(trace, "steps", "the trace"), key=lambda each: each["step"]):
        number, epochs = step["step"], step.get("epochs")
        if (not isinstance(epochs, list) or not epochs
                or not all(isinstance(each, int) and not isinstance(each, bool)
                           for each in epochs)):
            findings.append(f"step {number} prints epochs {epochs!r}, not a list of integers")
            continue
        if len(set(epochs)) != 1:
            findings.append(f"step {number} reads two epochs, {epochs}")
            continue
        epoch = epochs[0]
        if seen is None:
            if epoch != first_epoch:
                findings.append(f"step {number} is read in epoch {epoch}, not {first_epoch}")
        elif epoch < seen or epoch > seen + 1:
            findings.append(f"step {number} is read in epoch {epoch} after epoch {seen}")
        if seen is not None and epoch != seen and boundary is None:
            boundary = number - 1
            if epoch != epoch_after:
                findings.append(f"step {number} is read in epoch {epoch}, not {epoch_after}")
        seen = epoch
    return boundary, findings


def shuffle_findings(shuffle: object, cells: Sequence[tuple[int, int, int]],
                     record: Mapping[tuple[int, int, int], tuple[str, int]] | None = None
                     ) -> list[str]:
    """What is wrong with the ``shuffle`` register-run printed for a cyclic segment, against
    the (launch, epoch, rank) cells the run's launches read (the table's, in that order):
    one entry per cell, each ``equal`` and with the digest and the count of the record, and
    the epoch 0 digests of the launches equal rank by rank (a digest covers the whole order
    of one rank for one epoch, whatever part of the epoch a launch read).

    ``record`` is the run record's own ``shuffle_digests`` read from the file the replica
    wrote, as (launch, epoch, rank) -> (digest, count): the digest and count the tool says
    it recorded, and the digest and count it computed, must each be the file's, so that a
    tool that prints its own digest as the recorded one finds no pass."""
    if not isinstance(shuffle, list) or not all(isinstance(each, Mapping) for each in shuffle):
        return [f"shuffle is not a list of objects: {shuffle!r}"]
    findings = []
    got = [(each.get("launch"), each.get("epoch"), each.get("rank")) for each in shuffle]
    if got != list(cells):
        findings.append(f"the entries are the (launch, epoch, rank) cells {got}, the record's "
                        f"launches read {list(cells)}")
    for each in shuffle:
        cell = (each.get("launch"), each.get("epoch"), each.get("rank"))
        digest, recorded = each.get("digest"), each.get("recorded_digest")
        if not isinstance(recorded, str) or not recorded:
            findings.append(f"{cell}: no recorded digest ({recorded!r})")
        if digest != recorded:
            findings.append(f"{cell}: digest {digest!r}, recorded {recorded!r}")
        count = each.get("count")
        if count != each.get("recorded_count") or not isinstance(count, int) or count <= 0:
            findings.append(f"{cell}: count {count!r}, recorded {each.get('recorded_count')!r}")
        if each.get("equal") is not True:
            findings.append(f"{cell}: equal is {each.get('equal')!r}")
        if record is not None and cell in record:
            wanted_digest, wanted_count = record[cell]
            if (recorded, each.get("recorded_count")) != (wanted_digest, wanted_count):
                findings.append(f"{cell}: the tool says it recorded {recorded!r} and "
                                f"{each.get('recorded_count')!r}, the run record holds "
                                f"{wanted_digest!r} and {wanted_count!r}")
            if (digest, count) != (wanted_digest, wanted_count):
                findings.append(f"{cell}: the tool computed {digest!r} and {count!r}, the run "
                                f"record holds {wanted_digest!r} and {wanted_count!r}")
    by_cell = {(each.get("launch"), each.get("epoch"), each.get("rank")): each
               for each in shuffle}
    for rank in sorted({cell[2] for cell in cells}):
        digests = [by_cell[(launch, 0, rank)].get("digest")
                   for launch in sorted({cell[0] for cell in cells})
                   if (launch, 0, rank) in by_cell]
        if len(set(digests)) > 1:
            findings.append(f"rank {rank}: the epoch 0 digests of the launches differ: {digests}")
    return findings


# --- The replayed arrays against the kept cache files ---------------------------------------


def cache_files(cache: Path, run_record: Mapping) -> dict[tuple[str, int | None], Path]:
    """The kept cache file of each (kind, shard position), shard None for the blend's.

    A cache folder is the one Megatron wrote with ``path_to_cache`` set (docs/interfaces.md
    8.4): per dataset a ``description.txt`` that names the dataset's path, and its
    ``.npy`` arrays under the same stem; the blend's arrays under the blend's stem.
    The shard of a dataset is the position of the run record's shard whose prefix ends
    with the name its ``description.txt`` gives."""
    by_name = {Path(shard["prefix"]).name: shard["position"]
               for shard in run_record["shards"]}
    out: dict[tuple[str, int | None], Path] = {}
    for description in sorted(Path(cache).glob("*-GPTDataset-train-description.txt")):
        held = json.loads(description.read_text(encoding="utf-8"))
        name = Path(held["dataset_path"]).name
        if name not in by_name:
            raise TraceError(f"{description.name} names the dataset {held['dataset_path']!r}, "
                             f"which no shard of the run record ends with: {sorted(by_name)}")
        stem = description.name[:-len("description.txt")]
        for array in sorted(Path(cache).glob(f"{stem}*.npy")):
            out[(array.name[len(stem):-len(".npy")], by_name[name])] = array
    blend = "-BlendedDataset-train-"
    for array in sorted(Path(cache).glob(f"*{blend}*.npy")):
        out[(array.name.split(blend, 1)[1][:-len(".npy")], None)] = array
    return out


def array_differences(replayed: Path, kept: Path) -> list[str]:
    """How two ``.npy`` arrays differ: dtype, shape, bytes. Empty when they are the same."""
    a = np.load(replayed, allow_pickle=False)
    b = np.load(kept, allow_pickle=False)
    if a.dtype != b.dtype:
        return [f"dtype {a.dtype} against {b.dtype}"]
    if a.shape != b.shape:
        return [f"shape {a.shape} against {b.shape}"]
    if a.tobytes() != b.tobytes():
        first = int(np.flatnonzero(a.reshape(-1) != b.reshape(-1))[0])
        return [f"bytes differ, the first differing value at flat index {first} of {a.size}"]
    return []


def compare_replay(printed: Mapping, out: Path, kept: Mapping[tuple[str, int | None], Path],
                   kinds: Sequence[str]) -> list[str]:
    """The findings where the arrays ``replay`` wrote differ from the kept cache files:
    a kind the row lists that was not written, an array written that no kept file
    matches, a kept file that was not written, and each pair that differs. The empty
    list is "byte-identical" for every array."""
    findings = []
    entries = _need(printed, "arrays", "what replay printed")
    written = {(entry["kind"], entry["shard"]) for entry in entries}
    if len(written) != len(entries):
        findings.append("replay printed one array twice")
    for kind in kinds:
        if not any(entry_kind == kind for entry_kind, _ in written):
            findings.append(f"no array of the kind {kind} was written")
    for entry in entries:
        key = (entry["kind"], entry["shard"])
        path = Path(entry["file"])
        path = path if path.is_absolute() else Path(out) / path
        if key not in kept:
            findings.append(f"{key}: written, and the kept cache holds no file of that kind")
        elif not path.is_file():
            findings.append(f"{key}: the file {entry['file']} is not in the folder replay wrote to")
        else:
            findings += [f"{key}: {text}" for text in array_differences(path, kept[key])]
    findings += [f"{key}: kept in the cache and not written" for key in sorted(
        kept, key=lambda k: (k[0], -1 if k[1] is None else k[1])) if key not in written]
    return findings
