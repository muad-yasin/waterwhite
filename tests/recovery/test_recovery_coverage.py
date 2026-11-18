"""Recovered links, read through the coverage proof (plan 2.2, 3.2).

A pipeline of two unobserved copy steps is recovered step by step. The proof then
walks the recovered edges like any other: a chain of them shows "contains", and a
file whose every record is traced to other sources gets "does not contain".
"""

from __future__ import annotations

import recovery_kit as kit

from waterwhite import capture, recovery
from waterwhite import vocabulary as v

MERGE = capture.step_ref("merge-shuffle", "not recorded")
DEAL = capture.step_ref("deal-shuffle", "not recorded")


def root(record, files):
    """Follow the kit's bookkeeping back to the source line a record is a copy of."""
    name, line = record.origin
    while name in files:
        name, line = files[name][line - 1].origin
    return name, line


def test_a_recovered_chain_answers_contains_and_does_not_contain(store, folder):
    a = kit.source(store, folder, "A", ["a one", "a two", "a three"])
    b = kit.source(store, folder, "B", ["b one", "b two", "b three"])
    z = kit.source(store, folder, "Z", ["z one", "z two", "z three", "z four"])
    kit.source(store, folder, "C", ["c one"])

    # Step one, not observed: two merges, each shuffled.
    files = {
        "merged-az": kit.shuffled(a + z[:2], 31),
        "merged-bz": kit.shuffled(b + z[2:], 32),
    }
    read = {name: kit.found_file(store, folder, name, lines) for name, lines in files.items()}
    # Step two, not observed: one merged file dealt over two chunks, the other
    # shuffled into a third.
    first, second = kit.dealt(kit.shuffled(read["merged-az"], 33), 2)
    files["chunk-1"], files["chunk-2"] = kit.shuffled(first, 34), kit.shuffled(second, 35)
    files["chunk-3"] = kit.shuffled(read["merged-bz"], 36)
    for name in ("chunk-1", "chunk-2", "chunk-3"):
        kit.found_file(store, folder, name, files[name])

    sources = ["src/A", "src/B", "src/Z"]
    merges = ["merged-az", "merged-bz"]
    for name in merges:
        result = recovery.recover_copy_step(store, name, sources, step=MERGE)
        assert (result.recovered, result.no_edge) == (5, 0)
    for name in ("chunk-1", "chunk-2", "chunk-3"):
        result = recovery.recover_copy_step(store, name, merges, step=DEAL)
        assert (result.recovered, result.candidates, result.no_edge) == (len(files[name]), 0, 0)
    store.check_integrity()

    for name in ("chunk-1", "chunk-2", "chunk-3"):
        lines = files[name]
        roots = [root(record, files) for record in lines]
        for key in ("A", "B", "Z", "C"):
            answer = kit.ask(store, key, name)
            held = [n for n, (source, _line) in enumerate(roots, 1) if source == f"src/{key}"]
            if not held:
                # Every record is traced, through two recovered links, to a source
                # the request does not name.
                assert answer["answer"] == "does not contain", (name, key)
                assert answer["scope"]["walked"] == len(lines)
                continue
            # The chain shown is that of the first record in line order that holds.
            line = held[0]
            middle, origin = lines[line - 1].origin, roots[line - 1]
            assert answer["answer"] == "contains", (name, key)
            assert kit.hops(answer) == [
                f"{name}:L{line}", f"{middle[0]}:L{middle[1]}", f"{origin[0]}:L{origin[1]}"]
            assert [hop["link"] for hop in answer["chain"]] == [v.RECOVERED, v.RECOVERED, "member"]

    # The bookkeeping itself: A went into the first merge only, so the third chunk
    # holds nothing of it, and one of the first two chunks does.
    assert kit.ask(store, "A", "chunk-3")["answer"] == "does not contain"
    assert "contains" in (kit.ask(store, "A", "chunk-1")["answer"],
                          kit.ask(store, "A", "chunk-2")["answer"])
    assert kit.ask(store, "B", "chunk-3")["answer"] == "contains"


def test_a_candidates_link_down_a_chain_gives_may_contain_for_each_candidate_source(store, folder):
    shared = "a passage two sources supply word for word"
    a = kit.source(store, folder, "A", [shared, "a two"])
    b = kit.source(store, folder, "B", [shared, "b two"])
    z = kit.source(store, folder, "Z", ["z one", "z two"])
    kit.source(store, folder, "C", ["c one"])

    # The merge took A's copy of the passage, B's other line and Z. Nothing recorded
    # which copy it took.
    merged = kit.found_file(store, folder, "merged", kit.shuffled([a[0], b[1]] + z, 41))
    chunk = kit.shuffled(merged, 42)
    kit.found_file(store, folder, "chunk-1", chunk)
    first = recovery.recover_copy_step(store, "merged", ["src/A", "src/B", "src/Z"], step=MERGE)
    second = recovery.recover_copy_step(store, "chunk-1", ["merged"], step=DEAL)
    assert (first.recovered, first.candidates) == (3, 1)
    assert (second.recovered, second.candidates) == (4, 0)  # inside the chain the copy is unique

    # A is a candidate and no other record of A is in the file: "may contain", with
    # both candidate sources named. The failing link is the candidates edge, one
    # step up from the file asked about.
    passage = next(n for n, record in enumerate(merged, 1) if record.text == shared)
    answer = kit.ask(store, "A", "chunk-1")
    assert (answer["answer"], answer["names"], answer["candidates"]) == (
        "may contain", v.NAMES_CANDIDATES, ["A", "B"])
    assert (answer["failing"]["artifact"], answer["failing"]["record"]) == ("merged", f"L{passage}")
    # B has another record in the file, so its recorded chain wins (plan 2.2).
    assert kit.ask(store, "B", "chunk-1")["answer"] == "contains"
    # Asked about B's copy of the passage alone, the candidates link is all there is.
    narrowed = kit.ask(store, "B", "chunk-1", records=["src/B:L1"])
    assert (narrowed["answer"], narrowed["names"], narrowed["candidates"]) == (
        "may contain", v.NAMES_CANDIDATES, ["A", "B"])
    assert kit.ask(store, "C", "chunk-1")["answer"] == "does not contain"
