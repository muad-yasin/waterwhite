"""``show`` of a shard's named map, of an artifact's drops, and of a binding's
end-of-document settings (plan 3.3, 3.4, 5.5 Checks 3 and 6, F1; docs/cli.md).

Tiny files in a temporary folder, read through ``cli.main`` as a person's command
would; nothing here imports Megatron or torch.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest

from waterwhite import cli, ids
from waterwhite import vocabulary as v
from waterwhite.records import (
    DropRecord,
    MapRecord,
    MapRow,
    OccurrenceRecord,
    OccurrenceRef,
    SnapshotShard,
    StepRef,
)
from waterwhite.store import views
from waterwhite.store.store import Store

NORMALISER = "control-chars/1"
TEXTS = ["one", "two", "two", "three"]  # lines 2 and 3 are identical twins
DOCUMENTS = [[1], [2], [2], [3]]


def step(name: str) -> StepRef:
    return StepRef(name, "1", ids.param_digest(None))


def write_jsonl(path: Path, texts: list[str]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return len(texts), ids.file_digest(path)


def occurrences(texts: list[str]) -> list[OccurrenceRecord]:
    return [OccurrenceRecord(f"L{n}", ids.fingerprint(t)) for n, t in enumerate(texts, start=1)]


def write_shard(prefix: Path, documents: list[list[int]]) -> tuple[int, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(d) for d in documents]
    tokens = [t for d in documents for t in d]
    pointers = [4 * sum(lengths[:n]) for n in range(len(documents))]
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}i", *tokens))
    index = b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, len(lengths), len(lengths) + 1)
    index += struct.pack(f"<{len(lengths)}i", *lengths)
    index += struct.pack(f"<{len(pointers)}q", *pointers)
    index += struct.pack(f"<{len(lengths) + 1}q", *range(len(lengths) + 1))
    Path(f"{prefix}.idx").write_bytes(index)
    digest = ids.shard_digest(ids.file_digest(f"{prefix}.bin"), ids.file_digest(f"{prefix}.idx"))
    return len(lengths), digest


class Shards:
    """A store with an input file of four lines (two identical), the file a
    sanitizer wrote from it, a shard of four documents and a shard with no map."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / "evidence.db"
        self.store = Store.create(self.path, normaliser=NORMALISER)
        count, digest = write_jsonl(folder / "in" / "input.jsonl", TEXTS)
        self.input = self.store.register_jsonl(
            folder / "in" / "input.jsonl", name="input", role=v.ROLE_SOURCE, step=None,
            member_name="Z/r1", inputs=(), occurrences=occurrences(TEXTS),
            declared_count=count, declared_digest=digest)
        self.store.add_source("Z")
        self.store.add_member("Z", "Z/r1", "r1", views.PART_FILE, [(self.input, "")])
        kept = [TEXTS[0], TEXTS[1], TEXTS[3]]
        count, digest = write_jsonl(folder / "mid" / "clean.jsonl", kept)
        self.clean = self.store.register_jsonl(
            folder / "mid" / "clean.jsonl", name="clean", role=v.ROLE_INTERMEDIATE,
            step=step("sanitize"), member_name=None, inputs=[self.input],
            occurrences=occurrences(kept), declared_count=count, declared_digest=digest)
        self.count, self.digest = write_shard(folder / "shards" / "shard", DOCUMENTS)
        self.shard = self.store.register_shard(
            folder / "shards" / "shard", name="shard", step=step("tokenize"), inputs=[self.clean])
        write_shard(folder / "shards" / "bare", [[9]])
        self.store.register_shard(
            folder / "shards" / "bare", name="bare", step=step("tokenize"), inputs=[self.clean])

    def line(self, number: int) -> OccurrenceRef:
        return OccurrenceRef(self.input, f"L{number}")

    def map(self, verification: str = v.VERIFIED_PASSED, *, namespace: str = v.NAMESPACE_CAPTURE,
            derivation: str = v.DERIVED_POSITION, append_eod: bool = True,
            eod_in_span: bool = True, swap: bool = False) -> MapRecord:
        """Documents 1 and 2 are the identical twins: with a verified map each is
        labelled "candidates" and holds both lines. ``swap`` gives documents 0
        and 3 each other's line, to tell one map from another."""
        passed = verification == v.VERIFIED_PASSED
        parents = (4, 2, 3, 1) if swap else (1, 2, 3, 4)
        rows = []
        for n, parent in enumerate(parents):
            if not passed:
                rows.append(MapRow(n, n, n + 1, v.UNKNOWN))
            elif n in (1, 2):
                rows.append(MapRow(n, n, n + 1, v.CANDIDATES, None,
                                   (self.line(2), self.line(3))))
            else:
                rows.append(MapRow(n, n, n + 1, v.RECOVERED, self.line(parent)))
        label = v.RECOVERED if passed else v.UNKNOWN
        return MapRecord(namespace, derivation, verification, label, append_eod, eod_in_span,
                         tuple(rows))

    def write(self, record: MapRecord, **more) -> None:
        self.store.write_map(self.shard, record, declared_count=self.count,
                             declared_digest=self.digest, **more)

    def run(self, *argv: str) -> tuple[int, str, str]:
        """The command line in this process; the writer's connection is closed
        first, as a real caller's would be."""
        self.store.close()
        out, err = io.StringIO(), io.StringIO()
        code = cli.main([*argv, "--store", str(self.path)], stdout=out, stderr=err)
        self.store = Store.open(self.path)
        return code, out.getvalue(), err.getvalue()

    def show(self, *argv: str) -> dict:
        code, out, err = self.run("show", *argv)
        assert (code, err) == (0, "")
        return json.loads(out)


@pytest.fixture
def shards(tmp_path):
    built = Shards(tmp_path)
    yield built
    built.store.close()


def expect_error(result: tuple[int, str, str], code: int, name: str) -> dict:
    got, out, err = result
    assert out == ""
    error = json.loads(err)
    assert (got, error["error"]) == (code, name)
    return error


def record_of(artifact: str, instance: str, locator: str, members: list[str]) -> dict:
    return {"artifact": artifact, "instance": instance, "record": locator, "members": members}


# --- show SHARD --map NAMESPACE/DERIVATION -----------------------------------------------


def test_show_map_prints_the_named_map_and_its_rows(shards):
    s = shards
    s.write(s.map(namespace=v.NAMESPACE_CAPTURE, append_eod=True, eod_in_span=False))
    s.write(s.map(namespace=v.NAMESPACE_RECOVERY, derivation=v.DERIVED_POSITION,
                  append_eod=False, eod_in_span=True, swap=True))
    s.write(s.map(namespace=v.NAMESPACE_RECOVERY, derivation=v.DERIVED_TOKEN_MATCHING))

    shown = s.show("shard", "--map", "recovery/position")
    assert list(shown) == ["artifact", "instance", "kind", "map", "rows"]
    assert (shown["artifact"], shown["instance"], shown["kind"]) == ("shard", s.shard, "shard")
    assert shown["map"] == {
        "namespace": "recovery", "derivation": "position", "verification": "passed",
        "label": "recovered", "declared_count": 4, "append_eod": False, "eod_in_span": True}
    members = ["Z/r1"]
    twins = [record_of("input", s.input, "L2", members), record_of("input", s.input, "L3", members)]
    assert shown["rows"] == [
        {"record": "D0", "label": "recovered", "token_start": 0, "token_end": 1,
         "parent": record_of("input", s.input, "L4", members), "candidates": []},
        {"record": "D1", "label": "candidates", "token_start": 1, "token_end": 2, "parent": None,
         "candidates": twins},
        {"record": "D2", "label": "candidates", "token_start": 2, "token_end": 3, "parent": None,
         "candidates": twins},
        {"record": "D3", "label": "recovered", "token_start": 3, "token_end": 4,
         "parent": record_of("input", s.input, "L1", members), "candidates": []},
    ]

    # The capture map is the one a reader gets first. It is read by its name too,
    # and it is not the recovery map read above.
    first = s.show("shard", "--map", "capture/position")
    assert (first["map"]["namespace"], first["map"]["append_eod"], first["map"]["eod_in_span"]) == (
        "capture", True, False)
    assert first["rows"][0]["parent"]["record"] == "L1"
    assert first["rows"][3]["parent"]["record"] == "L4"
    assert s.show("shard", "--map", "recovery/token-matching")["map"]["derivation"] == "token-matching"

    # A row prints its parent and candidates as `show --record D<n>` prints them.
    for row in first["rows"]:
        one = s.show("shard", "--record", row["record"])
        assert (one["label"], one["token_start"], one["token_end"], one["parent"],
                one["candidates"]) == (row["label"], row["token_start"], row["token_end"],
                                       row["parent"], row["candidates"])


def test_show_map_names_the_map_that_did_not_verify_and_offers_the_other(shards):
    s = shards
    s.write(s.map(v.VERIFIED_FAILED, namespace=v.NAMESPACE_RECOVERY))
    only = s.show("shard", "--map", "recovery/position")
    assert only["map"]["verification"] == "failed"
    assert {r["label"] for r in only["rows"]} == {"unknown"}
    assert "other_maps" not in only  # no other map of this namespace and derivation
    # A result the namespace and derivation hold no map of is refused.
    expect_error(s.run("show", "shard", "--map", "recovery/position", "--verification", "not run"),
                 2, "request refused")
    # A map that was not run is named by its result, and the offer to read it holds
    # the word in quotes so that a copy of it works in a shell.
    s.write(s.map(v.VERIFIED_NOT_RUN, namespace=v.NAMESPACE_RECOVERY))
    unrun = s.show("shard", "--map", "recovery/position", "--verification", "not run")
    assert unrun["map"]["verification"] == "not run"
    offers = {o["verification"]: o["show_with"] for o in
              s.show("shard", "--map", "recovery/position")["other_maps"]}
    assert offers["not run"] == '--map recovery/position --verification "not run"'

    s.write(s.map(v.VERIFIED_PASSED, namespace=v.NAMESPACE_RECOVERY))
    got = s.show("shard", "--map", "recovery/position")
    assert got["map"]["verification"] == "passed"
    assert [r["label"] for r in got["rows"]] == ["recovered", "candidates", "candidates", "recovered"]
    assert got["other_maps"][0] == {
        "namespace": "recovery", "derivation": "position", "verification": "failed",
        "label": "unknown", "declared_count": 4,
        "show_with": "--map recovery/position --verification failed"}
    assert [o["verification"] for o in got["other_maps"]] == ["failed", "not run"]

    old = s.show("shard", "--map", "recovery/position", "--verification", "failed")
    assert old["map"]["verification"] == "failed"
    assert {r["label"] for r in old["rows"]} == {"unknown"}
    assert [(o["verification"], o["show_with"]) for o in old["other_maps"]] == [
        ("passed", "--map recovery/position --verification passed"),
        ("not run", '--map recovery/position --verification "not run"')]


def test_show_map_offers_only_a_map_the_flags_can_name(shards):
    """Two maps of one namespace and derivation that both failed: the flags reach
    the first written, so the second is listed and offers no flags."""
    s = shards
    first = s.map(v.VERIFIED_FAILED, namespace=v.NAMESPACE_RECOVERY)
    second = MapRecord(first.namespace, first.derivation, v.VERIFIED_FAILED, v.UNKNOWN, True, True,
                       tuple(MapRow(n, n, n + 1, v.UNKNOWN) for n in range(4))[:3]
                       + (MapRow(3, 3, 5, v.UNKNOWN),))
    s.write(first)
    s.write(second)
    s.write(s.map(v.VERIFIED_PASSED, namespace=v.NAMESPACE_RECOVERY))
    listed = s.show("shard", "--map", "recovery/position")["other_maps"]
    assert [(o["verification"], o["show_with"]) for o in listed] == [
        ("failed", "--map recovery/position --verification failed"), ("failed", None)]
    got = s.show("shard", "--map", "recovery/position", "--verification", "failed")
    assert got["rows"][3]["token_end"] == 4  # the first that was written
    assert [o["show_with"] for o in got["other_maps"]] == [
        "--map recovery/position --verification passed", None]


def test_show_map_has_a_text_form(shards):
    s = shards
    s.write(s.map(namespace=v.NAMESPACE_CAPTURE))
    code, out, err = s.run("show", "shard", "--map", "capture/position", "--format", "text")
    assert (code, err) == (0, "")
    assert "verification: passed" in out and "record: D1" in out and "append_eod: yes" in out


def test_show_map_refuses_what_it_cannot_print(shards):
    s = shards
    s.write(s.map(namespace=v.NAMESPACE_CAPTURE))
    for argv in (("shard", "--map", "recovery/position"),  # no such map
                 ("shard", "--map", "capture/token-matching"),
                 ("shard", "--map", "capture/position", "--verification", "failed"),
                 ("bare", "--map", "capture/position"),  # a shard with no map
                 ("input", "--map", "capture/position"),  # no shard
                 ("shard", "--map", "nonsense/words"),
                 ("no-such-name", "--map", "capture/position")):
        error = expect_error(s.run("show", *argv), 2, "request refused")
        assert error["detail"]
    held = expect_error(s.run("show", "shard", "--map", "recovery/position"), 2, "request refused")
    assert held["context"]["held"] == ["capture/position (passed)"]
    for bad in ("capture", "capture/", "/position", "a/b/c", ""):
        expect_error(s.run("show", "shard", "--map", bad), 1, "usage")
    expect_error(s.run("show", "shard", "--verification", "passed"), 1, "usage")
    expect_error(s.run("show", "shard", "--map", "capture/position", "--record", "D0"), 1, "usage")
    expect_error(s.run("show", "shard", "--map", "capture/position", "--verification", "maybe"),
                 1, "usage")


# --- drops ----------------------------------------------------------------------------------


def test_show_prints_the_drops_whose_output_is_the_artifact(shards):
    s = shards
    s.store.add_drops(s.clean, [DropRecord(s.line(3), v.DROP_SANITIZER_SKIP, "sanitize")])
    s.write(s.map(), drops=[DropRecord(OccurrenceRef(s.clean, "L2"), v.DROP_NO_TOKENS, "tokenize")])
    s.store.register_mixture(
        name="mix", shards=[SnapshotShard(0, s.shard, "shards/shard",
                                          s.store.artifact("shard").content_digest)],
        settings={"mix": 1})
    s.store.register_checkpoint(name="ck@1", segment="seg-1", step=1)

    assert s.show("clean")["drops"] == [
        {"artifact": "input", "instance": s.input, "record": "L3",
         "reason": "sanitizer skip", "step": "sanitize"}]
    assert s.show("shard")["drops"] == [
        {"artifact": "clean", "instance": s.clean, "record": "L2",
         "reason": "no tokens", "step": "tokenize"}]
    # The key is there for every registered artifact, with none to show as well.
    for name in ("input", "bare", "mix", "ck@1"):
        assert s.show(name)["drops"] == []
    # By instance id too, and a name the store has no row for prints no such key.
    assert s.show(s.clean)["drops"] == s.show("clean")["drops"]
    assert "drops" not in s.show("nothing-here")
    # The record's own view keeps its shape.
    assert s.show("input", "--record", "L3")["drops"] == [
        {"reason": "sanitizer skip", "step": "sanitize", "output": "clean"}]


# --- the bindings ----------------------------------------------------------------------------


def test_every_binding_says_its_end_of_document_settings(shards):
    s = shards
    s.write(s.map(namespace=v.NAMESPACE_CAPTURE, append_eod=True, eod_in_span=False))
    s.write(s.map(namespace=v.NAMESPACE_RECOVERY, append_eod=False, eod_in_span=True))
    got = {(b["namespace"], b["derivation"]): (b["append_eod"], b["eod_in_span"])
           for b in s.show("shard")["bindings"]}
    assert got == {("capture", "position"): (True, False), ("recovery", "position"): (False, True)}
    # A binding that is no map says null, not false.
    assert [(b["append_eod"], b["eod_in_span"]) for b in s.show("input")["bindings"]] == [(None, None)]
