"""A store in a temporary folder for the tokenization-map tests, with small
helpers to register an input file, run the wrapper with the toy tokenizer of
``tm_world.py`` and read a map back. Nothing here runs Megatron."""

from __future__ import annotations

from pathlib import Path

import tm_world

from waterwhite import capture, tokenmap
from waterwhite import vocabulary as v
from waterwhite.records import MapRow, OccurrenceRef
from waterwhite.store import Store

SETTINGS = tokenmap.TokenizerSettings(append_eod=True, eod_in_span=True,
                                      eod_token_id=tm_world.EOD)
STEP = tokenmap.tokenize_step("toy/1", {"tokenizer": "bytes", "append_eod": True})


class World:
    """One store and the folder beside it."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.store = Store.create(folder / "evidence.db", normaliser=tm_world.NORMALISER)
        self.store.add_source("Z")

    def register(self, path: Path, name: str) -> str:
        """Register a JSONL file under ``name`` and return its instance id."""
        return capture.ingest_jsonl(self.store, path, source="Z", member=f"Z/{name}",
                                    revision="r1", name=name)

    def input(self, name: str, texts: list[str], *, id_prefix: str = "U") -> str:
        """Write and register a tokenizer-input file with these texts."""
        path = tm_world.write_rows(self.folder / "tokenizer-input" / f"{name}.jsonl", texts,
                                   id_prefix=id_prefix)
        return self.register(path, name)

    def prefix(self, shard: str) -> Path:
        return self.folder / "shards" / shard

    def capture(self, shard: str, texts: list[str], **changes) -> tokenmap.ShardMaps:
        """Register an input with these texts and run the capture wrapper on it
        with the toy tokenizer."""
        self.input(f"{shard}.input", texts)
        arguments = dict(
            input=f"{shard}.input", prefix=self.prefix(shard), name=shard, step=STEP,
            run_tokenizer=tm_world.toy_preprocess, settings=SETTINGS,
            tokenize=tm_world.toy_tokenize,
        )
        arguments.update(changes)
        return tokenmap.tokenize_shard(self.store, **arguments)

    def bare_shard(self, shard: str, input_name: str, documents: list[list[int]]) -> str:
        """A shard written and registered with no map, as a run made without the
        wrapper leaves it."""
        tm_world.write_shard(self.prefix(shard), documents)
        return self.store.register_shard(self.prefix(shard), name=shard, step=STEP,
                                         inputs=[self.store.artifact(input_name).instance_id])

    def rows(self, shard: str, namespace: str, derivation: str) -> list[MapRow]:
        instance = self.store.artifact(shard).instance_id
        return list(self.store.map_rows(instance, namespace, derivation=derivation))

    def assigned(self, shard: str, namespace: str, derivation: str) -> list[tuple]:
        """Per ordinal: (label, the line of the parent or None, the sorted lines of
        the candidate set)."""
        return [(row.label, None if row.parent is None else row.parent.locator,
                 sorted(c.locator for c in row.candidates))
                for row in self.rows(shard, namespace, derivation)]

    def ref(self, name: str, line: int) -> OccurrenceRef:
        return OccurrenceRef(self.store.artifact(name).instance_id, f"L{line}")

    def drops(self, shard: str) -> list[tuple]:
        instance = self.store.artifact(shard).instance_id
        return [(d.input.locator, d.reason, d.step_name)
                for d in self.store.drops_by_output(instance)]

    def head(self, shard: str) -> list[tuple]:
        """The maps of a shard in the reader's order: (namespace, derivation,
        verification, label)."""
        instance = self.store.artifact(shard).instance_id
        return [(b.namespace, b.derivation, b.verification, b.label)
                for b in self.store.bindings(instance)]


CAPTURE, RECOVERY = v.NAMESPACE_CAPTURE, v.NAMESPACE_RECOVERY
POSITION, MATCHING = v.DERIVED_POSITION, v.DERIVED_TOKEN_MATCHING
