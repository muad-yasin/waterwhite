"""The files of one shard of one layout, as a build wrote them.

It names where a build puts the four things one shard is made of: the chunk as
the sanitizer read it, the sanitizer's log, the tokenizer input and the shard's
two files. It imports nothing of the tool, so a build with capture off can use
it.

Names in the store (docs/interfaces.md section 9.2): the tokenizer input of the
shard ``E/shard-1`` is ``E/shard-1.input``; the chunk as the sanitizer read it is
``E/shard-1.sanitizer-input``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUFFIX_TOKENIZER_INPUT = ".input"
SUFFIX_SANITIZER_INPUT = ".sanitizer-input"


@dataclass(frozen=True)
class ShardFiles:
    """One shard's files. ``out`` is the build folder and ``layout`` the layout's
    name; ``shard`` is the shard's file name and ``store_name`` its name in the
    store. ``listed`` names the record files the copy steps list for this shard,
    as paths relative to the build folder, in listing order."""

    out: Path
    layout: str
    shard: str
    store_name: str
    listed: tuple[str, ...]

    @property
    def relative(self) -> str:
        """The layout's folder, relative to the build folder."""
        return f"pipeline/{self.layout}"

    @property
    def tokenizer_input_name(self) -> str:
        return self.store_name + SUFFIX_TOKENIZER_INPUT

    @property
    def sanitizer_input_name(self) -> str:
        return self.store_name + SUFFIX_SANITIZER_INPUT

    @property
    def tokenizer_input_relative(self) -> str:
        return f"{self.relative}/tokenizer-input/{self.shard}.jsonl"

    @property
    def sanitizer_input_relative(self) -> str:
        return f"{self.relative}/sanitizer-input/{self.shard}.jsonl"

    @property
    def prefix_relative(self) -> str:
        return f"{self.relative}/shards/{self.shard}"

    @property
    def tokenizer_input(self) -> Path:
        return self.out / self.tokenizer_input_relative

    @property
    def sanitizer_input(self) -> Path:
        return self.out / self.sanitizer_input_relative

    @property
    def prefix(self) -> Path:
        """The shard's two files are ``<prefix>.bin`` and ``<prefix>.idx``."""
        return self.out / self.prefix_relative

    @property
    def sanitizer_log(self) -> Path:
        return self.out / self.relative / "sanitizer.log.jsonl"

    @property
    def log_input(self) -> str:
        """The value the sanitizer's log carries in ``input`` for this shard's
        chunk: the path of the chunk relative to the build folder."""
        return self.sanitizer_input_relative


def shard_files(out: Path, layout: str, entry: dict) -> ShardFiles:
    """The files of one shard from its entry in a build's summary."""
    return ShardFiles(out=Path(out), layout=layout, shard=entry["shard"],
                      store_name=entry["store_name"], listed=tuple(entry["listed"]))
