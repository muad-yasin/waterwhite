"""The fixture of the tokenization-map tests. The store it builds is in
``tm_store.py``, the shards and the toy tokenizer in ``tm_world.py``."""

from __future__ import annotations

import pytest
import tm_store


@pytest.fixture
def world(tmp_path):
    built = tm_store.World(tmp_path)
    yield built
    built.store.close()
