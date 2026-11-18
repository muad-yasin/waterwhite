"""Fixtures of the developer tests in this folder. The store they share is built by
hand in ``ww_world.py``; nothing here runs Megatron or the replica."""

from __future__ import annotations

import pytest
import ww_world

from waterwhite import coverage, requests


@pytest.fixture
def world(tmp_path):
    built = ww_world.build(tmp_path)
    yield built
    built.store.close()


@pytest.fixture
def ask(world):
    """Put a withdrawal to one artifact and return the answer's JSON form. The
    store is read as it is when the question is put, so a test can apply a fault
    first."""

    def put(source: str, artifact: str, *, members=(), records=()) -> dict:
        request = requests.withdraw(world.store, source, members=members, records=records)
        return coverage.answer(world.store, artifact, request).to_json()

    return put
