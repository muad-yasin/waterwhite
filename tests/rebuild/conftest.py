"""Fixtures of the tests of the rebuild report and its verification. The store is
the hand-built world of ``rb_kit.py``; nothing here runs Megatron or the replica."""

from __future__ import annotations

import pytest
import rb_kit


@pytest.fixture
def world(tmp_path):
    built = rb_kit.build(tmp_path / "w")
    yield built
    built.store.close()


@pytest.fixture
def swap_world(tmp_path):
    built = rb_kit.build(tmp_path / "w", swap=True)
    yield built
    built.store.close()


@pytest.fixture
def repair_world(tmp_path):
    """The world with the successors a repair registers."""
    built = rb_kit.build(tmp_path / "w", successors=True)
    yield built
    built.store.close()


@pytest.fixture
def unseeded_repair_world(tmp_path):
    built = rb_kit.build(tmp_path / "w", successors=True, unseeded=True)
    yield built
    built.store.close()
