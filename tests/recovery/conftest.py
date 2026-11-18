"""Fixtures of the recovery tests. Every test writes its own small files under
its temporary folder and registers them through capture and the recovery; nothing
here runs Megatron or the replica."""

from __future__ import annotations

import pytest
import recovery_kit as kit


@pytest.fixture
def folder(tmp_path):
    return tmp_path


@pytest.fixture
def store(folder):
    opened = kit.new_store(folder)
    yield opened
    opened.close()
