"""Shared set-up of the confined-harness tests.

Every test here carries the marker "sandbox". A test that needs a working sandbox
asks for the fixture ``sandbox``; where bubblewrap is missing or cannot create its
namespaces, such a test is skipped with the harness's own reason and is not
worked around."""

from __future__ import annotations

from pathlib import Path

import pytest

from waterwhite.confine import SandboxUnavailable, find_sandbox

JOBS = Path(__file__).parent / "jobs"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "sandbox: runs a job in the bubblewrap sandbox (Linux, unprivileged user namespaces)",
    )


@pytest.fixture(scope="session")
def sandbox():
    try:
        return find_sandbox()
    except SandboxUnavailable as unavailable:
        pytest.skip(str(unavailable))


@pytest.fixture
def jobs() -> Path:
    return JOBS
