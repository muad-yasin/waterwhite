"""The named errors of the confined harness.

Every named error of the tool is a class of ``waterwhite.errors``, and these
three are defined there. This module names them for code that imports them from
the package they belong to; the classes are the same objects.
docs/confine.md lists them.
"""

from __future__ import annotations

from ..errors import ConfinedJobFailed, JobProtocolViolation, SandboxUnavailable

__all__ = ["ConfinedJobFailed", "JobProtocolViolation", "SandboxUnavailable"]
