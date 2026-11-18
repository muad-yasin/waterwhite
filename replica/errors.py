"""The replica's own exceptions.

The replica is test infrastructure, so it does not raise the tool's named errors.
Every failure stops the build with one of these; nothing is patched over.
"""

from __future__ import annotations


class ReplicaError(Exception):
    """Base class of every error the replica raises on purpose."""


class SpecError(ReplicaError):
    """The fixture spec or a set file says something the replica cannot build."""


class InputError(ReplicaError):
    """An input file does not hold what the spec or the set file says it holds."""


class OutputError(ReplicaError):
    """The output folder cannot take a build (it exists and is not empty)."""


class TokenizerError(ReplicaError):
    """The Megatron environment is missing, or the preprocessing run failed, hung or
    wrote a shard that does not match its input."""


class CaptureError(ReplicaError):
    """Capture was asked for and cannot run as asked."""


class ManifestError(ReplicaError):
    """A manifest, or the list of files a build registered, does not hold what was
    asked of it: a name the build did not register, a handle on no line or on
    several."""


class RunError(ReplicaError):
    """The run wrapper cannot make or register a run: the environment does not
    match the spec's Megatron release, a shard of the fixture build is missing, the
    worker failed, or a segment would not register."""
