"""Exceptions of the oracle. Each message says what failed and where.

The oracle is test infrastructure, so these are its own classes and not the named
errors of the tool. Nothing here is caught and replaced by a default: a failure
stops the run.
"""

from __future__ import annotations


class OracleError(Exception):
    """Base class of every failure the oracle raises itself."""


class ConfigError(OracleError):
    """A config file that is missing a key, holds a wrong value or names a bad path."""


class EnvironmentNotReady(OracleError):
    """The Megatron environment is not set up as the oracle needs it."""


class CacheFolderNotEmpty(OracleError):
    """An index build was asked to write into a folder that already holds files."""


class UnexpectedCacheContent(OracleError):
    """The cache folder does not hold exactly the index files the build should write."""


class ReadMismatch(OracleError):
    """What Megatron's item getter returned differs from the reads that were logged."""


class SamplerRefused(OracleError):
    """Megatron's sampler class refused its arguments or gave no batch."""


class LaunchOutrunsDataset(OracleError):
    """A launch with the sequential sampler asks for more steps than its dataset
    holds from the launch's consumed count on, and the config does not allow the
    pass to be started again."""
