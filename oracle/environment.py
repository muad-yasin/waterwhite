"""What the oracle states about the environment it ran in.

The run record names the Megatron release, the megatron-core and torch versions
and, when Megatron-Bridge's sampler file was used, its version (plan 3.1, F6). The
release is read from the Megatron-LM checkout in the environment folder, and the
installed package is compared with that checkout file by file, so the tag that is
written down is the code that ran.

A checkout's tag is read from its ``.git`` folder directly. No ``git`` command is
run.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

from oracle.config import megatron_env
from oracle.digests import file_sha256
from oracle.errors import EnvironmentNotReady

# The files whose behaviour the oracle's rules rest on (docs/oracle.md). Paths are
# relative to the checkout root and to the installed package root alike.
RULE_FILES = (
    "megatron/core/datasets/gpt_dataset.py",
    "megatron/core/datasets/blended_dataset.py",
    "megatron/core/datasets/indexed_dataset.py",
    "megatron/core/datasets/megatron_dataset.py",
    "megatron/core/datasets/blended_megatron_dataset_builder.py",
    "megatron/core/datasets/helpers.cpp",
    "megatron/training/datasets/data_samplers.py",
)


def megatron_lm_checkout() -> Path:
    path = megatron_env() / "src" / "Megatron-LM"
    if not path.is_dir():
        raise EnvironmentNotReady("no Megatron-LM checkout under src/ of the environment folder")
    return path


def _git_dir(checkout: Path) -> Path:
    dot_git = checkout / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            target = Path(text[len("gitdir:") :].strip())
            return target if target.is_absolute() else (checkout / target).resolve()
    raise EnvironmentNotReady(f"not a git checkout: {checkout.name}")


def _packed_refs(git_dir: Path) -> list[tuple[str, str, str | None]]:
    """(commit or tag object, ref name, peeled commit or None) of each packed ref."""
    path = git_dir / "packed-refs"
    refs: list[tuple[str, str, str | None]] = []
    if not path.is_file():
        return refs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("^"):
            if refs:
                sha, name, _ = refs[-1]
                refs[-1] = (sha, name, line[1:].strip())
            continue
        sha, _, name = line.partition(" ")
        refs.append((sha.strip(), name.strip(), None))
    return refs


def checkout_ref(checkout: Path) -> dict:
    """The commit a checkout is at and the tags that name that commit.

    Returns ``{"commit": <hex>, "tags": [<tag>, ...]}``. An annotated tag is
    matched through its peeled line in ``packed-refs``; a loose annotated tag
    object cannot be resolved without git and is then not listed."""
    git_dir = _git_dir(checkout)
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    packed = _packed_refs(git_dir)
    if head.startswith("ref:"):
        ref = head[len("ref:") :].strip()
        loose = git_dir / ref
        if loose.is_file():
            commit = loose.read_text(encoding="utf-8").strip()
        else:
            found = [sha for sha, name, _ in packed if name == ref]
            if not found:
                raise EnvironmentNotReady(f"cannot resolve HEAD of checkout {checkout.name}")
            commit = found[0]
    else:
        commit = head
    tags = set()
    for sha, name, peeled in packed:
        if name.startswith("refs/tags/") and commit in (sha, peeled):
            tags.add(name[len("refs/tags/") :])
    tags_dir = git_dir / "refs" / "tags"
    if tags_dir.is_dir():
        for root, _dirs, files in os.walk(tags_dir):
            for file in files:
                path = Path(root) / file
                if path.read_text(encoding="utf-8").strip() == commit:
                    tags.add(str(path.relative_to(tags_dir)).replace(os.sep, "/"))
    return {"commit": commit, "tags": sorted(tags)}


def ref_name(checkout: Path) -> str:
    """One name for the state of a checkout: its tag when exactly one tag names the
    commit it is at, else the commit."""
    ref = checkout_ref(checkout)
    if len(ref["tags"]) == 1:
        return ref["tags"][0]
    return ref["commit"]


def checkout_of(path: Path) -> Path:
    """The checkout a file belongs to: the nearest folder above it with ``.git``."""
    for folder in path.resolve().parents:
        if (folder / ".git").exists():
            return folder
    raise EnvironmentNotReady(f"{path.name} lies in no git checkout, so its version is unknown")


def verify_installed_matches_checkout() -> list[str]:
    """Compare the installed Megatron files the rules rest on with the checkout.

    Returns the list of files compared. Raises when one differs or is missing: the
    release tag in the run record would then name code that did not run."""
    import megatron  # the installed package

    roots = [Path(p) for p in megatron.__path__]
    checkout = megatron_lm_checkout()
    for rel in RULE_FILES:
        source = checkout / rel
        installed = [root.parent / rel for root in roots if (root.parent / rel).is_file()]
        if not source.is_file() or not installed:
            raise EnvironmentNotReady(f"cannot compare {rel}: missing in the checkout or the install")
        if file_sha256(source) != file_sha256(installed[0]):
            raise EnvironmentNotReady(
                f"the installed {rel} differs from the Megatron-LM checkout; "
                "the release tag would not name the code that ran"
            )
    return list(RULE_FILES)


def versions(bridge_sampler_file: Path | None) -> dict:
    """The ``versions`` object of a run record (docs/interfaces.md 8.1).

    ``megatron_bridge`` is null unless Megatron-Bridge's sampler file was used; it
    is then the tag, or the commit, of the checkout that holds the file."""
    import torch

    verify_installed_matches_checkout()
    return {
        "release_tag": ref_name(megatron_lm_checkout()),
        "megatron_core": metadata.version("megatron-core"),
        "megatron_bridge": (
            None if bridge_sampler_file is None else ref_name(checkout_of(bridge_sampler_file))
        ),
        "torch": torch.__version__,
    }
