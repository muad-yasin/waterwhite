"""The recovery of a run made without capture (plan 5.4, Check 6).

A build with capture off calls nothing of the tool but its normaliser and leaves
only its files. ``recover_build`` puts such a build into a store afterwards,
through the tool's public calls and nothing else:

- each source file is ingested (``waterwhite.capture.ingest_jsonl``);
- each generation output is registered as a plain file with no edge
  (``waterwhite.recovery.register_found_file``). No job ran in a harness, and a
  generation's parent is never recovered by similarity (plan L5), so its records
  have no recorded origin;
- each release is registered as a found file and its rows are linked to the
  records of the files its inputs come from by fingerprint
  (``waterwhite.recovery.recover_copy_step``). The dedup selection was not
  observed, so a row whose text two sources supply gets both as candidates (L3);
- per shard: the chunk as the sanitizer read it and the copy steps before it, the
  tokenizer input and the sanitizer between the two
  (``replica.pipeline_links``), then the shard itself (``Store.register_shard``)
  and its map in the recovery namespace (``waterwhite.tokenmap.recover_map``).

No mixture is registered: a run made without capture has no mixture snapshot
(plan L6), and whether one may be registered afterwards is open (plan 5.8, B-4).

``add_recovery_maps`` recovers the maps of shards a captured build registered, so
that one store holds a shard's capture map and its recovery maps side by side
(plan Check 3; DECISIONS.md D-20).

A dropped line is recorded once (``replica.pipeline_links``). Where a run kept its
tokenizer input the sanitizer's skips are drops whose output is that file; where
it kept only the sanitizer's input (``kept="sanitizer-input"``) the map is
recovered over the chunk with the sanitizer's log, and the skips are drops whose
output is the shard.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from waterwhite import capture as ww_capture
from waterwhite import recovery as ww_recovery
from waterwhite import tokenmap as ww_tokenmap

from . import megatron_tokenizer, pipeline_links
from .build import WORK_DIR, BuildOptions
from .errors import SpecError
from .plan import BuildPlan
from .shard_files import shard_files

KEPT_TOKENIZER_INPUT = "tokenizer-input"
KEPT_SANITIZER_INPUT = "sanitizer-input"
KEPT = (KEPT_TOKENIZER_INPUT, KEPT_SANITIZER_INPUT)

STEP_VERSION = "1"
STEP_RELEASE = "replica.release"


@dataclass(frozen=True)
class Uncaptured:
    """A build made with capture off: what it was built from, its folder and its
    summary. ``moved_to`` gives the same build at another place, for a caller that
    copied the folder to change a file before a recovery."""

    plan: BuildPlan
    out: Path
    summary: dict
    options: BuildOptions
    spec: Path | None = None

    def moved_to(self, out: Path) -> "Uncaptured":
        return replace(self, out=Path(out))


class _Registry:
    """The names a recovery registered, as ``StoreCapture`` keeps them."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.instances: dict[str, str] = {}
        self.files: dict[str, str] = {}
        self.name_of_file: dict[str, str] = {}

    def name(self, store_name: str) -> str:
        return self.prefix + store_name

    def add(self, relative: str, name: str, instance: str) -> None:
        self.instances[name] = instance
        self.files[name] = relative
        self.name_of_file[relative] = name

    def earlier(self, listed: Sequence[str]) -> list[str]:
        return [self.name_of_file[r] for r in listed if r in self.name_of_file]


def _release_files(plan: BuildPlan, options: BuildOptions, release) -> list[str]:
    """The record files a release's inputs come from, in the order the inputs
    first name them: the files its copy step can have read."""
    file_of: dict[str, str] = {}
    for source in plan.sources:
        for record_id in source.records:
            file_of[record_id] = f"sources/{source.key}.jsonl"
    for gfile in plan.generation:
        for row in gfile.rows:
            file_of[row.id] = f"gen/{gfile.name}.jsonl"
    wanted = [options.changes.swap.get(record_id, record_id) for record_id in release.inputs]
    return list(dict.fromkeys(file_of[record_id] for record_id in wanted
                              if record_id in file_of))


def _tokenized(built: Uncaptured) -> bool:
    return bool(built.summary.get("tokenized"))


def recover_build(
    store_path: Path,
    built: Uncaptured,
    *,
    kept: str = KEPT_TOKENIZER_INPUT,
    also_token_matching: bool = False,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> dict:
    """Put a build made without capture into the store at ``store_path``.

    ``kept`` says which input of each shard the run kept: its tokenizer input (the
    default), or only the chunk as the sanitizer read it. ``also_token_matching``
    recovers the token-matching map beside the position-mode map of every shard,
    so that the two can be compared (plan Check 3). A map whose verification
    fails is stored as it is, "unknown", and reported; it does not stop the
    recovery.

    Returns ``{"instances", "files", "recovery"}``: each registered name with its
    instance id, each name with the file it was registered from (relative to the
    build folder; for a shard the prefix of its two files), and what each call
    found."""
    if kept not in KEPT:
        raise SpecError(f"a run kept its {' or its '.join(KEPT)}, got {kept!r}")
    plan, options, out = built.plan, built.options, Path(built.out)
    if built.summary.get("capture"):
        raise SpecError("this build was made with capture on; a recovery is for a run "
                        "made without it")
    tokenizer = plan.tokenizer
    if _tokenized(built):
        pipeline_links.map_settings(tokenizer)
        if megatron_env is None:
            megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    registry = _Registry(options.store_name_prefix)
    report: dict = {"kept": kept, "releases": {}, "shards": {}, "generation": []}
    store = pipeline_links.open_store(store_path, plan.normaliser)
    try:
        for source in plan.sources:
            relative = f"sources/{source.key}.jsonl"
            name = registry.name(f"src/{source.key}")
            registry.add(relative, name, ww_capture.ingest_jsonl(
                store, out / relative, source=source.source, member=source.member,
                revision=source.revision, name=name))
        for gfile in plan.generation:
            relative = f"gen/{gfile.name}.jsonl"
            if not (out / relative).is_file():
                continue
            name = registry.name(f"gen/{gfile.name}")
            registry.add(relative, name,
                         ww_recovery.register_found_file(store, out / relative, name=name))
            report["generation"].append(name)
        for release in plan.releases:
            if release.name not in built.summary["releases"]:
                continue
            relative = f"releases/{release.name}.jsonl"
            name = registry.name(release.name)
            # A release found afterwards keeps its pipeline role, so that a listing
            # covers it; every other found file is an intermediate one.
            registry.add(relative, name, ww_recovery.register_found_file(
                store, out / relative, name=name, role=ww_capture.ROLE_RELEASE))
            step = ww_capture.step_ref(STEP_RELEASE, STEP_VERSION,
                                       {"dedup": release.dedup, "normaliser": plan.normaliser})
            earlier = registry.earlier(_release_files(plan, options, release))
            result = ww_recovery.recover_copy_step(store, name, earlier, step=step)
            report["releases"][name] = {
                "earlier": earlier, "recovered": result.recovered,
                "candidates": result.candidates, "no_edge": result.no_edge}
        for imported in plan.imports:
            relative = f"imports/{imported.name.split('/')[-1]}.jsonl"
            if not (out / relative).is_file():
                continue
            # A text-only import is registered with its file family in every kind
            # of run: the family is the team's statement, not a wrapper's.
            name = registry.name(imported.name)
            registry.add(relative, name, ww_capture.import_jsonl(
                store, out / relative, name=name, member_name=imported.name,
                file_family=(imported.family_member, imported.evidence)))
        _recover_layouts(store, built, registry, report, kept=kept,
                         also_token_matching=also_token_matching, megatron_env=megatron_env)
    finally:
        store.close()
    report["mixtures_not_registered"] = sorted(built.summary.get("mixtures", {}))
    return {"instances": registry.instances, "files": registry.files, "recovery": report}


def _recover_layouts(store, built: Uncaptured, registry: _Registry, report: dict, *,
                     kept: str, also_token_matching: bool, megatron_env) -> None:
    plan, out = built.plan, Path(built.out)
    entries = [(layout, entry)
               for layout in built.summary["layouts"]
               for entry in built.summary["layouts"][layout]["shards"]
               if entry["chunk_lines"]]
    if not entries:
        return
    tokenize = None
    if _tokenized(built):
        tokenize = megatron_tokenizer.line_tokenizer(
            megatron_env, plan.tokenizer, out / WORK_DIR / "recover" / "line-tokenizer")
    try:
        for layout, entry in entries:
            # The summary's store name already carries the build's name prefix.
            files = shard_files(out, layout, entry)
            shard_report: dict = {}
            report["shards"][files.store_name] = shard_report
            shard_report["sanitizer_input"] = pipeline_links.register_sanitizer_input(
                store, files, registry.earlier(files.listed))
            registry.add(files.sanitizer_input_relative, files.sanitizer_input_name,
                         shard_report["sanitizer_input"]["instance"])
            input_name = files.sanitizer_input_name
            sanitizer = None
            if entry["empty"]:
                continue  # every line was skipped: no tokenizer input and no shard
            if kept == KEPT_TOKENIZER_INPUT:
                shard_report["tokenizer_input"] = pipeline_links.register_tokenizer_input(
                    store, files, plan.normaliser)
                registry.add(files.tokenizer_input_relative, files.tokenizer_input_name,
                             shard_report["tokenizer_input"]["instance"])
                input_name = files.tokenizer_input_name
            else:
                sanitizer = pipeline_links.sanitizer_run(files)
            if tokenize is None:
                continue
            instance = store.register_shard(
                files.prefix, name=files.store_name,
                step=pipeline_links.tokenize_step(plan.tokenizer),
                inputs=[registry.instances[input_name]])
            registry.add(files.prefix_relative, files.store_name, instance)
            result = ww_tokenmap.recover_map(
                store, shard=files.store_name, input=input_name,
                settings=pipeline_links.map_settings(plan.tokenizer), tokenize=tokenize,
                sanitizer=sanitizer, also_token_matching=also_token_matching)
            shard_report.update(pipeline_links.maps_report(result))
    finally:
        if tokenize is not None:
            tokenize.close()


def add_recovery_maps(
    store_path: Path,
    out: Path,
    plan: BuildPlan,
    summary: dict,
    *,
    also_token_matching: bool = True,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> dict:
    """Recover the maps of the shards a captured build registered, into the
    recovery namespace of the same store, over the same tokenizer inputs. A shard
    then holds its capture map, its position-mode recovery map and, with
    ``also_token_matching``, its token-matching recovery map side by side
    (plan Check 3; DECISIONS.md D-20). Returns, per shard name, what the call
    found."""
    if not summary.get("capture") or not summary.get("tokenized"):
        raise SpecError("recovery maps are added beside the capture maps of a build that "
                        "was captured and tokenized")
    if megatron_env is None:
        megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    out = Path(out)
    report: dict = {}
    store = pipeline_links.open_store(store_path, plan.normaliser)
    try:
        with megatron_tokenizer.line_tokenizer(
                megatron_env, plan.tokenizer,
                out / WORK_DIR / "recover" / "line-tokenizer") as tokenize:
            for layout in summary["layouts"]:
                for entry in summary["layouts"][layout]["shards"]:
                    if entry["empty"]:
                        continue
                    files = shard_files(out, layout, entry)
                    result = ww_tokenmap.recover_map(
                        store, shard=files.store_name, input=files.tokenizer_input_name,
                        settings=pipeline_links.map_settings(plan.tokenizer),
                        tokenize=tokenize, also_token_matching=also_token_matching)
                    report[files.store_name] = pipeline_links.maps_report(result)
    finally:
        store.close()
    return report
