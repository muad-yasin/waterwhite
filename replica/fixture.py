"""The hand-checked fixture and the stores of the hand-written table, in one call.

    from replica.fixture import build_captured_fixture, build_table_store

    built = build_captured_fixture(folder)               # everything the spec holds
    built = build_captured_fixture(folder, tokenize=True)  # needs the Megatron environment
    built = build_table_store(folder, "main")            # one store of the hand-written table

``folder`` receives two things side by side: ``store.sqlite``, the evidence store,
and ``build/``, the build folder. The store keeps the build's paths relative to its
own folder, so ``folder`` can be moved or copied as a whole. Every record file
under ``build/`` has its manifest beside it (``<file>.manifest.jsonl``).

A run made without capture is two calls: ``build_uncaptured`` writes the files
and no store, and ``recover_uncaptured`` puts them into a store afterwards
(``replica.recover``). Between the two a caller may change a file
(``replica.setups``).

docs/replica.md section 5 says what each store holds afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import megatron_tokenizer
from .build import HARNESS_CONFINED, BuildOptions, build, check_build, require_new_folder
from .errors import ManifestError, OutputError, SpecError
from .plan import (
    LAYOUT_PARTITIONED,
    LAYOUT_ROUND_ROBIN,
    LAYOUTS,
    BuildPlan,
    load_fixture_spec,
)
from .run_plan import load_runs_plan
from .shard_files import SUFFIX_TOKENIZER_INPUT
from .shard_index import read_shard_index

STORE_FILE = "store.sqlite"
BUILD_DIR = "build"

HARNESS_IN_PROCESS = "in-process"


def default_spec_path() -> Path:
    """The fixture spec of the repository this package lies in."""
    return Path(__file__).resolve().parent.parent / "fixture" / "spec" / "fixture_spec.toml"


@dataclass(frozen=True)
class StoreVariant:
    """One store of the hand-written table (fixture/table/README.md section 4), as
    a selection of what the fixture spec holds.

    ``generation`` False leaves the generation outputs out of the store: their
    files are written and nothing about them is registered. ``tokenize`` says
    whether the store holds shards, so that building it runs Megatron's
    preprocessing. ``capture`` False is the run made without capture: the files
    are made with no wrapper and unseeded shuffles, and recovered afterwards.
    ``scratch`` names a scratch store, which is no build of the fixture's
    pipeline (``replica.scratch``). ``runs`` names the runs of the spec that are
    registered in the store (by ``replica.runs`` and ``replica.register_runs``, not
    by the build): the store lists the joins those runs use as verified."""

    releases: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    layouts: tuple[str, ...] = ()
    generation: bool = True
    tokenize: bool = False
    capture: bool = True
    scratch: str | None = None
    runs: tuple[str, ...] = ()


# The stores of the table. Each holds the four source files unless it is a scratch
# store; what else it holds is listed here.
TABLE_STORES: dict[str, StoreVariant] = {
    # gen/P, gen/T, gen/S, R1 and R2; the three shards of the partitioned layout
    # with their capture maps and the five mixtures. No swapped release, no import.
    "main": StoreVariant(releases=("R1", "R2"), layouts=(LAYOUT_PARTITIONED,), tokenize=True,
                         runs=("E", "D", "Y", "K")),
    # Run E's data in the round-robin layout: the generation outputs (the chain of
    # T runs through them), the three shards E-rr/shard-1 to E-rr/shard-3 with
    # their capture maps, and the two mixtures. No release.
    "rr": StoreVariant(layouts=(LAYOUT_ROUND_ROBIN,), tokenize=True, runs=("E-rr",)),
    # The main store's sources and generation outputs, with R1-swap beside R1.
    "swap": StoreVariant(releases=("R1", "R1-swap")),
    # The sources plus the text-only import of T. The generation outputs are not
    # registered, so the store knows T only through the import's file family.
    "import": StoreVariant(imports=("import/T",), generation=False),
    # The main store's files made with no wrapper and recovered afterwards.
    "no-capture": StoreVariant(releases=("R1", "R2"), layouts=(LAYOUT_PARTITIONED,),
                               tokenize=True, capture=False),
    "scratch-500": StoreVariant(tokenize=True, scratch="scratch-500"),
    "scratch-notokens": StoreVariant(tokenize=True, scratch="scratch-notokens"),
}

# The stores that are a store of the table plus the release of one extension (not stores of the
# table's own list, so they are not in ``TABLE_STORES``): the key is the build's. Each maps to the
# store of the table it is built from and the extension releases added to it afterwards
# (``[extension_releases.<name>]`` of the fixture spec). ``main+generating-model`` is the main
# build, then the release R3 of the generating-model rule (docs/replica.md). The main build is
# made exactly as before: R3 writes files of its own and touches none of main's.
EXTENSION_STORES: dict[str, tuple[str, tuple[str, ...]]] = {
    "main+generating-model": ("main", ("R3",)),
}

# The stores of the table the replica cannot build at all, each with what is
# missing. There is none now; the name stays for a caller that asks.
TABLE_STORES_NOT_BUILT: dict[str, str] = {}

# What the call that builds a store of the table does not itself register of what
# the table lists for it (README section 4), each with where it comes from. A row
# that needs one of these needs the runs made and registered after the build.
TABLE_STORES_LACK: dict[str, str] = {
    "main": "runs E, D, Y and K with their segments, run records and step traces, and "
            "the checkpoints E@6, E@9, D@3, Y@6, K@3, K@8 and K@40: the build registers no "
            "run; replica.runs.build_runs makes them and replica.register_runs.register_runs "
            "registers them through the tool's register-run (docs/replica.md section 13)",
    "rr": "the two segments E-rr/seg-1 and E-rr/seg-2 with their step traces, and the "
          "checkpoint E-rr@6: the build registers no run; replica.runs.build_runs makes "
          "them and replica.register_runs.register_runs registers them through the tool's "
          "register-run (docs/replica.md section 13)",
}


@dataclass(frozen=True)
class CapturedFixture:
    """What a build with a store made.

    ``store`` is the store file. ``out`` is the build folder: every record file in
    it has its manifest beside it, and ``manifest(relative)`` gives the path of
    one. ``instances`` maps each name registered in the store (``src/A``,
    ``gen/P``, ``R1``, ``E/shard-1.input``, ``E/shard-1``, ``E/mix-1``) to its
    instance id, and ``files`` maps the same names to what each was registered
    from, relative to the build folder: a record file, for a shard the prefix of
    its two files, for a mixture its folder. ``summary`` is the build's summary
    (docs/replica.md section 9). ``spec`` and ``options`` are what the build ran
    with, so that the same build can be run again (``register_again``); ``plan``
    is what it was built from."""

    store: Path
    out: Path
    instances: dict[str, str]
    summary: dict
    files: dict[str, str] = field(default_factory=dict)
    spec: Path | None = None
    options: BuildOptions | None = None
    plan: BuildPlan | None = None

    def manifest(self, relative: str) -> Path:
        """The manifest of the record file at ``relative`` under the build folder,
        for example ``releases/R1.jsonl``."""
        return self.out / f"{relative}.manifest.jsonl"

    def file(self, name: str) -> Path:
        """What was registered under ``name``: a record file, the prefix of a
        shard's two files, or a mixture's folder."""
        try:
            return self.out / self.files[name]
        except KeyError:
            raise ManifestError(f"this build registered nothing under the name {name!r}; "
                                f"registered: {sorted(self.files)}") from None

    def manifest_rows(self, name: str) -> list[dict]:
        """The manifest rows of the file registered under ``name``, one per line of
        the file: ``{"line", "handle", "origin"}``."""
        path = Path(f"{self.file(name)}.manifest.jsonl")
        if not path.is_file():
            raise ManifestError(f"{name} was not registered from a record file, so it has "
                                "no manifest")
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f.read().split("\n") if line != ""]

    def line_of(self, name: str, handle: str) -> int:
        """The line, counted from 1, that holds the record with ``handle`` in the
        file registered under ``name``. The answer comes from the manifest the
        replica wrote beside the file, never from the store. A handle that no line
        or more than one line carries is an error."""
        lines = [row["line"] for row in self.manifest_rows(name) if row["handle"] == handle]
        if len(lines) != 1:
            raise ManifestError(f"the manifest of {name} holds the handle {handle!r} on "
                                f"{len(lines)} lines; exactly one was expected")
        return lines[0]

    def ordinal_of(self, shard: str, handle: str) -> int:
        """The ordinal, counted from 0, of the document of ``shard`` that holds the
        record with ``handle``. The answer comes from the replica's own records:
        the manifest of the shard's tokenizer input gives the line, and the
        shard's index file, read by the replica's own reader, says which lines
        gave tokens. A line that gave no tokens is no document and has no
        ordinal."""
        line = self.line_of(shard + SUFFIX_TOKENIZER_INPUT, handle)
        tokens = read_shard_index(Path(f"{self.file(shard)}.idx")).document_tokens
        if line > len(tokens) or tokens[line - 1] == 0:
            raise ManifestError(f"line {line} of the tokenizer input of {shard} gave no "
                                "tokens, so it is no document")
        return sum(1 for count in tokens[:line - 1] if count)


def build_captured_fixture(
    folder: Path,
    *,
    tokenize: bool = False,
    spec: Path | None = None,
    layouts: tuple[str, ...] = LAYOUTS,
    releases: tuple[str, ...] | None = None,
    imports: tuple[str, ...] | None = None,
    capture_generation: bool = True,
    generation_harness: str = HARNESS_IN_PROCESS,
    parallel: int = 1,
    tokenizer_timeout: float = 900.0,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
    verified_joins: tuple[tuple[str, str, str], ...] = (),
    extension_releases: tuple[str, ...] = (),
) -> CapturedFixture:
    """Build the hand-checked fixture into ``folder`` with capture on.

    ``verified_joins`` are the (release tag, sampler type, sampler class) rows the
    new store lists as joins verified (plan F6); none by default. ``extension_releases``
    names the releases of an extension (``[extension_releases.<name>]`` of the spec) that the
    build makes after everything else, in the same store (``replica.capture_hooks``,
    ``build_extension_release``): none by default; a release made by models needs the
    confined harness.

    ``folder`` must be new or empty. With ``tokenize`` False the build stops before
    Megatron's preprocessing: no shards and no mixtures. With ``tokenize`` True it
    needs the Megatron environment (``WATERWHITE_MEGATRON_ENV``, or ``megatron_env``)
    and registers each shard with its capture map and each mixture's snapshot.
    ``releases`` and ``imports`` name the releases and imports to build and register
    (None: every one the spec holds), so a caller can make a store without
    ``R1-swap`` or without ``import/T``. ``capture_generation`` False writes the
    generation outputs and registers nothing about them. ``generation_harness``
    "confined" runs each generation job as a program in the tool's sandbox and is
    refused where the sandbox cannot be built."""
    folder = Path(folder)
    require_new_folder(folder, "a captured fixture")
    spec = Path(spec) if spec is not None else default_spec_path()
    if tokenize and megatron_env is None:
        # Refuse a missing environment before the store file is made.
        megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    options = BuildOptions(layouts=layouts, tokenize=tokenize, parallel=parallel,
                           tokenizer_timeout=tokenizer_timeout, releases=releases,
                           imports=imports, capture_generation=capture_generation,
                           generation_harness=generation_harness)
    return _captured_build(folder / STORE_FILE, folder / BUILD_DIR, spec, options, megatron_env,
                           verified_joins=verified_joins, extension_releases=extension_releases)


def build_table_store(
    folder: Path,
    store: str = "main",
    *,
    tokenize: bool = False,
    spec: Path | None = None,
    parallel: int = 1,
    tokenizer_timeout: float = 900.0,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
    generation_harness: str = HARNESS_CONFINED,
) -> CapturedFixture:
    """Build one store of the hand-written table into ``folder``.

    ``store`` is a name of fixture/table/README.md section 4 (``TABLE_STORES``).
    A name the table does not define is refused.

    ``tokenize`` False (the default, as before shards were registered) stops
    before Megatron's preprocessing and leaves the shards, their maps and the
    mixtures out: a caller that asks for ``main``, ``rr`` or ``no-capture`` with no
    flag gets the store without them, and runs no tokenizer. A scratch store is
    nothing but a shard and is refused by name then. ``tokenize`` True builds the
    store as the table lists it (``StoreVariant.tokenize``): a store that holds
    shards (``main``, ``rr``, ``no-capture`` and the two scratch stores) is
    tokenized and needs the Megatron environment.

    ``generation_harness`` is "confined" by default: the table's rows compare
    answers on these stores, and only a log the confined harness wrote establishes a
    bound (Decision 13), so a store built with capture runs its generation jobs in
    the tool's sandbox, and the build is refused where the sandbox cannot be made.

    ``TABLE_STORES_LACK`` says what the call does not register of the table's list
    (the runs). A store that is to hold runs (``StoreVariant.runs``) lists the joins
    of those runs as verified, so that registering them gives a trace route.

    A name of ``EXTENSION_STORES`` (``main+generating-model``) builds the store of the table it
    names and then the releases of an extension in the same store, so that the main build is
    exactly the one of ``main``."""
    extension_releases: tuple[str, ...] = ()
    if store in EXTENSION_STORES:
        store, extension_releases = EXTENSION_STORES[store]
    if store in TABLE_STORES_NOT_BUILT:
        raise SpecError(f"the store {store!r} of the hand-written table cannot be built yet: "
                        f"{TABLE_STORES_NOT_BUILT[store]}")
    if store not in TABLE_STORES:
        raise SpecError(f"the hand-written table defines no store {store!r}; it defines "
                        f"{sorted(TABLE_STORES) + sorted(TABLE_STORES_NOT_BUILT)}")
    variant = TABLE_STORES[store]
    folder = Path(folder)
    spec = Path(spec) if spec is not None else default_spec_path()
    if variant.scratch is not None:
        if not tokenize:
            raise SpecError(f"the store {store!r} holds one tokenized shard and nothing that "
                            "can be built without tokenization")
        return _scratch_store(folder, variant.scratch, spec, tokenizer_timeout, megatron_env)
    if not variant.capture:
        options = BuildOptions(layouts=variant.layouts, seeded=False, tokenize=tokenize,
                               parallel=parallel, tokenizer_timeout=tokenizer_timeout,
                               releases=variant.releases, imports=variant.imports)
        built = build_uncaptured(folder, load_fixture_spec(spec), options, spec=spec,
                                 megatron_env=megatron_env)
        return recover_uncaptured(folder, built, megatron_env=megatron_env)
    joins = load_runs_plan(spec).verified_joins(variant.runs) if variant.runs else ()
    return build_captured_fixture(
        folder, tokenize=tokenize, spec=spec, layouts=variant.layouts, verified_joins=joins,
        releases=variant.releases, imports=variant.imports,
        capture_generation=variant.generation,
        # A store that captures no generation step runs no job in a harness.
        generation_harness=generation_harness if variant.generation else HARNESS_IN_PROCESS,
        parallel=parallel, tokenizer_timeout=tokenizer_timeout, megatron_env=megatron_env,
        extension_releases=extension_releases)


SET_STORES = ("no-capture",)


def build_set_store(
    folder: Path,
    set_folder: Path,
    store: str = "no-capture",
    *,
    layout: str = LAYOUT_ROUND_ROBIN,
    tokenizer_spec: Path | None = None,
    parallel: int = 1,
    tokenizer_timeout: float = 900.0,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> CapturedFixture:
    """Build a generated set into ``folder`` as a store of the same kind as a store
    of the table, so that Check 6's generated rows can ask it.

    Only ``"no-capture"`` is built: the set is built with capture off and unseeded
    shuffles, tokenized with ``tokenizer_spec`` (default: this repository's fixture
    spec), and recovered by the rule of the table's store of that name
    (``recover_uncaptured`` with the tokenizer input kept and the position-mode map
    only). The store is ``folder/store.sqlite`` and the build folder
    ``folder/build``. The names are the replica's (docs/replica.md section 5):
    ``set-rr/<shard>`` and ``set-rr/<shard>.input`` for each shard of the layout,
    ``R1``, ``src/<source>`` and ``gen/<job>``. ``folder`` must be new or empty;
    a store kind other than ``"no-capture"`` is refused by name."""
    from .sets import build_uncaptured_set

    if store not in SET_STORES:
        raise SpecError(f"a generated set is built as the store {SET_STORES[0]!r}, not "
                        f"{store!r}; the table's other stores are built by build_table_store")
    if layout not in LAYOUTS:
        raise SpecError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    folder = Path(folder)
    built = build_uncaptured_set(folder, set_folder, tokenizer_spec=tokenizer_spec,
                                 layouts=(layout,), parallel=parallel,
                                 tokenizer_timeout=tokenizer_timeout, megatron_env=megatron_env)
    return recover_uncaptured(folder, built, megatron_env=megatron_env)


def build_uncaptured(folder: Path, plan: BuildPlan, options: BuildOptions, *,
                     spec: Path | None = None,
                     megatron_env: megatron_tokenizer.MegatronEnv | None = None):
    """Build ``plan`` into ``folder/build`` with capture off: the run made without
    capture (plan Check 6). No store is made and nothing of the tool is called but
    its normaliser. ``folder`` must be new or empty. Returns the
    ``replica.recover.Uncaptured`` build, which ``recover_uncaptured`` takes."""
    from .recover import Uncaptured

    folder = Path(folder)
    require_new_folder(folder, "a build made without capture")
    out = folder / BUILD_DIR
    summary = build(plan, out, options, megatron_env=megatron_env)
    return Uncaptured(plan=plan, out=out, summary=summary, options=options, spec=spec)


def recover_uncaptured(
    folder: Path,
    built,
    *,
    kept: str = "tokenizer-input",
    also_token_matching: bool = False,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
) -> CapturedFixture:
    """Put a build made without capture into the new store ``folder/store.sqlite``
    through the tool's recovery calls (``replica.recover.recover_build``).

    ``built`` is what ``build_uncaptured`` returned, or the same build at the
    place a caller copied it to (``built.moved_to``); its folder is expected at
    ``folder/build``, beside the store, so that the store's paths are relative.
    ``kept`` and ``also_token_matching`` are those of ``recover_build``."""
    from .recover import recover_build

    folder = Path(folder)
    store_path = folder / STORE_FILE
    if store_path.exists():
        raise OutputError(f"{store_path} exists; a recovery writes a new store")
    recovered = recover_build(store_path, built, kept=kept,
                              also_token_matching=also_token_matching,
                              megatron_env=megatron_env)
    summary = dict(built.summary)
    summary["instances"] = dict(recovered["instances"])
    summary["files"] = dict(recovered["files"])
    summary["recovery"] = recovered["recovery"]
    return CapturedFixture(store=store_path, out=Path(built.out),
                           instances=dict(recovered["instances"]), summary=summary,
                           files=dict(recovered["files"]), spec=None, options=None,
                           plan=built.plan)


def add_recovery_maps(built: CapturedFixture, *, also_token_matching: bool = True,
                      megatron_env: megatron_tokenizer.MegatronEnv | None = None) -> dict:
    """Recover the maps of every shard of a captured, tokenized build into the
    recovery namespace of its store, beside the capture maps (plan Check 3;
    ``replica.recover.add_recovery_maps``). Returns, per shard, what was found."""
    from .recover import add_recovery_maps as add

    if built.plan is None:
        raise SpecError("this build does not say what it was built from")
    return add(built.store, built.out, built.plan, built.summary,
               also_token_matching=also_token_matching, megatron_env=megatron_env)


def register_again(built: CapturedFixture, name: str = "again", *,
                   megatron_env: megatron_tokenizer.MegatronEnv | None = None) -> CapturedFixture:
    """Run the build of ``built`` a second time into the same store.

    The second build writes its files into a new folder ``build-<name>`` beside
    the first build folder and registers every artifact again under the same
    names, with the same steps, inputs and edges. The files have the same bytes at
    another path; a path is part of no instance id. ``megatron_env`` is the
    environment a tokenizing build runs with (default: the process's). Returns the
    second build."""
    if built.spec is None or built.options is None:
        raise SpecError("this build does not say what it ran with, so it cannot be run again")
    out = built.store.parent / f"{BUILD_DIR}-{name}"
    if out.exists():
        raise OutputError(f"{out} exists; a second build needs a new folder")
    if built.options.tokenize and megatron_env is None:
        megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    return _captured_build(built.store, out, built.spec, built.options,
                           megatron_env if built.options.tokenize else None)


def _captured_build(store_path: Path, out: Path, spec: Path, options: BuildOptions,
                    megatron_env, plan: BuildPlan | None = None,
                    verified_joins: tuple[tuple[str, str, str], ...] = (),
                    extension_releases: tuple[str, ...] = ()) -> CapturedFixture:
    # The store, the wrappers and the harness are imported here and not at the top:
    # a build with capture off never loads them.
    from .capture_hooks import StoreCapture

    if plan is None:
        plan = load_fixture_spec(spec)
    # Everything the build would refuse is refused before the store file is made.
    check_build(plan, out, options, capturing=True, megatron_env=megatron_env)
    if extension_releases and (options.generation_harness != HARNESS_CONFINED
                               or not options.capture_generation):
        raise SpecError("a release made by models runs its jobs in the confined harness "
                        "with the generation step captured")
    capture = StoreCapture.open(store_path, normaliser=plan.normaliser,
                                verified_joins=verified_joins)
    try:
        summary = build(plan, out, options, capture=capture, megatron_env=megatron_env)
        if extension_releases:
            # The R3 step: after the main build, in the same store; new files only.
            from .capture_hooks import build_extension_release

            for release in extension_releases:
                build_extension_release(capture, spec, out, release,
                                        timeout=options.generation_timeout)
            summary["instances"] = dict(capture.instances)
            summary["files"] = dict(capture.files)
    finally:
        capture.close()
    return CapturedFixture(store=store_path, out=out, instances=dict(summary["instances"]),
                           summary=summary, files=dict(summary["files"]), spec=spec,
                           options=options, plan=plan)


def _scratch_store(folder: Path, which: str, spec: Path, tokenizer_timeout: float,
                   megatron_env) -> CapturedFixture:
    from . import scratch

    plan = load_fixture_spec(spec)
    if megatron_env is None:
        megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    scratch_builder = {"scratch-500": scratch.build_scratch_500,
                       "scratch-notokens": scratch.build_scratch_notokens}[which]
    made = scratch_builder(folder, plan, megatron_env, store_file=STORE_FILE,
                           build_dir=BUILD_DIR, tokenizer_timeout=tokenizer_timeout)
    summary = dict(made["summary"])
    summary["instances"] = dict(made["instances"])
    summary["files"] = dict(made["files"])
    return CapturedFixture(store=folder / STORE_FILE, out=folder / BUILD_DIR,
                           instances=dict(made["instances"]), summary=summary,
                           files=dict(made["files"]), spec=None, options=None)
