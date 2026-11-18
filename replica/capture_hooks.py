"""The capture-on path of the replica.

The build calls an object of ``StoreCapture`` at its capture points (plan 5.4):

- ingest of each source file          -> ``waterwhite.capture.ingest_jsonl``
- each generation job                 -> ``waterwhite.harness.run_job``, or with the
  build option for it ``waterwhite.confine.run_confined_job``; the output file is
  then registered with one generation edge per row, written by
  ``waterwhite.capture.generation_edge`` from what the job's output states it used
- the dedup selection of a release    -> ``waterwhite.capture.record_jsonl`` with
  the edges the selection step states: for a group of equivalent copies an "any"
  edge that names every copy with the kept one selected, and a drop for each
  loser; for a row with no equivalent copy a single-parent edge
- the text-only import                -> ``waterwhite.capture.import_jsonl``
  (not among the capture points plan 5.4 lists: the fixture's import of plan 5.1
  and L1)
- each tokenized shard                -> ``waterwhite.tokenmap.map_tokenized_shard``:
  the shard is registered and its capture map written over its tokenizer input
- each mixture                        -> ``Store.register_mixture``: the snapshot of
  the shard files the trainer is handed, in its order, each with its digest

The copy steps (list, shuffle, chunk, merge) and the sanitizer are recovery points
(plan 5.4): the build states no edge for the files they write. At the tokenization
capture point those files are registered as files found afterwards and their
links are recovered through ``waterwhite.recovery`` (``replica.pipeline_links``),
because the capture map names the lines of the tokenizer input.

Run registration is no part of the build: the runs are made and registered after
it, by ``replica.runs`` and ``replica.register_runs`` (docs/replica.md section 13).
The last method of the class is a hook the build still calls; it holds no code.

It is written against docs/interfaces.md sections 4 and 5, docs/capture.md,
docs/store.md, docs/tokenmap.md, docs/recovery.md, docs/confine.md and
DECISIONS.md D-14, D-15 and D-21. The build imports this module only when capture
is on. Beside it, ``replica.pipeline_links``, ``replica.recover`` and
``replica.scratch`` write to the tool; apart from ``replica.probe_run``, which opens a
store to register one mixture, nothing else in the replica imports the store, the
wrappers or the harness.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from waterwhite import capture as ww_capture
from waterwhite import harness as ww_harness
from waterwhite import tokenmap as ww_tokenmap
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from . import pipeline_links
from .build import json_row
from .built_from import MANIFEST_FILE, _row_text
from .errors import CaptureError, InputError, SpecError
from .records import Record, read_records
from .shard_files import ShardFiles
from .steps import Kept

STEP_VERSION = "1"
STEP_GENERATION = "replica.generation"
STEP_RELEASE = "replica.release"

# How a generation job is run (``BuildOptions.generation_harness``).
HARNESS_IN_PROCESS = "in-process"
HARNESS_CONFINED = "confined"
HARNESSES = (HARNESS_IN_PROCESS, HARNESS_CONFINED)

# The fixed-output job as a program of its own, for the confined harness.
_FIXED_OUTPUT_JOB = Path(__file__).with_name("jobs") / "fixed_output.py"

# The model-calling job of the R3 step, as a program of its own for the confined harness.
_MODEL_CALL_JOB = Path(__file__).with_name("jobs") / "model_call.py"

# The name the R3 step gives the generated file in the store (gen/<name>). The plan names the
# release R3; the name of the file it copies from is the replica's (spec F24).
EXTENSION_GENERATED_FILE = "R3-models"

# What a mixture's settings say beside its shards: the trainer is handed the files
# of one folder (plan 3.3 "Mixture selection", Decision 27).
MIXTURE_SELECTION = "directory"


def require_sandbox() -> None:
    """Refuse, before anything is written, a build that asks for the confined
    harness on a machine where its sandbox cannot be built. The refusal carries
    the tool's own error name, "sandbox unavailable", and what the sandbox said.
    No job is ever run unconfined in its place."""
    from waterwhite import confine as ww_confine

    try:
        ww_confine.find_sandbox()
    except ww_confine.SandboxUnavailable as unavailable:
        raise CaptureError(f"{unavailable.name}: {unavailable.detail}") from unavailable


class StoreCapture:
    """Writes one build's records into a store.

    ``instances`` maps each name this object registered to its instance id, and
    ``files`` maps the same names to what each was registered from, as a path
    relative to the build folder: a record file, for a shard the prefix of its two
    files, for a mixture its folder."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.instances: dict[str, str] = {}
        self.files: dict[str, str] = {}
        self._by_file: dict[str, str] = {}  # build-relative file path -> instance id
        self._name_of_file: dict[str, str] = {}  # build-relative file path -> store name
        # output id -> the harness result of the job that made it
        self._jobs: dict[str, ww_harness.JobResult] = {}

    @classmethod
    def open(cls, store_path: Path, *, normaliser: str,
             verified_joins=()) -> "StoreCapture":
        """Create the store at ``store_path`` with ``normaliser``, or open the one
        that is there. A store whose fingerprints were taken after another
        normaliser is refused: the sanitizer of this build and the store's
        fingerprints would not agree (plan 3.1, F8).

        A build's paths are stored relative to the store's folder when the build
        folder lies in it (docs/interfaces.md section 4). ``verified_joins`` are the
        (release tag, sampler type, sampler class) rows a new store lists (plan F6);
        a store that exists keeps the list it was made with."""
        return cls(pipeline_links.open_store(Path(store_path), normaliser,
                                             verified_joins=verified_joins))

    def close(self) -> None:
        self.store.close()

    # --- References ----------------------------------------------------------------

    def _registered(self, artifact: str, name: str, instance: str) -> None:
        self._by_file[artifact] = instance
        self._name_of_file[artifact] = name
        self.instances[name] = instance
        self.files[name] = artifact

    def _ref(self, record: Record):
        instance = self._by_file.get(record.artifact or "")
        if instance is None:
            raise CaptureError(f"{record.artifact} was not registered before a step named "
                               f"its line {record.line} as a parent")
        return ww_capture.occurrence(instance, record.line)

    @staticmethod
    def _inputs(refs) -> list[str]:
        """The artifacts a step read, each once, in the order it first named them.
        They cover every parent of every edge the step states (DECISIONS.md D-15)."""
        return list(dict.fromkeys(ref.instance_id for ref in refs))

    # --- Capture points --------------------------------------------------------------

    def ingest(self, path: Path, artifact: str, source, name: str) -> None:
        self._registered(artifact, name, ww_capture.ingest_jsonl(
            self.store, path, source=source.source, member=source.member,
            revision=source.revision, name=name))

    def generate(self, row, handed_records, log_path: Path, *,
                 harness: str = HARNESS_IN_PROCESS, timeout: float = 120.0) -> str:
        """Run one fixed-output job in the closed-input harness and return the text
        it emits. The job is handed the records as they stand in their registered
        files, reads every one of them through the harness in hand-off order, calls
        no model and states the records the row says it used.

        ``harness`` picks how the job runs: in this process
        (``waterwhite.harness.run_job``), or as a program of its own in the sandbox
        (``waterwhite.confine.run_confined_job``), where it has no file of the
        build and no network and gets each record by asking for it. The job makes
        the same requests in the same order either way; the two logs differ in the
        harness word of the job entry and in nothing else (DECISIONS.md D-30)."""
        handed = tuple(ww_harness.Handed(ref=self._ref(r), text=r.text) for r in handed_records)
        used = tuple(h.ref for h, r in zip(handed, handed_records) if r.ref in row.used)
        if harness == HARNESS_CONFINED:
            result = self._generate_confined(row, handed, handed_records, log_path, timeout)
        elif harness == HARNESS_IN_PROCESS:
            def job(context):
                for ref in context.handed():
                    context.read(ref)
                return ww_harness.JobOutput(text=row.text, used=used)

            result = ww_harness.run_job(job, job_name=row.job_name, handed=handed,
                                        log_path=log_path)
        else:
            raise CaptureError(f"a generation job runs {' or '.join(HARNESSES)}, "
                               f"got {harness!r}")
        if len(result.outputs) != 1:
            raise CaptureError(f"job {row.job_name}: expected one output, "
                               f"got {len(result.outputs)}")
        self._jobs[row.id] = result
        return result.outputs[0].text

    @staticmethod
    def _generate_confined(row, handed, handed_records, log_path: Path, timeout: float):
        """The same job as a program in the sandbox. Its two arguments are the
        fixed output, which stands in for what a model would have written and is
        no handed record, and the places in the hand-off of the records the output
        states it used. The texts of the handed records reach it only through the
        harness."""
        from waterwhite import confine as ww_confine

        positions = [place for place, record in enumerate(handed_records)
                     if record.ref in row.used]
        program = ww_confine.python_program(
            _FIXED_OUTPUT_JOB, [json.dumps(row.text), json.dumps(positions)])
        try:
            return ww_confine.run_confined_job(
                program, job_name=row.job_name, handed=handed, log_path=log_path,
                timeout_seconds=timeout)
        except ww_confine.SandboxUnavailable as unavailable:
            raise CaptureError(f"{unavailable.name}: {unavailable.detail}") from unavailable

    def generate_with_model(self, row: "ModelRow", handed_records, log_path: Path, *,
                            remove_model_entry: bool = False, timeout: float = 120.0) -> str:
        """Run one model-calling job in the confined harness and return the text it emits (the
        R3 step). The job is handed the records as they stand in their registered files, reads
        every one through the harness, calls ``row.model`` once through the client below and
        emits the answer. The client is the model: it reports the row's identifier and answers
        the row's text.

        With ``remove_model_entry`` the model entry line is removed from the harness's own log
        file afterwards, the log is read back, and the texts of the real run are re-attached to
        it: the registered log has model None and is marked compared when its edge is registered
        (``capture.generation_edge`` and the store, the owner's answer on compared texts). A log
        read back from its file alone would carry no texts, establish no bound, and the answer
        would name condition (g) instead of the missing entry. This injects the one specified
        fault into a real log and makes no evidence up."""
        from waterwhite import confine as ww_confine

        handed = tuple(ww_harness.Handed(ref=self._ref(r), text=r.text) for r in handed_records)
        places = [place for place, record in enumerate(handed_records) if record.ref in row.used]
        program = ww_confine.python_program(
            _MODEL_CALL_JOB, [json.dumps(row.model), json.dumps(places)])

        def client(model: str, prompt: str):
            return ww_harness.ModelReply(text=row.text, model=row.model)

        try:
            result = ww_confine.run_confined_job(
                program, job_name=row.job_name, handed=handed, log_path=log_path,
                timeout_seconds=timeout, model_client=client)
        except ww_confine.SandboxUnavailable as unavailable:
            raise CaptureError(f"{unavailable.name}: {unavailable.detail}") from unavailable
        if remove_model_entry:
            result = _without_model_entry(result, log_path)
        if len(result.outputs) != 1:
            raise CaptureError(f"job {row.job_name}: expected one output, "
                               f"got {len(result.outputs)}")
        self._jobs[row.id] = result
        return result.outputs[0].text

    def record_generation(self, path: Path, artifact: str, name: str, made) -> None:
        """Register one generation output file. ``made`` holds, per line, the row,
        the records the job was handed and the record written. Each line gets the
        generation edge of its job: every handed record is an input, one the output
        states it used is a parent, and the edge carries the job's harness log."""
        step = ww_capture.step_ref(STEP_GENERATION, STEP_VERSION,
                                   {"jobs": [row.job_name for row, _, _ in made]})
        edges = []
        handed = []
        for number, (row, _handed_records, _record) in enumerate(made, 1):
            result = self._jobs[row.id]
            edges.append(ww_capture.generation_edge(number, result.outputs[0], result.log, step))
            handed.extend(result.log.handed)
        self._registered(artifact, name, ww_capture.record_jsonl(
            self.store, path, name=name, role=v.ROLE_INTERMEDIATE, step=step,
            inputs=self._inputs(handed), edges=edges))

    def record_release(self, path: Path, artifact: str, name: str, params: dict, lines) -> None:
        """Register a release with the edges its selection step states. ``lines``
        holds one ``steps.Kept`` per line. A line with one copy gets a single-parent
        edge. A line the dedup kept out of several equivalent copies gets an "any"
        edge that names every copy, the kept one selected, and each other copy is a
        drop with the reason "dedup loser"."""
        step = ww_capture.step_ref(STEP_RELEASE, STEP_VERSION, params)
        edges = []
        drops = []
        parents = []
        for number, kept in enumerate(lines, 1):
            if len(kept.copies) == 1:
                ref = self._ref(kept.record)
                parents.append(ref)
                edges.append(ww_capture.copy_edge(number, ref, step))
                continue
            group = [self._ref(copy) for copy in kept.copies]
            edge, losers = ww_capture.selection_edge(number, group, self._ref(kept.record), step)
            parents.extend(group)
            edges.append(edge)
            drops.extend(losers)
        self._registered(artifact, name, ww_capture.record_jsonl(
            self.store, path, name=name, role=v.ROLE_RELEASE, step=step,
            inputs=self._inputs(parents), edges=edges, drops=drops))

    def import_text(self, path: Path, artifact: str, name: str, member_name: str,
                    family_member: str, evidence: str) -> None:
        self._registered(artifact, name, ww_capture.import_jsonl(
            self.store, path, name=name, member_name=member_name,
            file_family=(family_member, evidence)))

    # --- Tokenization and mixtures -----------------------------------------------------

    def shard_tokenized(self, files: ShardFiles, tokenizer, tokenize, normaliser: str) -> dict:
        """The tokenization capture point. A chunk was tokenized into the shard's
        two files from the lines of its tokenizer input.

        In order: the chunk as the sanitizer read it is registered as a found file
        and the copy steps are recovered against the registered files the layout
        lists for this shard; the tokenizer input is registered under
        ``<shard>.input`` and the sanitizer is recovered between the two, with each
        line of the sanitizer's log as a drop whose output is the tokenizer input;
        the shard is registered and its capture map written over the tokenizer
        input, which records a line that gives no tokens as a drop whose output is
        the shard. So a dropped line is recorded once.

        ``tokenizer`` holds the settings of the run and ``tokenize`` is the same
        tokenizer as a function of one text, for the verification. Returns what
        was written, for the build's summary. A capture map that does not verify
        stops the build."""
        store = self.store
        earlier = [self._name_of_file[relative] for relative in files.listed
                   if relative in self._name_of_file]
        report: dict = {
            "listed_not_registered": [r for r in files.listed if r not in self._name_of_file],
            "sanitizer_input": pipeline_links.register_sanitizer_input(store, files, earlier),
        }
        self._registered(files.sanitizer_input_relative, files.sanitizer_input_name,
                         report["sanitizer_input"]["instance"])
        report["tokenizer_input"] = pipeline_links.register_tokenizer_input(
            store, files, normaliser)
        self._registered(files.tokenizer_input_relative, files.tokenizer_input_name,
                         report["tokenizer_input"]["instance"])
        result = ww_tokenmap.map_tokenized_shard(
            store, input=files.tokenizer_input_name, prefix=files.prefix,
            name=files.store_name, step=pipeline_links.tokenize_step(tokenizer),
            settings=pipeline_links.map_settings(tokenizer), tokenize=tokenize)
        self._registered(files.prefix_relative, files.store_name, result.shard_instance_id)
        report.update(pipeline_links.maps_report(result))
        if not result.verified:
            reasons = [reason for outcome in result.maps
                       for reason in outcome.verification.reasons]
            raise CaptureError(f"the capture map of {files.store_name} did not verify: "
                               f"{'; '.join(reasons) or 'no document was compared'}")
        return report

    def mixture_selected(self, folder: Path, relative: str, name: str, shards) -> str:
        """The mixture capture point. A mixture folder was filled. ``shards`` is the
        ordered list the trainer is handed, each entry the shard's store name and
        its file prefix inside the folder; ``relative`` is the folder relative to
        the build folder.

        The snapshot names each shard by its registered instance, in that order,
        with the digest of the two files that lie in the mixture folder (plan
        Decision 27). The store refuses an entry whose digest differs from the
        digest its shard is registered with. The settings say that the trainer is
        handed the files of one folder, and which folder; two mixtures that select
        the same shards from two folders are therefore two mixtures."""
        entries = []
        for position, (store_name, prefix) in enumerate(shards):
            instance = self.instances.get(store_name)
            if instance is None:
                raise CaptureError(f"mixture {name}: the shard {store_name} was not "
                                   "registered before the mixture that selects it")
            entries.append(pipeline_links.snapshot_shard(self.store, position, instance, prefix))
        instance = self.store.register_mixture(
            name=name, shards=entries,
            settings={"selection": MIXTURE_SELECTION, "directory": relative})
        self.instances[name] = instance
        self.files[name] = relative
        return instance

    # --- A hook the build calls: no code ---------------------------------------------------

    def run_segment_ready(self, segment, folder: Path, mixture_name: str) -> None:
        """Hook: the mixture of one run segment of the spec is built. It writes
        nothing: a segment, its run record and its checkpoints are registered after
        the build, once the run wrapper has written them (``replica.register_runs``)."""


# --- The R3 step: a release made by models (the generating-model extension) -----------------


@dataclass(frozen=True)
class ModelRow:
    """One row of the generated file of an extension release (fixture/sources/generation_r3.jsonl):
    the job, the records it is handed (ids of the source files), the ones its output states it
    used, the model the job calls and the text the model answers. It has the fields
    ``StoreCapture.record_generation`` reads of a generation row (``id``, ``job_name``)."""

    id: str
    job_name: str
    handed: tuple[str, ...]
    used: tuple[str, ...]
    model: str
    text: str


def _without_model_entry(result, log_path: Path):
    """The result of a job with the model entry line removed from its log file, the log read back
    and the handed texts of the real run put back on it. The call and reply entries stay, so
    ``read_log`` accepts the file and reads model None (a log with no model entry)."""
    lines = Path(log_path).read_text(encoding="utf-8").split("\n")
    kept = [raw for raw in lines if raw != "" and json.loads(raw).get("kind") != ww_harness.ENTRY_MODEL]
    if len(kept) == len([raw for raw in lines if raw != ""]):
        raise CaptureError(f"{log_path} holds no model entry to remove")
    Path(log_path).write_text("\n".join(kept) + "\n", encoding="utf-8")
    edited = ww_harness.read_log(log_path)
    return dataclasses.replace(
        result, log=dataclasses.replace(edited, handed_texts=result.log.handed_texts))


def build_extension_release(capture: StoreCapture, spec: Path, out: Path, name: str, *,
                            timeout: float = 120.0) -> None:
    """The R3 step of a build (``[extension_releases.<name>]`` of the fixture spec, the rows in the
    file it names): run each row's job in the confined harness with a model client that reports the
    row's model, register the generated records as a file of generation edges (every log carries the
    texts it was handed, so the store compares them and marks the log compared), then register the
    release that copies them, with the copy edges a selection step states.

    It runs after the main build, inside the same open store. It writes new files and appends the
    two rows of R3 to the build's ``built-from.jsonl`` (docs/replica.md section 16); the files,
    manifests, rows and counts of the main build are not changed. The log of every row named in
    ``model_entry_removed`` has its model entry removed (``_without_model_entry``). The job names,
    models and texts are the file's; the registered names are ``gen/R3-models`` and the release's."""
    spec = Path(spec).resolve()
    with open(spec, "rb") as f:
        tables = tomllib.load(f).get("extension_releases", {})
    if name not in tables:
        raise SpecError(f"the fixture spec holds no [extension_releases.{name}]")
    table = tables[name]
    repo = spec.parent.parent.parent  # the repository root, as the spec's own paths read it
    rows_file = repo / table["file"]
    if not rows_file.is_file():
        raise InputError(f"{rows_file} is missing")
    rows = []
    for number, raw in enumerate(rows_file.read_text(encoding="utf-8").split("\n"), 1):
        if raw.strip() == "":
            continue
        row = json.loads(raw)
        try:
            rows.append(ModelRow(id=row["id"], job_name=row["job"], handed=tuple(row["handed"]),
                                 used=tuple(row["used"]), model=row["model"], text=row["text"]))
        except KeyError as missing:
            raise InputError(f"{rows_file}, line {number}: no field {missing}") from None
    if [row.id for row in rows] != list(table["records"]):
        raise SpecError(f"{rows_file} holds the ids {[row.id for row in rows]}; the spec lists "
                        f"{list(table['records'])} for {name}")
    removed = set(table.get("model_entry_removed", ()))
    if not removed <= {row.id for row in rows}:
        raise SpecError(f"model_entry_removed names {sorted(removed - {row.id for row in rows})}, "
                        f"which no row of {rows_file} has")

    index: dict[str, Record] = {}
    for path in sorted((Path(out) / "sources").glob("*.jsonl")):
        if path.name.endswith(".manifest.jsonl"):
            continue  # the manifest beside a source file, not a source file
        for record in read_records(path, artifact=f"sources/{path.name}", source=path.stem,
                                   strict=True):
            index.setdefault(record.ref, record)

    generated = f"gen/{EXTENSION_GENERATED_FILE}"
    relative = f"{generated}.jsonl"
    lines: list[str] = []
    manifest: list[tuple[str, list[str]]] = []
    made = []
    for row in rows:
        missing = [h for h in row.handed if h not in index]
        if missing:
            raise InputError(f"job {row.job_name}: no record {missing} among the source files")
        handed = [index[h] for h in row.handed]
        log_path = Path(out) / "gen" / "harness" / EXTENSION_GENERATED_FILE / f"{row.id}.log.jsonl"
        text = capture.generate_with_model(row, handed, log_path,
                                           remove_model_entry=row.id in removed, timeout=timeout)
        line = json_row({"id": row.id, "text": text})
        lines.append(line)
        manifest.append((row.id, list(row.used)))
        made.append((row, handed, Record(ref=row.id, text=text, raw=line, artifact=relative,
                                         line=len(lines), source=None)))
    path = _write_record_file(Path(out) / relative, lines, manifest)
    capture.record_generation(path, relative, generated, made)
    # The file-level manifest (docs/replica.md section 16): a generation output is built from the
    # files of the records its jobs were handed.
    _add_built_from(Path(out), generated, [relative],
                    [record.artifact for _, handed, _ in made for record in handed])

    release = f"releases/{name}.jsonl"
    release_path = _write_record_file(
        Path(out) / release, [json_row({"text": record.text}) for _, _, record in made],
        [(row.id, [row.id]) for row, _, _ in made])
    kept = [Kept(record=record, copies=(record,), rule="") for _, _, record in made]
    capture.record_release(release_path, release, name, {"dedup": False, "extension": name}, kept)
    # A release is built from the files of the copies it kept.
    _add_built_from(Path(out), name, [release], [record.artifact for _, _, record in made])


def _add_built_from(out: Path, name: str, files: list[str], read_files: list[str]) -> None:
    """Append the row of one artifact to the build's ``built-from.jsonl``, the way
    ``built_from.BuiltFrom.add_from_files`` does: ``built_from`` is the names of the artifacts the
    files ``read_files`` belong to, each once, in the order first met, and the row is written in the
    recorder's own form. The recorder of the main build is closed with that build, so the rows of
    the R3 step are appended to the file it wrote; a build with no manifest (capture off) gets no
    row. A name that already has a row is refused, as the recorder refuses it."""
    path = out / MANIFEST_FILE
    if not path.is_file():
        return
    names: dict[str, str] = {}
    known: set[str] = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line != "":
            row = json.loads(line)
            known.add(row["name"])
            for relative in row["files"]:
                names.setdefault(relative, row["name"])
    if name in known:
        raise CaptureError(f"the manifest holds two rows for {name!r}")
    built_from = list(dict.fromkeys(names[f] for f in read_files if f in names))
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(_row_text(name, files, built_from))


def _write_record_file(path: Path, lines: list[str], manifest: list[tuple[str, list[str]]]) -> Path:
    """A JSONL record file and the manifest beside it, in the form of ``build._Build.write``: one
    manifest row per line with the line number, the handle and the origin ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    with open(f"{path}.manifest.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for number, (handle, origin) in enumerate(manifest, 1):
            f.write(json_row({"line": number, "handle": handle, "origin": list(origin)}) + "\n")
    return path
