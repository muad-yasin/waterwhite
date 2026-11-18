"""The two scratch stores of the hand-written table (fixture/table/README.md
section 4): ``scratch-notokens`` for Check 3 and ``scratch-500`` for Check 4.

Neither is a build of the fixture's pipeline. Each writes one tokenizer input by
hand, tokenizes it with Megatron's preprocessing under the fixture's settings and
registers what its check needs. Both need the Megatron environment.

``scratch-notokens`` (set-up ``t.no-tokens-line``): ``src/Z`` ingested; a
tokenizer input of five lines, of which lines 1, 2, 4 and 5 hold the sanitized
texts of the first four records of Z and line 3 a record whose text is empty. The
line is put there past the sanitizer, which would have dropped it. The input is
registered as ``c03/shard-notokens.input`` and linked to ``src/Z`` by re-running
the sanitizer; the shard ``c03/shard-notokens`` is registered with its capture
map; its position-mode recovery map is then recovered from the same input.

``scratch-500``: ``src/A`` ingested; a file of 500 lines ingested under a source
of its own, ``c04``, as ``c04/shard-500.input``; that file's tokenized shard
registered as ``c04/shard-500``, with no map. The rows of Check 4 write the map
themselves: one is stopped part way, one declares a wrong count. Nothing of this
store needs a run registration.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from waterwhite import capture as ww_capture
from waterwhite import recovery as ww_recovery
from waterwhite import tokenmap as ww_tokenmap

from . import megatron_tokenizer, pipeline_links
from .build import WORK_DIR, json_row, require_new_folder
from .errors import InputError, SpecError
from .plan import BuildPlan
from .records import read_records
from .shard_files import ShardFiles
from .shard_index import read_shard_index
from .steps import normaliser

LAYOUT = "scratch"

NOTOKENS_SHARD = "c03/shard-notokens"
NOTOKENS_RECORDS = 4
NOTOKENS_EMPTY_LINE = 3
NOTOKENS_EMPTY_ID = "no-tokens-line"
HANDLE_NO_TOKENS = "no_tokens_line"

S500_SHARD = "c04/shard-500"
S500_SOURCE = "c04"
S500_MEMBER = "c04/r1"
S500_REVISION = "r1"
S500_LINES = 500


def _source(plan: BuildPlan, key: str):
    for source in plan.sources:
        if source.key == key:
            return source
    raise SpecError(f"the spec holds no source {key}")


def _ingest_source(store, plan: BuildPlan, out: Path, key: str) -> tuple[str, str, str]:
    """Copy one source file of the fixture into the scratch folder and ingest it."""
    source = _source(plan, key)
    relative = f"sources/{key}.jsonl"
    (out / "sources").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.path, out / relative)
    name = f"src/{key}"
    instance = ww_capture.ingest_jsonl(store, out / relative, source=source.source,
                                       member=source.member, revision=source.revision,
                                       name=name)
    return name, relative, instance


def _write_input(files: ShardFiles, rows: list[tuple[dict, str, list[str]]]) -> None:
    """Write a tokenizer input and its manifest: one (row, handle, origin) a line."""
    path = files.tokenizer_input
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        for row, _, _ in rows:
            f.write(json_row(row) + "\n")
    with open(f"{path}.manifest.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for number, (_, handle, origin) in enumerate(rows, 1):
            f.write(json_row({"line": number, "handle": handle, "origin": origin}) + "\n")


def _tokenize(files: ShardFiles, plan: BuildPlan, env, timeout: float) -> None:
    megatron_tokenizer.preflight(env, plan.tokenizer,
                                 files.out / WORK_DIR / "tokenize" / "preflight")
    megatron_tokenizer.tokenize_shards(env, plan.tokenizer, [megatron_tokenizer.ShardJob(
        name=files.shard, input_path=files.tokenizer_input, prefix=files.prefix,
        work_dir=files.out / WORK_DIR / "tokenize" / LAYOUT / files.shard)], timeout=timeout)


def _summary(plan: BuildPlan, files: ShardFiles, store_name: str) -> dict:
    index = read_shard_index(Path(f"{files.prefix}.idx"))
    return {
        "mode": "scratch",
        "capture": True,
        "tokenized": True,
        "shard": store_name,
        "input": files.tokenizer_input_name,
        "input_lines": len(index.document_tokens),
        # A line that gave no tokens has an entry of 0 here and is no document.
        "line_tokens": list(index.document_tokens),
        "documents": sum(1 for count in index.document_tokens if count),
        "tokens": index.tokens,
        "tokenizer": {"type": plan.tokenizer.type, "name": plan.tokenizer.name,
                      "revision": plan.tokenizer.revision,
                      "append_eod": plan.tokenizer.append_eod,
                      "eod_token_id": plan.tokenizer.eod_token_id,
                      "eod_in_span": plan.tokenizer.eod_in_span},
    }


def build_scratch_notokens(folder: Path, plan: BuildPlan, env, *,
                           store_file: str, build_dir: str,
                           tokenizer_timeout: float = 900.0) -> dict:
    """Build the store ``scratch-notokens`` into ``folder``. Returns
    ``{"instances", "files", "summary"}`` as a captured build does."""
    require_new_folder(folder, "a scratch store")
    pipeline_links.map_settings(plan.tokenizer)
    out = Path(folder) / build_dir
    files = ShardFiles(out=out, layout=LAYOUT, shard="shard-notokens",
                       store_name=NOTOKENS_SHARD, listed=("sources/Z.jsonl",))
    clean = normaliser(plan.normaliser)
    z = _source(plan, "Z")
    records = read_records(z.path, artifact="sources/Z.jsonl", source="Z",
                           strict=True)[:NOTOKENS_RECORDS]
    if len(records) != NOTOKENS_RECORDS:
        raise InputError(f"{z.path}: fewer than {NOTOKENS_RECORDS} records")
    rows = [({"id": r.ref, "text": clean(r.text)}, plan.handle_of(r.ref), [r.ref])
            for r in records]
    rows.insert(NOTOKENS_EMPTY_LINE - 1,
                ({"id": NOTOKENS_EMPTY_ID, "text": ""}, HANDLE_NO_TOKENS, []))
    out.mkdir(parents=True, exist_ok=True)
    _write_input(files, rows)
    _tokenize(files, plan, env, tokenizer_timeout)

    instances: dict[str, str] = {}
    registered: dict[str, str] = {}
    store = pipeline_links.open_store(Path(folder) / store_file, plan.normaliser)
    try:
        name, relative, instance = _ingest_source(store, plan, out, "Z")
        instances[name], registered[name] = instance, relative
        input_name = files.tokenizer_input_name
        instances[input_name] = ww_recovery.register_found_file(
            store, files.tokenizer_input, name=input_name)
        registered[input_name] = files.tokenizer_input_relative
        # The four texts are sanitized texts of records of Z: the link is the
        # sanitizer's, re-run. There is no sanitizer's log: the fifth line was put
        # in past the sanitizer and has no text to match.
        links = ww_recovery.recover_sanitizer(
            store, input_name, name, step=pipeline_links.sanitize_step(plan.normaliser))
        with megatron_tokenizer.line_tokenizer(
                env, plan.tokenizer, out / WORK_DIR / "tokenize" / "line-tokenizer",
                timeout=tokenizer_timeout) as tokenize:
            settings = pipeline_links.map_settings(plan.tokenizer)
            captured = ww_tokenmap.map_tokenized_shard(
                store, input=input_name, prefix=files.prefix, name=NOTOKENS_SHARD,
                step=pipeline_links.tokenize_step(plan.tokenizer), settings=settings,
                tokenize=tokenize)
            instances[NOTOKENS_SHARD] = captured.shard_instance_id
            registered[NOTOKENS_SHARD] = files.prefix_relative
            recovered = ww_tokenmap.recover_map(
                store, shard=NOTOKENS_SHARD, input=input_name, settings=settings,
                tokenize=tokenize)
    finally:
        store.close()
    summary = _summary(plan, files, NOTOKENS_SHARD)
    summary.update({
        "input_links": {"recovered": links.recovered, "no_edge": links.no_edge,
                        "no_text": links.no_text},
        "capture_map": pipeline_links.maps_report(captured),
        "recovery_map": pipeline_links.maps_report(recovered),
    })
    return {"instances": instances, "files": registered, "summary": summary}


def scratch_500_text(number: int) -> str:
    """The text of line ``number`` of the 500-line file: plain, self-authored, and
    different on every line, so that no two documents of the shard are twins."""
    return (f"Scratch line {number} of five hundred. It belongs to the store that Check 4 "
            f"writes a map for, and it names its own number, {number}, so that no other "
            "line holds the same text.")


def build_scratch_500(folder: Path, plan: BuildPlan, env, *,
                      store_file: str, build_dir: str,
                      tokenizer_timeout: float = 900.0) -> dict:
    """Build the store ``scratch-500`` into ``folder``: the shard is registered and
    has no map. Returns ``{"instances", "files", "summary"}``."""
    require_new_folder(folder, "a scratch store")
    out = Path(folder) / build_dir
    files = ShardFiles(out=out, layout=LAYOUT, shard="shard-500", store_name=S500_SHARD,
                       listed=())
    rows = [({"id": f"{S500_SOURCE}-{n:03d}", "text": scratch_500_text(n)},
             f"{S500_SOURCE}-{n:03d}", [f"{S500_SOURCE}-{n:03d}"])
            for n in range(1, S500_LINES + 1)]
    out.mkdir(parents=True, exist_ok=True)
    _write_input(files, rows)
    _tokenize(files, plan, env, tokenizer_timeout)
    index = read_shard_index(Path(f"{files.prefix}.idx"))
    if index.documents != S500_LINES or 0 in index.document_tokens:
        raise InputError(f"{files.prefix}: {index.documents} documents for {S500_LINES} lines")

    instances: dict[str, str] = {}
    registered: dict[str, str] = {}
    store = pipeline_links.open_store(Path(folder) / store_file, plan.normaliser)
    try:
        name, relative, instance = _ingest_source(store, plan, out, "A")
        instances[name], registered[name] = instance, relative
        input_name = files.tokenizer_input_name
        instances[input_name] = ww_capture.ingest_jsonl(
            store, files.tokenizer_input, source=S500_SOURCE, member=S500_MEMBER,
            revision=S500_REVISION, name=input_name)
        registered[input_name] = files.tokenizer_input_relative
        instances[S500_SHARD] = store.register_shard(
            files.prefix, name=S500_SHARD, step=pipeline_links.tokenize_step(plan.tokenizer),
            inputs=[instances[input_name]])
        registered[S500_SHARD] = files.prefix_relative
    finally:
        store.close()
    summary = _summary(plan, files, S500_SHARD)
    summary["map"] = None
    return {"instances": instances, "files": registered, "summary": summary}
