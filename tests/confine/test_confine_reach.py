"""What a confined job can reach of the build it runs in (plan F10's first
condition, read as given by the hand-off over a pipe; plan 3.3 "Generation",
Decision 13; DECISIONS.md D-27).

A store with a source ingested lies on this machine, its source file and its store
file beside it. A job handed one record of that source tries to open the source
file and the store file by their paths: neither is there for it. The one record
it was handed comes over the pipe, and the log lists that read and no other. The
network is covered by test_confine_job.py::test_the_job_has_no_network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from waterwhite import capture
from waterwhite.confine import python_program, run_confined_job
from waterwhite.harness import Handed
from waterwhite.records import OccurrenceRef
from waterwhite.store import Store

pytestmark = pytest.mark.sandbox

LIMIT = 60


def test_a_confined_job_reaches_no_source_file_and_no_store_file_of_its_build(sandbox, jobs, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    source = build / "A.jsonl"
    source.write_text('{"text": "the handed record"}\n{"text": "a record nobody handed"}\n',
                      encoding="utf-8")
    store_file = build / "store.sqlite"
    with Store.create(store_file, normaliser="control-chars/1") as store:
        instance = capture.ingest_jsonl(store, source, source="A", member="A/r1", revision="r1",
                                        name="src/A")
    handed = OccurrenceRef(instance, "L1")

    result = run_confined_job(
        python_program(jobs / "open_paths.py", [str(source), str(store_file), str(build)]),
        job_name="open-paths",
        handed=[Handed(handed, "the handed record")],
        log_path=tmp_path / "job.jsonl",
        timeout_seconds=LIMIT,
    )
    (output,) = result.outputs
    report = json.loads(output.text)

    # Both files are there for this process, which is not confined.
    assert source.stat().st_size > 0 and store_file.stat().st_size > 0
    # For the job neither the source file, nor the store file, nor their folder exists.
    for path in (source, store_file, build):
        assert report[str(path)]["ok"] is False, path
        assert report[str(path)]["error"] == "FileNotFoundError", path
    # The handed record came over the pipe, and the log lists that one read.
    assert report["handed"] == "the handed record"
    assert [read.occurrence for read in result.log.reads] == [handed]
    assert result.log.bound_stated()
