"""Tokenize chunk files with Megatron's preprocessing script, as a subprocess.

The replica imports nothing from Megatron. It runs ``tools/preprocess_data.py`` of
a Megatron-LM checkout with the interpreter of the test-time environment, under
the conditions a run must repeat (plan F1, 5.3):

- one partition, so that document order is line order;
- ``--append-eod``, recorded with the run;
- a new, empty input folder for each run (the script writes partition files beside
  its input and reuses them when they are there);
- the vocabulary file and the merges file handed over by path, and ``HF_HOME``
  and ``HF_HUB_OFFLINE=1`` set for every subprocess: with a file missing,
  Megatron's tokenizer class tries a download into the home folder's cache;
- the tokenizer built once before the first run, offline, at the revision the
  spec records, because a tokenizer error inside the script's worker pool makes
  the script hang instead of exit;
- a time limit on every run, for the same reason.

``LineTokenizer`` is the same tokenizer as a function from one text to its token
ids, for the verification of a tokenization map (plan 3.3): one more subprocess of
the environment, which builds the tokenizer as the script does and answers one
text per line.

The environment folder is read from the variable ``WATERWHITE_MEGATRON_ENV``. It
holds ``venv/``, ``src/Megatron-LM`` and ``hf_home/`` (tools/setup_megatron_env.sh
builds it). The spec's ``[tokenizer]`` table names the two tokenizer files by
paths relative to that folder, and the hub revision they belong to.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .errors import SpecError, TokenizerError
from .plan import TokenizerSettings

ENV_VARIABLE = "WATERWHITE_MEGATRON_ENV"
SUPPORTED_TOKENIZER = "GPT2BPETokenizer"
HUB_NAME = "gpt2"
EOD_TOKEN_ID = 50256

# What the script appends to its output prefix for the "text" key of each line.
_OUTPUT_SUFFIX = "_text_document"

# Run with the hub name and the revision as its two arguments.
_PREFLIGHT = (
    "import sys\n"
    "from transformers import AutoTokenizer\n"
    "t = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])\n"
    "print(t.eos_token_id)\n"
)


@dataclass(frozen=True)
class MegatronEnv:
    """The pieces of the test-time environment the replica uses."""

    root: Path

    @property
    def python(self) -> Path:
        return self.root / "venv" / "bin" / "python"

    @property
    def script(self) -> Path:
        return self.root / "src" / "Megatron-LM" / "tools" / "preprocess_data.py"

    @property
    def hf_home(self) -> Path:
        return self.root / "hf_home"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "MegatronEnv":
        environ = os.environ if environ is None else environ
        value = environ.get(ENV_VARIABLE, "")
        if value == "":
            raise TokenizerError(
                f"{ENV_VARIABLE} is not set: it must name the Megatron environment folder "
                "(tools/setup_megatron_env.sh builds one); or build with tokenization off")
        env = cls(Path(value).resolve())
        for piece in (env.python, env.script, env.hf_home):
            if not piece.exists():
                raise TokenizerError(f"{ENV_VARIABLE}={value}: {piece} is missing")
        return env

    def tokenizer_files(self, settings: TokenizerSettings) -> tuple[Path, Path]:
        """The vocabulary file and the merges file the spec names, as paths inside
        this environment. Every preprocessing run is handed both: without them
        Megatron's tokenizer class tries to download them."""
        vocab, merges = self.root / settings.vocab_file, self.root / settings.merges_file
        for name, piece in (("vocab_file", vocab), ("merges_file", merges)):
            if not piece.is_file():
                raise TokenizerError(
                    f"the spec's tokenizer {name} is {piece.relative_to(self.root)} and the "
                    f"environment folder holds no such file: the {settings.name} tokenizer "
                    f"files of revision {settings.revision} were not fetched into it")
        return vocab, merges

    def child_environment(self) -> dict[str, str]:
        child = dict(os.environ)
        child.pop("PYTHONPATH", None)
        child["HF_HOME"] = str(self.hf_home)
        child["HF_HUB_OFFLINE"] = "1"
        child["TRANSFORMERS_OFFLINE"] = "1"
        child["TOKENIZERS_PARALLELISM"] = "false"
        return child


@dataclass(frozen=True)
class ShardJob:
    """One chunk to tokenize: ``input_path`` becomes ``<prefix>.bin`` and
    ``<prefix>.idx``. ``work_dir`` must not exist yet; it receives the copy of the
    input the script reads and the script's log, and is kept."""

    name: str
    input_path: Path
    prefix: Path
    work_dir: Path


# Set when a build is being abandoned: every waiting run then stops its subprocess.
_stop = threading.Event()

_TRACEBACK = b"Traceback (most recent call last)"
_POLL_SECONDS = 1.0


def stop_all() -> None:
    """Ask every tokenizer run of this process to stop its subprocess."""
    _stop.set()


def _log_tail(log_path: Path, size: int = 2000) -> str:
    return log_path.read_bytes()[-size:].decode("utf-8", errors="replace")


def _run(cmd: Sequence[str], *, cwd: Path, env: dict[str, str], log_path: Path,
         timeout: float, what: str) -> None:
    """Run one subprocess to its end, with its output in ``log_path``.

    The script is known to hang when one of its worker processes dies: the error is
    printed and the parent waits for ever. So the run is stopped, loudly, when a
    traceback appears in the log of a process that has not exited, when the time
    limit passes, or when the build is abandoned."""
    waited = 0.0
    with open(log_path, "wb") as log:
        # A session of its own: the script and its workers form one process group,
        # which can be stopped by the group's id without touching anything else.
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                                   start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    waited += _POLL_SECONDS
                if _stop.is_set():
                    raise TokenizerError(f"{what} was stopped because the build was abandoned")
                if waited >= timeout:
                    raise TokenizerError(
                        f"{what} did not finish within {timeout:.0f} seconds and was stopped; "
                        f"see {log_path}")
                if _TRACEBACK in log_path.read_bytes():
                    raise TokenizerError(
                        f"{what} printed an error and did not exit; it was stopped; "
                        f"see {log_path}\n{_log_tail(log_path)}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if code != 0:
        raise TokenizerError(f"{what} exited with code {code}; see {log_path}\n"
                             f"{_log_tail(log_path)}")


def check_settings(settings: TokenizerSettings) -> None:
    if settings.type != SUPPORTED_TOKENIZER:
        raise SpecError(f"the replica tokenizes with {SUPPORTED_TOKENIZER} only, "
                        f"got {settings.type!r}")
    if settings.name != HUB_NAME:
        raise SpecError(f"the replica's tokenizer files are those of {HUB_NAME!r}, "
                        f"got {settings.name!r}")
    if settings.partitions != 1:
        raise SpecError("the replica tokenizes in one partition")
    if settings.revision == "":
        raise SpecError("the tokenizer's hub revision is not recorded")


def preflight(env: MegatronEnv, settings: TokenizerSettings, work_dir: Path, *,
              timeout: float = 120.0) -> None:
    """Build the tokenizer once, offline, at the recorded revision, before any
    preprocessing run, and see that both tokenizer files are where the spec says."""
    env.tokenizer_files(settings)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    log_path = work_dir / "preflight.log"
    _run([str(env.python), "-I", "-c", _PREFLIGHT, settings.name, settings.revision],
         cwd=work_dir, env=env.child_environment(),
         log_path=log_path, timeout=timeout, what="building the tokenizer")
    lines = log_path.read_text(encoding="utf-8", errors="replace").split()
    if not lines or lines[-1] != str(EOD_TOKEN_ID):
        raise TokenizerError(f"the tokenizer's end-of-document id is not {EOD_TOKEN_ID}; "
                             f"see {log_path}")


def tokenize_shard(env: MegatronEnv, settings: TokenizerSettings, job: ShardJob, *,
                   timeout: float) -> None:
    work_dir = job.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    job.prefix.parent.mkdir(parents=True, exist_ok=True)
    input_copy = work_dir / f"{job.name}.jsonl"
    shutil.copyfile(job.input_path, input_copy)
    vocab, merges = env.tokenizer_files(settings)
    cmd = [
        str(env.python), "-I", str(env.script),
        "--input", str(input_copy),
        "--output-prefix", str(work_dir / job.name),
        "--tokenizer-type", settings.type,
        "--vocab-file", str(vocab),
        "--merge-file", str(merges),
        "--workers", str(settings.workers),
        "--partitions", str(settings.partitions),
    ]
    if settings.append_eod:
        cmd.append("--append-eod")
    _run(cmd, cwd=work_dir, env=env.child_environment(),
         log_path=work_dir / "preprocess.log", timeout=timeout,
         what=f"preprocessing {job.name}")
    for suffix in (".bin", ".idx"):
        written = work_dir / f"{job.name}{_OUTPUT_SUFFIX}{suffix}"
        if not written.exists():
            raise TokenizerError(f"preprocessing {job.name} wrote no {written.name}; "
                                 f"see {work_dir / 'preprocess.log'}")
        target = Path(str(job.prefix) + suffix)
        if target.exists():
            raise TokenizerError(f"{target} exists already")
        os.replace(written, target)


def tokenize_shards(env: MegatronEnv, settings: TokenizerSettings, jobs: Sequence[ShardJob], *,
                    parallel: int = 1, timeout: float = 900.0) -> None:
    """Tokenize every job, ``parallel`` at a time. Each run is its own subprocess
    with its own input folder, so the order they finish in changes nothing."""
    check_settings(settings)
    if parallel < 1:
        raise SpecError("the number of parallel tokenizer runs must be at least 1")
    _stop.clear()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(tokenize_shard, env, settings, job, timeout=timeout)
                   for job in jobs]
        try:
            for future in futures:
                future.result()
        except BaseException:
            # One run failed, or the build was interrupted: stop the others too.
            for future in futures:
                future.cancel()
            stop_all()
            raise


# --- The tokenizer as a function of one text ---------------------------------------------

_LINE_WORKER = Path(__file__).with_name("megatron_line_worker.py")


class LineTokenizer:
    """The build's tokenizer as a function: ``tokenizer(text)`` gives the token ids
    of one text, without the end-of-document token, as Megatron's own tokenizer
    class gives them to the preprocessing script.

    It is one subprocess of the Megatron environment (``megatron_line_worker.py``),
    started offline with both tokenizer files handed over by path. ``work_dir``
    receives its log. A reply that does not come within ``timeout`` seconds stops
    the subprocess and the build. Use it as a context manager, or call ``close``."""

    def __init__(self, env: MegatronEnv, settings: TokenizerSettings, work_dir: Path, *,
                 timeout: float = 120.0) -> None:
        check_settings(settings)
        vocab, merges = env.tokenizer_files(settings)
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = work_dir / "line-tokenizer.log"
        self.timeout = timeout
        self._buffer = b""
        self._log = open(self.log_path, "wb")
        self._process: subprocess.Popen | None = subprocess.Popen(
            [str(env.python), "-I", str(_LINE_WORKER), str(env.script.parent.parent),
             settings.type, str(vocab), str(merges)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, cwd=work_dir,
            env=env.child_environment(), start_new_session=True)
        try:
            ready = self._reply("building the tokenizer")
            if not isinstance(ready, dict) or ready.get("eod") != EOD_TOKEN_ID:
                raise TokenizerError(f"the tokenizer's end-of-document id is not "
                                     f"{EOD_TOKEN_ID}; see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def _reply(self, what: str):
        process = self._process
        waited = 0.0
        while b"\n" not in self._buffer:
            ready, _, _ = select.select([process.stdout], [], [], _POLL_SECONDS)
            if ready:
                block = os.read(process.stdout.fileno(), 1 << 16)
                if block == b"":
                    raise TokenizerError(
                        f"{what}: the tokenizer process ended (exit code {process.wait()}); "
                        f"see {self.log_path}\n{_log_tail(self.log_path)}")
                self._buffer += block
                continue
            waited += _POLL_SECONDS
            if _stop.is_set():
                raise TokenizerError(f"{what} was stopped because the build was abandoned")
            if waited >= self.timeout:
                raise TokenizerError(f"{what} did not answer within {self.timeout:.0f} seconds "
                                     f"and was stopped; see {self.log_path}")
        line, _, self._buffer = self._buffer.partition(b"\n")
        return json.loads(line)

    def __call__(self, text: str) -> list[int]:
        if self._process is None:
            raise TokenizerError("the tokenizer process was closed")
        try:
            self._process.stdin.write(json.dumps(text).encode("ascii") + b"\n")
            self._process.stdin.flush()
            reply = self._reply("tokenizing a line")
        except TokenizerError:
            self.close()
            raise
        except OSError as exc:
            self.close()
            raise TokenizerError(f"the tokenizer process is gone ({exc}); see "
                                 f"{self.log_path}") from exc
        if not isinstance(reply, list):
            self.close()
            raise TokenizerError(f"the tokenizer process answered {reply!r}; see {self.log_path}")
        return reply

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # A session of its own: stopped by its own process group id.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        self._log.close()

    def __enter__(self) -> "LineTokenizer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def line_tokenizer(env: MegatronEnv, settings: TokenizerSettings, work_dir: Path, *,
                   timeout: float = 120.0) -> LineTokenizer:
    """Start the build's tokenizer as a function of one text (``LineTokenizer``)."""
    return LineTokenizer(env, settings, work_dir, timeout=timeout)
