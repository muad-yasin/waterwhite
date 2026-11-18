# The confined harness

`waterwhite.confine` runs a generation job out of process, in a sandbox, and
serves it the records it was handed over a pipe. It is the closed-input harness
of plan 3.3 "Generation" for a job that is not trusted to stay inside its
hand-off by itself. The in-process harness (`waterwhite.harness`,
docs/capture.md section 3) logs what passes its context and cannot say what else
a job opened. This module adds the part that was left open there: the job has
nothing else to open.

Plan sections served: 3.3 "Generation", 2.2(g), Decision 13, Decision 25, limit
L8, assumption F10, and 5.8 item B-1, which leaves the design of the confinement
to the build. `DECISIONS.md` entries D-07 (a refused read) and D-21 (a sole
handed record) apply unchanged.

Code: `src/waterwhite/confine/`. Tests: `tests/confine/`, all marked `sandbox`.
The package is part of the tool: its three errors are classes of
`waterwhite.errors`, and docs/interfaces.md section 5.3 lists what it exports.

Read section 5 before relying on a log this module wrote. It says what the
sandbox establishes and what it does not, and which of the two requirements of
plan F10 the pipe meets.

## 1. In short

```python
from waterwhite import capture
from waterwhite.confine import python_program, run_confined_job
from waterwhite.harness import Handed

result = run_confined_job(
    python_program("jobs/paraphrase.py"),
    job_name="paraphrase",
    handed=[Handed(ref, text)],
    log_path="logs/paraphrase.jsonl",
    timeout_seconds=600,
    model_client=None,            # or a function (model, prompt) -> text or ModelReply
)
result.outputs                    # harness.JobOutput, in the job's order
result.log                        # records.HarnessLog, read back from the log file
result.log.harness                # "confined"
edges = capture.generation_edges(result, child_lines=[1], step=step)
```

The job, `jobs/paraphrase.py`:

```python
import waterwhite_job

with waterwhite_job.connect() as job:
    (only,) = job.handed          # the references it was handed; not a read
    text = job.read(only)         # logged by the harness before it is answered
    job.emit(rewrite(text), used=[only])
```

`run_confined_job` returns the same `JobResult` as `harness.run_job`, and the log
file is the same format, written by the same code. `HarnessLog`, its
`bound_stated()` (the store's own mark decides `bound_established()`),
`capture.generation_edge` and the store take a confined
result as they take any other. One word differs: the job entry of every log
names the harness that wrote it, "confined" here and "in-process" for
`harness.run_job`. A test runs one job both ways and compares the two log files:
they are equal entry for entry but for that word.

## 2. How a run goes

1. The time limit and the program's mounts are checked. Then the sandbox
   command is found and checked (section 3). If the sandbox cannot be built, the
   call ends with "sandbox unavailable". Nothing has been started and no log
   file has been written.
2. The harness's own code (`harness._run_job`, which is not part of its surface)
   is called with one fixed in-process job: a broker that is part of this module.
   The harness opens the log file, writes the job entry with the word "confined"
   and hands the broker its `JobContext`. The public `harness.run_job` takes no
   argument for that word and always writes "in-process", so no caller of the
   public function can state a confinement it does not give: only this module
   writes "confined".
3. The broker makes two pipes, starts the job in the sandbox with the job's ends
   of the pipes, and writes a greeting: the protocol, the job's name and the
   references of the hand-off.
4. The job sends requests, one per line. The broker answers each from the
   context:
   - a read goes to `JobContext.read`, which writes the read to the log file and
     syncs it to disk before it returns the text. Only then is the text sent.
     Anything that was not handed is written to the log as a refused attempt and
     answered with a refusal. The job may go on;
   - a model call goes to `JobContext.call_model`, which writes the model's
     identifier to the log and then calls the model client the caller gave. That
     client runs in the caller's process, outside the sandbox. Where it returns a
     `harness.ModelReply`, the identifier it reports is written to the log after
     the call, and the log's model entry names that one;
   - an output is kept in memory with the handed records it states it used.
5. The job sends its end message, closes its pipes and exits with status 0. The
   broker returns the outputs to the harness, which writes the output entries,
   the model entry and the end entry, closes the log and reads it back.

A run counts only when all of this happened: the end message was received,
nothing came after it, and the exit status was 0. In every other case a named
error is raised (section 4), the log is closed with status "failed", and no
`JobResult` exists. With no result there is no output to write and nothing to
build an edge from. The failed log stays on disk and still shows every read up
to the failure.

The whole run has one time limit, `timeout_seconds`, counted from the start of
the job with model calls included. A job still running then is stopped. The
limit has no default: the caller knows how long its model takes. Time decides
only whether a run is stopped. No time is written to the log, so the same job
with the same hand-off and the same requests gives the same log bytes.

### The protocol `waterwhite.confine/1`

JSON objects, one per line, UTF-8, on two pipes the job inherits. The variable
`WATERWHITE_JOB_FDS` holds `<request fd>,<reply fd>`. A reference is
`{"instance_id": ..., "locator": ...}`.

| The job sends | The harness answers | In the log |
|---|---|---|
| (nothing; the harness writes first) | `{"protocol": "waterwhite.confine/1", "job_name": ..., "handed": [references]}` | the `job` entry, with `harness` "confined", written before the job starts |
| `{"op": "read", "ref": reference}` for a handed record | `{"ok": true, "text": ...}` | a `read` entry, on disk before the answer |
| `{"op": "read", "ref": anything else}` | `{"ok": false, "error": "read outside the hand-off", "detail": ...}` | a `read` entry with `refused` true and `outside_ref` |
| `{"op": "model", "model": id, "prompt": text}` | `{"ok": true, "text": ...}` | a `model_call` entry, on disk before the client is called; a `model_reply` entry after it when the client reported which model answered |
| `{"op": "output", "text": ..., "used": [references]}` | `{"ok": true, "index": n}` | an `output` entry, after the job has ended |
| `{"op": "end"}` | `{"ok": true}` | the `model` and `end` entries |

A request for something that was not handed is named in the log by what was
asked: the reference as `instance:locator`, a text as it is, any other JSON
value in its JSON form. The job chooses that text, so it is bounded: beyond 200
characters the log holds the first 200 and the SHA-256 of the whole. A model
identifier longer than 200 characters is answered with an error and is neither
logged nor called. So a job cannot grow the log or copy a record into it at
length. (Up to 200 characters of the job's choosing can still stand in a refused
entry. The in-process harness has no such bound.)

The job side of the protocol is `waterwhite_job.py`, one file that imports only
the standard library (`json`, `os`, `typing`). It gives `connect()`, and on the
connection `job_name`, `handed`, `read(ref)`, `call_model(model, prompt)`,
`emit(text, used)` and `end()`. A refused read raises `waterwhite_job.Refused`;
any other error answer raises `waterwhite_job.RequestFailed` with the error's
name. A job in another language can speak the protocol itself; the client is a
convenience and not part of the confinement.

## 3. The sandbox

The job is started by bubblewrap, the command `bwrap`, found on PATH. An unset
PATH is read as an empty one; no default folders are searched. This is what the
command line builds (`Sandbox.argv` returns it, for a caller who wants to keep
it beside the log):

| What | How | What it shuts |
|---|---|---|
| Network | a new network namespace, required. It holds a loopback device of its own and no other device | every connection to this machine or beyond: other hosts, services on the machine's own loopback, abstract Unix sockets |
| Files | a new mount namespace with an empty root. Mounted read-only into it: `/usr/lib`, `/usr/lib64` (with `/lib`, `/lib64` as on the machine), the interpreter's own folder at its path, the job's script, the client. The root is then made read-only | every other path: the rest of the home folder, data folders, the store, the log, `/etc`, `/tmp`, `/run`, `/var`, mounted disks, Unix sockets that live in the file system |
| Kernel views | `/proc` and `/sys` are not mounted. `/dev` is bubblewrap's small one (null, zero, random and the like) | process lists, mount tables, kernel settings, device files |
| Processes | a new pid namespace and a new ipc namespace | signals to and tracing of other processes; shared memory and message queues of the machine |
| Scratch | `/scratch`, a memory file system of 64 MiB (`scratch_bytes`), the working folder, `HOME` and `TMPDIR` | nothing; it is what the job may write. It exists only inside the sandbox and is gone when the job ends, so no job leaves anything for the next |
| Environment | emptied, then set to `LANG`, `HOME`, `TMPDIR`, `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE` and `WATERWHITE_JOB_FDS` | keys, tokens and paths in the caller's environment |
| Open files | the job inherits standard input (the null device), standard output and standard error (a pipe the harness drains) and its two pipes | every other open file of the caller, the log file among them |
| Privilege | a new user namespace with no capability in it; making a further user namespace is disabled; a new session with no terminal | mounting, changing the namespaces, typing into the caller's terminal |
| Lifetime | the sandbox dies with the harness, and every process in it dies with the sandbox | a job that outlives its run |

Each namespace is asked for by name and is required. bubblewrap's one option for
"all namespaces" only tries the user and the cgroup namespace and goes on
without them, so it is not used.

**The check before every run.** `find_sandbox()` starts the sandbox once with
the same namespaces and system mounts and runs bubblewrap's own version print
inside it. A failure is "sandbox unavailable" with what bubblewrap said. That is
what happens when user namespaces are switched off for unprivileged users, or
when the installed bubblewrap lacks an option used here.

**There is no fallback.** The job is never run without the sandbox, and no
weaker sandbox is tried. A log from an unconfined run would be the same file,
and its bound would be claimed without its basis.

**A Python job.** `python_program(script, args=(), code=())` builds the program
for the interpreter the caller runs on. It mounts the interpreter's base
installation at its own path (in a virtual environment: the base interpreter,
without the environment's packages) and the script alone as
`/waterwhite/job/<name>`. The job has the standard library and the client.

**Code mounts.** A job that needs more code names it:
`code=[CodeMount(source, target)]`, each mounted read-only. `Program(argv,
mounts)` describes any other program the same way. A mount is the one place
where the caller decides what the job can read without the harness seeing it.
See section 5.

## 4. Errors

| Error name | Class | When |
|---|---|---|
| "sandbox unavailable" | `SandboxUnavailable` | `bwrap` is not on PATH, or the check run failed. Nothing was started, no log was written |
| "confined job failed" | `ConfinedJobFailed` | the job exited with a status other than 0, ran past its time limit, or closed its pipe without the end message. The detail quotes the end of what the job printed |
| "job protocol violation" | `JobProtocolViolation` | a request line that is not a JSON object, an unknown request, a missing or mistyped field, a line longer than `max_request_bytes` (64 MiB), anything after the end message |
| "invalid record" | `InvalidRecord` | as in the in-process harness: an output that states a record that was not handed, or states one twice; a hand-off that names a record twice; no job name. Also here: no time limit, a mount target that is not allowed, a model client that returns something other than text |
| "no model client" | `ModelClientMissing` | the job called a model and no client was given. The call is in the log |
| "write refused" | `WriteRefused` | a log file already exists at `log_path` |

The first three belong to this module. They have exit code 1 and no family, like
"no model client": the harness was set up or used in a way that cannot work.
They are classes of `waterwhite.errors`, as every named error of the tool is;
`confine.errors` and the package name the same classes. A model client that
returns a `ModelReply` whose model cannot be logged (empty, "none", one that
starts with "[") or whose text is not text ends the run with "invalid record".

An error the caller's model client raises is passed on as it is, after the job
has been stopped.

The detail of "confined job failed" quotes up to 4096 bytes of the job's last
output. A job can print a record there, so the detail of this error is not free
of record text. The log never holds it.

## 5. What this establishes, and what it does not

### What it establishes

On a machine where the tests of this module pass, for a job started by
`run_confined_job`:

- The job has no path to a record file. The data folders and the store are not
  in its file system, and of the home folder only the interpreter's own folder
  is there, read-only, at its path (where the interpreter lives under it). A test gives the job the path of a
  file that exists, and the job finds neither the file nor its folder.
- The job has no network. A test starts a listener on the machine's loopback and
  the job cannot reach it; it cannot resolve a name or reach another address,
  and it sees no device but its own loopback.
- The job's one way to the text of a record is a read request, and its one way
  to a model is a model request. The harness writes each to the log, on disk,
  before it answers.
- A request for anything outside the hand-off delivers nothing and is in the log
  as a refused attempt. The bound still holds (D-07).
- The log names every model the job asked for through the client. A job that
  reaches a model by another route would need a network, which it does not
  have.

So "the log lists no read outside the hand-off" here says more than it does for
an in-process job: the reads in the log are all the records the job received,
not just the ones it chose to ask the context for. The log says which of the two
it is: its job entry carries `harness` "confined", the store keeps the word, and
an answer names it beside every log a negative rests on (docs/coverage.md
section 6). Condition 2.2(g) reads the word: only a log the confined harness
wrote establishes a bound (Decision 13: a bound is established only by a
closed-input harness whose access log lists every record the job read). A log
the in-process harness wrote, or one that names no harness, establishes none,
and past its edge the answer is "may contain" naming the job (plan F10's
fallback).

What the store can check stops at the log's own words. A log file written by
hand outside the tool, one whose job entry says "confined" though the tool never
ran the job, is outside what the store can check: the store keeps the word the
file gives, and the proof reads it. Whoever registers a log vouches for it. What
the tool does guarantee is its own side: the public in-process harness cannot
write that word, and only this module does.

### The two requirements of plan F10, and the pipe

Plan F10 asks two things of a harness. It words the first with a read-only mount
of the handed records. This module mounts no record at all and serves the handed
records over a pipe. Which of the two requirements the pipe meets, and how:

1. **The job can reach no other record or record source.** This is met by the
   sandbox, not by the pipe. The job's file system holds the interpreter, the
   system's library folders, the job's code and nothing else; it has no network
   and inherits no open file but its two pipes (section 3). The pipe's part is
   narrow: it is the one channel that carries a record's text into the sandbox,
   and the harness answers a request on it only for a record of the hand-off. A
   request for anything else delivers nothing. So no record other than a handed
   one is reachable through the pipe, and nothing else of the machine is
   reachable at all. The requirement holds under the conditions of "What it
   rests on" below (code mounts that hold no records, a model client that
   reaches a model and nothing else).
2. **The log lists every record the job read.** This is met by the pipe, and it
   is the reason records are served and not mounted. Each delivery of a record's
   text is an answer the harness gives, and the harness writes the read to the
   log, on disk, before it sends the answer. A read the log does not list did
   not happen: the job has no other way to the text. With a read-only mount of
   the handed records the first requirement would hold in the same way, but the
   job would open the files itself, and the harness could list what was handed,
   not what was read.

What the pipe does not give is the mount itself. The first build reads the
hand-off over a pipe as giving F10's first condition (only the handed records
reachable, every read logged before it is answered); F10 itself stays
unverified, as the plan tags it, and its model-client half is not settled here.

Two limits follow, each with F10's fallback, "may contain" naming the job:

- **A job that must read its records as files** (a tool that takes a path)
  cannot run here as it is. Run in process, its log establishes no bound.
- **A job that calls a model over the network** cannot run here: the sandbox has
  no network, and only a model client the caller hands the harness is reached.
  Run in process, its log establishes no bound.

### What it rests on

These are conditions, not things the harness checks:

- **The code mounts hold no records.** The job reads its script, its code
  mounts, the interpreter and the system's library folders without a log entry.
  The harness cannot tell code from data. A caller who mounts a folder with
  records in it, or who keeps records under `/usr/lib`, has handed them to the
  job unseen, and the log will still give an established bound. The same holds
  for a script that carries records in its own text, and for record text passed
  in the job's arguments.
- **The hand-off is what it says, and the store compares it where it can.**
  `Handed(ref, text)` hands the job `text` as the record at `ref`. The broker holds
  no store, so `run_confined_job` runs whatever it is handed; the log it returns
  carries the handed texts (`log.handed_texts`; the log file holds none). When the
  output is registered with the edge `generation_edge` builds from that result, the
  store compares each text with the record its reference names, by fingerprint
  after its normaliser, and refuses the registration, naming the job and the
  reference, when one is not that record. A "does not contain" past a confined log
  therefore rests on the hand-off being the records it names for a result that is
  registered as `run_confined_job` returned it. For a log registered without its
  texts, on an edge with none (read from its file with `read_log`, fetched from the
  store with `store.harness_log`, or built by hand), nothing is compared and the log is
  stored as not compared: it establishes no bound, and a negative past its edge is "may
  contain" until the original texts are shown (`compare-hand-off`, docs/cli.md).
  A reference to a record that holds no text takes only a text that is empty after
  the normaliser; any other is refused. docs/capture.md, section 3, lists the routes.
- **The model client reaches a model and nothing else.** The client runs
  outside the sandbox with the caller's network. If it retrieves documents,
  searches, or calls tools, the job has a record-bearing source behind an
  allowed door, and the first condition of plan F10 is broken although the log
  shows nothing but a model call. Plan F10 names this case: a hosted job with
  its own retrieval may not fit the harness.
- **The model client says truly which model answered.** The identifier in the
  job's call is written to the log before the call is forwarded; the job chooses
  that text. Which model answers is up to the caller's client. A client that
  returns `harness.ModelReply(text, model)` reports it: the reported identifier
  is written to the log after the call, tied to the call's entry, and the log's
  model entry names the reported one. The job cannot set it. A client that
  returns plain text reports nothing, and the log then names what the job asked
  for; with such a client, one that ignores the identifier or maps it to another
  model makes the log name a model that did not answer. Either way the
  identifier is the client's or the job's word. The harness sees no model.
- **The log is not changed afterwards.** The store keeps the log's digest; the
  file itself is an ordinary file of the caller.

### What it does not claim

- **A kernel exploit.** The sandbox is made of kernel namespaces. The job can
  make every system call an unprivileged process can make; there is no system
  call filter. A job that breaks the kernel is outside every statement here.
- **Covert channels.** The job can read the clock and use the processor, memory
  and its scratch folder. Timing, load and cache effects between the job and
  another process on the machine are not shut. Processor time, memory and the
  number of processes are not limited beyond the time limit and the size of the
  scratch folder.
- **What the job sends out.** Outputs, prompts, the names in refused requests
  (bounded), what it prints, and its exit status all leave the sandbox. The
  confinement is about what the job can read, not about what it can say.
- **What the model knows.** A model the job calls may hold records it was
  trained on and may write them into its answer. No sandbox around the job sees
  that. It is the generating-model question (plan L11, 2.2(h)),
  An extension: while it is off the model is a logged attribute of the edge; with it on the proof reads the logged model (docs/coverage.md, section 6). A negative answer past a generation edge carries the words "generating-model
  exposure not assessed" while it is off.
- **That an output depends on what it states.** `used` is what the output
  states. It is process evidence (plan Decision 25).
- **Other machines.** This was built and tried on one laptop: Linux 7.3 (Fedora
  44), bubblewrap 0.12.0, unprivileged user namespaces on, with CPython 3.12 from
  a standalone build and CPython 3.14 from the system. On macOS and Windows
  there is no `bwrap` and the answer is "sandbox unavailable". A Linux with
  another layout of its library folders may fail to start the job; that is a
  failed run, never an unconfined one.

### The fixture, and plan F10

The fixture's generation jobs are fixed outputs with no model call. They run in
the replica, and their code is in this repository. A build whose answers the
table grades runs them in the confined harness (`build_table_store` defaults to it,
docs/replica.md "The confined harness"); only a build that no graded row reads may run
them in process. No fixture job runs in this sandbox, and no check of the first build
shows a real model being called through the client: the test here uses a
stand-in function.

Plan F10 is an assumption about a team's own jobs: that the job can be run so
that it reaches only the records handed to it. This module shows one way to run
a job like that on one kind of machine. It does not show that a given team's job
can run this way. A job that needs a GPU, a network service, or a library that
reads from disk may not start in this sandbox at all, or may need mounts that
break the conditions above. F10 stays an assumption for a team's job until that
job has run confined on that team's machine and the conditions of "What it rests
on" have been checked there. Until then the plan's fallback applies: the bound
is not established and the answer is "may contain" naming the job.

## 6. Choices

Each with the other option and what would change it.

**Records served over a pipe, not mounted read-only.** Plan F10 words the
condition as a read-only mount of the handed records. With a mount, the job
opens the files itself and the harness sees nothing: the log could only repeat
the hand-off, it could not tell a record the job read from one it never opened,
and a request outside the hand-off would be a failed `open` that leaves no
trace. With a pipe, each read entry is a delivery the harness made, a refusal is
an event the harness logged, and the in-process log format carries over. Section
5 says which of F10's two requirements the pipe meets: the second, that the log
lists every record read; the first, that no other record is reachable, is the
sandbox's. A job that must have its records as files (a tool that takes a path)
would change this; such a job could be given a mount in addition, with every
handed record logged as read at the start.

**The harness forwards model calls; the job has no network.** The other option
is a network namespace with one allowed address, the model's. That needs a
network device into the sandbox (a virtual pair set up as root, or a user-mode
network helper) and a filter on it. A filter by address cannot tell a model call
from a retrieval call to the same host, the job would hold the model's key, and
the model's identifier would have to be read from the job's traffic, which is
normally encrypted, or taken on the job's word. With forwarding, the job's
network is simply absent, the identifier logged is the one in the call the
harness forwards, and the caller's client holds the keys, which the job never
sees. A client library that must run inside the job and cannot be given a
transport would change this.

**No sandbox, no run.** When `bwrap` is missing or cannot make its namespaces,
the harness raises "sandbox unavailable". The other option, running the job in
process or in a plain subprocess with a warning, would write the same log and
give an established bound that rests on nothing. A second supported sandbox
would be added as a second checked sandbox, not as a fallback.

**Code mounts are allowed.** The other option is a job that gets the standard
library and its one script and nothing else. That shuts the one route by which a
caller can hand the job unlogged files, and it also shuts out every job that
imports a library. Mounts are allowed, read-only, and named by the caller, and
section 5 says what that leaves to the caller. A log that recorded its mounts
would let a later reader see them; the log format has no place for that today.

**One log format, and a word for the harness.** A confined log and an in-process
log are the same kind of file, read by the same reader. The job entry names the
harness that wrote the log, and the store keeps the word. The other option is a
log that does not say: a stored bound would then look the same whether the job
could have opened other files or not, and plan F10's first requirement could
not be told from the log. The word changes no answer: condition 2.2(g) reads
the bound for both, because the fixture's jobs, which the table's negatives rest
on, run in process. A ruling that an in-process log establishes no bound would
change that, and the word is what such a rule would read.

**The model the client reports.** The log names the identifier the client
reports where it reports one, and the identifier the job asked for otherwise.
The other option is the asked identifier alone, written before the call. It is
kept as well, because it is in the log whether or not the call returns; but it
is a text of the job's choosing, and the generating-model rule (plan 2.2(h), an
extension, off by default) looks that text up as a checkpoint name, exactly, with the switch on. A client that
cannot say which model answered would leave the asked identifier as the only
one.

## 7. Running the tests

```
python -m pytest tests/confine -q -m sandbox
```

The tests need Linux and a `bwrap` that can create user namespaces. Where it
cannot, the tests that start a sandbox are skipped with the harness's own
reason, and are not worked around; the tests of the refusal and of the command
line still run. The whole folder takes a few seconds; the longest test waits 1.5
seconds for a job to pass its time limit.

What the tests cover: reads logged in order and the digest of the file; the log
equal to the in-process log but for the harness it names; the word kept by the
store; the model a client reports named by the log; edges built from a confined
result; a
request outside the hand-off refused and logged with the bound standing; an
outside file absent for the job; no network; nothing of the caller's environment
and no open file but the two pipes; a model call forwarded to a stand-in client
with its identifier logged; a failed exit, garbage on the pipe, a missing end
message and a passed time limit each giving a named error and no outputs; and
the refusal to run when the sandbox command is missing or cannot build the
sandbox.
