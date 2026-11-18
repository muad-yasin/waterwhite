# The part of the tool that answers

A team withdraws a source. For each artifact (a JSONL file, a tokenized shard, a
mixture, a checkpoint) the tool gives one of four answers:

| Answer | Meaning |
|---|---|
| contains | a recorded chain of evidence leads from the artifact to a record of the source |
| may contain | the artifact has its own record, and a link on the way is missing, weak or unverified |
| does not contain | a proof holds: every record of the artifact is traced to its origin, and none of the origins is in the request |
| not covered | the artifact's own record is missing, or the store was never told of the artifact |

This file describes the rules as they are built. "Plan" is the build plan, cited by
section. Code:

| File | What it holds |
|---|---|
| `src/waterwhite/requests.py` | the withdraw request and the refusals |
| `src/waterwhite/answers.py` | the answer type, its JSON form, the proof object |
| `src/waterwhite/coverage/__init__.py` | `prove`, `answer`, the integrity pass |
| `src/waterwhite/coverage/_walk.py` | the walk over records, shards and mixtures |
| `src/waterwhite/coverage/_checkpoint.py` | a checkpoint's answer |
| `src/waterwhite/listing.py` | listings |
| `src/waterwhite/cli.py` | the command line (see `docs/cli.md`) |

The proof reads the store through the read methods `docs/store.md` documents and
runs no SQL of its own. Every word an answer carries (answers, causes, reasons,
notes, bases, levels, the link words of a hop) is a constant of
`src/waterwhite/vocabulary.py`, and every error a class of
`src/waterwhite/errors.py`.

Tests: `tests/coverage/`. They build a small store by hand and need no Megatron.

Plan sections served: 2.1 to 2.4, 3.2, 3.3, 3.5, 4.4, limits L1, L3, L6 to L8, L10
and L12, assumptions F6 and F9, Decisions 1, 3, 8, 12, 13, 18, 20, 21, 22, 24, 25,
26 and 28; DECISIONS.md D-05, D-07, D-18, D-20, D-21, D-22 and D-24.

## 1. Requests

The first build answers one kind of request: the withdrawal of a source (plan 2.3,
Decision 24).

A source is a named set of members. A member is a revision of the source or a part
of one (a file, a folder, a list of records), registered in the store. A request
names the source. It is expanded when the query runs:

```python
from waterwhite import requests

request = requests.withdraw(store, "A")                          # every member of A
request = requests.withdraw(store, "A", members=["A/r1"])        # named members only
request = requests.withdraw(store, "A", records=["src/A:L3"])    # one record
request.members        # ('A/r1',): what the request expanded to
request.to_json()      # {"kind": "withdraw", "source": "A", "members": [...], "records": [...]}
```

- With no member named, the request covers every member registered under the
  source at that moment. Every answer prints the list, so a revision the team
  forgot to group under the source shows as missing from it.
- A record is named `ARTIFACT:LOCATOR`: a name of a registered JSONL file (or its
  instance id) and a line, `L<n>` counted from 1. A request with records is a
  record-level request. It puts in scope those records and everything derived
  from them, and nothing else of the source.
- Scope: everything whose chain of evidence leads to a requested record. A record
  that arrived through an input registered under another source is outside the
  withdrawal, even when its text is the same (plan L12).

Refusals. A request that cannot be answered exactly is refused with a named error
(exit code 2) before any answer exists. Nothing is approximated (plan L7).

| Request | Error name |
|---|---|
| a source with no registered member | source not registered |
| a named member that is not a member of the source | member not registered |
| a record no covered member holds, or that the store does not hold | record not in source |
| a record text that is not `ARTIFACT:L<n>`; a selector on metadata such as a URL prefix; an unknown request kind | unsupported selector |
| a content ban, a request that starts from flagged records, while their extension is off | extension off |
| the same two with their extension switched on | request refused (this build holds only the off state) |

Each name is a class of `waterwhite.errors` (`MemberNotRegistered`,
`RecordNotInSource` and the others); `requests.py` defines none.

The plan names one refusal for a withdrawal, "source not registered" (plan 2.3).
"member not registered" and "record not in source" are additions. The plan does
not name them: error names beyond those it fixes are left to the builder (plan
5.8, B-7), and DECISIONS.md D-26 records these two.

## 2. An answer

`coverage.answer(store, artifact, request)` returns an `answers.Answer`. Its JSON
form (`Answer.to_json()`):

| Field | When | Content |
|---|---|---|
| `artifact` | always | the name the query used |
| `instance`, `kind` | always | the instance id and kind; null for an artifact the store has no row for |
| `request` | always | kind, source, the members the request expanded to, the records as `name:locator` |
| `answer` | always | one of the four answers |
| `reason` | not covered | "record missing" (another record names the artifact) or "not registered" |
| `names` | may contain | the named cause, a `NAMES_*` word of `vocabulary.py` |
| `names_detail` | may contain | the reason beside the cause where the vocabulary fixes one (for "data-order trace missing") |
| `failing` | may contain | the first failing link, see below |
| `candidates` | may contain | candidate sources, see below |
| `chain` | contains | the evidence chain, one hop per item |
| `basis` | a checkpoint that contains | "whole-sample read assumed (F9, L9)" or "operator note" |
| `notes` | contains, does not contain | fixed sentences, see below |
| `untraced_steps` | a checkpoint | `[segment, step]` pairs of its history |
| `first_exposure` | a checkpoint that contains | `{"segment": ..., "step": ...}` |
| `level` | a checkpoint that contains | "consumed" (its own segment read the record) or "inherited" (an ancestor's did) |
| `scope` | does not contain | what the negative covers, see below |

Every answer has every field of the table, in that order; a field that does not
apply is null or empty. `docs/interfaces.md` section 6 fixes the same form.

A hop of `chain` is an object. `hop` is its short form: `artifact:locator` for a
record (`L<n>` a line, `D<n>` a document of a shard), `segment:step n` for a step.
The other keys say where it lies (`artifact`, `instance`, `record`, or `segment`,
`step`) and how it is tied to the next hop (`link`: the evidence label of the edge
or map row, "member" with `member` on a record of a requested member, "file-level
link" where a whole-file link ends the chain, "consumed" on a step). A hop whose
edge is a generation edge has `generation: true` and its `job`. The chain runs from
the artifact to the source record.

`failing` holds the keys that apply:

| Key | Content |
|---|---|
| `condition` | the letter of the proof condition (section 4); absent for a missing record upstream |
| `artifact`, `instance`, `record` | where the named record lies |
| `segment`, `step` | a step of a run |
| `job` | the generation job of the edge the link sits on (section 6) |
| `reason` | for a missing record upstream: "record missing" or "not registered" |
| `detail` | one sentence for a person |

`candidates`: where the failing link is a set of candidate parents, the sources of
those candidates. Otherwise the sources among the inputs of the step that made the
artifact, followed back through the artifacts those inputs were made from (plan
2.1).

`notes`:

- on "contains", when a chain of the artifact passes a generation edge: the parent
  list is process evidence, not proof that the output depends on the source
  (Decision 25). The note is given when any record of the artifact is in scope
  through a generation edge, also when the chain the answer shows is another one;
- on "does not contain", "generating-model exposure not assessed" (section 6):
  A negative whose proof passed a generation edge, while the generating-model switch is off.
- on "not covered" for a published model the store holds while its extension is off:
  "the published-models extension is off" (section 6 and 7).

`scope` of a negative: `source` and `members`; `records` the request was narrowed
to; for a checkpoint `runs` and `steps` (`[segment, first step, last step]` per
segment of its history); `generation_edges`, the jobs whose established bound the
proof rests on, nearest first, and `harness_logs`, one entry per such job (log
digest, number of handed records and of reads, the attempts the harness refused,
the model entry, and `harness`: which harness wrote the log, "in-process" or
"confined", null when the log does not say); `walked`, how many records the proof
visited;
`files_not_checked`, the artifacts of the ancestor closure whose file was not at
its recorded place. For these no file was read: their declared digest was
compared with the registered one only, and their declared count with the store's
rows only, so records removed from the store together with a lowered count are
not found while the file is away (docs/store.md, "States that are not faults"). A
negative says nothing beyond its scope: not about content that arrived under
another source (plan L12), and not about what a generating model carries (plan
L11). While the generating-model switch is off the sentence stands as written; with the switch on, a
negative past a generation edge also rests on each edge's model being "none" or a
registered checkpoint that itself answers "does not contain" (section 6); a model
that is not registered gives no negative.

### How "does not contain" is closed off

The answer type cannot say "does not contain" without a proof, and a proof comes
from one place (plan W6, Decision 1). This is enforced in the types:

1. `answers.Proof` has no constructor. Calling it raises `ProofRequired`. A proof
   cannot be changed, copied or pickled.
2. The one function that makes proofs is handed out by
   `answers.claim_proof_mint()`, once, and only to the module
   `waterwhite.coverage`. That module takes it when it is imported and keeps it
   inside `prove`. A claim from any other module, or a second claim, raises.
3. The mint remembers every proof it made, with the names the artifact had, the
   store the walk read and the store's mark (`store.state_mark()`) when the walk
   began. `Answer` checks when it is built, and again when it is written as JSON,
   that its proof is one of those, that the proof is about the same instance and
   the same request, that the name the answer shows is a name of that artifact or
   its instance id, and that the notes are the proof's own. A proof shown under
   the name of another artifact is refused. When it is built, the answer also
   checks that the store is as the walk read it, after every check of the proof's
   identity: a proof the store has outlived (it was written, by this process or by
   another connection, after the walk) builds no answer and ends in the named error
   "store written during the query" (`StoreWrittenDuringQuery`, docs/interfaces.md
   section 2); a proof whose store is closed or gone stays `ProofRequired`, a mistake
   of the caller. (An answer already built is not asked again when it is written, so
   it can be written after its store is closed.) An object with the shape of a proof
   that the mint did not make is refused with `ProofRequired`, whatever the store
   holds.
4. `Answer` is frozen. `dataclasses.replace` runs the same checks. An answer whose
   field was changed around the type is refused when it is written.
5. What `prove` returns when the proof does not hold is a `Finding`, and a
   `Finding` refuses the answer "does not contain".
6. The kind that decides which rule walks an artifact is the kind the store holds
   for its instance id. An `Artifact` handed to `prove` or to a session is looked
   up again by instance id, and what the store holds is used, so an object built
   around the store (another kind, other names) decides nothing; an instance id the
   store has no row for is not covered or refused.
7. A session and an integrity pass are valid for one state of the store (below).

The one gap this leaves open on purpose: code in the process can set the private
fields of a `Session` (its table of finished verdicts). No query reaches them.

A program can still defeat this by reaching into the interpreter's internals on
purpose. No mistake in a query can.

## 3. The proof

```python
from waterwhite import coverage

result = coverage.prove(store, "E/shard-3", request)   # a Proof, or a Finding
answer = coverage.answer(store, "E/shard-3", request)  # the Answer
```

`prove` walks the artifact's ancestor closure on demand from the store. Nothing is
precomputed and nothing is stored: the store's tables are the only authority.

It returns a `Proof` when every condition of section 4 holds and no record in
scope was found. Otherwise it returns a `Finding`:

1. a recorded chain shows "contains". This wins over a failed condition anywhere
   else, for every kind of artifact (plan 2.2 closing paragraph, Decision 20);
2. else the artifact's own record is missing: "not covered";
3. else "may contain" naming the first failing link.

The walk never stops early. It reads every record of the artifact, so the answer
does not depend on where a record sits. "First" failing link means first in walk
order: a file's lines in order, a shard's documents in order, a mixture's shards in
the trainer's order, and for each record its parents in the order its edge names
them, a parent's chain before the next parent. For a checkpoint see section 8.

The loud checks. Before the walk, `prove` runs `store.check_integrity` over the
artifact's ancestor closure. A fault raises a named error of the integrity family
(exit code 3) and no answer exists (plan 3.4). `coverage.check_store(store, ids)`
runs the checks once and returns a pass that `prove` accepts in place of its own
run; a listing uses it for the whole store. Only `check_store` makes a pass that
covers anything: a pass made by calling `IntegrityPass` says nothing about the store,
and `prove` runs its own checks for it. A pass is valid for the state of the store
it was made on: once the store is written, by this connection or another, it covers
nothing and `prove` runs its own checks again. (A change to an artifact's file on
disk is not a write to the store; the pass does not see it.) The checks read every file of the
closure that is at its recorded place once. A file with its registered digest
must hold as many records as each of its bindings declares, so occurrence rows
or map rows removed from the store together with a lowered count stop the query
with "map count mismatch" and the walk never runs over the shortened record.

With the generating-model switch on, the ancestor closure of the asked artifact also holds every registered checkpoint that a generation log in the closure names as its model, with that checkpoint's own ancestry, to a fixpoint; the loud checks and the list of files not checked run over it on every query. With the switch off the closure is as before.

`coverage.Session(store, request)` holds what a walk has learned. Passing one
session to several `prove` calls lets a shard that several mixtures hold be walked
once. A session is valid for the state of the store it began on
(`store.state_mark()`): after any write to the store, `prove` called with it
refuses with the named error "store written during the query"
(`StoreWrittenDuringQuery`, exit code 2, docs/interfaces.md section 2) and a new
session reads the store as it is. The same refusal stands if the store is written
during one `prove` call, and if it is written after `prove` returned its proof and
before the answer is built from it: the answer type refuses a proof the store has
outlived by the same name. `query`, `report` and `verify` also compare the store's
mark once more after their last answer is built and before anything is printed.

## 4. The conditions

A proof holds only if all of these hold (plan 2.2). Each failure names a link.

| | Condition | Where it is checked | Link named, and cause |
|---|---|---|---|
| (a) | every record has a recorded chain back to ingested revisions | the walk | the record with no edge, no member and no deciding link, or whose edge names no input, a generation edge included: "missing origin" |
| (b) | each link's evidence level can decide the request | the walk, per record and requested source | the record whose link is too weak: "evidence level cannot decide the request"; a mixed candidate set: "candidate parents" |
| (c) | every parent an edge names exists | the store's loud check | error "edge parent missing", no answer |
| (d) | every map's digest and count match its artifact | the store's loud check: the declared count against the store's rows and, for a file that is present with its registered digest, against the records the file holds | error "map count mismatch" or "map digest mismatch", no answer |
| (e) | every step of a checkpoint's history has a complete trace | section 8; the count of a present trace is the store's loud check | the step: "step untraced", or the cause stored for a segment with no trace route |
| (f) | each label agrees with its derivation and verification | the walk | the map (no record), the map row or the edge: "label stronger than derivation" |
| (g) | past a generation edge that does not name the source, the edge's upper bound is established | the walk | the job and the edge's record: "upper bound not established"; a record handed and not stated as used: "in scope only" |
| (h) | the generating model of each such edge is clean | Applied while the generating-model switch is on, per record, where the walk meets a generation edge whose bound is established, and, for a record whose edge is a publisher's pointer, in place of (g); not applied while it is off (section 6). | |

Besides the conditions, a missing record upstream gives "may contain" naming it
(section 7).

(f) in detail. `vocabulary.labels_allowed(derivation, verification)` is the one
statement of which labels a link may carry. The store refuses a stronger label
when it writes. A label edited afterwards is caught here: for a map's own label
the failing link is the map (`artifact` with no `record`); for a map row, an edge
or a source link, the record. A chain through a map whose own label fails (f) can
still show "contains" through rows whose labels are allowed; it cannot support a
negative. A source link whose label fails (f) shows nothing, for the member it
names or for any other: its record answers "may contain" unless another chain of
the record shows "contains".

## 5. What a link decides

Evidence levels are read per record and per requested source (Decision 3). The
walk asks of each link what it says about the requested source. A strong link to
one source never stands in for a weak link to another: a record with an observed
edge to A and a file-level link to B is "contains" for A and, for a record of B,
"may contain".

A record has up to three kinds of evidence, and all are read:

- it is a record of a member (an ingested revision): its chain ends there. It is in
  scope when the member is one of the request's and, for a record-level request,
  the record is one of the request's;
- an edge to parent records;
- source links: a link from the record, or from its whole file, to a member, with
  no parent record behind it.

By label (plan 3.2):

| Label | On an edge or map row | As a source link |
|---|---|---|
| observed | record-level: the parents are walked | (not allowed) |
| recovered | record-level: the parents are walked | (not allowed) |
| mapped | record-level for the parents it names; never a negative for a source it does not name (g) | "contains" for a whole-source request about its member; (b) for a record-level request; (g) for any other source |
| candidates | every candidate in scope: "contains". None in scope and all clear: a link to other sources. Otherwise "may contain" (b), with the candidates' sources | (not allowed) |
| file-level | for a request about the whole source: "contains" when a parent is in scope, otherwise (b). For a record-level request: (b) | to a requested member: "contains" for a whole-source request, (b) for a record-level request. To another member: says nothing |
| unknown | (b) | to a requested member: (b). To another member: says nothing |

A file-level link is knowledge about a file ("this file holds translations of
A"). It shows "contains" for a request about all of A. It never supports "does
not contain" for another source (plan 2.2(a), 3.2; DECISIONS.md D-24): the link
names what the file derives from and does not vouch for what else the file holds.
A record whose only evidence is a file-level link has no recorded origin of its
own, so for every source the link does not name it fails (a) with "missing
origin". Where the record also has an edge, the edge is what a negative rests on
and the link adds nothing to it. Tests:
`test_a_file_level_link_decides_the_whole_source_and_no_record` and
`test_an_observed_link_to_a_never_masks_a_file_level_link_to_b` in
`tests/coverage/test_coverage_proof.py`.

Parent modes (plan 3.1):

- "all": the child is in scope when any parent is. Every parent is walked.
- "any" (equivalent copies, one selected by a dedup step): the child follows the
  parent the step selected. History keeps its attribution: a release that kept A's
  copy of a passage "contains" for A and not for B, which supplied the same text.
  The other copies decide what a rebuild does later, not this answer. An "any"
  edge with no selected parent is read as a candidate set.

## 6. Generation edges

A generation edge names the records its job was handed. A handed record the output
states it used is a parent. One it does not state is in scope only (Decision 25).
A job's sole handed record is a parent whatever is stated (DECISIONS.md D-21).
A generation edge that names no input at all, a job handed nothing, gives its output
record no recorded origin: condition (a) fails at the record ("missing origin", "the
edge names no parent"), for every source, and the record is never "does not contain"
(plan 2.3).

- A parent in scope: "contains", and the answer carries the process-evidence note.
- A record in scope only that is itself in scope of the request: "may contain"
  naming "in scope only", with the job and the handed record. Never "contains" on
  that basis.
- A source the edge does not name: a negative needs the edge's upper bound, that
  the job read nothing beyond what it was handed (plan 2.2(g), Decision 13). The
  bound holds only when all of these hold: the edge is an observed generation
  edge; its harness log is stored and was written by the confined harness; the log
  lists no read outside the hand-off that
  the harness let through (a refused attempt is listed in the scope and does not
  break the bound, DECISIONS.md D-07); the log is the log of the job the edge
  names; the edge's stored bound field says established; every input of the edge
  lies in the log's hand-off. The stored field alone never establishes a bound,
  and a field that says "not established" denies it. A mapped edge never has a
  bound. A generation edge with no harness log (a job that wrote its own parent
  list) has none. Otherwise: "may contain" naming "upper bound not established"
  with the job.

Which harness a bound rests on. Only a log the confined harness wrote
establishes a bound (Decision 13: a bound is established only by a closed-input
harness whose access log lists every record the job read; plan 3.3: the job can
reach no other record or record source). A confined job has no way to a record
but the harness; an in-process job can open files the log never sees
(docs/capture.md, docs/confine.md section 5). So a log the in-process harness
wrote, or one that names no harness (every log of a store made before the word
existed), establishes none, and past its edge the answer is "may contain" naming
"upper bound not established" with the job (plan F10's fallback). Every entry of
`scope.harness_logs` carries `harness`. Tests:
`test_only_a_log_the_confined_harness_wrote_establishes_a_bound` and
`test_a_negative_names_the_harness_behind_each_bound_it_rests_on`.

Which job an answer names (DECISIONS.md D-24). A generation edge keeps the name
of its job in a field of its own, beside its harness log. `failing.job`, the `job`
of a hop and the sentence in `failing.detail` are read from the edge. So when a
generation edge's log is gone, or its bound is not established for any other
reason, the answer names that job and not the step (plan Check 5: with the log
of P's job removed and the bound field left at established, shard 3 answers "may
contain" naming P's job). An edge that is not a generation edge has no job. For a
publisher's pointer that fails (g), `failing.job` holds the name of the edge's
step. Test: `test_a_bound_field_with_its_log_removed_is_no_bound`.

Condition (h), the generating-model rule (plan 2.2(h), Decision 28), is an
extension, built and off by default (DECISIONS.md D-E2-1).

- While the switch is off the model is a logged attribute of the edge and not an ancestor. A negative that passes a generation edge rests on conditions (a) to (g) and carries the words "generating-model exposure not assessed"; the scope lists the model entry of each log it rests on.
- While the switch is on, condition (h) is applied in the walk, per record. The model is read from the edge's stored harness log, never from the job's own statement. "none" lets the record stand. A registered checkpoint (a name the store holds for an artifact of the kind checkpoint) is answered for the same request, in the same session, and lets the record stand only if it answers "does not contain"; otherwise the record answers "may contain" naming the checkpoint ("generating model exposed"), or "upstream record missing" with the reason "record missing" when the checkpoint has no record of its own. Any other text, an artifact of another kind included, answers "may contain" naming the text ("generating model not registered"). A log with no model entry answers "may contain" naming the job ("generating-model entry missing"). The list form the harness writes is judged element by element in listed order. Condition (g) is applied first: a log whose texts were not shown and compared gives "upper bound not established" and (h) is not reached. A recorded "contains" wins whatever (h) says. The note is not printed on a negative.
- Where a publisher's pointer is the only evidence the generating model is not registered, so a record whose edge is a pointer and whose parents do not name the requested source answers "may contain" naming the model as unregistered (condition h); with the switch off it is condition g.
- A temporal ancestry cycle (a registered model checkpoint whose own history reads a record that the model's own output feeds) is a refused request, exit code 2, no answer printed, unless a recorded "contains" of the same artifact wins. The walk carries the cycle as a failed link and raises the error where the artifact's answer leaves the session. A store can hold such a cycle; the write-time refusal is not built.

A switch that holds neither "on" nor "off" (the store
writes no such value) is not read as off. The switches of all six extensions are
checked before anything else is done for a request, so a broken switch stops every
answer, the listing, the report and `verify`, including an answer that never reads it, and the
refusal names the extension and the value it found. The store is opened first (the switch is read
from it); only then is a `--layout` or `--placement` value interpreted, a placement file opened,
or the report or sides file of `verify` read. The parser still refuses a missing value or an
unknown option before the store is opened. So does a setting under the
`extension.` prefix that names no extension of this build. Writing the switch again
(`Store.set_setting`) mends a wrong value, and `Store.remove_setting` clears a key the
tool cannot read (it refuses to write one); `show` and `check` are not answers and keep
working on it.

## 7. Artifacts, and where "not covered" ends

"Not covered" is only for the artifact whose own record is missing (Decision 21).
Everything downstream of it answers "may contain" naming the missing record
(`names`: "upstream record missing", `failing.artifact` and `failing.reason`),
unless a recorded chain the gap does not touch shows "contains".

| Artifact | Its own record | "not covered" when | What the proof walks |
|---|---|---|---|
| JSONL file (source, generation output, release, import) | its occurrence record: one occurrence per line | the record is gone ("record missing") | every line |
| shard | its tokenization map | it has no map ("record missing") | every document, through the map a reader gets |
| mixture | its snapshot | the snapshot holds no entry ("record missing"). The store registers no mixture without one, so this is a snapshot that was removed; with no shard to walk nothing is proved | the shards of the snapshot, in the trainer's order |
| checkpoint | its run segment | the segment has no row ("record missing") | section 8 |
| any name the store has no row for | | "record missing" when another record names it by value, else "not registered" | nothing |

- Which map a reader gets is the store's rule (DECISIONS.md D-20 as amended by
  D-24): a verified map before one that is not, across namespaces; among verified
  maps the capture map, else the position-mode recovery map, else the
  token-matching one. The proof reads `store.binding(shard)` and
  `store.map_rows(shard)` with no namespace named. A shard whose only maps are
  not verified still has a record of its own: it is not "not covered", and each
  of its documents answers "may contain" for its "unknown" link (b).
- A snapshot entry that names a shard with no row, or a shard with no map: the
  mixture names it and answers "may contain". The listing shows the shard as "not
  covered".
- A JSONL file on the way whose occurrence record is gone: everything derived from
  it names it the same way.
- A published model is an extension that is off: a query naming one answers "not covered"
  with the reason "not registered" and the note "the published-models extension is off"
  (plan 1.4, 3.1, Section 7), if such an artifact is ever in a store. A listing of a store
  that holds one gives the same answer (section 6). With the switch on this build holds the
  off state only, and the query is refused.

## 8. Checkpoints

A checkpoint is answered from what its history consumed, not from what its mixture
held (plan 2.4). A mixture can hold a record of the source, and so "contain",
while a checkpoint trained on it never read the record.

History: the registered steps of the checkpoint's segment up to its step, then the
same for the segment of the checkpoint it started from, and so on to a segment
that started from scratch.

A step is untraced (condition (e)) when it has no trace; when a sample of its
trace has an uncovered token range; when a sample's coverage mark claims more
tokens than the sample's stored rows add up to (the rows that show what the step
read are gone: the mark is no evidence without them); when the operator's step record gives fewer
tokens of its last sample as read than the sample holds (the note is applied here
even if the stored coverage mark is whole: a note can only lower); or when its
number lies in the history and is not among the segment's registered steps (a gap
between two registered steps, steps between the first step a launch records and
the first registered one, or the checkpoint's own step). `coverage.untraced_steps(store,
segment)` applies this rule to a whole registered segment, as for a checkpoint at its
last step, and `waterwhite show <segment> --trace` lists its result (docs/cli.md).

The rule about the mark follows plan 3.5's own definition of it: the mark counts the
sample's tokens that its rows cover (DECISIONS.md D-50). A mark above what the sample's
rows add up to contradicts that definition. Such a step is untraced, as Check 5's table
treats a deleted or a half-covered trace (plan 5.5 Check 5), and is not a stop: nothing is
refused and `check_integrity` raises nothing; the checkpoints that hold the step answer
"may contain".

A segment whose trace route is "none" has no usable trace. Each of its steps is
untraced, and the cause is the one stored for the segment when the run was
registered: "data-order trace missing" with its reason in `names_detail`, "cyclic
order not recorded", or "cyclic order digest mismatch" (plan 3.5). The reason is one of
the six words of `vocabulary.TRACE_MISSING_STORED_REASONS` or none, and is stored at
registration, never derived when the answer is made. A segment with route "none"
answers so whether or not its run record holds a launch. The proof reads
`trace_route`, `trace_cause` and `trace_cause_detail`; it does not compute them.
They come from the run record and the files it names, at registration, through
`waterwhite register-run` (docs/join.md section 3), never from the caller: a
segment is "cache files" when the index arrays have the recorded digests,
"replay" when the arrays derived again have them, and "none" with a cause
otherwise.

Three cases the milestone rules (plan 2.4, 3.5; DECISIONS.md D-25, D-26). (a) A
segment read with the cyclic sampler whose run record holds no shuffle digest
supports no "does not contain" and no "contains": it is "may contain" naming
"cyclic order not recorded", whatever cache files were kept. The store refuses to
write such a segment with a route other than "none", and the proof reads such a
segment as having none even where a store edited by hand holds another route.
(b) A segment's route follows from its run record and nothing else, whoever calls
the store: the join computes it, no flag of the command states it, and the store
refuses ("invalid record", nothing written) a route of "cache files" or "replay"
for a run record that holds no digest of an index array the segment has (a replay
counts only against digests recorded at training time, plan 3.5, Check 2; kept cache
files count only against the same digests, a decision stricter than plan 3.5 and 4.6,
DECISIONS.md D-41), and for no run record at all. A cyclic segment whose digests are all recorded gets its
route only when the order the join computes has the recorded digest and count
("cyclic order digest mismatch" otherwise). The store also refuses a segment with
no trace route that names any cause but the five of `vocabulary.TRACE_CAUSES`, and
one that carries a reason beside any cause but "data-order trace missing" or a reason
outside the six words of `vocabulary.TRACE_MISSING_STORED_REASONS` (a null reason is
accepted: a direct caller may not know it).
(c) Nothing stronger than "may contain" rests on a segment whose run record holds no
launch (plan 2.2(e), 3.1, 3.5): nothing says which step comes first or where the
counter of its launch starts, so no step of it is known to be whole. This is closed at
the write, whoever calls the store. The store refuses ("invalid record", nothing
written) a trace route other than "none" for a run record whose `launches` is empty,
null or absent, and it refuses any trace, through `write_step_traces` or through
`register_run_segment(traces=...)`, for a segment whose route is "none". So no store the
tool writes holds a trace for a segment whose run record holds no launch, and a "contains"
cannot rest on one. Such a segment is registered with the route "none" and answers as
every segment with no trace route does: "may contain" naming its stored cause and
stored reason, at its first step, condition (e). Through `register-run` a run record
with no launch gets the cause and reason of the first question of the join that
applies (docs/join.md section 3): when nothing earlier applies, "data-order trace
missing" with the reason "run record incomplete" (question 5). The name an answer gives
follows from the record, never from a route a caller stored. A segment registered with
no run record at all keeps the cause stored for it ("data-order trace missing", "no
recorded digest"). Only a store edited by hand can hold a stored trace route and a run
record with no launch; the proof reads such a segment as the join would have stored
it, "may contain" naming "data-order trace missing" with the reason "run record
incomplete", and a cyclic one with no shuffle digest as "cyclic order not recorded",
the join's own order of questions (2 before 5).
Tests: `tests/coverage/test_register_run_command.py`,
`test_a_cyclic_segment_with_no_shuffle_digest_supports_no_answer_whatever_route_is_stored`
in `tests/coverage/test_coverage_checkpoints.py`,
`tests/store/test_route_follows_from_record.py` and
`tests/coverage/test_no_launch_segment.py`.
Traces stored under such a segment are not read. The store registers such a
segment only with one of the five trace causes. Where the stored cause is no named cause (it was
edited), the answer names "data-order trace missing", which the missing route
alone shows, says in `failing.detail` what it found, and passes no reason on in
`names_detail`. Where the stored cause is a named cause, `names_detail` is the reason
stored with it, as registered.

A segment is verified when the store lists its release tag, sampler type and
sampler class among its verified joins (assumption F6). The trace rows of an
unverified segment decide nothing. A store made by `waterwhite init` lists the joins
that plan Check 1 verified, shipped with the tool: release `core_v0.20.0` with the
sequential sampler (`MegatronPretrainingSampler`) and the cyclic one
(`MegatronPretrainingRandomSampler`; docs/cli.md, "init"). A store made by the
library lists what `Store.create` was given.

The answer:

- "contains" when a trace row of a step of the history reads a document whose
  chain is in scope, on a verified segment with a trace route and a mixture
  snapshot, and the row lies inside the covered part of its sample. One token is
  enough (Decision 18). `first_exposure` is the earliest such step in history
  order. `basis` is "operator note" when the operator's step record covers the
  sample that was read, else "whole-sample read assumed (F9, L9)". `level` is
  "inherited" when the exposure lies in an ancestor's segment. The answer lists
  the untraced steps of the whole history beside the chain, and holds whatever
  else is untraced.
- "does not contain" when the history reaches a segment that started from
  scratch; every step of it is registered and completely traced; every segment is
  verified, has a trace route and a mixture snapshot; conditions (a), (b), (f)
  and (g) hold for every document the steps read; and none is in scope. The scope
  lists the runs and steps.
- "may contain" otherwise, naming the first failing link.

Which link is first (plan 2.2: "the first failing link"; plan 2.4; DECISIONS.md
D-24). The order of causes is the order of the history. The history is read
oldest step first, and the first failing link is the one at the earliest step: an
untraced step, or a document a step read whose chain fails a condition. The order
in which plan 2.4 lists its causes decides only inside one step: the untraced
step, then the consumed record. So a checkpoint whose step 7 read a record with a
broken chain and whose step 8 is untraced names the record, and still lists step
8 in `untraced_steps`. Under the other reading, plan 2.4's list order over the
whole history, the faulted link of an early step would be hidden behind a later
untraced step. Tests: `test_the_first_failing_link_is_the_one_at_the_earliest_step`
and `test_at_one_step_the_untraced_step_comes_before_the_record_it_read` in
`tests/coverage/test_coverage_checkpoints.py`. A cause that belongs to a whole
segment stands where the segment stands:

| Cause | `names` | Where it stands |
|---|---|---|
| the history stops at a missing record (a parent checkpoint with no row, an ancestor's segment with no row) | upstream record missing | before everything: what came before it is unknown |
| a segment with trace route "none" | the stored cause and its stored reason (`names_detail`), whether or not its run record holds a launch (section 8 (c)) | at the segment's first step |
| a segment with a stored trace route whose run record holds no launch (a store edited by hand) | data-order trace missing, reason "run record incomplete" | at the segment's first step |
| an untraced step | step untraced | at the step |
| a segment with no mixture snapshot | mixture snapshot missing | at the segment's first step, as a record cause |
| a consumed document whose shard has no map | upstream record missing | at the first step that read the document |
| a consumed document whose chain fails (a), (b), (f) or (g) | that condition's cause | at the first step that read the document |
| the checkpoint's own segment is unverified | data-order trace unverified for this version, or for this sampler type | after the segment's own steps |
| an ancestor's segment is unverified | ancestor may contain, with the ancestor checkpoint | after that segment's steps |

"For this version" is named when the release tag is in no verified join, "for this
sampler type" when the tag is and the sampler type or class is not.

A fault on a record that no step of the history read leaves the checkpoint's
answer as it was (DECISIONS.md D-22). The loud checks are wider: they cover the
whole mixture of each segment.

A document counts as read by the first step whose trace holds a row for it, also
when that row lies in an uncovered range: it then orders the links, and it is not
evidence of exposure.

## 9. Listings

`listing.listing(store, request)` puts one request to every artifact the store
holds (plan 2.1, Decision 26; DECISIONS.md D-24):

```json
{"request": {...},
 "covers": {"runs": [], "mixtures": [], "shards": [], "checkpoints": [], "releases": []},
 "answers": [...],
 "not_covered": [{"artifact": "E/shard-1", "instance": "...", "kind": "shard",
                  "reason": "record missing", "named_by": [...], "detail": "..."}]}
```

- `covers` names what has a record of its own: every run, every mixture, every
  shard with a map, every checkpoint whose segment is registered, every JSONL file
  with the role "release".
- `answers` holds one answer for every registered artifact, whatever its kind and
  role: source files, the files between source and release, imports and releases;
  shards; mixtures; checkpoints. An artifact with no record of its own is
  registered too, and its answer is "not covered". The order is by kind (JSONL
  files, shards, mixtures, checkpoints) and inside a kind the order of
  registration. So the number of answers is the number of artifacts the store
  holds.
- `not_covered` names every artifact that has no record of its own, registered or
  only named by another record. Registered (each also has its "not covered"
  answer): a shard with a row and no map; a JSONL file with no occurrence record;
  a mixture whose snapshot holds no entry; a checkpoint whose segment has no row.
  Only named (no row, so no answer): a shard a snapshot names, a mixture a segment
  names, a checkpoint a segment started from, an input of a step. And, with
  `artifact` null and the reason "not registered", the mixture of a segment that
  was registered with no snapshot (plan L6). `artifact` is a name where the
  artifact has one, else the path its snapshot gives, else its instance id.
  `named_by` says which records name it; it is empty for a registered artifact
  that nothing names.
- A store that holds a published model while the published-models extension is off
  is listed: the model answers "not covered" with the reason "not registered" and the
  note "the published-models extension is off", and stands in `not_covered` with that
  sentence as its detail. A query that names the model gives the same answer. With the
  switch on, this build holds
  the off state only and the listing is refused with the error of a query: a listing
  passes no registered artifact over.
- An artifact nobody registered and no record names cannot appear (plan L10). The
  gap shows as the difference between `covers` and the team's own inventory.
- The loud checks run over the whole store first. One fault anywhere stops the
  listing.

## 10. What the walk reads

Nothing is loaded as a table. The walk reads the store through its documented
read methods and through nothing else, one row or one small group of rows at a
time.

Per JSONL artifact met (the artifact asked about, and each artifact that holds a
parent), once: its binding; whether any of its records has an edge; its whole-file
members (`file_members`); whether any member names single records of it
(`has_record_parts`); its whole-file source links; whether any record of it has a
source link of its own.

Per record of the artifact asked about, and per ancestor record: `edge_of` (one
edge with its inputs and harness log) when the artifact has edges; the members of
that one record (`record_members`) when the artifact has record members;
`source_links` when it has record links. For the candidate sources of a "may
contain", `sources_of_artifact` once per artifact of the closure. A record of a source file that is covered by a whole-file member
costs no query at all.

- A JSONL file: `occurrences` is read as an open cursor, one line at a time.
- A shard: `map_rows` is read as an open cursor, one document at a time.
- A mixture: its snapshot (one row per shard), then each shard as above.
- A checkpoint: its history (the step numbers of its segments); per segment the
  segment row, `trace_states` as a cursor, the snapshot, and `trace_documents` as
  a cursor (the distinct documents its steps read). Per document one `map_row`.
  Per document in scope, `trace_hits`. One step's trace is loaded only to place a
  row inside a partly covered sample, or to read the last sample of a step with an
  operator note.

Memory. The walk keeps the verdict of each record it has finished, so that a
record many others derive from is walked once. The table is a cache: at 200,000
entries (`_walk.MEMO_LIMIT`) it is emptied and filled again, and the answers do
not depend on it. A chain is walked with a stack of its own, so its length is not
bound by the interpreter's recursion limit. A chain that runs into itself fails
(a). What grows with the artifact and stays: the untraced steps of a checkpoint's
history, one entry per generation job a negative rests on, and the store's own
`History` (the step numbers).

Speed is not measured here. The lookup of an edge by its child is one indexed
query. The lookup of the members of one record is made only for an artifact that
has members naming single records, and it then scans the member parts, because the
schema has no index by artifact and locator; for a source whose members are whole
files it is never made. The store documents which of its own lookups scan.

Inside the walk a verdict is checked when it is made: a "may contain" with no
failing link, or a "contains" with no chain, cannot be built. The tallies that
add verdicts up read exactly those two fields, so a verdict that lacked one
could otherwise count as clear.

## 11. Not built

- The write-time refusal of a record-level cycle; the pairing of condition (h) with a content ban or flagged records; the published-model branch.
- The content ban, flagged records, published models, exposure counts, adapters:
  their off states only (a refusal, or "not covered").
- The proof does not compute the exposure join. It reads the step traces, trace routes
  and causes that the registration of a run writes (the join itself is built: docs/join.md).
- Checkpoint level "eligible" (the record was in a mixture the run used and no
  step is shown to have read it) is not reported.
- A measured walk time (plan 5.6).
