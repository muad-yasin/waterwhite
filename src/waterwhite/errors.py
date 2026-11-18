"""Named errors and exit codes.

Three families, each with its own exit code, so a caller never has to read text to
tell a refusal from a fault:

- a refused request (exit 2): the tool answers nothing and never approximates
  (plan 2.3, L7);
- a loud integrity fault (exit 3): the query stops, names the mismatch and prints
  no answer, whatever the answer would have been (plan 2.2, 3.4);
- a refused write (exit 4): the store rejects a record that does not match the
  artifact it describes (plan 3.4).

Each error has a fixed name. The name is what a test, the hand-written table and a
person compare; the detail after it is free text.

Every named error of the tool is a class of this module, whichever module raises
it. A module imports the class from here and defines none of its own. An error of
no family (exit 1) is a mistake in how the tool was called.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REFUSED = 2
EXIT_INTEGRITY = 3
EXIT_WRITE_REFUSED = 4


class WaterwhiteError(Exception):
    """Base class. ``name`` is the fixed error name, ``detail`` says what and where."""

    name = "error"
    exit_code = EXIT_USAGE

    def __init__(self, detail: str = "", **context: object) -> None:
        self.detail = detail
        self.context = context
        super().__init__(f"{self.name}: {detail}" if detail else self.name)

    def to_json(self) -> dict:
        return {"error": self.name, "detail": self.detail, "context": dict(self.context)}


# --- Refused requests (plan 2.3, L7) -----------------------------------------------


class Refused(WaterwhiteError):
    name = "request refused"
    exit_code = EXIT_REFUSED


class SourceNotRegistered(Refused):
    """A withdrawal that names a source with no registered member (plan 2.3)."""

    name = "source not registered"


class ExtensionOff(Refused):
    """A request of an extension that is switched off (plan 1.4, 2.3, L7)."""

    name = "extension off"


class UnsupportedSelector(Refused):
    """A selector the first build does not support, such as a URL prefix (plan 2.3)."""

    name = "unsupported selector"


class UnsupportedFormat(Refused):
    """A corpus format other than JSONL and Megatron binary shards (plan 2.3, 5.7)."""

    name = "unsupported format"


class MemberNotRegistered(Refused):
    """A withdrawal narrows to a member that is not registered under its source.

    The plan names one refusal for a withdrawal, "source not registered" (plan 2.3).
    This name is an addition: error names beyond those the plan fixes are left to
    the builder (plan 5.8, B-7), and DECISIONS.md D-26 records this one."""

    name = "member not registered"


class RecordNotInSource(Refused):
    """A withdrawal narrows to a record that no member it covers holds, or that the
    store does not hold.

    Like "member not registered" this name is an addition the plan leaves to the
    builder (plan 5.8, B-7; DECISIONS.md D-26). The plan itself names one refusal
    for a withdrawal, "source not registered" (plan 2.3)."""

    name = "record not in source"


class AmbiguousArtifact(Refused):
    """A query names an artifact by digest or path and more than one instance matches."""

    name = "ambiguous artifact"


class NotAStore(Refused):
    """The path given as a store holds no store: there is no file, the file is not
    an SQLite file, or it has no schema version of this tool (plan 3.4)."""

    name = "not a store"


class NewerSchemaVersion(Refused):
    """The store was written by a newer version of the tool: its schema version is
    above the one this build reads. Nothing is read from it (plan 3.4)."""

    name = "store of a newer schema version"


class StoreWrittenDuringQuery(Refused):
    """The store was written, by this process or by another connection, between the
    moment a read began and the moment its answer was to be made. Nothing is
    answered; the command can be run again (docs/interfaces.md section 2). A
    coverage session holds for one state of the store (DECISIONS.md D-45)."""

    name = "store written during the query"

    def __init__(self, detail: str = "the store changed after the query began; run the command again",
                 **context: object) -> None:
        super().__init__(detail, **context)


class HarnessRefused(Refused):
    """A job asked the closed-input harness for a record that was not handed to it
    (plan 3.3 "Generation", F10). Nothing is delivered, and the attempt is in the
    harness log with ``refused`` true."""

    name = "read outside the hand-off"


class TemporalAncestryCycle(Refused):
    """With the generating-model extension on, a registered model checkpoint whose own
    history reads a record that the model's own output feeds (plan 2.2(h), 3.3
    "Generation"). The answer for the artifact would rest on a cycle, which is invalid
    evidence, so the request is refused and no answer is printed; a recorded "contains"
    of the same artifact wins and is answered. The family is the one of the other refused
    requests (exit 2). The detail names the checkpoint and the record."""

    name = "temporal ancestry cycle"


# --- Loud integrity faults (plan 2.2 closing paragraph, 3.4) -----------------------


class IntegrityFault(WaterwhiteError):
    name = "integrity fault"
    exit_code = EXIT_INTEGRITY


class MapCountMismatch(IntegrityFault):
    """A map or occurrence record whose count does not match its artifact (2.2(d))."""

    name = "map count mismatch"


class MapDigestMismatch(IntegrityFault):
    """A map or occurrence record whose digest does not match its artifact (2.2(d))."""

    name = "map digest mismatch"


class EdgeParentMissing(IntegrityFault):
    """An edge whose parent occurrence is absent (2.2(c))."""

    name = "edge parent missing"


class TraceSampleCountMismatch(IntegrityFault):
    """A present trace whose consumed-sample count differs from its step's batch size."""

    name = "trace sample count mismatch"


class ArtifactDigestMismatch(IntegrityFault):
    """An artifact whose digest no longer matches its file."""

    name = "artifact digest mismatch"


class StoredDigestMismatch(IntegrityFault):
    """A digest the store keeps of its own records that the records no longer give: a
    run record whose text differs from the digest registered with it, a mixture whose
    snapshot or settings no longer give its content digest, parameter digest or
    instance id."""

    name = "stored digest mismatch"


class LocatorDomainMismatch(IntegrityFault):
    """A record of a binding whose number lies outside the count the binding declares:
    a line locator other than L1 to Ln of a JSONL artifact of n lines, an ordinal
    other than 0 to n-1 of a shard's map of n documents."""

    name = "locator domain mismatch"


class CacheDigestConflict(IntegrityFault):
    """A segment with more than one recorded digest for one array of the blend, which
    has one digest."""

    name = "cache digest conflict"


# --- Refused writes (plan 3.4) -----------------------------------------------------


class WriteRefused(WaterwhiteError):
    name = "write refused"
    exit_code = EXIT_WRITE_REFUSED


class DeclaredCountMismatch(WriteRefused):
    """The declared record count does not match the artifact on disk."""

    name = "declared count mismatch"


class DeclaredDigestMismatch(WriteRefused):
    """The declared artifact digest does not match the artifact on disk."""

    name = "declared digest mismatch"


class LabelStrongerThanDerivation(WriteRefused):
    """A label the link's derivation and verification result do not allow."""

    name = "label stronger than derivation"


class InvalidRecord(WriteRefused):
    """A record that breaks a rule of the record model (a parent outside the input
    scope of its job or outside the inputs of its step, an unknown word of the
    vocabulary, a missing field, a second registration that disagrees with the
    stored instance in a field the instance id does not cover)."""

    name = "invalid record"


class TorchUnavailable(WriteRefused):
    """A run read with the cyclic sampler is registered and the tool cannot import
    torch to compute its shuffle (plan 3.5: the join calls
    torch as the sampler does, and never a copy of it). The registration is
    refused and nothing is written; no cause word is stored for it, because a
    segment's cause says what its record holds, not what this machine has."""

    name = "torch not available"


class UnknownNormaliser(WriteRefused):
    """A normaliser name that this build does not hold (plan 3.1, F8). A
    fingerprint is only taken in order to write a record, so the write that needed
    it is refused. There is no default normaliser."""

    name = "unknown normaliser"


class StoreFileExists(WriteRefused):
    """``Store.create`` was given a path that already holds a file. A store is
    never created over a file, whatever the file is."""

    name = "a file already exists at the store path"


class DropAlreadyRecorded(WriteRefused):
    """A recovery was handed a sanitizer's log for a line whose skip the store
    already holds under another artifact. A skipped or empty line is recorded
    once, as a drop whose output is the first registered artifact that lacks the
    line (plan 3.1 "Drop", 3.3). Nothing of the call is written."""

    name = "drop already recorded"


# --- Mistakes in how the tool was called (exit 1) ---------------------------------


class UsageError(WaterwhiteError):
    """A command line the tool cannot read: an unknown command or flag, a shortened
    flag, a missing argument. Raised by the command line only."""

    name = "usage"


class ModelClientMissing(WaterwhiteError):
    """A job called a model through the harness and ``run_job`` was given no model
    client. The call is in the harness log before this is raised."""

    name = "no model client"


# The three errors of the confined harness (``waterwhite.confine``, plan F10, 5.8
# B-1). Like "no model client" they belong to no family: the harness was set up or
# used in a way that cannot work. Each means that the run gave no outputs and so
# no edges.


class SandboxUnavailable(WaterwhiteError):
    """The sandbox cannot be built on this machine: the bubblewrap command is not
    on PATH, or it cannot create the namespaces (user namespaces are switched off
    for unprivileged users, or the command is too old for an option the harness
    needs). The job was not started. It is never run unconfined instead: a log
    written by an unconfined job would claim a bound that has no basis."""

    name = "sandbox unavailable"


class ConfinedJobFailed(WaterwhiteError):
    """The confined job did not end as a job must: it exited with a status other
    than zero, was stopped by a signal, ran past its time limit, or closed its
    pipe without the end message. No output of the run is returned."""

    name = "confined job failed"


class JobProtocolViolation(WaterwhiteError):
    """The confined job wrote something on its request pipe that is not a request
    of the protocol: a line that is not a JSON object, an unknown request, a
    request with a missing or mistyped field, a line longer than the limit, or
    anything after the end message. The job is stopped and no output is returned."""

    name = "job protocol violation"


class ReplayNotPossible(WaterwhiteError):
    """The index arrays cannot be derived again: the run record lacks a setting
    the replay needs, or its settings describe an index the trainer could not
    have built. The detail says which. Like "usage" it belongs to no family: a
    function was asked for something it cannot give. The join never lets it out
    of ``compute_join``; it turns it into the cause of plan 3.5."""

    name = "replay not possible"


class TokenizerFailed(WaterwhiteError):
    """The team's tokenizer, run for a map recovery as an external command or
    loaded as a Python entry point, could not be started or loaded, ended early,
    ran past its time limit for one text, or answered with something that is not
    a list of token ids. No map is derived from such a run (plan 3.3
    "Tokenization": the tool ships no tokenizer and imports none)."""

    name = "tokenizer failed"
