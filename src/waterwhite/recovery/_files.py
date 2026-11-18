"""Reading the file of a registered JSONL artifact, as a stream.

A recovery reads record texts only to compare them (plan 3.1): it keeps no text.
A file is read in one of two ways:

- ``stream``: every line in order, once. Used for the later file of a step.
- ``raw_line``: one line by its number. The first call reads the file once, takes
  its digest and notes where each line starts (one whole number per line); a line
  is then read by a seek. Used for the candidate files, whose lines are asked for
  in the later file's order.

Either way the bytes read are held against the digest the artifact is registered
with, so a comparison is never made against a file that changed after it was
registered.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator
from pathlib import Path

from .. import capture, ids
from .. import vocabulary as v
from ..errors import ArtifactDigestMismatch, InvalidRecord, MapCountMismatch, WriteRefused
from ..store.views import Artifact


def shown_name(artifact: Artifact) -> str:
    """A name of the artifact for a message, or its instance id when it has none."""
    return artifact.names[0] if artifact.names else artifact.instance_id


def resolve_jsonl(store, ref: str, what: str) -> Artifact:
    """The registered JSONL artifact a caller names by a name or an instance id."""
    if not isinstance(ref, str) or not ref:
        raise InvalidRecord(f"{what} is named by a registered name or an instance id")
    artifact = store.artifact(ref)
    if artifact is None:
        raise InvalidRecord(
            f"{what} {ref!r} is not registered; a recovery links registered files only",
            artifact=ref,
        )
    if artifact.kind != v.KIND_JSONL:
        raise InvalidRecord(
            f"{what} {ref!r} is a {artifact.kind}; a copy step and a sanitizer read and "
            "write JSONL files",
            artifact=ref,
        )
    return artifact


def disk_path(store, artifact: Artifact) -> Path:
    """Where the artifact's file lies, as the store gives it (``Store.disk_path``)."""
    path = store.disk_path(artifact.instance_id)
    if path is None:
        raise WriteRefused(f"the store holds no path for {shown_name(artifact)}")
    return path


class RegisteredFile:
    """One registered JSONL artifact whose file a recovery reads."""

    def __init__(
        self,
        store,
        artifact: Artifact,
        *,
        function: Callable[[str], str],
        text_field: str,
    ) -> None:
        self.artifact = artifact
        self.instance_id = artifact.instance_id
        self.name = shown_name(artifact)
        self.path = disk_path(store, artifact)
        self.was_read = False
        self._function = function
        self._text_field = text_field
        self._offsets: array | None = None
        self._handle = None

    # -- the file ----------------------------------------------------------------------

    def _require_file(self) -> None:
        if not self.path.is_file():
            raise WriteRefused(
                f"the file of {self.name} is not at its recorded place ({self.path}); a "
                "recovery compares record texts and needs the file",
                artifact=self.name,
            )

    def _hold_against_registration(self, digest: str) -> None:
        if digest != self.artifact.content_digest:
            raise ArtifactDigestMismatch(
                f"{self.name}: the file at {self.path} has the digest {digest}, and the "
                f"artifact is registered with {self.artifact.content_digest}; nothing was "
                "recovered from it",
                artifact=self.name,
            )

    def verify_digest(self) -> None:
        """Read the whole file and hold its digest against the registered one."""
        self._require_file()
        self._hold_against_registration(ids.file_digest(self.path))

    def stream(self) -> Iterator[bytes]:
        """Every line of the file, in order, each without its line feed. When the
        last line has been given, the digest of the bytes read is held against the
        registered digest."""
        self._require_file()
        self.was_read = True
        hasher = ids.ContentHasher()
        yield from ids.read_lines(self.path, hasher)
        self._hold_against_registration(hasher.hexdigest())

    def _index(self) -> array:
        """Read the file once: its digest, and the byte at which each line starts."""
        self._require_file()
        self.was_read = True
        hasher = ids.ContentHasher()
        offsets = array("Q")
        position = 0
        for raw in ids.read_lines(self.path, hasher):
            offsets.append(position)
            position += len(raw) + 1
        self._hold_against_registration(hasher.hexdigest())
        self._handle = open(self.path, "rb")
        self._offsets = offsets
        return offsets

    def raw_line(self, number: int) -> bytes:
        """Line ``number`` (counted from 1) without its line feed."""
        offsets = self._offsets if self._offsets is not None else self._index()
        if not 1 <= number <= len(offsets):
            raise MapCountMismatch(
                f"{self.name}: the store holds a record at line {number} and the file has "
                f"{len(offsets)} lines",
                artifact=self.name,
            )
        assert self._handle is not None
        self._handle.seek(offsets[number - 1])
        raw = self._handle.readline()
        return raw[:-1] if raw.endswith(ids.LINE_FEED) else raw

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # -- a record ----------------------------------------------------------------------

    def record(self, raw: bytes, number: int) -> tuple[str, bytes | None]:
        """What capture took of this line when the file was registered: the text
        after the store's normaliser, and its fingerprint. A line that holds no
        text, or none after the normaliser, gives ``("", None)``."""
        text = capture.line_text(raw, self._text_field)  # the one definition
        if text is None:
            return "", None
        normalised = self._function(text)
        if not normalised:
            return "", None
        try:
            return normalised, ids.fingerprint(normalised)
        except UnicodeEncodeError:
            raise InvalidRecord(
                f"{self.name}, line {number}: the text holds a surrogate code point with "
                "no partner and has no fingerprint",
                artifact=self.name,
                line=number,
            ) from None

    def stored_fingerprint_differs(self, number: int) -> InvalidRecord:
        """The error for a line whose text does not give the fingerprint the store
        holds for it. A changed file is named as that first."""
        self.verify_digest()
        return InvalidRecord(
            f"{self.name}, line {number}: the text read does not give the fingerprint the "
            f"store holds for this line; the file was registered with another text field "
            f"than {self._text_field!r}, or after another normaliser function",
            artifact=self.name,
            line=number,
        )
