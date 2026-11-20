"""The command line (docs/interfaces.md section 7, docs/cli.md).

``waterwhite <command> --store <file> ...``. A command prints one JSON document on
standard output, or with ``--format text`` a few lines for a person. An error
prints ``{"error": name, "detail": ..., "context": ...}`` on standard error, exits
with the code of the error's family, and prints nothing on standard output:

    0 answered   1 usage   2 request refused   3 loud integrity fault   4 write refused

and, only for ``verify --require-held`` (docs/cli.md "Gating on the verification"):

    5 made and printed, its conditions not held

Nothing is printed before a command has finished: the output is built whole and
written once. A query runs the store's loud checks first (over the ancestor closure
of the artifact it asks about, over the whole store for a listing), so a loud fault
prints no answer (plan 2.2 closing paragraph, 3.4).

``recover`` is the one command that writes after ``init``: it registers files
found afterwards and derives the links of steps nobody observed (plan 3.3, 5.4).
With ``--dry-run`` it opens the store read-only and prints what it found.

Plan sections: 2.1 (answers and listings), 2.3 (requests and refusals), 3.3 and
5.4 (recovery), 3.4.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from . import capture, coverage, extensions, join, normalise, rebuild, recovery, requests, tokenmap
from . import listing as listings
from . import vocabulary as v
from .errors import (EXIT_OK, InvalidRecord, Refused, StoreWrittenDuringQuery, UnsupportedSelector,
                     UsageError, WaterwhiteError)
from .records import OccurrenceRef
from .shipped_joins import SHIPPED_JOINS
from .store import artifact_files
from .store.store import Store

FORMATS = ("json", "text")

# The exit code of `verify --require-held` for a verification that was made, printed
# and does not hold (docs/cli.md "Gating on the verification"). No command returns it
# otherwise, and the default of `verify` stays 0 for every verification that is made.
EXIT_NOT_HELD = 5


class _Parser(argparse.ArgumentParser):
    """An argument parser that raises instead of printing and leaving with its own
    exit code, so that every error takes the one way out."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message, usage=self.format_usage().strip())


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument("--store", default=argparse.SUPPRESS, metavar="FILE",
                        help="the evidence store file")
    common.add_argument("--format", choices=FORMATS, default=argparse.SUPPRESS,
                        help="json (default) or text for a person")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = _Parser(prog="waterwhite", allow_abbrev=False, parents=[common],
                     description="Trace training data into mixtures and checkpoints.")
    commands = parser.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)

    def add(to, name: str, run: Callable, text: str) -> argparse.ArgumentParser:
        made = to.add_parser(name, parents=[common], allow_abbrev=False, help=text, description=text)
        made.set_defaults(run=run)
        return made

    init = add(commands, "init", _init, "create a store")
    init.add_argument("--normaliser", required=True, metavar="NAME/VERSION",
                      help="the normalisation fingerprints are taken after, for example control-chars/1")
    init.add_argument("--verified-join", action="append", default=[], metavar="TAG,TYPE,CLASS",
                      help="name one join of the list the tool ships, as a Megatron release tag, "
                           "sampler type and sampler class; may be given more than once; given, "
                           "the store lists only the joins named; a join outside the shipped "
                           "list is refused")

    query = commands.add_parser("query", parents=[common], allow_abbrev=False,
                                help="put a request to one artifact or to all",
                                description="put a request to one artifact or to all")
    kinds = query.add_subparsers(dest="request", metavar="<request>", parser_class=_Parser)
    withdraw = add(kinds, "withdraw", _query_withdraw, "withdraw a source")
    withdraw.add_argument("source", help="the source's name")
    withdraw.add_argument("--member", action="append", default=[], metavar="MEMBER",
                          help="narrow to this member of the source; may be given more than once")
    withdraw.add_argument("--record", action="append", default=[], metavar="ARTIFACT:LOCATOR",
                          help="narrow to this record of the source, for example src/A:L3")
    withdraw.add_argument("--selector", action="append", default=[], metavar="KEY=VALUE",
                          help="a selector on metadata; none is supported, the request is refused")
    ban = add(kinds, "ban", _query_ban, "ban a text across origins (an extension)")
    ban.add_argument("text", help="the text to ban")
    flagged = add(kinds, "flagged", _query_flagged, "start from flagged records (an extension)")
    flagged.add_argument("path", help="a file of flagged records")
    for made in (withdraw, ban, flagged):
        made.add_argument("--artifact", default=None, metavar="NAME",
                          help="answer for this artifact; without it, the listing")

    report = commands.add_parser("report", parents=[common], allow_abbrev=False,
                                 help="print the rebuild report of a withdrawal",
                                 description="print the rebuild report of a withdrawal")
    asked = report.add_subparsers(dest="request", metavar="<request>", parser_class=_Parser)
    report_withdraw = add(asked, "withdraw", _report_withdraw, "withdraw a source")
    report_withdraw.add_argument("source", help="the source's name")
    report_withdraw.add_argument("--member", action="append", default=[], metavar="MEMBER",
                                 help="narrow to this member of the source; may be given more "
                                      "than once")
    report_withdraw.add_argument("--record", action="append", default=[], metavar="ARTIFACT:LOCATOR",
                                 help="narrow to this record of the source, for example src/A:L3")
    report_withdraw.add_argument("--selector", action="append", default=[], metavar="KEY=VALUE",
                                 help="a selector on metadata; none is supported, the request is "
                                      "refused")
    report_withdraw.add_argument("--layout", action="append", default=[], metavar="NAME=PREFIX",
                                 help="name a layout: the shards whose store name begins with "
                                      "PREFIX; may be given more than once")
    report_withdraw.add_argument("--placement", action="append", default=[], metavar="NAME=FILE",
                                 help="state where the layout NAME places the records that "
                                      "return to a release: a JSON file that names, for each "
                                      "source, the shard its records land in; one per layout")
    report_ban = add(asked, "ban", _report_ban, "ban a text across origins (an extension)")
    report_ban.add_argument("text", help="the text to ban")
    report_flagged = add(asked, "flagged", _report_flagged,
                         "start from flagged records (an extension)")
    report_flagged.add_argument("path", help="a file of flagged records")

    register = add(commands, "register-run", _register_run,
                   "register one run segment, with its steps, checkpoints and step traces")
    register.add_argument("--run-record", required=True, metavar="PATH",
                          help="the run record the training wrapper wrote for the segment")
    register.add_argument("--steps", default=None, metavar="PATH",
                          help="the operator's step record: batch sizes, tokens read of a step's "
                               "last sample, and the checkpoints")
    register.add_argument("--cache", default=None, metavar="DIR",
                          help="the data-order cache folder the run kept; without it the index "
                               "arrays are replayed from the run record")
    register.add_argument("--mixture", default=None, metavar="NAME",
                          help="the mixture the segment trained on, by the name it is registered under")
    register.add_argument("--parent-checkpoint", default=None, metavar="NAME",
                          help="the checkpoint a resume or warm start goes on from, by name; "
                               "defaults to the run record's own parent_checkpoint")

    replay = add(commands, "replay", _replay,
                 "derive a segment's index arrays again from its run record and write them to a folder")
    replay.add_argument("--run-record", required=True, metavar="PATH",
                        help="the run record the training wrapper wrote for the segment")
    replay.add_argument("--out", required=True, metavar="DIR",
                        help="the folder the arrays are written to, one .npy file each")

    show = add(commands, "show", _show, "what the store holds about an artifact")
    show.add_argument("artifact", help="a name, an instance id, a content digest or a path")
    show.add_argument("--record", default=None, metavar="LOCATOR",
                      help="one record of it: L<n> a line of a JSONL file, D<n> a document of a shard")
    show.add_argument("--map", default=None, metavar="NAMESPACE/DERIVATION",
                      help="the rows of this map of a shard, for example recovery/position; where "
                           "the namespace and derivation hold a map that did not verify beside a "
                           "verified one, the verified one is printed and the other is offered")
    show.add_argument("--verification", default=None, choices=v.VERIFICATIONS,
                      help="with --map: the map whose verification gave this result")
    show.add_argument("--trace", action="store_true",
                      help="the artifact is a registered run segment: print the rows its steps read, "
                           "in reading order, and its untraced steps")
    show.add_argument("--step", type=int, default=None, metavar="N",
                      help="with --trace: keep this step only")

    add(commands, "check", _check, "run the loud checks over the whole store")
    add(commands, "migrate", _migrate, "bring a store made by an earlier build to this schema version")

    compare = add(commands, "compare-hand-off", _compare_hand_off,
                  "show a stored job the texts it was handed and mark its harness log compared")
    compare.add_argument("--job", required=True, metavar="NAME", help="the name of the job")
    compare.add_argument("--texts", required=True, metavar="FILE",
                         help='a JSON file {"texts": [{"record": "ARTIFACT:L<n>", "text": "..."}]}: '
                              "one text for every record handed to the job, as the job was handed it")

    verify = add(commands, "verify", _verify, "verify a repair against the report it was driven by")
    verify.add_argument("--report", required=True, metavar="FILE",
                        help="the report the repair was driven by, read as a file")
    verify.add_argument("--sides", default=None, metavar="FILE",
                        help="the sides file: the repaired side's and the reference side's file "
                             "for each artifact; without it the sides are not compared")
    verify.add_argument("--mode", choices=v.BUILD_MODES, default=v.BUILD_UNSEEDED,
                        help="seeded when every step of the build was seeded (the shard digests "
                             "are then compared); the default compares the record multiset only")
    verify.add_argument("--text-field", default="text", metavar="FIELD",
                        help="the field of a JSONL line that holds its text (default: text)")
    verify.add_argument("--require-held", action="store_true",
                        help="print the verification as always, and exit with 5 when its "
                             "conditions are not all held (docs/cli.md, Gating on the verification)")

    recover = commands.add_parser(
        "recover", parents=[common], allow_abbrev=False,
        help="register files found afterwards and recover the links of unobserved steps",
        description="register files found afterwards and recover the links of unobserved steps")
    whats = recover.add_subparsers(dest="recovery", metavar="<what>", parser_class=_Parser)

    def dry(made: argparse.ArgumentParser) -> None:
        made.add_argument("--dry-run", action="store_true",
                          help="print the result and write nothing; the store is opened read-only")

    def text_field(made: argparse.ArgumentParser) -> None:
        made.add_argument("--text-field", default="text", metavar="FIELD",
                          help="the field of a line that holds its text (default: text)")

    def step_flags(made: argparse.ArgumentParser, what: str) -> None:
        made.add_argument("--step-version", default=STEP_VERSION_NOT_RECORDED, metavar="VERSION",
                          help=f"the version of {what} (default: {STEP_VERSION_NOT_RECORDED!r})")
        made.add_argument("--step-params", default=None, metavar="JSON",
                          help=f"the parameters of {what}, a JSON value")

    found = add(whats, "file", _recover_file, "register a JSONL file found afterwards")
    found.add_argument("path", help="the file")
    found.add_argument("--name", required=True, metavar="NAME",
                       help="the name it is registered under; also part of its instance id")
    found.add_argument("--role", choices=recovery.FOUND_ROLES, default=v.ROLE_INTERMEDIATE,
                       help="its place in the pipeline (default: intermediate)")
    text_field(found)
    dry(found)

    shard = add(whats, "shard", _recover_shard, "register a tokenized shard found afterwards")
    shard.add_argument("prefix", help="the shard's prefix: the path without .bin and .idx")
    shard.add_argument("--name", required=True, metavar="NAME",
                       help="the name the shard is registered under")
    shard.add_argument("--input", action="append", default=[], metavar="ARTIFACT",
                       help="a registered file the tokenizer read; may be given more than once")
    shard.add_argument("--tokenizer-version", required=True, metavar="VERSION",
                       help="the version of the tokenizer that wrote the shard")
    shard.add_argument("--tokenizer-params", default=None, metavar="JSON",
                       help="the tokenizer's parameters, a JSON value")
    dry(shard)

    copy = add(whats, "copy-step", _recover_copy_step,
               "recover the links of a copy step (shuffle, chunk, merge) for one file it wrote")
    copy.add_argument("later", help="the registered file the step wrote")
    copy.add_argument("--input", action="append", default=[], metavar="ARTIFACT",
                      help="a registered file the step can have read; give every one")
    copy.add_argument("--step-name", required=True, metavar="NAME", help="the step's name")
    step_flags(copy, "the step")
    text_field(copy)
    dry(copy)

    sanitizer = add(whats, "sanitizer", _recover_sanitizer,
                    "recover the links and the drops of the sanitizer for one file it wrote")
    sanitizer.add_argument("later", help="the registered file the sanitizer wrote")
    sanitizer.add_argument("--input", required=True, metavar="ARTIFACT",
                           help="the registered file the sanitizer read")
    sanitizer.add_argument("--log", default=None, metavar="FILE",
                           help="the sanitizer's own log of the lines it skipped")
    sanitizer.add_argument("--log-input", default=None, metavar="VALUE",
                           help="the value the log's rows carry in 'input' for this file")
    sanitizer.add_argument("--step-name", default=v.STEP_SANITIZE, metavar="NAME",
                           help=f"the step's name (default: {v.STEP_SANITIZE})")
    step_flags(sanitizer, "the sanitizer")
    text_field(sanitizer)
    dry(sanitizer)

    mapped = add(whats, "map", _recover_map, "recover the tokenization map of a registered shard")
    mapped.add_argument("shard", help="the registered shard")
    mapped.add_argument("--input", required=True, metavar="ARTIFACT",
                        help="the registered file whose lines became the documents")
    mapped.add_argument("--mode", choices=MAP_MODES, default=MODE_POSITION,
                        help="position (default; token matching takes over when its "
                             "verification fails), token-matching, or both")
    mapped.add_argument("--append-eod", required=True, choices=YES_NO,
                        help="whether the tokenizer run appended the end-of-document token")
    mapped.add_argument("--eod-in-span", required=True, choices=YES_NO,
                        help="whether that token lies inside every document's span")
    mapped.add_argument("--eod-token-id", type=int, default=None, metavar="ID",
                        help="the end-of-document token; needed with --eod-in-span yes")
    mapped.add_argument("--tokenizer-command", default=None, metavar="COMMAND",
                        help="the team's tokenizer as an external command that answers one "
                             "text per line (docs/cli.md)")
    mapped.add_argument("--tokenizer-entry", default=None, metavar="MODULE:FUNCTION",
                        help="the team's tokenizer as a Python function, imported by name")
    mapped.add_argument("--tokenizer-timeout", type=float,
                        default=tokenmap.DEFAULT_TOKENIZER_TIMEOUT, metavar="SECONDS",
                        help="how long the tokenizer command may take for one text")
    mapped.add_argument("--sanitizer-log", default=None, metavar="FILE",
                        help="the sanitizer's log, when --input is the file the sanitizer read")
    mapped.add_argument("--log-shard", default=None, metavar="VALUE",
                        help="read only the log's rows with this value in 'shard'")
    mapped.add_argument("--log-input", default=None, metavar="VALUE",
                        help="read only the log's rows with this value in 'input'")
    mapped.add_argument("--sanitizer", default=None, metavar="NAME/VERSION",
                        help="the sanitizer to run again (default: the store's normaliser)")
    mapped.add_argument("--sanitizer-step", default=v.STEP_SANITIZE, metavar="NAME",
                        help=f"the step the log's drops are recorded for (default: {v.STEP_SANITIZE})")
    text_field(mapped)
    dry(mapped)
    return parser


STEP_VERSION_NOT_RECORDED = "not recorded"
YES_NO = ("yes", "no")
MODE_POSITION, MODE_TOKEN_MATCHING, MODE_BOTH = "position", "token-matching", "both"
MAP_MODES = (MODE_POSITION, MODE_TOKEN_MATCHING, MODE_BOTH)


# --- Commands: each returns (JSON value, text for a person) --------------------------


def _store_path(args: argparse.Namespace) -> str:
    path = getattr(args, "store", None)
    if not path:
        raise UsageError("the command needs --store FILE")
    return path


def _init(args: argparse.Namespace) -> tuple[dict, str]:
    joins = []
    for text in args.verified_join:
        parts = tuple(p.strip() for p in text.split(","))
        if len(parts) != 3 or not all(parts):
            raise UsageError(f"--verified-join takes TAG,TYPE,CLASS; got {text!r}")
        if parts not in SHIPPED_JOINS:  # the whole triple: the tool lists only what Check 1 verified
            raise UsageError(f"--verified-join {text!r} is not a join the tool ships "
                             "(Megatron release tag, sampler type and sampler class as verified)")
        joins.append(parts)
    if not joins:
        joins = list(SHIPPED_JOINS)  # what Check 1 verified (plan 3.5, F6); a flag names a subset
    normalise.get(args.normaliser)  # an unknown normaliser is refused before a file is made
    path = _store_path(args)
    with Store.create(path, normaliser=args.normaliser, verified_joins=joins) as store:
        out = {
            "created": str(store.path),
            "schema_version": store.schema_version,
            "normaliser": store.normaliser,
            "verified_joins": [list(j) for j in store.verified_joins()],
        }
    return out, _plain(out)


def _unchanged_since(store: Store, mark: tuple[int, int]) -> None:
    """The last step of a command that answers from the store: refuse, with the named
    error and before anything is printed, when the store was written since ``mark``,
    the mark taken as the command opened it. It reaches a write that fell between the
    last answer and the output, and one the answers' own checks could not see; a write
    after the output is printed is outside the command (docs/interfaces.md section 2)."""
    if store.state_mark() != mark:
        raise StoreWrittenDuringQuery()


def _answer_or_listing(args: argparse.Namespace, build: Callable) -> tuple[dict, str]:
    with Store.open(_store_path(args), readonly=True) as store:
        mark = store.state_mark()
        request = build(store)  # a refused request ends here, before any answer
        if args.artifact is None:
            made = listings.listing(store, request)
            out, text = made.to_json(), _listing_text(made.to_json())
        else:
            answer = coverage.answer(store, args.artifact, request)
            out, text = answer.to_json(), _answer_text(answer.to_json())
        _unchanged_since(store, mark)
        return out, text


def _query_withdraw(args: argparse.Namespace) -> tuple[dict, str]:
    return _answer_or_listing(args, lambda store: requests.withdraw(
        store, args.source, members=args.member, records=args.record, selectors=args.selector))


def _query_ban(args: argparse.Namespace) -> tuple[dict, str]:
    return _answer_or_listing(args, lambda store: requests.ban(store, args.text))


def _query_flagged(args: argparse.Namespace) -> tuple[dict, str]:
    return _answer_or_listing(args, lambda store: requests.flagged(store, args.path))


def _layouts(given: Sequence[str]) -> dict[str, str]:
    """``--layout NAME=PREFIX`` as a mapping. Both parts are needed, and a name is given
    once."""
    layouts: dict[str, str] = {}
    for text in given:
        name, equals, prefix = text.partition("=")
        if not equals or not name or not prefix:
            raise UsageError(f"--layout takes NAME=PREFIX; got {text!r}")
        if name in layouts:
            raise UsageError(f"--layout names {name!r} twice")
        layouts[name] = prefix
    return layouts


def _placements(given: Sequence[str], layouts: dict[str, str]) -> dict[str, rebuild.placing.Placement]:
    """``--placement NAME=FILE`` as a mapping of layout name to the placement read from the
    file. The layout must be one ``--layout`` names, and a layout is given one file."""
    placements: dict[str, rebuild.placing.Placement] = {}
    for text in given:
        name, equals, path = text.partition("=")
        if not equals or not name or not path:
            raise UsageError(f"--placement takes NAME=FILE; got {text!r}")
        if name not in layouts:
            raise UsageError(f"--placement names the layout {name!r}, which --layout does not name")
        if name in placements:
            raise UsageError(f"--placement names {name!r} twice")
        placements[name] = rebuild.placing.read_placement(name, path)
    return placements


def _report_withdraw(args: argparse.Namespace) -> tuple[dict, str]:
    with Store.open(_store_path(args), readonly=True) as store:
        # A switch that cannot be read is named before a layout is read or a placement file opened
        # (D-CORE-6; the parser has already refused a missing value or an unknown option).
        extensions.check_switches(store.settings())
        layouts = _layouts(args.layout)
        placements = _placements(args.placement, layouts)
        mark = store.state_mark()
        request = requests.withdraw(
            store, args.source, members=args.member, records=args.record, selectors=args.selector)
        document = rebuild.build_report(store, request, layouts=layouts, placements=placements)
        text = _report_text(document)
        _unchanged_since(store, mark)
    return document, text


def _report_ban(args: argparse.Namespace) -> tuple[dict, str]:
    with Store.open(_store_path(args), readonly=True) as store:
        requests.ban(store, args.text)  # refused while the extension is off
    raise Refused("the report of a content ban is not built")


def _report_flagged(args: argparse.Namespace) -> tuple[dict, str]:
    with Store.open(_store_path(args), readonly=True) as store:
        requests.flagged(store, args.path)  # refused while the extension is off
    raise Refused("the report of flagged records is not built")


def _migrate(args: argparse.Namespace) -> tuple[dict, str]:
    found, reached = Store.migrate(_store_path(args))
    out = {"schema_version_found": found, "schema_version": reached, "steps_run": reached - found}
    return out, _plain(out)


def _compare_hand_off(args: argparse.Namespace) -> tuple[dict, str]:
    """Compare the texts a stored job was handed with the records they name and mark its
    log compared (docs/cli.md "compare-hand-off"). The output names the job and the number
    of references and never a text."""
    document = join.read_json(args.texts, "texts file")
    listed = document.get("texts")
    if not isinstance(listed, list) or not all(
            isinstance(e, dict) and isinstance(e.get("record"), str) and isinstance(e.get("text"), str)
            for e in listed):
        raise InvalidRecord('the texts file holds {"texts": [{"record": "ARTIFACT:L<n>", "text": ...}]}')
    with Store.open(_store_path(args)) as store:
        texts = []
        for entry in listed:
            try:
                name, locator = requests.split_record(entry["record"])
            except UnsupportedSelector:
                raise InvalidRecord(
                    f"a record of the texts file is written ARTIFACT:L<n>, for example src/A:L3; "
                    f"got {entry['record']!r}", record=entry["record"]) from None
            held = store.resolve(name)
            if held is None:
                raise InvalidRecord(f"{entry['record']} is not a record of the store", record=entry["record"])
            texts.append((OccurrenceRef(held.instance_id, locator), entry["text"]))
        store.verify_hand_off(args.job, texts)
    out = {"job": args.job, "references": len(texts), "hand_off_checked": True}
    return out, _plain(out)


def _check(args: argparse.Namespace) -> tuple[dict, str]:
    # Opened for writing: the check reads the two files of every shard that is at its place and
    # records what it found (``Store.recheck_shard_files``), before the loud checks run, so that
    # the result stays when a loud check stops the command.
    with Store.open(_store_path(args), migrate_older=False) as store:
        store.recheck_shard_files()
        store.check_integrity(None)
        out = {
            "check": "passed",
            "artifacts": sum(1 for _ in store.artifacts()),
            "missing_files": [
                {"instance": m.instance_id, "kind": m.kind, "path": m.path, "state": m.state}
                for m in store.missing_files()
            ],
        }
    return out, _plain(out)


def _show(args: argparse.Namespace) -> tuple[dict, str]:
    if args.step is not None and not args.trace:
        raise UsageError("--step keeps one step of a segment's trace; give --trace too")
    if args.trace:
        if args.record is not None or args.map is not None or args.verification is not None:
            raise UsageError("--trace prints the rows a segment's steps read; it takes no --record, "
                             "--map or --verification")
        with Store.open(_store_path(args), readonly=True) as store:
            out = describe_trace(store, args.artifact, args.step)
        return out, _plain(out)
    if args.map is None and args.verification is not None:
        raise UsageError("--verification names a result of --map's namespace and derivation; "
                         "give --map too")
    if args.map is not None and args.record is not None:
        raise UsageError("--map prints the rows of a map and --record one record of the map a "
                         "reader gets; give one of them")
    named = None if args.map is None else _map_name(args.map)
    with Store.open(_store_path(args), readonly=True) as store:
        if named is not None:
            out = describe_map(store, args.artifact, *named, verification=args.verification)
        elif args.record is None:
            out = describe(store, args.artifact)
        else:
            out = describe_record(store, args.artifact, args.record)
    return out, _plain(out)


def _map_name(text: str) -> tuple[str, str]:
    parts = text.split("/")
    if len(parts) != 2 or not all(parts):
        raise UsageError(f"--map takes NAMESPACE/DERIVATION, for example capture/position; "
                         f"got {text!r}")
    return parts[0], parts[1]


# --- recover: files found afterwards and the links of unobserved steps -----------------


def _json_flag(text: str | None, flag: str) -> object:
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError as error:
        raise UsageError(f"{flag} takes a JSON value: {error}") from None


def _recover_file(args: argparse.Namespace) -> tuple[dict, str]:
    with Store.open(_store_path(args), readonly=args.dry_run) as store:
        held = store.artifact(args.name)
        out: dict = {"artifact": args.name, "role": args.role, "already_registered": held is not None}
        if args.dry_run:
            scan = capture.scan_jsonl(args.path, normaliser=store.normaliser,
                                      text_field=args.text_field)
            out |= {"instance": None, "records": scan.count, "content_digest": scan.digest}
        else:
            instance = recovery.register_found_file(
                store, args.path, name=args.name, role=args.role, text_field=args.text_field)
            artifact = store.artifact(instance)
            out |= {"instance": instance, "role": artifact.role,
                    "records": store.occurrence_count(instance),
                    "content_digest": artifact.content_digest}
        out["dry_run"] = args.dry_run
    return out, _plain(out)


def _recover_shard(args: argparse.Namespace) -> tuple[dict, str]:
    step = tokenmap.tokenize_step(
        args.tokenizer_version, _json_flag(args.tokenizer_params, "--tokenizer-params"))
    with Store.open(_store_path(args), readonly=args.dry_run) as store:
        inputs = []
        for ref in args.input:
            artifact = store.artifact(ref)
            if artifact is None:
                raise Refused(f"the input {ref!r} is not registered; register it first "
                              "(recover file)", artifact=ref)
            inputs.append(artifact.instance_id)
        held = store.artifact(args.name)
        out: dict = {"artifact": args.name, "already_registered": held is not None}
        if args.dry_run:
            on_disk = artifact_files.read_shard(args.prefix)
            out |= {"instance": None, "documents": on_disk.document_count,
                    "content_digest": on_disk.digest}
        else:
            instance = store.register_shard(args.prefix, name=args.name, step=step, inputs=inputs)
            artifact = store.artifact(instance)
            out |= {"instance": instance,
                    "documents": artifact_files.read_shard(args.prefix).document_count,
                    "content_digest": artifact.content_digest}
        out |= {"inputs": [_name(store, i) for i in inputs],
                "has_map": held is not None and store.binding(held.instance_id) is not None,
                "dry_run": args.dry_run}
    return out, _plain(out)


def _recover_copy_step(args: argparse.Namespace) -> tuple[dict, str]:
    if not args.input:
        raise UsageError("recover copy-step needs --input ARTIFACT, once for every file the "
                         "step can have read")
    step = capture.step_ref(args.step_name, args.step_version,
                            _json_flag(args.step_params, "--step-params"))
    with Store.open(_store_path(args), readonly=args.dry_run) as store:
        result = recovery.recover_copy_step(
            store, args.later, args.input, step=step, text_field=args.text_field,
            write=not args.dry_run)
    out = result.to_json()
    return out, _plain(out)


def _recover_sanitizer(args: argparse.Namespace) -> tuple[dict, str]:
    if args.log is None and args.log_input is not None:
        raise UsageError("--log-input names a file of a log; give the log with --log FILE")
    step = capture.step_ref(args.step_name, args.step_version,
                            _json_flag(args.step_params, "--step-params"))
    with Store.open(_store_path(args), readonly=args.dry_run) as store:
        result = recovery.recover_sanitizer(
            store, args.later, args.input, step=step, log_path=args.log,
            log_input=args.log_input, text_field=args.text_field, write=not args.dry_run)
    out = result.to_json()
    return out, _plain(out)


def _recover_map(args: argparse.Namespace) -> tuple[dict, str]:
    if args.tokenizer_command is not None and args.tokenizer_entry is not None:
        raise UsageError("give the tokenizer once: --tokenizer-command or --tokenizer-entry")
    has_tokenizer = args.tokenizer_command is not None or args.tokenizer_entry is not None
    if args.mode != MODE_POSITION and not has_tokenizer:
        raise UsageError(f"--mode {args.mode} tokenizes every line again and needs "
                         "--tokenizer-command or --tokenizer-entry")
    if args.sanitizer_log is None and (
            args.log_shard is not None or args.log_input is not None or args.sanitizer is not None):
        raise UsageError("--log-shard, --log-input and --sanitizer belong to a sanitizer's log; "
                         "give it with --sanitizer-log FILE")
    settings = tokenmap.TokenizerSettings(
        append_eod=args.append_eod == "yes", eod_in_span=args.eod_in_span == "yes",
        eod_token_id=args.eod_token_id)
    sanitizer = None
    if args.sanitizer_log is not None:
        skips = tokenmap.read_sanitizer_log(
            args.sanitizer_log, shard=args.log_shard, input=args.log_input)
        sanitizer = tokenmap.SanitizerRun(skips, normaliser=args.sanitizer,
                                          step_name=args.sanitizer_step)
    command = None
    with contextlib.ExitStack() as stack:
        store = stack.enter_context(Store.open(_store_path(args), readonly=args.dry_run))
        tokenize = None
        if args.tokenizer_entry is not None:
            tokenize = tokenmap.entry_point_tokenizer(args.tokenizer_entry)
        elif args.tokenizer_command is not None:
            command = stack.enter_context(tokenmap.command_tokenizer(
                args.tokenizer_command, timeout_seconds=args.tokenizer_timeout))
            tokenize = command
        result = tokenmap.recover_map(
            store, shard=args.shard, input=args.input, settings=settings, tokenize=tokenize,
            sanitizer=sanitizer, f1_holds=args.mode != MODE_TOKEN_MATCHING,
            also_token_matching=args.mode == MODE_BOTH, text_field=args.text_field,
            write=not args.dry_run)
        names = {"artifact": _name(store, result.shard_instance_id),
                 "input_artifact": _name(store, result.input_instance_id)}
    out = names | {"mode": args.mode} | result.to_json() | {"dry_run": args.dry_run}
    if command is not None and command.exit_status != 0:
        ended = ("had to be stopped" if command.exit_status is None
                 else f"ended with exit status {command.exit_status}")
        out["notes"] = out["notes"] + [
            f"the tokenizer command {ended} after its last reply; every reply was given"]
    return out, _plain(out)


# --- register-run: one segment of a training run, with its step traces ---------------


def _registered(store, name: str, kind: str, what: str) -> str:
    """The instance id of a registered artifact named ``name``, which must be of
    ``kind``. A name the store does not hold cannot be turned into an instance id,
    and a segment that names it by value would be left with a dangling reference,
    so the registration is refused."""
    held = store.resolve(name)
    if held is None or held.kind != kind:
        raise InvalidRecord(
            f"{what} {name!r} is not a registered {kind}; register it first", name=name)
    return held.instance_id


def _malformed(error: InvalidRecord) -> InvalidRecord:
    """A run record or step record the command was handed that is not one. It is
    named "invalid record" and exits with 4, like every refusal of that name: one
    name, one exit code (docs/interfaces.md section 2)."""
    return error


def _register_run(args: argparse.Namespace) -> tuple[dict, str]:
    try:
        document = join.read_json(args.run_record, "run record")
        names = {}
        for key in ("run", "segment", "start_kind"):
            value = document.get(key)
            if not isinstance(value, str) or not value:
                raise InvalidRecord(f"the run record names no {key.replace('_', ' ')}", key=key)
            names[key] = value
        named_parent = document.get("parent_checkpoint")
        if named_parent is not None and not isinstance(named_parent, str):
            raise InvalidRecord("the run record's parent_checkpoint is a name or null")
        step_record = None if args.steps is None else join.read_json(args.steps, "step record")
    except InvalidRecord as error:
        raise _malformed(error) from None
    if args.parent_checkpoint is not None and named_parent not in (None, args.parent_checkpoint):
        raise InvalidRecord(
            f"--parent-checkpoint {args.parent_checkpoint!r} and the run record's parent_checkpoint "
            f"{named_parent!r} name different checkpoints")
    parent_name = args.parent_checkpoint if args.parent_checkpoint is not None else named_parent
    with Store.open(_store_path(args)) as store:
        mixture = (None if args.mixture is None
                   else _registered(store, args.mixture, v.KIND_MIXTURE, "the mixture"))
        parent = (None if parent_name is None
                  else _registered(store, parent_name, v.KIND_CHECKPOINT, "the parent checkpoint"))
        held = store.segment(names["segment"])
        if held is not None:
            position = held.position
        else:
            taken = [store.segment(name).position for name in store.segments(names["run"])]
            position = max(taken, default=-1) + 1
        try:
            result = join.compute_join(
                document, verified_joins=store.verified_joins(),
                record_dir=Path(args.run_record).parent, cache_dir=args.cache,
                step_record=step_record)
        except InvalidRecord as error:
            raise _malformed(error) from None
        join.register_segment(
            store,
            join.SegmentRegistration(
                run=names["run"], segment=names["segment"], position=position,
                start_kind=names["start_kind"], mixture_instance_id=mixture,
                parent_checkpoint=parent),
            document, result)
    out = {
        "segment": names["segment"],
        "route": result.route,
        "cause": result.cause,
        "cause_detail": result.cause_detail,
        "steps": len(result.steps),
        "traced_steps": len(result.traces),
        "untraced_steps": [[names["segment"], step] for step in result.untraced_steps],
        "checkpoints": [name for name, _ in result.checkpoints],
        "replay": {
            "attempted": result.replay.attempted,
            "accepted": result.replay.accepted if result.replay.attempted else None,
            "reason": result.replay.reason,
        },
        "shuffle": [{
            "launch": check.launch,
            "epoch": check.epoch,
            "rank": check.rank,
            "digest": check.digest,
            "count": check.count,
            "recorded_digest": check.recorded_digest,
            "recorded_count": check.recorded_count,
            "equal": check.equal,
        } for check in result.shuffle],
    }
    return out, _plain(out)


def _replay(args: argparse.Namespace) -> tuple[dict, str]:
    """Derive a segment's index arrays again and write them (docs/cli.md "replay").
    Reads the run record and the shards' index files and writes the arrays to a
    folder; it opens no store."""
    document = join.read_json(args.run_record, "run record")
    out = join.write_replay(document, Path(args.run_record).parent, args.out)
    return out, _plain(out)


# --- show: what the store holds --------------------------------------------------------


def describe_trace(store, name: str, step: int | None = None) -> dict:
    """The rows the join wrote for a registered segment, in reading order (plan 3.5;
    docs/cli.md "show"). ``step`` keeps one step. A segment with the route "none" has
    no rows. ``untraced_steps`` are the steps the proof reads as untraced (plan
    2.2(e)), whatever the route."""
    held = store.segment(name)
    if held is None:
        raise Refused(f"no run segment {name!r} is registered; show it without --trace to see "
                      "what the store holds under that name")
    shards: dict[int, str] = {}
    if held.mixture_instance_id is not None:
        for entry in store.mixture_shards(held.mixture_instance_id):
            registered = store.artifact(entry.shard_instance_id)
            shards[entry.position] = (registered.names[0] if registered is not None and registered.names
                                      else entry.path)
    steps = []
    epochs: dict[int, list[int] | None] = {}
    if held.trace_route != v.ROUTE_NONE:
        if held.sampler_type == v.SAMPLER_CYCLIC and held.run_record is not None:
            epochs = join.step_epochs(
                join.parse_run_record(held.run_record),
                {registered.step: registered.batch_size for registered in held.steps},
                None if step is None else (step,))
        wanted = store.step_traces(name) if step is None else store.step_traces(
            name, first_step=step, last_step=step)
        for trace in wanted:
            steps.append({
                "step": trace.step,
                "consumed_samples": trace.consumed_samples,
                **({"epochs": epochs.get(trace.step)} if held.sampler_type == v.SAMPLER_CYCLIC else {}),
                "samples": [{
                    "sample_pos": sample.sample_pos,
                    "global_index": sample.global_index,
                    "rank": sample.rank,
                    "sample_tokens": sample.sample_tokens,
                    "covered_tokens": sample.covered_tokens,
                    "rows": [{
                        "shard": shards.get(row.shard_position),
                        "shard_position": row.shard_position,
                        "ordinal": row.ordinal,
                        "token_start": row.token_start,
                        "token_end": row.token_end,
                    } for row in sample.rows],
                } for sample in trace.samples],
            })
    return {
        "segment": name,
        "route": held.trace_route,
        "cause": held.trace_cause,
        "cause_detail": held.trace_cause_detail,
        "launches": [{
            "position": launch.position,
            "world_size": launch.world_size,
            "data_parallel_size": launch.data_parallel_size,
            "consumed_at_start": launch.consumed_at_start,
            "step_at_start": launch.step_at_start,
            "first_step": launch.first_step,
            "last_step": launch.last_step,
        } for launch in held.launches],
        "steps": steps,
        "untraced_steps": [[name, number] for number, _ in coverage.untraced_steps(store, name)
                           if step is None or number == step],
    }


def _reference(store, reference) -> dict:
    held = store.artifact(reference.referrer)
    return {
        "relation": reference.relation,
        "referrer": reference.referrer if held is None or not held.names else held.names[0],
        "position": reference.position,
    }


def describe(store, ref: str) -> dict:
    """What the store holds about one artifact. For a name the store has no row
    for, whether another record names it (plan 2.1)."""
    artifact = store.resolve(ref)
    if artifact is None:
        named = [r for r in store.references_to(ref) if not r.registered]
        return {
            "artifact": ref,
            "registered": False,
            "reason": v.REASON_RECORD_MISSING if named else v.REASON_NOT_REGISTERED,
            "named_by": [_reference(store, r) for r in named],
        }
    instance = artifact.instance_id
    bindings = [
        {
            "namespace": b.namespace,
            "declared_count": b.declared_count,
            "declared_digest": b.declared_digest,
            "derivation": b.derivation,
            "verification": b.verification,
            "label": b.label,
            "append_eod": b.append_eod,
            "eod_in_span": b.eod_in_span,
        }
        for b in store.bindings(instance)
    ]
    out: dict = {
        "artifact": ref if ref in artifact.names else (artifact.names[0] if artifact.names else instance),
        "registered": True,
        "instance": instance,
        "kind": artifact.kind,
        "role": artifact.role,
        "names": list(artifact.names),
        "content_digest": artifact.content_digest,
        "edge_digest": artifact.edge_digest,
        "path": artifact.path,
        "file": store.file_state(instance),
        "step": None if artifact.step is None else {
            "name": artifact.step.name, "version": artifact.step.version,
            "param_digest": artifact.step.param_digest,
        },
        "member": artifact.member_name,
        "inputs": [_input(store, i) for i in artifact.inputs],
        "bindings": bindings,
        "drops": [_drop_of_output(store, d) for d in store.drops_by_output(instance)],
        "named_by": [_reference(store, r) for r in store.references_to(instance)],
    }
    if artifact.kind in (v.KIND_JSONL, v.KIND_SHARD):
        out["own_record"] = bool(bindings)
        out["stored_rows"] = store.stored_row_count(instance)
    if artifact.kind == v.KIND_JSONL:
        out["source_links"] = [_link(l) for l in store.file_source_links(instance)]
    elif artifact.kind == v.KIND_SHARD:
        out["mixtures"] = [_name(store, m) for m in store.mixtures_holding(instance)]
    elif artifact.kind == v.KIND_MIXTURE:
        out["shards"] = [
            {
                "position": s.position, "instance": s.shard_instance_id,
                "name": _name(store, s.shard_instance_id), "path": s.path,
                "content_digest": s.content_digest,
                "registered": store.artifact(s.shard_instance_id) is not None,
                "has_map": store.binding(s.shard_instance_id) is not None,
            }
            for s in store.mixture_shards(instance)
        ]
        out["segments"] = list(store.segments_using(instance))
    elif artifact.kind == v.KIND_CHECKPOINT:
        mark = store.checkpoint(instance)
        history = store.checkpoint_history(instance)
        out["segment"] = None if mark is None else mark.segment
        out["at_step"] = None if mark is None else mark.step
        out["own_record"] = mark is not None and store.segment(mark.segment) is not None
        out["history"] = None if history is None else {
            "complete": history.complete,
            "missing": None if history.missing is None else _reference(store, history.missing)
            | {"referred": history.missing.referred},
            "segments": [
                {"segment": p.segment, "upto_step": p.upto_step, "steps": len(p.steps),
                 "first_step": p.steps[0] if p.steps else None,
                 "last_step": p.steps[-1] if p.steps else None}
                for p in history.segments
            ],
        }
        out["started_segments"] = list(store.segments_started_from(instance))
    return out


def _drop_of_output(store, drop) -> dict:
    """A drop whose output is the artifact shown: the artifact and record name the
    line that was dropped."""
    return {"artifact": _name(store, drop.input.instance_id), "instance": drop.input.instance_id,
            "record": drop.input.locator, "reason": drop.reason, "step": drop.step_name}


def _name(store, instance_id: str) -> str:
    held = store.artifact(instance_id)
    return instance_id if held is None or not held.names else held.names[0]


def _input(store, instance_id: str) -> dict:
    held = store.artifact(instance_id)
    return {"instance": instance_id, "name": _name(store, instance_id), "registered": held is not None}


def _link(link) -> dict:
    return {"record": link.locator, "member": link.member, "source": link.source,
            "label": link.label, "evidence": link.evidence}


def _record(store, ref: OccurrenceRef) -> dict:
    return {"artifact": _name(store, ref.instance_id), "instance": ref.instance_id,
            "record": ref.locator,
            "members": [m.name for m in store.members_covering(ref)]}


def _candidate_sources(store, refs) -> list[str]:
    return sorted({m.source for r in refs for m in store.members_covering(r)})


def describe_record(store, ref: str, locator: str) -> dict:
    """What the store holds about one record: a line of a JSONL file with its edge
    and links, or a document of a shard with its map row."""
    artifact = store.resolve(ref)
    if artifact is None:
        raise Refused(f"no artifact {ref!r} is registered; show it without --record to see "
                      "whether a record names it")
    instance = artifact.instance_id
    out: dict = {"artifact": _name(store, instance), "instance": instance, "kind": artifact.kind,
                 "record": locator}
    if artifact.kind == v.KIND_JSONL and locator.startswith("L"):
        target = OccurrenceRef(instance, locator)
        occurrence = store.occurrence(target)
        if occurrence is None:
            raise Refused(f"{ref}:{locator} is not a record of the store")
        edge = store.edge_of(target)
        out["fingerprint"] = None if occurrence.fingerprint is None else occurrence.fingerprint.hex()
        out["members"] = [m.name for m in store.members_covering(target)]
        out["edge"] = None
        if edge is not None:
            log = edge.harness_log
            out["edge"] = {
                "transform": edge.transform.name,
                "parent_mode": edge.parent_mode,
                "derivation": edge.derivation,
                "verification": edge.verification,
                "label": edge.label,
                "parents": [_record(store, p) for p in edge.parents],
                "selected_parent": None if edge.selected is None else _record(store, edge.selected),
                "in_scope_only": [_record(store, p) for p in edge.in_scope_only],
                "candidate_sources": _candidate_sources(store, edge.parents),
                "is_generation": edge.is_generation,
                "job": edge.job_name,
                "model": None if log is None else log.model,
                "harness": None if log is None else log.harness,
                "harness_log": log is not None,
                "bound_field": edge.bound_field,
                "bound_from_log": None if log is None else log.bound_established(),
            }
        out["source_links"] = [_link(l) for l in store.source_links(target)]
        out["drops"] = [
            {"reason": d.reason, "step": d.step_name,
             "output": None if d.output_instance_id is None else _name(store, d.output_instance_id)}
            for d in store.drops_of(target)
        ]
        return out
    if artifact.kind == v.KIND_SHARD and locator.startswith("D") and locator[1:].isdigit():
        binding = store.binding(instance)
        row = store.map_row(instance, int(locator[1:]))
        if binding is None or row is None:
            raise Refused(f"{ref}:{locator}: the shard's map holds no such document"
                          if binding is not None else f"{ref} has no tokenization map")
        out["map"] = {"namespace": binding.namespace, "derivation": binding.derivation,
                      "verification": binding.verification, "label": binding.label}
        out["label"] = row.label
        out["token_start"], out["token_end"] = row.token_start, row.token_end
        out["parent"] = None if row.parent is None else _record(store, row.parent)
        out["candidates"] = [_record(store, c) for c in row.candidates]
        out["candidate_sources"] = _candidate_sources(store, row.candidates)
        return out
    raise Refused(f"{locator!r} names no record of a {artifact.kind}: a line of a JSONL file is "
                  "L<n>, a document of a shard is D<n>")


def _map_words(binding) -> str:
    return f"{binding.namespace}/{binding.derivation}"


def _map_head(binding) -> dict:
    return {
        "namespace": binding.namespace,
        "derivation": binding.derivation,
        "verification": binding.verification,
        "label": binding.label,
        "declared_count": binding.declared_count,
        "append_eod": binding.append_eod,
        "eod_in_span": binding.eod_in_span,
    }


def _reachable(binding, slot: list) -> bool:
    """Whether the flags can name this map: ``--verification`` reaches the first,
    in the reader's order, of the maps of the slot with that result."""
    return next(b for b in slot if b.verification == binding.verification) is binding


def _show_with(binding) -> str:
    result = binding.verification
    quoted = f'"{result}"' if " " in result else result
    return f"--map {_map_words(binding)} --verification {quoted}"


def describe_map(store, ref: str, namespace: str, derivation: str, *,
                 verification: str | None = None) -> dict:
    """The rows of one named map of a shard (plan 3.3, 5.5 Checks 3 and 6). The
    namespace and derivation are named, so a recovery map beside a capture map is
    read, which a reader's own choice would never give. Where they hold a verified
    map beside one that did not verify, the verified one is printed (the store's
    rule, DECISIONS.md D-28) unless ``verification`` names the other; the map
    printed says its result, and ``other_maps`` offers each other map of the same
    namespace and derivation with the flag that reads it."""
    artifact = store.resolve(ref)
    if artifact is None:
        raise Refused(f"no artifact {ref!r} is registered; show it without --map to see whether "
                      "a record names it")
    instance = artifact.instance_id
    if artifact.kind != v.KIND_SHARD:
        raise Refused(f"{ref} is a {artifact.kind}, and only a shard holds a tokenization map",
                      artifact=ref, kind=artifact.kind)
    held = store.bindings(instance)
    slot = [b for b in held if b.namespace == namespace and b.derivation == derivation]
    chosen = [b for b in slot if verification is None or b.verification == verification]
    if not chosen:
        raise Refused(
            f"{ref} holds no map {namespace}/{derivation}"
            + ("" if verification is None else f" whose verification is {verification!r}")
            + ("; it holds no tokenization map at all" if not held else ""),
            artifact=ref, map=f"{namespace}/{derivation}", verification=verification,
            held=[f"{_map_words(b)} ({b.verification})" for b in held])
    binding = chosen[0]  # the first in the reader's order, as store.binding gives it
    others = [b for b in slot if b is not binding]
    seen: dict[OccurrenceRef, dict] = {}  # twins share their candidates: read each record once

    def record(ref: OccurrenceRef) -> dict:
        if ref not in seen:
            seen[ref] = _record(store, ref)
        return dict(seen[ref], members=list(seen[ref]["members"]))

    rows = [
        {
            "record": f"D{row.ordinal}",
            "label": row.label,
            "token_start": row.token_start,
            "token_end": row.token_end,
            "parent": None if row.parent is None else record(row.parent),
            "candidates": [record(c) for c in row.candidates],
        }
        for row in store.map_rows(instance, namespace, derivation=derivation,
                                  verification=binding.verification)
    ]
    out: dict = {"artifact": _name(store, instance), "instance": instance, "kind": artifact.kind,
                 "map": _map_head(binding), "rows": rows}
    if others:
        out["other_maps"] = [
            {"namespace": b.namespace, "derivation": b.derivation,
             "verification": b.verification, "label": b.label,
             "declared_count": b.declared_count,
             "show_with": _show_with(b) if _reachable(b, slot) else None}
            for b in others
        ]
    return out


# --- Text for a person -------------------------------------------------------------------


def _plain(value: object, indent: int = 0) -> str:
    """Any JSON value as indented lines."""
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:")
                lines.append(_plain(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                body = _plain(item, indent + 1).lstrip()
                lines.append(f"{pad}- {body}")
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_scalar(value)}"


def _scalar(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return "none"
    return str(value)


def _verify(args: argparse.Namespace) -> tuple[dict, str]:
    from .rebuild import verify_sides

    with Store.open(_store_path(args), readonly=True) as store:
        extensions.check_switches(store.settings())  # before either document is read (D-CORE-6)
        report = verify_sides.read_document(args.report, v.FORM_REPORT, "report")
        sides = None if args.sides is None else verify_sides.read_document(
            args.sides, v.FORM_SIDES, "sides file")
        mark = store.state_mark()
        made = verify_sides.verify(
            store, report, sides=sides, base=None if args.sides is None else Path(args.sides).parent,
            mode=args.mode, text_field=args.text_field)
        text = _verification_text(made)
        _unchanged_since(store, mark)
    if args.require_held:
        args.unmet = _unmet_conditions(made, report, sides_given=args.sides is not None)
    return made, text


def _unmet_conditions(made: dict, report: dict, *, sides_given: bool) -> list[str]:
    """The conditions of docs/cli.md "Gating on the verification" that the verification
    does not hold, each named as that list names it, in the order of that list. Nothing
    counts as held that the verification did not establish: a null or a "not compared"
    where the condition asks for a comparison is unmet."""
    from .rebuild import verify_sides

    unmet = []
    kinds = {entry["artifact"]: entry["kind"] for entry in report["artifacts"]}
    has_files = v.KIND_JSONL in kinds.values()
    has_shards = v.KIND_SHARD in kinds.values()
    # Without a sides file no file or shard is compared at all.
    if not sides_given and (has_files or has_shards):
        unmet.append("sides: not given")
    # "not compared" is unmet only where the report names something of that kind to compare.
    if made["multisets"] != v.SIDES_EQUAL and (has_files or made["multisets"] != v.SIDES_NOT_COMPARED):
        unmet.append(f"multisets: {made['multisets']}")
    shards_asked = made["mode"] == v.BUILD_SEEDED and has_shards
    if made["per_shard_digests"] == v.SIDES_DIFFER or (
            shards_asked and made["per_shard_digests"] != v.SIDES_EQUAL):
        unmet.append(f"per_shard_digests: {made['per_shard_digests']}")
    nothing_to_compare = set(verify_sides.NOTE_EMPTIED.values())
    for entry in made["differences"]:
        if entry.get("note") and entry["note"] not in nothing_to_compare:
            unmet.append(f"differences {entry['artifact']}: {entry['note']}")
    for entry in made["kept"]:
        if entry["identical"] is False:
            unmet.append(f"kept {entry['artifact']}: not identical")
        elif entry["identical"] is None and kinds.get(entry["artifact"]) in (v.KIND_JSONL, v.KIND_SHARD):
            unmet.append(f"kept {entry['artifact']}: not compared")
    for entry in made["successors"]:
        name = entry["successor"]
        # A successor is the same kind of artifact as the one it replaces: a mixture
        # registered under the name of a shard's successor has no file to compare and
        # reads "matches_sides" null, which is no evidence that it replaces the shard.
        replaced = kinds.get(entry["artifact"])
        if entry["registered"] and replaced is not None and entry["kind"] != replaced:
            unmet.append(f"successor {name}: kind {entry['kind']}, but the artifact it replaces is of kind {replaced}")
        if not entry["registered"]:
            unmet.append(f"successor {name}: not registered")
        elif not v.is_negative_word(entry["answer"]):
            unmet.append(f"successor {name}: {entry['answer']}")
        elif entry["matches_sides"] is False:
            unmet.append(f"successor {name}: not the file compared")
    for entry in made["history"]:
        if entry["instance_unchanged"] is False:
            unmet.append(f"history {entry['artifact']}: instance changed")
    for entry in made["caches"]:
        if entry["stale"] is None:
            unmet.append(f"caches {entry['segment']}: no answer")
        elif entry["stale"] and not entry["rebuilt"]:
            unmet.append(f"caches {entry['segment']}: stale, not rebuilt")
    return unmet


def _verification_text(made: dict) -> str:
    lines = [f"verification of {_request_text(made['request'])}, mode {made['mode']}",
             f"  record multisets: {made['multisets']} ({made['differing_records']} differing records)",
             f"  shard digests: {made['per_shard_digests']}"]
    if made["unresolved_records"]:
        lines.append(f"  unresolved records in the report: {made['unresolved_records']}")
    for entry in made["kept"]:
        lines.append(f"  kept {entry['artifact']}: identical {entry['identical']}")
    for entry in made["successors"]:
        lines.append(f"  successor {entry['successor']}: {entry['answer'] or '(none registered)'}")
    for entry in made["history"]:
        lines.append(f"  history {entry['artifact']}: {entry['answer']}")
    for entry in made["caches"]:
        if entry["stale"]:
            lines.append(f"  stale caches: {entry['segment']} (rebuilt: {entry['rebuilt']})")
        elif entry["stale"] is None:
            lines.append(f"  no answer for the caches of {entry['segment']}: mixture {entry['mixture']}, "
                         f"{entry['reason']}")
    for entry in made["restarts"]:
        lines.append(f"  restart of {entry['run']}: {entry['segment']} step {entry['step']} "
                     f"({entry['basis']}; comparison {entry['step_comparison']})")
    return "\n".join(lines)


def _request_text(request: dict) -> str:
    text = f"{request['kind']} {request['source']} (members: {', '.join(request['members'])})"
    if request["records"]:
        text += f", records: {', '.join(request['records'])}"
    return text


def _answer_text(answer: dict) -> str:
    lines = [f"{answer['artifact']} ({answer['kind'] or 'unknown'}) under {_request_text(answer['request'])}"]
    head = f"  {answer['answer']}"
    if answer["reason"]:
        head += f": {answer['reason']}"
    if answer["names"]:
        head += f": {answer['names']}"
        if answer["names_detail"]:
            head += f" ({answer['names_detail']})"
    lines.append(head)
    failing = answer["failing"]
    if failing:
        where = ", ".join(f"{key} {failing[key]}" for key in
                          ("condition", "artifact", "record", "segment", "step", "job", "reason")
                          if key in failing)
        lines.append(f"  failing link: {where}")
        if "detail" in failing:
            lines.append(f"    {failing['detail']}")
    if answer["candidates"]:
        lines.append(f"  candidate sources: {', '.join(answer['candidates'])}")
    if answer["chain"]:
        lines.append("  chain: " + " <- ".join(hop["hop"] for hop in answer["chain"]))
    if answer["basis"]:
        lines.append(f"  basis: {answer['basis']}")
    if answer["first_exposure"]:
        exposure = answer["first_exposure"]
        lines.append(f"  first exposure: {exposure['segment']} step {exposure['step']}"
                     + (f" ({answer['level']})" if answer["level"] else ""))
    if answer["untraced_steps"]:
        lines.append("  untraced steps: "
                     + ", ".join(f"{s} step {n}" for s, n in answer["untraced_steps"]))
    scope = answer["scope"]
    if scope:
        lines.append(f"  scope: source {scope['source']}, members {', '.join(scope['members'])}")
        if scope["records"]:
            lines.append(f"    records: {', '.join(scope['records'])}")
        if scope["steps"]:
            lines.append("    steps: " + ", ".join(f"{s} {a} to {b}" for s, a, b in scope["steps"]))
        if scope["generation_edges"]:
            lines.append("    rests on the harness logs of: " + ", ".join(scope["generation_edges"]))
        lines.append(f"    records walked: {scope['walked']}")
    for note in answer["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _listing_text(made: dict) -> str:
    lines = [f"listing under {_request_text(made['request'])}", "covers:"]
    for key in listings.COVERS_KEYS:
        lines.append(f"  {key}: {', '.join(made['covers'][key]) or '-'}")
    lines.append("not covered:")
    if not made["not_covered"]:
        lines.append("  -")
    for gap in made["not_covered"]:
        by = "; ".join(f"{n['relation']} of {n['referrer']}" for n in gap["named_by"])
        lines.append(f"  {gap['artifact'] or '(unnamed)'} ({gap['kind']}): {gap['reason']}"
                     + (f", {gap['detail']}" if gap["detail"] else "")
                     + (f"; named by {by}" if by else ""))
    lines.append("answers:")
    for answer in made["answers"]:
        lines.append(_answer_text(answer))
    return "\n".join(lines)


def _exact_rerun_text(restart: dict) -> list[str]:
    """The line of a run that has a stale segment: its exact-rerun restart, as the
    report states it (docs/interfaces.md 7.1). No line for a run that has none."""
    exact = restart["exact_rerun"]
    if exact is None:
        return []
    line = f"    exact rerun, run {restart['run']}, segment {exact['segment']}: "
    if exact["basis"] == v.RESTART_EXACT_BY_VERIFY:
        return [line + exact["basis"]]
    return [line + f"from step {exact['step']}, {exact['basis']} "
                   f"(step comparison {exact['step_comparison']})"]


def _report_text(document: dict) -> str:
    counts = document["counts"]
    lines = [f"rebuild report under {_request_text(document['request'])}"]
    records = counts["records"]
    lines.append("records: " + ", ".join(f"{word} {records[word]}" for word in records)
                 + f"; removed {counts['records_removed']}")
    shards, mixtures = counts["shards"], counts["mixtures"]
    lines.append(f"shards: keep {shards[v.ACTION_KEEP]}, rewrite {shards[v.ACTION_REWRITE]}; "
                 f"mixtures: keep {mixtures[v.ACTION_KEEP]}, rewrite {mixtures[v.ACTION_REWRITE]}")
    unstated = counts["place_not_stated"]
    if unstated["shards"] or unstated["mixtures"]:
        lines.append(f"place not stated: shards {len(unstated['shards'])}, "
                     f"mixtures {len(unstated['mixtures'])}")
    lines.append(f"bytes to rewrite: {counts['bytes_to_rewrite']}"
                 + (f" (unknown for {', '.join(counts['bytes_unknown'])})"
                    if counts["bytes_unknown"] else ""))
    lines.append("artifacts:")
    for entry in document["artifacts"]:
        line = f"  {entry['artifact']} ({entry['kind']}): {entry['action']}"
        if entry["successor"]:
            line += f" -> {entry['successor']}"
        lines.append(line)
    for entry in document["artifacts"]:
        if entry["kind"] == v.KIND_SHARD and entry["action"] == v.ACTION_CLEAN_PLACE_NOT_STATED:
            lines.append(f"  {entry['artifact']}: its own contents are clean; it may receive "
                         f"{', '.join(entry['returning'])}; their place is {v.PLACE_NOT_STATED}")
    if counts["unresolved_artifacts"]:
        lines.append("unresolved: " + ", ".join(counts["unresolved_artifacts"]))
    lines.append("files upstream (not in the counts):")
    for entry in document["upstream_files"]:
        lines.append(f"  {entry['artifact']} ({entry['role']}): {entry['action'] or 'no action'}"
                     f" - {entry['reason']}")
    if not document["upstream_files"]:
        lines.append("  -")
    layouts = document["groups"]["layouts"]
    if not layouts["counted"]:
        lines.append(f"per-layout count not made: {layouts['reason']}")
    elif layouts["unmatched_shards"]:
        lines.append("shards in no named layout: " + ", ".join(layouts["unmatched_shards"]))
    for name, placed in document["groups"]["placements"].items():
        if placed["placement"] is not None:
            lines.append(f"placement of {name}: {placed['placement']['path']} "
                         f"(sha256 {placed['placement']['sha256']})")
        for landing in placed["landing"]:
            lines.append(f"  lands in {landing['shard']} (not in the store): "
                         + ", ".join(landing["receives"]))
    lines.append("restarts:")
    for restart in document["restarts"]:
        where = (f"{restart['checkpoint']} (step {restart['step']})" if restart["proven"]
                 else restart["basis"])
        lines.append(f"  run {restart['run']}: {where}")
        lines.extend(_exact_rerun_text(restart))
    if not document["restarts"]:
        lines.append("  -")
    for exposed in document["exposed_checkpoints"]:
        lines.append(f"exposed: {exposed['checkpoint']} ({exposed['answer']})")
    for uncovered in document["checkpoints_not_covered"]:
        lines.append(f"not covered: {uncovered['checkpoint']} ({uncovered['reason']})")
    return "\n".join(lines)


# --- The one way in and out -----------------------------------------------------------


MIGRATION_NOTES = (
    (4, "its harness logs now read as not compared, so a clean answer past a generation step reads "
        "\"may contain\" until the texts the jobs were handed are shown (waterwhite compare-hand-off)"),
    (5, "its shards read as not checked, so a negative answer that rests on the map of a shard whose "
        "files are not at their recorded place is not given until the files are put back and "
        "waterwhite check has read them"),
)


def migration_note(found: int, reached: int) -> str:
    """What the steps from ``found`` to ``reached`` change in the answers of the store: one
    clause for each step that does."""
    clauses = [text for version, text in MIGRATION_NOTES if found < version <= reached]
    return (f"this store was migrated from schema version {found} to {reached}: "
            + "; and ".join(clauses))


def _wants_text(argv: Sequence[str]) -> bool:
    """Whether the arguments ask for text, read without the parser, so that a
    usage error can be printed in the format that was asked for."""
    for position, item in enumerate(argv):
        if item == "--format=text":
            return True
        if item == "--format" and position + 1 < len(argv) and argv[position + 1] == "text":
            return True
    return False


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None,
         stderr: TextIO | None = None) -> int:
    """Run one command. Returns the exit code."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    arguments = list(sys.argv[1:] if argv is None else argv)
    as_text = _wants_text(arguments)
    migrated: list[tuple[int, int]] = []
    try:
        try:
            args = build_parser().parse_args(arguments)
        except SystemExit as leaving:  # --help prints and leaves
            return int(leaving.code or 0)
        run = getattr(args, "run", None)
        if run is None:
            command = getattr(args, "command", None)
            if not command:
                raise UsageError("name a command: init, query, report, show, check, register-run, "
                                 "replay, verify or recover")
            if command == "recover":
                raise UsageError("name what to recover: recover file, shard, copy-step, "
                                 "sanitizer or map")
            raise UsageError(f"name a request: {command} withdraw <source>")
        Store.migration_hook = lambda found, reached: migrated.append((found, reached))
        try:
            value, text = run(args)
        finally:
            Store.migration_hook = None
        if migrated and isinstance(value, dict):
            found, reached = migrated[0]
            note = migration_note(found, reached)
            value = {**value, "migrated": {"from": found, "to": reached, "note": note}}
            text = f"{text}\n{note}"
        printed = text if as_text else json.dumps(value, indent=2, ensure_ascii=False)
    except WaterwhiteError as error:
        # A store that was migrated before the command was refused stays migrated: say so here too.
        note = migration_note(migrated[0][0], migrated[0][1]) if migrated else None
        if as_text:
            err.write(f"error: {error.name}" + (f": {error.detail}" if error.detail else "") + "\n")
            if note:
                err.write(note + "\n")
        else:
            payload = error.to_json()
            if note:
                payload = {**payload, "migrated": {"from": migrated[0][0], "to": migrated[0][1], "note": note}}
            err.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return error.exit_code
    out.write(printed + "\n")
    unmet = getattr(args, "unmet", None)
    if unmet:
        err.write(f"not held: {'; '.join(unmet)}\n" if as_text else
                  json.dumps({"verification": "not held", "unmet": unmet}, ensure_ascii=False) + "\n")
        return EXIT_NOT_HELD
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
