"""The outside review of 504bf64 (Sol's second review): a store made before the offsets of a
shard's index were held to the token reader gave "does not contain" for a checkpoint whose
shard files had been moved away, though the offsets stored in the index could name another
document than the one the map names, and nothing in the store could say whether they did.

Schema 5 records for every shard what a check of its two files found. A shard registered by
this version is checked by its registration. A shard of an older store has no record, so a
negative answer that rests on its map, with its files not at their place, is not given
(``ShardFilesNotChecked``, exit 3) until ``waterwhite check`` has read the files: it records
"passed" for a shard that is a shard of this layout and "failed", for good, for one whose
index gives offsets the reader refuses. With the files present the loud checks read them at
every query, as before.

The expected answers come from the world of ``ww_world`` (shard-1 holds the records of R2, and
the withdrawal of C is not in it: its negative is what the review saw) and from the text of
the review's fixture, not from running the command.
"""

from __future__ import annotations

import io
import json
import sqlite3
import struct

import pytest
import ww_world

from waterwhite import cli, coverage, errors, requests
from waterwhite.store import Store, artifact_files


def forget_checks(world) -> None:
    """A store as an earlier schema left it: its shards have no record of a check. The table
    is the one schema 5 makes, so a store of that schema is brought to the earlier state; where
    the table is not there (the code under test has none) nothing is done."""
    has_table = world.scalar("SELECT COUNT(*) FROM sqlite_master WHERE name = 'shard_files_check'")
    if has_table:
        world.sql("DELETE FROM shard_files_check")


def put_away(world, name: str = "shard-1"):
    folder = world.folder / "away"
    folder.mkdir(exist_ok=True)
    for suffix in (".bin", ".idx"):
        (world.folder / f"shards/{name}{suffix}").rename(folder / f"{name}{suffix}")


def put_back(world, name: str = "shard-1"):
    for suffix in (".bin", ".idx"):
        (world.folder / f"away/{name}{suffix}").rename(world.folder / f"shards/{name}{suffix}")


def question(world, artifact: str = "shard-1") -> dict:
    request = requests.withdraw(world.store, "C")
    return coverage.answer(world.store, artifact, request).to_json()


def check_command(world) -> tuple[int, dict]:
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    try:
        code = cli.main(["check", "--store", str(world.path)], stdout=out, stderr=err)
    finally:
        world.store = Store.open(world.path)
    return code, json.loads(out.getvalue() or err.getvalue())


def swap_the_offsets(world, name: str = "shard-1") -> None:
    """The review's fixture: the index of the shard gives the offsets of its first two
    sequences the other way round (a011cd5 registered it), and the store's digests are the ones
    of the file as it is, as a store that registered it holds them."""
    idx = world.folder / f"shards/{name}.idx"
    data = bytearray(idx.read_bytes())
    sequences = artifact_files.read_shard_index(idx).sequence_count
    first = 34 + 4 * sequences  # where the offsets start
    one, two = struct.unpack_from("<2q", data, first)
    assert one != two
    struct.pack_into("<2q", data, first, two, one)
    idx.write_bytes(bytes(data))
    cut = artifact_files.shard_digest_only(world.folder / f"shards/{name}")
    row = world.scalar("SELECT artifact_id FROM artifact WHERE instance_id = ?", (world[name],))
    world.sql(("UPDATE artifact SET content_digest = ? WHERE artifact_id = ?", (cut, row)),
              ("UPDATE binding SET declared_digest = ? WHERE artifact_id = ?", (cut, row)))
    with pytest.raises(errors.UnsupportedFormat, match="offset"):
        artifact_files.read_shard(world.folder / f"shards/{name}")


def test_a_shard_registered_by_this_version_gives_its_negative_with_its_files_away(world):
    assert world.store.shard_files_state(world["shard-1"]) == "passed"
    put_away(world)
    answer = question(world)
    assert answer["answer"] == "does not contain"
    assert answer["scope"]["files_not_checked"] == ["shard-1"]


def test_a_legacy_shard_with_its_files_away_gives_no_negative_until_a_check_has_read_them(world):
    """Right offsets. Before the check: the loud fault, naming the shard and the command. After
    the files are put back and ``check`` has run: the negative, with the files away again."""
    forget_checks(world)
    put_away(world)
    with pytest.raises(errors.ShardFilesNotChecked, match="no check of them has passed") as caught:
        question(world)
    assert caught.value.exit_code == errors.EXIT_INTEGRITY
    assert caught.value.context["instance"] == world["shard-1"] and caught.value.context["state"] is None
    assert "waterwhite check" in caught.value.detail
    put_back(world)
    assert question(world)["answer"] == "does not contain"  # present: the loud checks read the files
    assert check_command(world)[0] == 0
    assert world.store.shard_files_state(world["shard-1"]) == "passed"
    put_away(world)
    answer = question(world)
    assert answer["answer"] == "does not contain"
    assert answer["scope"]["files_not_checked"] == ["shard-1"]


def test_a_legacy_shard_whose_offsets_the_check_finds_wrong_is_refused_for_good(world):
    """Sol's fixture: swapped offsets. The check with the files present stops with the fault it
    gave before and records "failed"; the files away again and the answer is refused, naming
    that they were found not to be a shard of this layout, however often it is asked."""
    swap_the_offsets(world)
    forget_checks(world)
    code, printed = check_command(world)
    assert code == errors.EXIT_INTEGRITY, printed
    assert world.store.shard_files_state(world["shard-1"]) == "failed"
    put_away(world)
    for _ in range(2):
        with pytest.raises(errors.ShardFilesNotChecked, match="not a shard of this layout") as caught:
            question(world)
        assert caught.value.context["state"] == "failed"
    # A later check cannot make it pass: the files are away and, back, they are the same bytes.
    put_back(world)
    assert check_command(world)[0] == errors.EXIT_INTEGRITY
    assert world.store.shard_files_state(world["shard-1"]) == "failed"


def test_the_check_command_refuses_a_store_of_an_earlier_schema_and_does_not_migrate_it(world):
    world.store.close()
    raw = sqlite3.connect(world.path)
    raw.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["check", "--store", str(world.path)], stdout=out, stderr=err)
    assert code == errors.EXIT_REFUSED and "waterwhite migrate" in json.loads(err.getvalue())["detail"]
    raw = sqlite3.connect(world.path)
    assert raw.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "4"
    raw.execute("UPDATE meta SET value = '5' WHERE key = 'schema_version'")  # the file is a schema 5 file
    raw.commit()
    raw.close()
    world.store = Store.open(world.path)


def cli_run(world, *argv: str) -> tuple[int, str, str]:
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    try:
        code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    finally:
        world.store = Store.open(world.path)
    return code, out.getvalue(), err.getvalue()


def test_the_query_through_the_command_line_exits_3_names_the_error_and_prints_no_answer(world):
    """What docs/cli.md ("check") promises of the refusal: exit 3, the name "shard files not
    checked", and nothing on standard output."""
    forget_checks(world)
    put_away(world)
    code, out, err = cli_run(world, "query", "withdraw", "C", "--artifact", "shard-1")
    assert (code, out) == (errors.EXIT_INTEGRITY, "")
    assert json.loads(err)["error"] == "shard files not checked"


def test_check_says_which_shards_are_not_passed_and_whether_their_files_are_away(world):
    """The audit of a078390: ``check`` printed "passed" for a shard that was never checked or
    was recorded "failed" and whose files are away, and said nothing of which shards the
    refusal above is waiting for."""
    code, out, err = cli_run(world, "check")
    assert code == 0 and json.loads(out)["shards_not_passed"] == []
    forget_checks(world)
    put_away(world)
    code, out, err = cli_run(world, "check")
    said = json.loads(out)
    assert code == 0 and said["check"] == "passed"
    assert said["shards_not_passed"] == [{"instance": world["shard-1"], "name": "shard-1",
                                          "state": "unchecked", "files": "away"}]
    put_back(world)
    code, out, err = cli_run(world, "check")
    assert code == 0 and json.loads(out)["shards_not_passed"] == []


def test_check_lists_a_failed_shard_whose_files_are_away_and_one_whose_files_are_back_fails_the_check(world):
    swap_the_offsets(world)
    forget_checks(world)
    code, out, err = cli_run(world, "check")  # present: the loud check stops, the record stays
    assert code == errors.EXIT_INTEGRITY
    put_away(world)
    code, out, err = cli_run(world, "check")
    said = json.loads(out)
    assert code == 0
    assert said["shards_not_passed"] == [{"instance": world["shard-1"], "name": "shard-1",
                                          "state": "failed", "files": "away"}]


def test_check_reads_the_shard_files_outside_its_write_transaction_and_writes_nothing_when_nothing_changed(
        world, monkeypatch):
    """The audit of a078390: ``check`` held ``BEGIN IMMEDIATE`` while it hashed every shard (a
    second writer waited out the busy timeout) and wrote a row for every shard on every run (a
    query in another process then ended with "store written during the query")."""
    world.store.close()
    seen = []
    original = artifact_files.read_shard_once

    def watched(prefix):
        other = sqlite3.connect(world.path, timeout=0)
        try:
            other.execute("BEGIN IMMEDIATE")  # raises "database is locked" while the check holds the write lock
            other.execute("ROLLBACK")
            seen.append("free")
        except sqlite3.OperationalError:
            seen.append("locked")
        finally:
            other.close()
        return original(prefix)

    monkeypatch.setattr(artifact_files, "read_shard_once", watched)
    reader = sqlite3.connect(world.path)
    before = reader.execute("PRAGMA data_version").fetchone()[0]
    out, err = io.StringIO(), io.StringIO()
    assert cli.main(["check", "--store", str(world.path)], stdout=out, stderr=err) == 0, err.getvalue()
    assert seen and set(seen) == {"free"}, seen
    assert reader.execute("PRAGMA data_version").fetchone()[0] == before  # nothing was committed
    reader.close()
    world.store = Store.open(world.path)
