"""A placement target that lies outside its layout's prefix (docs/cli.md "report" and
`--placement`; docs/interfaces.md 7.1; DECISIONS.md D-144).

``report withdraw A --layout NAME=PREFIX --placement NAME=FILE``: a placement file that
names a target shard, a value of ``shards`` or ``otherwise`` where that is not null, whose
store name does not begin with the layout's PREFIX is a usage error. The command ends with
exit code 1, the usage error document on standard error names the placement file, the
layout, its prefix and the target, no report document is printed, and the store file is
the same bytes before and after. A target inside the prefix is accepted as before: a
shard of the store that the placement names receives the copy, and a shard the store does
not hold is a landing, named in ``groups.placements[NAME].landing`` and in no artifact
entry.

The layout under test is ``x = zone-x/`` and the targets outside it lie under ``zone-y/``,
names that neither hold the other nor occur in the path of a placement file, so that a
message that names the prefix and the target can be told from one that names neither.

Expected values come from the documents named above and from the construction of the
worlds in ``keep_kit.py``: under "withdraw A" the world's one returning copy is
``src/B:L1``, which a placement files by its source, ``src/B``.
"""

from __future__ import annotations

import hashlib
import json

from waterwhite import vocabulary as v

from keep_kit import B_COPY, add_shard, entry, names, placements_of, write_placement
from rb_kit import run

PREFIX = "zone-x/"
LAYOUTS = {"x": PREFIX}
HELD = "zone-x/other"
OUTSIDE = "zone-y/elsewhere"
NOT_HELD = "zone-x/not-in-the-store"


def a_world_with_the_layouts_shard(world) -> None:
    """The world and one shard of the layout, ``zone-x/other``: one document, R2:L3."""
    add_shard(world, HELD, [("R2", 3)])


def store_digest(world) -> str:
    """The sha256 of the store file with no connection open on it."""
    world.store.close()
    return hashlib.sha256(world.path.read_bytes()).hexdigest()


def report_of(world, placement) -> tuple[int, str, str]:
    """``report withdraw A --layout x=zone-x/ --placement x=FILE``: the exit code, standard
    output and standard error."""
    return run(world, "report", "withdraw", "A", "--layout", f"x={PREFIX}",
               "--placement", f"x={placement}")


def assert_refused_as_a_usage_error(world, placement, target: str) -> None:
    before = store_digest(world)
    code, out, err = report_of(world, placement)
    assert code == 1, f"exit code {code}, not 1; the report began: {out[:120]!r}; stderr: {err!r}"
    assert out.strip() == "", f"a report document was printed: {out[:120]!r}"
    document = json.loads(err)
    assert document["error"] == "usage", err
    assert target in err, f"the usage error does not name the target {target!r}: {err}"
    assert PREFIX in err, f"the usage error does not name the prefix {PREFIX!r}: {err}"
    assert store_digest(world) == before, "the store file changed"


# --- A target outside the prefix is refused ---------------------------------------------------


def test_a_shards_target_outside_the_prefix_is_a_usage_error_and_the_store_is_unchanged(
        world, tmp_path):
    a_world_with_the_layouts_shard(world)
    placement = write_placement(tmp_path, "layoutx", {"src/B": OUTSIDE})
    assert_refused_as_a_usage_error(world, placement, OUTSIDE)


def test_an_otherwise_target_outside_the_prefix_is_a_usage_error_and_the_store_is_unchanged(
        world, tmp_path):
    a_world_with_the_layouts_shard(world)
    # every `shards` target lies inside the prefix: only `otherwise` is outside it
    placement = write_placement(tmp_path, "layoutx", {"src/B": HELD}, otherwise=OUTSIDE)
    assert_refused_as_a_usage_error(world, placement, OUTSIDE)


# --- A target inside the prefix is accepted as before --------------------------------------------


def test_every_target_inside_the_prefix_and_otherwise_null_is_accepted(world, tmp_path):
    a_world_with_the_layouts_shard(world)
    placement = write_placement(tmp_path, "layoutx", {"src/B": HELD}, otherwise=None)
    code, out, err = report_of(world, placement)
    assert code == 0, f"a placement with every target inside the prefix was refused: {err}"
    document = json.loads(out)
    assert document["form"] == v.FORM_REPORT
    # the shard of the store that the placement names receives the copy (docs/interfaces.md 7.1)
    shard = entry(document, HELD)
    assert shard["action"] == "rewrite" and shard["receives"] == [B_COPY]
    assert placements_of(document)["x"]["landing"] == []


def test_a_target_inside_the_prefix_that_the_store_does_not_hold_is_still_a_landing(
        world, tmp_path):
    a_world_with_the_layouts_shard(world)
    placement = write_placement(tmp_path, "layoutx", {"src/B": NOT_HELD})
    code, out, err = report_of(world, placement)
    assert code == 0, f"a target inside the prefix that the store does not hold was refused: {err}"
    document = json.loads(out)
    assert placements_of(document)["x"]["landing"] == [
        {"shard": NOT_HELD, "in_store": False, "receives": [B_COPY]}]
    # a landing is named under the placements only: no artifact entry carries its name
    assert NOT_HELD not in names(document)
