"""A registered checkpoint artifact that has no checkpoint row is named in the report
(DECISIONS.md D-142; docs/interfaces.md 7.1, the list ``checkpoints_not_covered``).

The artifact of the kind "checkpoint" is registered by ``register_checkpoint``, which
also writes the row that says which segment and step it is. A store file edited from
outside the write calls can hold the artifact without the row. Its answer is "not
covered", reason "record missing" (the coverage rule of docs/coverage.md section 7: a
checkpoint whose own record is missing), so the report must name it, with no segment,
no step and no run to give, whatever way the list is built.

The edit is the direct deletion of the one row of ``Y@2`` in the test's own copy of the
world (nothing starts from ``Y@2``, so no other answer of the world moves). The expected
entry comes from the vocabulary's words for the answer and the reason, and from the name
the world registers the checkpoint under (``tests/coverage/ww_world.py``).

The test is green on the code that names every artifact of the kind. Its red is a
version of ``checkpoints_not_covered`` that skips an artifact with no checkpoint row
before the answer is asked: with that version the list is empty and the assertion fails.
"""

from __future__ import annotations

from waterwhite import requests
from waterwhite import vocabulary as v

from rb_kit import need_attr


def test_a_checkpoint_artifact_with_no_checkpoint_row_is_named_with_the_answers_reason(world):
    world.sql((
        "DELETE FROM checkpoint WHERE artifact_id = "
        "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["Y@2"],)))
    build_report = need_attr("waterwhite.rebuild", "build_report")
    document = build_report(world.store, requests.withdraw(world.store, "A"))
    assert document["checkpoints_not_covered"] == [{
        "checkpoint": "Y@2", "segment": None, "step": None, "run": None,
        "answer": v.NOT_COVERED, "reason": v.REASON_RECORD_MISSING}]
    # It is not among the exposed ones: its answer is not an exposure.
    assert "Y@2" not in [entry["checkpoint"] for entry in document["exposed_checkpoints"]]
