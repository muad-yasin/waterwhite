-- Step 3 of the store's schema (plan 3.4: versioned and migrated stepwise).
-- Version 2 is schema.sql with schema_2.sql; this script brings a version-2 store to
-- version 3. A new store runs steps 1, 2 and this one, so a new store and a migrated
-- one hold the same schema. The migration step (_migrate.py) refuses a store that
-- holds two rows for one array of a blend before it runs any statement here.

-- The hand-off of a harness log is a list: the position of a handed record in the
-- order it was handed. The primary key stays (log_id, occurrence_id), so a record is
-- handed once per log; the position is unique per log beside it. NULL for a log
-- written under version 1 or 2, whose hand-off was stored as a set and has no order
-- to give back (such a log is read in the order of its occurrence ids, and a second
-- registration of its job is compared with the hand-off as a set).
ALTER TABLE harness_handed ADD COLUMN position INTEGER
    CHECK (position IS NULL OR position >= 0);
CREATE UNIQUE INDEX harness_handed_one_per_position
    ON harness_handed(log_id, position) WHERE position IS NOT NULL;

-- One array of the blend (shard_position NULL) has one recorded digest. The UNIQUE
-- constraint of schema.sql does not hold for a NULL position, because SQLite reads
-- two NULLs as different, so the rule is a partial index over the two columns that
-- remain.
CREATE UNIQUE INDEX segment_cache_digest_one_blend
    ON segment_cache_digest(segment_id, kind) WHERE shard_position IS NULL;
