-- Step 4 of the store's schema (plan 3.4: versioned and migrated stepwise).
-- This script brings a version-3 store to version 4. A new store runs steps 1 to 4,
-- so a new store and a migrated one hold the same schema.

-- Whether the texts handed to a job were compared with the records their references
-- name: 1 when the registration that stored the log carried a text for every
-- reference of the hand-off and every text matched its record (or when
-- ``Store.verify_hand_off`` later did the same), 0 otherwise. Every log written
-- under version 3 or earlier is 0: nothing says its texts were ever compared. The
-- stored log establishes no bound while this is 0 (``HarnessLog.bound_established``).
ALTER TABLE harness_log ADD COLUMN hand_off_checked INTEGER NOT NULL DEFAULT 0
    CHECK (hand_off_checked IN (0, 1));
