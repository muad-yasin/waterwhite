-- Step 2 of the store's schema (plan 3.4: versioned and migrated stepwise).
-- Version 1 is schema.sql; this script brings a version-1 store to version 2.
-- A new store runs step 1 and then this step, so a new store and a migrated one
-- hold the same schema.

-- A harness log says which harness wrote it. NULL when the log does not say
-- (every log written under version 1).
ALTER TABLE harness_log ADD COLUMN harness TEXT
    CHECK (harness IN ('in-process', 'confined'));

-- Per artifact, namespace and derivation: at most one binding whose verification
-- passed. A map is never overwritten and never deleted. A map that did not verify
-- (failed, or not run) can be followed by another map of the same namespace and
-- derivation, written beside it; a verified map is final for its namespace and
-- derivation (DECISIONS.md D-28). The derivation and the verification of a JSONL
-- binding are NULL, and NULLs never collide in a UNIQUE constraint, so the rule
-- is an index over the derivation with NULL read as '', kept for the bindings
-- that are verified or carry no verification. A JSONL artifact therefore has one
-- binding, and a map that did not verify is outside the index. The reader's
-- order of schema.sql takes the verified map first.
DROP INDEX binding_one_per_mode;
CREATE UNIQUE INDEX binding_one_verified_per_mode
    ON binding(artifact_id, namespace, COALESCE(derivation, ''))
    WHERE verification IS NULL OR verification = 'passed';
