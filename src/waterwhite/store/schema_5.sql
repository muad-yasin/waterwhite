-- Step 5 of the store's schema (plan 3.4: versioned and migrated stepwise).
-- This script brings a version-4 store to version 5. A new store runs steps 1 to 5,
-- so a new store and a migrated one hold the same schema.

-- The segment a trace row belongs to, so that the rows of one document in one segment are
-- found without reading the rows of the same document in every other segment (the index
-- below). The row's trace names its segment (``step_trace.segment_id``); the column is that
-- value, written with the row, and filled here for the rows already stored.
ALTER TABLE trace_row ADD COLUMN segment_id INTEGER;
UPDATE trace_row SET segment_id =
    (SELECT t.segment_id FROM step_trace t WHERE t.trace_id = trace_row.trace_id);
CREATE INDEX trace_row_segment_document ON trace_row(segment_id, shard_position, ordinal);

-- What a check of a shard's two files found: 'passed' when they are the pair the registered
-- digest names and their index gives the offsets and the .bin size a Megatron reader and
-- this tool's token reader agree on (artifact_files.read_shard), 'failed' when they are
-- not. A shard registered by this version is 'passed' (registration reads the pair that
-- way). A shard with no row was registered under an earlier schema, before the offsets and
-- the size were held to that reading; nothing says its files were ever checked, so a
-- negative answer that rests on a stored map of it, with its files not at their place, is
-- not given until ``waterwhite check`` has read the files (``Store.recheck_shard_files``).
CREATE TABLE shard_files_check (
    artifact_id INTEGER PRIMARY KEY REFERENCES artifact(artifact_id),
    state       TEXT NOT NULL CHECK (state IN ('passed', 'failed'))
);
