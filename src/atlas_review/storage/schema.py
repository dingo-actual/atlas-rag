"""Authoritative SQLite schema for a version 3 workspace."""

from __future__ import annotations

SCHEMA_VERSION = 3

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE generations (
    generation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    collection_version TEXT NOT NULL,
    format_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    build_fingerprint TEXT NOT NULL,
    source_bytes BLOB NOT NULL,
    manifest_msgpack BLOB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('building', 'ready')),
    artifact_directory TEXT,
    artifact_manifest_sha256 TEXT,
    passage_count INTEGER NOT NULL DEFAULT 0 CHECK (passage_count >= 0),
    CHECK (status <> 'ready' OR
           (artifact_directory IS NOT NULL AND artifact_manifest_sha256 IS NOT NULL))
);

CREATE TABLE workspace_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    active_generation_id TEXT REFERENCES generations(generation_id)
);
INSERT INTO workspace_state(singleton, active_generation_id) VALUES (1, NULL);

CREATE TABLE configuration (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    payload_msgpack BLOB NOT NULL
);

CREATE TABLE entities (
    generation_id TEXT NOT NULL REFERENCES generations(generation_id),
    entity_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN
        ('collection', 'matrix', 'tactic', 'technique', 'mitigation',
         'case_study', 'software', 'entity')),
    name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    payload_msgpack BLOB NOT NULL,
    PRIMARY KEY (generation_id, entity_id)
);
CREATE INDEX entities_by_kind ON entities(generation_id, kind);

CREATE TABLE edges (
    generation_id TEXT NOT NULL REFERENCES generations(generation_id),
    edge_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN
        ('employs', 'mitigates', 'specializes', 'achieves', 'sequences',
         'uses', 'attributed-to', 'implements', 'reported-by')),
    step_id TEXT,
    source_locator TEXT NOT NULL,
    payload_msgpack BLOB NOT NULL,
    PRIMARY KEY (generation_id, edge_id),
    FOREIGN KEY (generation_id, source_id)
        REFERENCES entities(generation_id, entity_id),
    FOREIGN KEY (generation_id, target_id)
        REFERENCES entities(generation_id, entity_id)
);
CREATE INDEX edges_from ON edges(generation_id, source_id, relation);
CREATE INDEX edges_to ON edges(generation_id, target_id, relation);

CREATE TABLE context_cache (
    cache_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    document_sha256 TEXT NOT NULL,
    chunk_sha256 TEXT NOT NULL,
    contextualizer_fingerprint TEXT NOT NULL,
    context_text TEXT NOT NULL CHECK (length(context_text) > 0),
    payload_msgpack BLOB NOT NULL
);

CREATE TABLE passages (
    passage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id TEXT NOT NULL REFERENCES generations(generation_id),
    entity_id TEXT,
    edge_id TEXT,
    section TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    bm25_ordinal INTEGER NOT NULL CHECK (bm25_ordinal >= 0),
    source_text TEXT NOT NULL,
    context_text TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    context_cache_key TEXT NOT NULL REFERENCES context_cache(cache_key),
    source_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    embedding_tokens INTEGER NOT NULL CHECK (embedding_tokens > 0),
    reranker_tokens INTEGER NOT NULL CHECK (reranker_tokens > 0),
    vector_f32_le BLOB NOT NULL CHECK (length(vector_f32_le) = 3072),
    payload_msgpack BLOB NOT NULL,
    CHECK ((entity_id IS NOT NULL AND edge_id IS NULL) OR
           (entity_id IS NULL AND edge_id IS NOT NULL)),
    FOREIGN KEY (generation_id, entity_id)
        REFERENCES entities(generation_id, entity_id),
    FOREIGN KEY (generation_id, edge_id)
        REFERENCES edges(generation_id, edge_id)
);
CREATE UNIQUE INDEX passage_entity_chunk
    ON passages(generation_id, entity_id, section, ordinal)
    WHERE entity_id IS NOT NULL;
CREATE UNIQUE INDEX passage_edge_chunk
    ON passages(generation_id, edge_id, section, ordinal)
    WHERE edge_id IS NOT NULL;
CREATE INDEX passages_generation ON passages(generation_id);
CREATE UNIQUE INDEX passages_keyword_position ON passages(generation_id, bm25_ordinal);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL REFERENCES generations(generation_id),
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('retrieved', 'complete', 'failed')),
    request_msgpack BLOB NOT NULL,
    config_msgpack BLOB NOT NULL,
    trace_msgpack BLOB NOT NULL,
    evidence_msgpack BLOB NOT NULL,
    report_msgpack BLOB
);

CREATE TABLE review_attempts (
    review_id TEXT NOT NULL REFERENCES reviews(review_id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    started_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN
        ('started', 'complete', 'failed', 'interrupted')),
    payload_msgpack BLOB NOT NULL,
    PRIMARY KEY (review_id, attempt_number)
);

PRAGMA user_version = 3;
"""
