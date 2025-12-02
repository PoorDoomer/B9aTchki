-- =============================================================================
-- Cross-Lingual Reclamation De-duplication Engine
-- Initial Database Schema with pgvector support
-- =============================================================================

-- Enable Vector Extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Main Table: reclamations
-- Stores the raw reclamation tickets with free-text in French/Arabic
-- =============================================================================
CREATE TABLE IF NOT EXISTS reclamations (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,                    -- The "User Reclame" (Grouping Key)
    message_libre TEXT NOT NULL,                -- The raw free text (Arabic/French)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'PENDING',       -- PENDING, DUPLICATE, POTENTIAL_DUPLICATE, PROCESSED
    
    -- Constraints
    CONSTRAINT chk_status CHECK (status IN ('PENDING', 'DUPLICATE', 'POTENTIAL_DUPLICATE', 'PROCESSED'))
);

-- Index for user_id filtering (used in grouping queries)
CREATE INDEX IF NOT EXISTS idx_reclamations_user_id ON reclamations(user_id);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_reclamations_status ON reclamations(status);

-- Index for time-based queries (7-day window)
CREATE INDEX IF NOT EXISTS idx_reclamations_created_at ON reclamations(created_at DESC);

-- Composite index for common query pattern (user_id + created_at)
CREATE INDEX IF NOT EXISTS idx_reclamations_user_created ON reclamations(user_id, created_at DESC);

-- =============================================================================
-- Vector Store Table: reclamation_embeddings
-- Stores the 768-dimensional LaBSE embeddings for semantic search
-- Separated for performance optimization
-- =============================================================================
CREATE TABLE IF NOT EXISTS reclamation_embeddings (
    reclamation_id BIGINT PRIMARY KEY REFERENCES reclamations(id) ON DELETE CASCADE,
    embedding vector(768)                       -- 768 dimensions matches LaBSE architecture
);

-- HNSW Index for State-of-the-Art approximate nearest neighbor search
-- Parameters:
--   m = 16: Maximum number of connections per layer (higher = more accurate, slower build)
--   ef_construction = 64: Size of dynamic candidate list during index construction
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON reclamation_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Audit Log Table: duplication_logs
-- Tracks decisions for transparency and debugging
-- =============================================================================
CREATE TABLE IF NOT EXISTS duplication_logs (
    id BIGSERIAL PRIMARY KEY,
    source_reclamation_id BIGINT REFERENCES reclamations(id) ON DELETE SET NULL,
    matched_reclamation_id BIGINT REFERENCES reclamations(id) ON DELETE SET NULL,
    similarity_score DECIMAL(5, 4),             -- e.g., 0.9850
    action VARCHAR(50) NOT NULL,                -- AUTO_MARK_DUPLICATE, FLAG_FOR_REVIEW
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_action CHECK (action IN ('AUTO_MARK_DUPLICATE', 'FLAG_FOR_REVIEW'))
);

-- Index for querying by source reclamation
CREATE INDEX IF NOT EXISTS idx_duplication_logs_source ON duplication_logs(source_reclamation_id);

-- Index for querying by matched reclamation
CREATE INDEX IF NOT EXISTS idx_duplication_logs_matched ON duplication_logs(matched_reclamation_id);

-- Index for time-based audit queries
CREATE INDEX IF NOT EXISTS idx_duplication_logs_detected_at ON duplication_logs(detected_at DESC);

-- =============================================================================
-- Helper function: Calculate cosine similarity score
-- Note: pgvector's <=> operator returns cosine distance, so we compute: 1 - distance
-- =============================================================================
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- Comments for documentation
-- =============================================================================
COMMENT ON TABLE reclamations IS 'Main table storing reclamation tickets with free-text in French/Arabic';
COMMENT ON TABLE reclamation_embeddings IS 'Vector store for 768-dim LaBSE embeddings, separated for performance';
COMMENT ON TABLE duplication_logs IS 'Audit log tracking duplicate detection decisions';
COMMENT ON FUNCTION cosine_similarity IS 'Helper function to compute cosine similarity from pgvector distance';


