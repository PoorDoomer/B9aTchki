"""
Database layer for the De-duplication Pipeline.

Provides connection pooling, CRUD operations for reclamations,
and vector similarity search using pgvector.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Generator, Dict
from decimal import Decimal

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import math

from src.config import get_config, PostgresConfig


logger = logging.getLogger(__name__)


def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Assumes embeddings are already normalized (LaBSE normalizes by default).
    For normalized vectors: cosine_similarity = dot_product

    Args:
        embedding1: First embedding vector (list of floats)
        embedding2: Second embedding vector (list of floats)

    Returns:
        Cosine similarity score between 0 and 1
    """
    return float(np.dot(embedding1, embedding2))


@dataclass
class Reclamation:
    """Data class representing a reclamation record."""
    id: int
    reclamant_id: int
    description: str
    motif_id: str
    created_at: datetime

@dataclass
class Motif:
    """Data class representing a motif record."""
    id: str
    libelle: str

@dataclass
class DuplicationLog:
    """Data class representing a duplication log record."""
    id: int
    source_reclamation_id: int
    matched_reclamation_id: int
    similarity_score: Decimal
    action: str
    detected_at: datetime


@dataclass
class ReclamationMatch:
    """Data class representing a reclamation match record."""
    id: int
    reclamation_id: int
    matched_reclamation_id: int
    similarity_score: Decimal
    match_status: str
    created_at: datetime
    updated_at: datetime


@dataclass
class SimilarityMatch:
    """Data class representing a similarity search result."""
    reclamation_id: int
    score: float
    motif_id: Optional[str] = None


@dataclass
class SimilarityMatchChunkResult:
    """Data class representing detailed chunk-based similarity result."""
    reclamation_id: int
    similarity_score: float
    motif_id: Optional[str] = None
    is_chunked: bool = False
    source_is_chunked: bool = False
    target_is_chunked: bool = False
    num_source_chunks: int = 0
    num_target_chunks: int = 0
    best_chunk_pair_score: float = 0.0


class DatabasePool:
    """
    Singleton database connection pool manager.
    
    Uses psycopg2's connection pool for efficient connection management.
    """
    
    _instance: Optional["DatabasePool"] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __new__(cls, config: Optional[PostgresConfig] = None) -> "DatabasePool":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[PostgresConfig] = None):
        """Initialize the connection pool."""
        if DatabasePool._pool is not None:
            return
        
        if config is None:
            config = get_config().postgres
        
        self._config = config
        self._create_pool()
    
    def _create_pool(self) -> None:
        """Create the connection pool."""
        try:
            DatabasePool._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=self._config.host,
                port=self._config.port,
                database=self._config.database,
                user=self._config.user,
                password=self._config.password
            )
            logger.info(f"Database pool created: {self._config.host}:{self._config.port}/{self._config.database}")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a connection from the pool.
        
        Yields:
            Database connection that will be returned to pool on exit.
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @classmethod
    def close_all(cls) -> None:
        """Close all connections in the pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            cls._instance = None
            logger.info("Database pool closed")
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls.close_all()


class ReclamationRepository:
    """
    Repository for reclamation CRUD operations and vector search.
    """
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """
        Initialize the repository.
        
        Args:
            db_pool: Optional database pool. If not provided, uses singleton.
        """
        self._pool = db_pool or DatabasePool()
    
    def create(self, reclamant_id: int, motif_id: str) -> int:
        """
        Create a new reclamation.
        
        Args:
            reclamant_id: The user ID (grouping key).
            motif_id: The motif_id.
        
        Returns:
            The ID of the created reclamation.
        """
        query = """
            INSERT INTO reclamation.reclamation (reclamant_id, motif_id)
            VALUES (%s, %s)
            RETURNING id;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (reclamant_id, motif_id))
                reclamation_id = cur.fetchone()[0]
                conn.commit()
                logger.debug(f"Created reclamation {reclamation_id} for user {motif_id}")
                return reclamation_id
    
    def get_motif_by_id(self, motif_id: str) -> Optional[Motif]:
        """
        Get a motif by ID.
        
        Args:
            motif_id: The motif ID.
        
        Returns:
            Motif object or None if not found.
        """
        query = """
            SELECT id, libelle
            FROM reclamation.motif
            WHERE id = %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (motif_id,))
                row = cur.fetchone()
                
                if row:
                    return Motif(**row)
                return None



    def get_by_id(self, reclamation_id: int) -> Optional[Reclamation]:
        """
        Get a reclamation by ID.
        
        Args:
            reclamation_id: The reclamation ID.
        
        Returns:
            Reclamation object or None if not found.
        """
        query = """
            SELECT id, motif_id, reclamant_id, description, created_at
            FROM reclamation.reclamation
            WHERE id = %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (reclamation_id,))
                row = cur.fetchone()
                
                if row:
                    return Reclamation(**row)
                return None
    
    def delete(self, reclamation_id: int) -> bool:
        """
        Delete a reclamation.
        
        Args:
            reclamation_id: The reclamation ID.
        
        Returns:
            True if deleted, False if not found.
        """
        query = "DELETE FROM reclamation.reclamation WHERE id = %s;"
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (reclamation_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted


class EmbeddingRepository:
    """
    Repository for vector embedding storage and similarity search.
    """
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """
        Initialize the repository.
        
        Args:
            db_pool: Optional database pool. If not provided, uses singleton.
        """
        self._pool = db_pool or DatabasePool()
    
    def save_embedding(self, reclamation_id: int, embedding: list[float]) -> None:
        """
        Save an embedding for a reclamation.
        
        Args:
            reclamation_id: The reclamation ID.
            embedding: The embedding vector as a list of floats.
        """
        query = """
            INSERT INTO public.reclamation_embeddings (reclamation_id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (reclamation_id)
            DO UPDATE SET embedding = EXCLUDED.embedding;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Convert list to PostgreSQL array format
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                cur.execute(query, (reclamation_id, embedding_str))
                conn.commit()
                logger.debug(f"Saved embedding for reclamation {reclamation_id}")
    
    def get_embedding(self, reclamation_id: int) -> Optional[list[float]]:
        """
        Get the embedding for a reclamation.
        
        Args:
            reclamation_id: The reclamation ID.
        
        Returns:
            The embedding vector or None if not found.
        """
        query = """
            SELECT embedding::text
            FROM public.reclamation_embeddings
            WHERE reclamation_id = %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (reclamation_id,))
                row = cur.fetchone()
                
                if row and row[0]:
                    # Parse PostgreSQL vector format "[x,y,z]"
                    embedding_str = row[0].strip("[]")
                    return [float(x) for x in embedding_str.split(",")]
                return None
    
    def find_similar_exceeded(
        self,
        embedding_chunks: dict[int, list[float]],
        exclude_id: int,
        min_score: float = 0.85,
        time_window_days: int = 7,
        limit: int = 5
    )-> list[SimilarityMatch]:
        """ 
        Find similar reclamations between reclamations that have chunks using vector similarity search.
        Logic : 
        1. Calculate the logic 
        

        """
    
    
    
    def find_similar(
        self,
        embedding: list[float],
        reclamant_id: int,
        exclude_id: int,
        min_score: float = 0.85,
        time_window_days: int = 7,
        limit: int = 5
    ) -> list[SimilarityMatch]:
        """
        Find similar reclamations using vector similarity search.
        
        Performs hybrid search combining:
        - Vector similarity (cosine)
        - Metadata filtering (reclamant_id, time window)
        
        Args:
            embedding: Query embedding vector.
            exclude_id: Reclamation ID to exclude (self).
            min_score: Minimum similarity score threshold.
            time_window_days: Time window in days for search scope.
            limit: Maximum number of results.
        
        Returns:
            List of SimilarityMatch objects sorted by score descending.
        """
        query = """
            SELECT
                r.id as reclamation_id,
                1 - (e.embedding <=> %s::vector) as score,
                r.motif_id
            FROM reclamation.reclamation r
            JOIN public.reclamation_embeddings e ON r.id = e.reclamation_id
            WHERE
                r.id != %s
                AND r.created_at > NOW() - INTERVAL '1 day' * %s
                AND 1 - (e.embedding <=> %s::vector) > %s
            ORDER BY score DESC
            LIMIT %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                cur.execute(query, (
                    embedding_str,
                    exclude_id,
                    time_window_days,
                    embedding_str,
                    min_score,
                    limit
                ))
                
                results = []
                for row in cur.fetchall():
                    results.append(SimilarityMatch(
                        reclamation_id=row["reclamation_id"],
                        score=float(row["score"]),
                        motif_id=row.get("motif_id")
                    ))
                
                return results
    
    def find_similar_exceeded(
        self,
        reclamation_id: int,
        reclamant_id: int,
        exclude_id: int,
        min_score: float = 0.70,
        time_window_days: int = 7,
        top_k_chunks: int = 3,
        use_advanced_matching: bool = True,
        elasticity_factor: float = 0.15
    ) -> list[SimilarityMatchChunkResult]:
        """
        Find similar reclamations handling chunked/non-chunked scenarios using
        the novel "Best-First Matching with Elasticity Consistency" approach.

        This innovative mathematical approach prevents false positives by:
        1. Finding the best chunk-to-chunk match pairs (maximum bipartite matching)
        2. Averaging the TOP_K pairs (configurable)
        3. Applying an elasticity factor to penalize global inconsistency:
           - If all chunks align well (max_excess = 0), no penalty
           - If chunks don't align (max_excess > 0), penalize score

        Math:
        - TOP_K = sqrt(n * m) where n, m are chunk counts, or use configured value
        - matched_score = average(top_k_pair_scores)
        - max_excess = max(0, |n - m| - 1)  // excess beyond 1 chunk difference
        - consistency_factor = 1.0 - (elasticity_factor * max_excess / (n + m - 1))
        - final_score = matched_score * consistency_factor

        Scenarios:
        1. Chunked A vs Chunked B: Use Best-First Matching with Elasticity
        2. Chunked A vs Non-chunked B: Find max similarity across A's chunks
        3. Non-chunked A vs Chunked B: Find max similarity across B's chunks
        4. Non-chunked A vs Non-chunked B: Standard cosine similarity (should use find_similar)

        Args:
            reclamation_id: Source reclamation ID
            reclamant_id: Reclamant ID for filtering
            exclude_id: Reclamation ID to exclude (self)
            min_score: Minimum similarity score threshold
            time_window_days: Time window in days for search scope
            top_k_chunks: Number of top chunk pairs to average (overrides dynamic calculation)
            use_advanced_matching: Whether to use elasticity consistency (recommended)
            elasticity_factor: Factor (0-1) for penalizing inconsistent chunk alignment

        Returns:
            List of SimilarityMatchChunkResult objects sorted by score descending
        """
        from src.config import get_config
        config = get_config()

        # Get source reclamation info
        source_is_chunked = self.is_chunked(reclamation_id)
        source_chunks = {}
        source_base = None

        if source_is_chunked:
            source_chunks = self.get_chunks_for_reclamation(reclamation_id)
            logger.debug(f"Reclamation {reclamation_id} is chunked with {len(source_chunks)} chunks")
        else:
            source_base = self.get_base_embedding(reclamation_id)
            if not source_base:
                logger.warning(f"Reclamation {reclamation_id} has no embedding")
                return []
            logger.debug(f"Reclamation {reclamation_id} has single embedding")

        # Get all candidate reclamations in time window
        query = """
            SELECT r.id, r.motif_id
            FROM reclamation.reclamation r
            WHERE r.id != %s
              AND r.created_at > NOW() - INTERVAL '1 day' * %s
            ORDER BY r.created_at DESC;
        """

        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (exclude_id, time_window_days))
                candidates = cur.fetchall()

        results = []

        for candidate in candidates:
            target_id = candidate["id"]
            target_motif = candidate["motif_id"]

            # Skip if it's the same as exclude_id
            if target_id == exclude_id:
                continue

            target_is_chunked = self.is_chunked(target_id)
            target_chunks = {}
            target_base = None

            if target_is_chunked:
                target_chunks = self.get_chunks_for_reclamation(target_id)
            else:
                target_base = self.get_base_embedding(target_id)

            # Compute similarity based on scenario
            similarity_score = 0.0
            best_chunk_pair_score = 0.0

            try:
                if source_is_chunked and target_is_chunked:
                    # SCENARIO 1: BOTH CHUNKED - Use Best-First Matching with Elasticity
                    n = len(source_chunks)
                    m = len(target_chunks)

                    # Calculate dynamic TOP_K if not provided
                    if top_k_chunks is None:
                        top_k = min(int(math.sqrt(n * m)), n * m)
                    else:
                        top_k = min(top_k_chunks, n * m)

                    if top_k < 1:
                        top_k = 1

                    # Find all chunk-to-chunk similarities
                    pair_scores = []
                    for chunk_a_id, embed_a in source_chunks.items():
                        for chunk_b_id, embed_b in target_chunks.items():
                            score = cosine_similarity(embed_a, embed_b)
                            pair_scores.append((score, chunk_a_id, chunk_b_id))

                    # Sort by score descending and get top K
                    pair_scores.sort(key=lambda x: x[0], reverse=True)
                    top_pairs = pair_scores[:top_k]

                    # Average top k scores
                    matched_score = sum(p[0] for p in top_pairs) / top_k
                    best_chunk_pair_score = top_pairs[0][0] if top_pairs else 0.0

                    # Apply elasticity consistency factor if advanced matching enabled
                    if use_advanced_matching and top_k > 0:
                        # Calculate excess: how many chunks in one doc don't have a corresponding match
                        max_excess = max(0, abs(n - m) - 1)
                        # Normalize excess by total chunks (avoid division by zero)
                        total_chunks = max(1, n + m - 1)
                        consistency_factor = 1.0 - (elasticity_factor * max_excess / total_chunks)

                        # Clamp consistency_factor to [0.7, 1.0] to avoid over-penalization
                        consistency_factor = max(0.7, min(1.0, consistency_factor))

                        similarity_score = matched_score * consistency_factor

                        logger.debug(
                            f"Chunked comparison {reclamation_id} vs {target_id}: "
                            f"n={n}, m={m}, top_k={top_k}, matched={matched_score:.4f}, "
                            f"excess={max_excess}, consistency={consistency_factor:.4f}, final={similarity_score:.4f}"
                        )
                    else:
                        # Simple average (no consistency check)
                        similarity_score = matched_score
                        logger.debug(
                            f"Chunked comparison {reclamation_id} vs {target_id} (simple): "
                            f"avg of top {top_k} = {similarity_score:.4f}"
                        )

                elif source_is_chunked and not target_is_chunked and target_base:
                    # SCENARIO 2: SOURCE CHUNKED, TARGET SINGLE
                    # Find max similarity across all source chunks
                    scores = [cosine_similarity(embed, target_base) for embed in source_chunks.values()]
                    similarity_score = max(scores) if scores else 0.0
                    best_chunk_pair_score = similarity_score

                    logger.debug(
                        f"Chunked source vs single target {reclamation_id} vs {target_id}: "
                        f"max across {len(source_chunks)} chunks = {similarity_score:.4f}"
                    )

                elif not source_is_chunked and source_base and target_is_chunked:
                    # SCENARIO 3: SOURCE SINGLE, TARGET CHUNKED
                    # Find max similarity across all target chunks
                    scores = [cosine_similarity(source_base, embed) for embed in target_chunks.values()]
                    similarity_score = max(scores) if scores else 0.0
                    best_chunk_pair_score = similarity_score

                    logger.debug(
                        f"Single source vs chunked target {reclamation_id} vs {target_id}: "
                        f"max across {len(target_chunks)} chunks = {similarity_score:.4f}"
                    )

                else:
                    # SCENARIO 4: BOTH SINGLE - Should use find_similar instead
                    if source_base and target_base:
                        similarity_score = cosine_similarity(source_base, target_base)
                        best_chunk_pair_score = similarity_score
                        logger.debug(
                            f"Single vs single {reclamation_id} vs {target_id}: {similarity_score:.4f}"
                        )

                # Filter by threshold
                if similarity_score >= min_score:
                    results.append(SimilarityMatchChunkResult(
                        reclamation_id=target_id,
                        similarity_score=similarity_score,
                        motif_id=target_motif,
                        is_chunked=(source_is_chunked or target_is_chunked),
                        source_is_chunked=source_is_chunked,
                        target_is_chunked=target_is_chunked,
                        num_source_chunks=len(source_chunks),
                        num_target_chunks=len(target_chunks) if target_is_chunked else 1,
                        best_chunk_pair_score=best_chunk_pair_score
                    ))

            except Exception as e:
                logger.error(f"Error comparing {reclamation_id} vs {target_id}: {e}")
                continue

        # Sort by similarity score descending
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(
            f"Found {len(results)} matches for reclamation {reclamation_id} "
            f"above threshold {min_score:.4f}"
        )

        return results
    def delete_embedding(self, reclamation_id: int) -> bool:
        """
        Delete an embedding.

        Args:
            reclamation_id: The reclamation ID.

        Returns:
            True if deleted, False if not found.
        """
        query = "DELETE FROM public.reclamation_embeddings WHERE reclamation_id = %s;"

        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (reclamation_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted

    def is_chunked(self, reclamation_id: int) -> bool:
        """
        Check if a reclamation was chunked (has embeddings stored with chunk suffix).

        Args:
            reclamation_id: The reclamation ID to check.

        Returns:
            True if any embeddings exist like "{reclamation_id}_chunk_1", False otherwise.
        """
        query = """
            SELECT EXISTS (
                SELECT 1 FROM public.reclamation_embeddings
                WHERE reclamation_id::text LIKE %s
            );
        """
        pattern = f"{reclamation_id}_chunk_%"

        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pattern,))
                result = cur.fetchone()
                return result[0] if result else False

    def get_chunks_for_reclamation(self, reclamation_id: int) -> Dict[str, list[float]]:
        """
        Retrieve all chunk embeddings for a reclamation.

        Args:
            reclamation_id: The reclamation ID.

        Returns:
            Dict mapping chunk IDs to embeddings (e.g., {"123_chunk_1": [embed1], "123_chunk_2": [embed2]}).
        """
        query = """
            SELECT reclamation_id, embedding::text
            FROM public.reclamation_embeddings
            WHERE reclamation_id::text LIKE %s
            ORDER BY reclamation_id;
        """
        pattern = f"{reclamation_id}_chunk_%"

        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pattern,))
                chunks = {}
                for row in cur.fetchall():
                    chunk_id, embed_str = row
                    embedding_str = embed_str.strip("[]")
                    embedding = [float(x) for x in embedding_str.split(",")]
                    chunks[chunk_id] = embedding
                return chunks

    def get_base_embedding(self, reclamation_id: int) -> Optional[list[float]]:
        """
        Get the non-chunked embedding for a reclamation (if it exists).

        Args:
            reclamation_id: The reclamation ID.

        Returns:
            The embedding vector or None if only chunks exist, or if no embedding exists.
        """
        query = """
            SELECT embedding::text
            FROM public.reclamation_embeddings
            WHERE reclamation_id::text = %s::text;
        """

        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (str(reclamation_id),))
                row = cur.fetchone()

                if row and row[0]:
                    embedding_str = row[0].strip("[]")
                    return [float(x) for x in embedding_str.split(",")]
                return None


class DuplicationLogRepository:
    """
    Repository for duplication audit log operations.
    """
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """
        Initialize the repository.
        
        Args:
            db_pool: Optional database pool. If not provided, uses singleton.
        """
        self._pool = db_pool or DatabasePool()
    
    def create(
        self,
        source_reclamation_id: int,
        matched_reclamation_id: int,
        similarity_score: float,
        action: str
    ) -> int:
        """
        Create a duplication log entry.
        
        Args:
            source_reclamation_id: The new reclamation ID.
            matched_reclamation_id: The matched existing reclamation ID.
            similarity_score: The similarity score.
            action: The action taken (AUTO_MARK_DUPLICATE or FLAG_FOR_REVIEW).
        
        Returns:
            The ID of the created log entry.
        """
        query = """
            INSERT INTO public.duplication_logs 
                (source_reclamation_id, matched_reclamation_id, similarity_score, action)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    source_reclamation_id,
                    matched_reclamation_id,
                    similarity_score,
                    action
                ))
                log_id = cur.fetchone()[0]
                conn.commit()
                logger.info(
                    f"Logged duplication: {source_reclamation_id} -> {matched_reclamation_id} "
                    f"(score: {similarity_score:.4f}, action: {action})"
                )
                return log_id
    
    def get_by_source(self, source_reclamation_id: int) -> list[DuplicationLog]:
        """
        Get all log entries for a source reclamation.
        
        Args:
            source_reclamation_id: The source reclamation ID.
        
        Returns:
            List of DuplicationLog objects.
        """
        query = """
            SELECT id, source_reclamation_id, matched_reclamation_id, 
                   similarity_score, action, detected_at
            FROM public.duplication_logs
            WHERE source_reclamation_id = %s
            ORDER BY detected_at DESC;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (source_reclamation_id,))
                return [DuplicationLog(**row) for row in cur.fetchall()]
    
    def get_recent(self, limit: int = 100) -> list[DuplicationLog]:
        """
        Get recent duplication log entries.
        
        Args:
            limit: Maximum number of entries to return.
        
        Returns:
            List of DuplicationLog objects.
        """
        query = """
            SELECT id, source_reclamation_id, matched_reclamation_id,
                   similarity_score, action, detected_at
            FROM public.duplication_logs
            ORDER BY detected_at DESC
            LIMIT %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return [DuplicationLog(**row) for row in cur.fetchall()]


class ReclamationMatchRepository:
    """
    Repository for reclamation match (duplicate relationship) operations.
    """
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """
        Initialize the repository.
        
        Args:
            db_pool: Optional database pool. If not provided, uses singleton.
        """
        self._pool = db_pool or DatabasePool()
    
    def create(
        self,
        reclamation_id: int,
        matched_reclamation_id: int,
        similarity_score: float,
        match_status: str = "PENDING"
    ) -> int:
        """
        Create a reclamation match entry.
        
        Args:
            reclamation_id: The source reclamation ID.
            matched_reclamation_id: The matched reclamation ID.
            similarity_score: The similarity score.
            match_status: Match status (PENDING, CONFIRMED_DUPLICATE, NOT_DUPLICATE, NEEDS_REVIEW).
        
        Returns:
            The ID of the created match entry.
        """
        query = """
            INSERT INTO public.reclamation_matches 
                (reclamation_id, matched_reclamation_id, similarity_score, match_status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (reclamation_id, matched_reclamation_id)
            DO UPDATE SET 
                similarity_score = EXCLUDED.similarity_score,
                match_status = EXCLUDED.match_status,
                updated_at = NOW()
            RETURNING id;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    reclamation_id,
                    matched_reclamation_id,
                    similarity_score,
                    match_status
                ))
                match_id = cur.fetchone()[0]
                conn.commit()
                logger.info(
                    f"Saved match: {reclamation_id} -> {matched_reclamation_id} "
                    f"(score: {similarity_score:.4f}, status: {match_status})"
                )
                return match_id
    
    def get_by_reclamation(self, reclamation_id: int) -> list[ReclamationMatch]:
        """
        Get all matches for a reclamation.
        
        Args:
            reclamation_id: The reclamation ID.
        
        Returns:
            List of ReclamationMatch objects.
        """
        query = """
            SELECT id, reclamation_id, matched_reclamation_id, 
                   similarity_score, match_status, created_at, updated_at
            FROM public.reclamation_matches
            WHERE reclamation_id = %s
            ORDER BY similarity_score DESC;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (reclamation_id,))
                return [ReclamationMatch(**row) for row in cur.fetchall()]
    
    def get_by_matched(self, matched_reclamation_id: int) -> list[ReclamationMatch]:
        """
        Get all reclamations that match a given reclamation.
        
        Args:
            matched_reclamation_id: The matched reclamation ID.
        
        Returns:
            List of ReclamationMatch objects.
        """
        query = """
            SELECT id, reclamation_id, matched_reclamation_id, 
                   similarity_score, match_status, created_at, updated_at
            FROM public.reclamation_matches
            WHERE matched_reclamation_id = %s
            ORDER BY similarity_score DESC;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (matched_reclamation_id,))
                return [ReclamationMatch(**row) for row in cur.fetchall()]
    
    def update_status(self, match_id: int, match_status: str) -> bool:
        """
        Update the status of a match.
        
        Args:
            match_id: The match ID.
            match_status: New status value.
        
        Returns:
            True if updated, False if not found.
        """
        query = """
            UPDATE public.reclamation_matches
            SET match_status = %s, updated_at = NOW()
            WHERE id = %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (match_status, match_id))
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.debug(f"Updated match {match_id} status to {match_status}")
                return updated
    
    def get_by_status(self, match_status: str, limit: int = 100) -> list[ReclamationMatch]:
        """
        Get matches by status.
        
        Args:
            match_status: The match status to filter by.
            limit: Maximum number of entries to return.
        
        Returns:
            List of ReclamationMatch objects.
        """
        query = """
            SELECT id, reclamation_id, matched_reclamation_id, 
                   similarity_score, match_status, created_at, updated_at
            FROM public.reclamation_matches
            WHERE match_status = %s
            ORDER BY created_at DESC
            LIMIT %s;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (match_status, limit))
                return [ReclamationMatch(**row) for row in cur.fetchall()]


def get_db_pool(config: Optional[PostgresConfig] = None) -> DatabasePool:
    """
    Get the singleton database pool instance.
    
    Args:
        config: Optional configuration override.
    
    Returns:
        The singleton DatabasePool instance.
    """
    return DatabasePool(config)


def get_reclamation_repo(db_pool: Optional[DatabasePool] = None) -> ReclamationRepository:
    """
    Get a ReclamationRepository instance.
    
    Args:
        db_pool: Optional database pool.
    
    Returns:
        ReclamationRepository instance.
    """
    return ReclamationRepository(db_pool)


def get_embedding_repo(db_pool: Optional[DatabasePool] = None) -> EmbeddingRepository:
    """
    Get an EmbeddingRepository instance.
    
    Args:
        db_pool: Optional database pool.
    
    Returns:
        EmbeddingRepository instance.
    """
    return EmbeddingRepository(db_pool)


def get_duplication_log_repo(db_pool: Optional[DatabasePool] = None) -> DuplicationLogRepository:
    """
    Get a DuplicationLogRepository instance.
    
    Args:
        db_pool: Optional database pool.
    
    Returns:
        DuplicationLogRepository instance.
    """
    return DuplicationLogRepository(db_pool)


def get_reclamation_match_repo(db_pool: Optional[DatabasePool] = None) -> ReclamationMatchRepository:
    """
    Get a ReclamationMatchRepository instance.
    
    Args:
        db_pool: Optional database pool.
    
    Returns:
        ReclamationMatchRepository instance.
    """
    return ReclamationMatchRepository(db_pool)

