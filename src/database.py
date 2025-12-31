"""
Database layer for the De-duplication Pipeline.

Provides connection pooling, CRUD operations for reclamations,
and vector similarity search using pgvector.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Generator
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

from src.config import get_config, PostgresConfig


logger = logging.getLogger(__name__)


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

