#!/usr/bin/env python
"""
Find Duplicates Script for the De-duplication Pipeline.

Finds duplicates among reclamations that already have embeddings
and populates the reclamation_matches table.

Usage:
    python scripts/find_duplicates.py              # Find all duplicates
    python scripts/find_duplicates.py --dry-run   # Show what would be found
    python scripts/find_duplicates.py --limit 100  # Process max 100 records
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.database import (
    DatabasePool,
    EmbeddingRepository,
    ReclamationMatchRepository,
    DuplicationLogRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuplicateFinder:
    """Finds duplicates among existing embeddings."""
    
    def __init__(
        self,
        limit: Optional[int] = None,
        dry_run: bool = False
    ):
        self.limit = limit
        self.dry_run = dry_run
        self._pool = None
        self._embedding_repo = None
        self._match_repo = None
        self._log_repo = None
        self._config = get_config().detection
    
    def _get_pool(self) -> DatabasePool:
        """Get or create database pool."""
        if self._pool is None:
            self._pool = DatabasePool()
        return self._pool
    
    def _get_embedding_repo(self) -> EmbeddingRepository:
        """Get or create embedding repository."""
        if self._embedding_repo is None:
            self._embedding_repo = EmbeddingRepository(self._get_pool())
        return self._embedding_repo
    
    def _get_match_repo(self) -> ReclamationMatchRepository:
        """Get or create match repository."""
        if self._match_repo is None:
            self._match_repo = ReclamationMatchRepository(self._get_pool())
        return self._match_repo
    
    def _get_log_repo(self) -> DuplicationLogRepository:
        """Get or create duplication log repository."""
        if self._log_repo is None:
            self._log_repo = DuplicationLogRepository(self._get_pool())
        return self._log_repo
    
    def get_reclamations_with_embeddings(self) -> list[dict]:
        """Get all reclamations that have embeddings."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT r.id, r.reclamant_id, e.embedding::text
                        FROM reclamation.reclamation r
                        JOIN public.reclamation_embeddings e ON r.id = e.reclamation_id
                        ORDER BY r.id
                    """
                    if self.limit:
                        query += f" LIMIT {self.limit}"
                    
                    cur.execute(query)
                    results = []
                    for row in cur.fetchall():
                        # Parse embedding
                        embedding_str = row[2].strip("[]")
                        embedding = [float(x) for x in embedding_str.split(",")]
                        results.append({
                            "id": row[0],
                            "reclamant_id": row[1],
                            "embedding": embedding
                        })
                    return results
        except Exception as e:
            logger.error(f"Failed to get reclamations with embeddings: {e}")
            return []
    
    def get_existing_matches(self) -> set[tuple[int, int]]:
        """Get set of existing match pairs to avoid duplicates."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT reclamation_id, matched_reclamation_id
                        FROM public.reclamation_matches
                    """)
                    return {(row[0], row[1]) for row in cur.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get existing matches: {e}")
            return set()
    
    def find_duplicates_for_record(self, record: dict) -> list[dict]:
        """Find duplicate matches for a record using existing embeddings."""
        embedding_repo = self._get_embedding_repo()
        
        matches = embedding_repo.find_similar(
            embedding=record["embedding"],
            reclamant_id=record["reclamant_id"],
            exclude_id=record["id"],
            min_score=self._config.threshold_review,
            time_window_days=self._config.time_window_days,
            limit=5
        )
        
        result = []
        for match in matches:
            if match.score >= self._config.threshold_auto_duplicate:
                status = "CONFIRMED_DUPLICATE"
                action = "AUTO_MARK_DUPLICATE"
            else:
                status = "NEEDS_REVIEW"
                action = "FLAG_FOR_REVIEW"
            
            result.append({
                "matched_id": match.reclamation_id,
                "score": match.score,
                "status": status,
                "action": action
            })
        
        return result
    
    def run_health_checks(self) -> bool:
        """Run health checks before processing."""
        print("\n" + "=" * 60)
        print("DUPLICATE FINDER HEALTH CHECK")
        print("=" * 60)
        
        config = get_config().postgres
        print(f"\nTarget: {config.host}:{config.port}/{config.database}")
        
        # Check connection
        print("\n[1] Testing database connection...", end=" ")
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    print("✓ OK")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False
        
        # Check embeddings count
        print("[2] Checking embeddings...", end=" ")
        records = self.get_reclamations_with_embeddings()
        print(f"✓ {len(records)} records with embeddings")
        
        # Check existing matches
        print("[3] Checking existing matches...", end=" ")
        existing = self.get_existing_matches()
        print(f"✓ {len(existing)} existing matches")
        
        # Show thresholds
        print(f"\n[Config] Thresholds: auto={self._config.threshold_auto_duplicate}, review={self._config.threshold_review}")
        print(f"[Config] Time window: {self._config.time_window_days} days")
        
        return len(records) > 0
    
    def process(self) -> bool:
        """Run the duplicate finding process."""
        if not self.run_health_checks():
            print("\n❌ Health checks failed or no embeddings found.")
            return False
        
        records = self.get_reclamations_with_embeddings()
        existing_matches = self.get_existing_matches()
        total = len(records)
        
        print("\n" + "-" * 60)
        print("FINDING DUPLICATES")
        print("-" * 60)
        
        if self.dry_run:
            print("\n[dry-run mode] Will analyze without saving.")
        
        match_repo = self._get_match_repo()
        log_repo = self._get_log_repo()
        
        total_matches = 0
        new_matches = 0
        skipped = 0
        
        for i, record in enumerate(records):
            rec_id = record["id"]
            
            # Find duplicates
            duplicates = self.find_duplicates_for_record(record)
            
            for dup in duplicates:
                total_matches += 1
                
                # Check if already exists
                pair = (rec_id, dup["matched_id"])
                if pair in existing_matches:
                    skipped += 1
                    continue
                
                if self.dry_run:
                    print(f"  Would save: {rec_id} -> {dup['matched_id']} (score={dup['score']:.4f}, {dup['status']})")
                else:
                    # Save to reclamation_matches
                    match_repo.create(
                        reclamation_id=rec_id,
                        matched_reclamation_id=dup["matched_id"],
                        similarity_score=dup["score"],
                        match_status=dup["status"]
                    )
                    # Save to duplication_logs
                    log_repo.create(
                        source_reclamation_id=rec_id,
                        matched_reclamation_id=dup["matched_id"],
                        similarity_score=dup["score"],
                        action=dup["action"]
                    )
                    existing_matches.add(pair)
                
                new_matches += 1
            
            # Progress
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  Progress: {i + 1}/{total} ({((i + 1) / total) * 100:.1f}%)")
        
        # Summary
        print("\n" + "=" * 60)
        print("DUPLICATE FINDING COMPLETE")
        print("=" * 60)
        print(f"\n✓ Records analyzed: {total}")
        print(f"✓ Total duplicate pairs found: {total_matches}")
        print(f"✓ New matches saved: {new_matches}")
        if skipped > 0:
            print(f"⚠ Skipped (already exists): {skipped}")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find duplicates among existing embeddings"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to analyze"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be found without saving"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        finder = DuplicateFinder(
            limit=args.limit,
            dry_run=args.dry_run
        )
        success = finder.process()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=args.verbose)
        sys.exit(1)
    finally:
        DatabasePool.reset()


if __name__ == "__main__":
    main()
