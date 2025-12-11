#!/usr/bin/env python
"""
Reclamation Processing Script for the De-duplication Pipeline.

Processes all unprocessed reclamations by generating embeddings.
Skips already processed records and provides progress reporting.

Usage:
    python scripts/process_reclamations.py              # Process all
    python scripts/process_reclamations.py --dry-run   # Show what would be processed
    python scripts/process_reclamations.py --limit 100  # Process max 100 records
    python scripts/process_reclamations.py --batch-size 25  # Custom batch size
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.database import DatabasePool, EmbeddingRepository, ReclamationMatchRepository, DuplicationLogRepository
from src.preprocessing import normalize_text
from src.embeddings import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReclamationProcessor:
    """Handles processing of unprocessed reclamations."""
    
    def __init__(
        self,
        batch_size: int = 50,
        limit: Optional[int] = None,
        dry_run: bool = False
    ):
        self.batch_size = batch_size
        self.limit = limit
        self.dry_run = dry_run
        self._pool = None
        self._embedding_repo = None
        self._match_repo = None
        self._log_repo = None
        self._model = None
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
    
    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info("Loading LaBSE embedding model...")
            self._model = get_embedding_model()
        return self._model
    
    def check_connection(self) -> bool:
        """Test database connectivity."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def check_table_exists(self, table_name: str, schema: str = "public") -> bool:
        """Check if a table exists."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_schema = %s AND table_name = %s
                        );
                    """, (schema, table_name))
                    return cur.fetchone()[0]
        except Exception:
            return False
    
    def check_pgvector(self) -> bool:
        """Check if pgvector extension is installed."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        );
                    """)
                    return cur.fetchone()[0]
        except Exception:
            return False
    
    def get_reclamation_count(self) -> int:
        """Get total number of reclamations."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM reclamation.reclamation;")
                    return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count reclamations: {e}")
            return 0
    
    def get_unprocessed_reclamations(self) -> list[dict]:
        """Get all reclamations without embeddings."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT r.id, r.reclamant_id, r.description, r.motif_id
                        FROM reclamation.reclamation r
                        LEFT JOIN public.reclamation_embeddings e ON r.id = e.reclamation_id
                        WHERE e.reclamation_id IS NULL
                        ORDER BY r.id
                    """
                    if self.limit:
                        query += f" LIMIT {self.limit}"
                    
                    cur.execute(query)
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get unprocessed reclamations: {e}")
            return []
    
    def get_motif_libelle(self, motif_id: str) -> Optional[str]:
        """Get the libelle for a motif ID."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT libelle FROM reclamation.motif WHERE id = %s;",
                        (motif_id,)
                    )
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception:
            return None
    
    def run_health_checks(self) -> bool:
        """Run all health checks and return overall status."""
        print("\n" + "=" * 60)
        print("PROCESSING HEALTH CHECK")
        print("=" * 60)
        
        config = get_config().postgres
        print(f"\nTarget: {config.host}:{config.port}/{config.database}")
        
        all_ok = True
        
        # Connection check
        print("\n[1] Testing database connection...", end=" ")
        if self.check_connection():
            print("✓ OK")
        else:
            print("✗ FAILED")
            return False
        
        # pgvector check
        print("[2] Checking pgvector extension...", end=" ")
        if self.check_pgvector():
            print("✓ Installed")
        else:
            print("✗ Not installed")
            all_ok = False
        
        # reclamation table check (in reclamation schema)
        print("[3] Checking reclamation.reclamation table...", end=" ")
        if self.check_table_exists("reclamation", "reclamation"):
            print("✓ Exists")
        else:
            print("✗ Missing")
            all_ok = False
        
        # reclamation_embeddings table check (in public schema)
        print("[4] Checking public.reclamation_embeddings table...", end=" ")
        if self.check_table_exists("reclamation_embeddings", "public"):
            print("✓ Exists")
        else:
            print("✗ Missing")
            all_ok = False
        
        # Data check
        print("[5] Checking reclamations data...", end=" ")
        count = self.get_reclamation_count()
        if count > 0:
            print(f"✓ {count} records found")
        else:
            print("⚠ No reclamations found")
            all_ok = False
        
        return all_ok
    
    def find_duplicates_for_record(self, record: dict, embedding: list[float]) -> list[dict]:
        """
        Find duplicate matches for a record.
        
        Returns:
            List of match dicts with matched_id, score, and status.
        """
        reclamant_id = record["reclamant_id"]
        reclamation_id = record["id"]
        embedding_repo = self._get_embedding_repo()
        
        # Search for similar reclamations
        min_score = self._config.threshold_review
        matches = embedding_repo.find_similar(
            embedding=embedding,
            reclamant_id=reclamant_id,
            exclude_id=reclamation_id,
            min_score=min_score,
            time_window_days=self._config.time_window_days,
            limit=5  # Get top 5 matches
        )
        
        result = []
        for match in matches:
            # Determine status based on score
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
    
    def process_batch(self, records: list[dict]) -> tuple[int, int, int]:
        """
        Process a batch of records.
        
        Returns:
            Tuple of (success_count, error_count, match_count)
        """
        model = self._get_model()
        embedding_repo = self._get_embedding_repo()
        match_repo = self._get_match_repo()
        log_repo = self._get_log_repo()
        
        success = 0
        errors = 0
        match_count = 0
        
        for record in records:
            try:
                rec_id = record["id"]
                motif_id = record.get("motif_id")
                description = record.get("description") or ""
                
                # Get motif libelle if available
                libelle = ""
                if motif_id:
                    libelle = self.get_motif_libelle(motif_id) or ""
                
                # Combine text for embedding
                text = f"{libelle} {description}".strip()
                
                if not text:
                    logger.warning(f"Reclamation {rec_id} has no text content, skipping")
                    errors += 1
                    continue
                
                # Normalize and encode
                normalized = normalize_text(text)
                embedding = model.encode_to_list(normalized)
                
                # Save embedding
                embedding_repo.save_embedding(rec_id, embedding)
                
                # Find and save duplicates
                duplicates = self.find_duplicates_for_record(record, embedding)
                for dup in duplicates:
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
                    match_count += 1
                
                success += 1
                
            except Exception as e:
                logger.error(f"Failed to process reclamation {record.get('id')}: {e}")
                errors += 1
        
        return success, errors, match_count
    
    def process(self) -> bool:
        """Run the complete processing pipeline."""
        # Run health checks
        if not self.run_health_checks():
            print("\n❌ Health checks failed. Please run db_setup.py first.")
            return False
        
        # Get unprocessed records
        print("\n" + "-" * 60)
        print("FINDING UNPROCESSED RECORDS")
        print("-" * 60)
        
        unprocessed = self.get_unprocessed_reclamations()
        total = len(unprocessed)
        
        if total == 0:
            print("\n✅ All reclamations already have embeddings. Nothing to process.")
            return True
        
        print(f"\nFound {total} unprocessed reclamation(s)")
        if self.limit:
            print(f"  (limited to {self.limit} by --limit flag)")
        
        if self.dry_run:
            print("\n[dry-run mode] Would process the following:")
            for i, rec in enumerate(unprocessed[:10]):
                desc = (rec.get("description") or "")[:50]
                print(f"  • ID {rec['id']}: {desc}...")
            if total > 10:
                print(f"  ... and {total - 10} more")
            return True
        
        # Process in batches
        print("\n" + "-" * 60)
        print("PROCESSING RECLAMATIONS")
        print("-" * 60)
        
        total_success = 0
        total_errors = 0
        total_matches = 0
        
        for i in range(0, total, self.batch_size):
            batch = unprocessed[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} records)...", end=" ")
            
            success, errors, matches = self.process_batch(batch)
            total_success += success
            total_errors += errors
            total_matches += matches
            
            print(f"✓ {success} success, {errors} errors, {matches} matches")
            
            # Progress update
            processed = min(i + self.batch_size, total)
            percent = (processed / total) * 100
            print(f"  Progress: {processed}/{total} ({percent:.1f}%)")
        
        # Summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\n✓ Successfully processed: {total_success}")
        print(f"✓ Duplicate matches found: {total_matches}")
        if total_errors > 0:
            print(f"✗ Failed: {total_errors}")
        
        return total_errors == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process unprocessed reclamations and generate embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of records to process per batch (default: 50)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
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
        processor = ReclamationProcessor(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run
        )
        success = processor.process()
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
