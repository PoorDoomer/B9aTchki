#!/usr/bin/env python
"""
Backward migration script to reprocess existing reclamations.

This script processes existing reclamations to:
1. Create embeddings for reclamations that don't have any
2. Optionally, recreate embeddings for all reclamations to apply the new chunking logic

Usage:
    python scripts/reprocess_existing_reclamations.py --help
    python scripts/reprocess_existing_reclamations.py --all
    python scripts/reprocess_existing_reclamations.py --missing-only
    python scripts/reprocess_existing_reclamations.py --reclamation-id 123
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.database import (
    DatabasePool,
    ReclamationRepository,
    EmbeddingRepository,
)
from src.preprocessing import normalize_text
from src.chunking import Chunker
from src.embeddings import get_embedding_model
from src.sentences import TextCleaner, sentences, count_tokens


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReclamationsReprocessor:
    """Handler for reprocessing existing reclamations."""
    
    def __init__(self):
        """Initialize the processor."""
        self._config = get_config()
        self._pool = DatabasePool()
        self._reclamation_repo = ReclamationRepository(self._pool)
        self._embedding_repo = EmbeddingRepository(self._pool)
        self._ml_config = self._config.ml
        self._chunker = Chunker()
        self._cleaner = TextCleaner(sentences)
    
    def get_reclamations_without_embeddings(self) -> list:
        """
        Get all reclamations that don't have embeddings.
        
        Returns:
            List of reclamation records
        """
        query = """
            SELECT r.id, r.description, r.motif_id, r.reclamant_id, r.created_at
            FROM reclamation.reclamation r
            WHERE r.id NOT IN (
                SELECT DISTINCT CAST(reclamation_id::text AS INTEGER)
                FROM public.reclamation_embeddings
                WHERE reclamation_id::text ~ '^[0-9]+$'
            )
            ORDER BY r.created_at DESC;
        """
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    
    def get_all_reclamations(self, limit: Optional[int] = None) -> list:
        """
        Get all reclamations.
        
        Args:
            limit: Optional limit on number of results
        
        Returns:
            List of reclamation records
        """
        query = """
            SELECT r.id, r.description, r.motif_id, r.reclamant_id, r.created_at
            FROM reclamation.reclamation r
            WHERE r.description IS NOT NULL AND r.description != ''
            ORDER BY r.created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit};"
        else:
            query += ";"
        
        with self._pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    
    def process_reclamation(self, reclamation_id: int) -> dict:
        """
        Process a single reclamation for duplicate detection.
        
        Args:
            reclamation_id: The reclamation ID to process
        
        Returns:
            Dict with processing results
        """
        result = {
            'reclamation_id': reclamation_id,
            'success': False,
            'is_chunked': False,
            'num_chunks': 0,
            'error': None
        }
        
        try:
            # Get reclamation
            reclamation = self._reclamation_repo.get_by_id(reclamation_id)
            if not reclamation:
                result['error'] = "Reclamation not found"
                return result
            
            logger.info(f"Processing reclamation {reclamation_id}")
            
            # Step 1: Preprocess the text
            normalized_text = normalize_text(reclamation.description)
            if not normalized_text:
                result['error'] = "Empty text after normalization"
                return result
            
            # Step 2: Clean text
            number_of_tokens = count_tokens(normalized_text)
            logger.debug(f"Token count before cleaning: {number_of_tokens}")
            
            normalized_text = self._cleaner.remove_sentences(normalized_text)
            number_of_tokens = count_tokens(normalized_text)
            logger.debug(f"Token count after cleaning: {number_of_tokens}")
            
            # Step 3: Check if needs chunking
            if number_of_tokens > self._ml_config.max_seq_tokens:
                logger.info(f"Reclamation {reclamation_id} exceeds max token limit, chunking ({number_of_tokens} > {self._ml_config.max_seq_tokens})")
                result['is_chunked'] = True
                
                # Chunk the text
                chunks = self._chunker.chunk_text(normalized_text)
                result['num_chunks'] = len(chunks)
                
                # Generate and save embeddings for each chunk
                model = get_embedding_model()
                for i, chunk in enumerate(chunks, 1):
                    logger.debug(f"Reclamation {reclamation_id}: Chunk {i} token count: {count_tokens(chunk)}")
                    embedding = model.encode_to_list(chunk)
                    self._embedding_repo.save_embedding(f"{reclamation_id}_chunk_{i}", embedding)
                
                logger.info(f"Reclamation {reclamation_id}: Processed {len(chunks)} chunks")
                
            else:
                # Generate single embedding
                model = get_embedding_model()
                embedding = model.encode_to_list(normalized_text)
                self._embedding_repo.save_embedding(reclamation_id, embedding)
                logger.info(f"Reclamation {reclamation_id}: Saved single embedding (tokens: {number_of_tokens})")
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error processing reclamation {reclamation_id}: {e}", exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def batch_process(self, reclamation_ids: list[int]) -> list[dict]:
        """
        Process multiple reclamations.
        
        Args:
            reclamation_ids: List of reclamation IDs to process
        
        Returns:
            List of processing results
        """
        results = []
        total = len(reclamation_ids)
        
        logger.info(f"Starting batch processing of {total} reclamations")
        
        for i, rec_id in enumerate(reclamation_ids, 1):
            logger.info(f"Progress: {i}/{total}")
            result = self.process_reclamation(rec_id)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failures = total - successful
        
        logger.info(f"Batch processing complete:")
        logger.info(f"  Total: {total}")
        logger.info(f"  Success: {successful}")
        logger.info(f"  Failed: {failures}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reprocess existing reclamations for duplicate detection"
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Reprocess all reclamations (will recreate embeddings)'
    )
    parser.add_argument(
        '--missing-only', '-m',
        action='store_true',
        help='Only process reclamations without embeddings (default)'
    )
    parser.add_argument(
        '--reclamation-id', '-r',
        type=int,
        help='Process a specific reclamation ID'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit number of reclamations to process (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        processor = ReclamationsReprocessor()
        
        # Determine which reclamations to process
        if args.reclamation_id:
            # Process single reclamation
            logger.info(f"Processing single reclamation: {args.reclamation_id}")
            results = [processor.process_reclamation(args.reclamation_id)]
            
        elif args.all:
            # Reprocess all reclamations
            reclamations = processor.get_all_reclamations(limit=args.limit)
            reclamation_ids = [r[0] for r in reclamations]
            
            if args.dry_run:
                logger.info(f"DRY RUN: Would process {len(reclamation_ids)} reclamations")
                for rec_id in reclamation_ids[:5]:  # Show first 5
                    logger.info(f"  - Reclamation {rec_id}")
                if len(reclamation_ids) > 5:
                    logger.info(f"  ... and {len(reclamation_ids) - 5} more")
            else:
                results = processor.batch_process(reclamation_ids)
            
        else:
            # Default: process only reclamations without embeddings
            reclamations = processor.get_reclamations_without_embeddings()
            reclamation_ids = [r[0] for r in reclamations]
            
            if not reclamation_ids:
                logger.info("No reclamations without embeddings found")
                sys.exit(0)
            
            logger.info(f"Found {len(reclamation_ids)} reclamations without embeddings")
            
            if args.dry_run:
                logger.info("DRY RUN: Would process the following reclamations:")
                for rec_id in reclamation_ids[:10]:  # Show first 10
                    logger.info(f"  - Reclamation {rec_id}")
                if len(reclamation_ids) > 10:
                    logger.info(f"  ... and {len(reclamation_ids) - 10} more")
            else:
                results = processor.batch_process(reclamation_ids)
        
        if not args.dry_run and 'results' in locals():
            # Show summary
            chunked = sum(1 for r in results if r.get('is_chunked'))
            logger.info(f"\nSummary:")
            logger.info(f"  Total processed: {len(results)}")
            logger.info(f"  Chunked: {chunked}")
            logger.info(f"  Single: {len(results) - chunked}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=args.verbose)
        sys.exit(1)
    finally:
        DatabasePool.reset()


if __name__ == "__main__":
    main()