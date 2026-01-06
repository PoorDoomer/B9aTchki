#!/usr/bin/env python
"""Script to update duplicate_detector.py for chunked comparison support"""

import re

# Read the file
with open('src/duplicate_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add MLConfig import and also import SimilarityMatchChunkResult
content = content.replace(
    'from src.database import (',
    '''from src.config import get_config, DuplicateDetectionConfig, MLConfig
from src.database import ('''
)

# Fix 2: Add SimilarityMatchChunkResult to database imports
content = content.replace(
    '    SimilarityMatch,\n)',
    '    SimilarityMatch,\n    SimilarityMatchChunkResult,\n)'
)

# Fix 3: Update __init__ to include ML config and chunk parameters
old_init = '''        # Thresholds
        self.threshold_auto_duplicate = self._config.threshold_auto_duplicate
        self.threshold_review = self._config.threshold_review
        self.time_window_days = self._config.time_window_days'''

new_init = '''        # Get full config
        full_config = get_config()
        self._ml_config = full_config.ml
        
        # Thresholds
        self.threshold_auto_duplicate = self._config.threshold_auto_duplicate
        self.threshold_review = self._config.threshold_review
        self.time_window_days = self._config.time_window_days
        self.top_k_chunks = self._config.top_k_chunks
        self.use_advanced_matching = self._config.use_advanced_matching
        self.elasticity_factor = self._config.elasticity_factor'''

content = content.replace(old_init, new_init)

# Fix 4: Fix number_of_tokens usage and add ML max_seq_tokens
content = content.replace(
    '#Step 2.2 : REMOVE KEYWORDS b7al salama 3alaikum\n        logger.debug(f"Reclamation {reclamation_id}: Token count before cleaning: {number_of_tokens}")\n        number_of_tokens=count_tokens(normalized_text)',
    '''#Step 2.2 : REMOVE KEYWORDS b7al salama 3alaikum
        number_of_tokens=count_tokens(normalized_text)
        logger.debug(f"Reclamation {reclamation_id}: Token count before cleaning: {number_of_tokens}")'''
)

# Fix 5: Change max_seq_tokens reference
content = content.replace(
    'if number_of_tokens > self._config.max_seq_tokens:',
    'if number_of_tokens > self._ml_config.max_seq_tokens:'
)

# Fix 6: Replace entire chunked section and non-chunked section to use find_similar_exceeded
old_section = '''        if number_of_tokens > self._ml_config.max_seq_tokens:
            logger.warning(f"Reclamation {reclamation_id} exceeds max token limit after cleaning")
            chunker=Chunker()
            chunks = chunker.chunk_text(normalized_text)
            embedding_chunks = {}
            for i in range(len(chunks)):
                logger.debug(f"EMBEDDING : Reclamation {reclamation_id}: Chunk {i+1} token count: {count_tokens(chunks[i])}")
                model = get_embedding_model()
                embedding_chunks[reclamation_id] = model.encode_to_list(chunks[i])
                self._embedding_repo.save_embedding(f"{reclamation_id}_chunk_{i+1}", embedding_chunks[reclamation_id])
            logger.info(f"Reclamation {reclamation_id}: Processed {len(chunks)} chunks due to token limit")
            

        else:
            # Step 3: Generate embedding
            model = get_embedding_model()
            embedding = model.encode_to_list(normalized_text)
        
            # Step 4: Store the embedding
            self._embedding_repo.save_embedding(reclamation_id, embedding)

            # Step 5: Search for similar reclamations
            matches = self._embedding_repo.find_similar(
                embedding=embedding,
                reclamant_id=reclamation.reclamant_id,
                exclude_id=reclamation_id,
                min_score=self.threshold_review,
                time_window_days=self.time_window_days,
                limit=1  # We only need the best match
            )
        
            # Step 6: Apply business rules
            if not matches:
                logger.info(f"Reclamation {reclamation_id}: No duplicates found")
                return DuplicateDetectionResult(
                    reclamation_id=reclamation_id,
                    is_duplicate=False,
                    action=DuplicateAction.NO_ACTION,
                    message="No similar reclamations found"
                )
            
            best_match = matches[0]
            action = self.determine_action(best_match.score)
            
            # Step 7: Create match record and log based on action
            if action != DuplicateAction.NO_ACTION:
                # Create match record with status
                match_status = self.get_match_status_for_action(action)
                self._match_repo.create(
                    reclamation_id=reclamation_id,
                    matched_reclamation_id=best_match.reclamation_id,
                    similarity_score=best_match.score,
                    match_status=match_status.value
                )
                logger.info("AUDIT LOG")
                # Create audit log
                self._log_repo.create(
                    source_reclamation_id=reclamation_id,
                    matched_reclamation_id=best_match.reclamation_id,
                    similarity_score=best_match.score,
                    action=action.value
                )
                
                logger.info(
                    f"Reclamation {reclamation_id}: {action.value} "
                    f"(matched with {best_match.reclamation_id}, score={best_match.score:.4f})"
                )
            
            return DuplicateDetectionResult(
                reclamation_id=reclamation_id,
                is_duplicate=(action == DuplicateAction.AUTO_MARK_DUPLICATE),
                action=action,
                matched_id=best_match.reclamation_id if action != DuplicateAction.NO_ACTION else None,
                similarity_score=best_match.score if action != DuplicateAction.NO_ACTION else None,
                message=f"Matched with reclamation {best_match.reclamation_id}" if action != DuplicateAction.NO_ACTION else "Unique reclamation"
            )'''

new_section = '''        # Step 3-4: Generate and store embeddings (chunked or single)
        is_chunked = False
        if number_of_tokens > self._ml_config.max_seq_tokens:
            logger.warning(f"Reclamation {reclamation_id} exceeds max token limit after cleaning")
            is_chunked = True
            chunker=Chunker()
            chunks = chunker.chunk_text(normalized_text)
            model = get_embedding_model()
            for i, chunk in enumerate(chunks, 1):
                embedding = model.encode_to_list(chunk)
                self._embedding_repo.save_embedding(f"{reclamation_id}_chunk_{i}", embedding)
                logger.debug(f"Saved chunk {i} for reclamation {reclamation_id}")
            logger.info(f"Reclamation {reclamation_id}: Split into {len(chunks)} chunks")

            # Step 5: Search for matches using chunk-aware comparison
            matches = self._embedding_repo.find_similar_exceeded(
                reclamation_id=reclamation_id,
                reclamant_id=reclamation.reclamant_id,
                exclude_id=reclamation_id,
                min_score=self.threshold_review,
                time_window_days=self.time_window_days,
                top_k_chunks=self.top_k_chunks,
                use_advanced_matching=self.use_advanced_matching,
                elasticity_factor=self.elasticity_factor
            )
            
        else:
            # Step 3: Generate embedding
            model = get_embedding_model()
            embedding = model.encode_to_list(normalized_text)
        
            # Step 4: Store the embedding
            self._embedding_repo.save_embedding(reclamation_id, embedding)

            # Step 5: Search for similar reclamations (no limit - get ALL matches above threshold)
            matches = self._embedding_repo.find_similar(
                embedding=embedding,
                reclamant_id=reclamation.reclamant_id,
                exclude_id=reclamation_id,
                min_score=self.threshold_review,
                time_window_days=self.time_window_days,
                limit=None  # Get ALL matches above threshold
            )
        
        # Step 6-7: Process ALL matches (not just best match)
        if not matches:
            logger.info(f"Reclamation {reclamation_id}: No duplicates found")
            return DuplicateDetectionResult(
                reclamation_id=reclamation_id,
                is_duplicate=False,
                action=DuplicateAction.NO_ACTION,
                message="No similar reclamations found"
            )
        
        logger.info(f"Reclamation {reclamation_id}: Found {len(matches)} potential match(es)")
        
        # Process each match
        for match in matches:
            # Handle both SimilarityMatch and SimilarityMatchChunkResult
            if isinstance(match, SimilarityMatchChunkResult):
                matched_id = match.reclamation_id
                score = match.similarity_score
                chunk_info = f" (chunked: source={match.source_is_chunked}, target={match.target_is_chunked})"
            else:
                matched_id = match.reclamation_id
                score = match.score
                chunk_info = ""
            
            action = self.determine_action(score)
            
            if action != DuplicateAction.NO_ACTION:
                match_status = self.get_match_status_for_action(action)
                self._match_repo.create(
                    reclamation_id=reclamation_id,
                    matched_reclamation_id=matched_id,
                    similarity_score=score,
                    match_status=match_status.value
                )
                
                logger.info("AUDIT LOG")
                self._log_repo.create(
                    source_reclamation_id=reclamation_id,
                    matched_reclamation_id=matched_id,
                    similarity_score=score,
                    action=action.value
                )
                
                logger.info(
                    f"Reclamation {reclamation_id}: {action.value} "
                    f"(matched with {matched_id}, score={score:.4f}{chunk_info})"
                )
        
        # Determine overall result using highest scoring match
        if isinstance(matches[0], SimilarityMatchChunkResult):
            best_match_score = matches[0].similarity_score
            best_match_id = matches[0].reclamation_id
        else:
            best_match_score = matches[0].score
            best_match_id = matches[0].reclamation_id
            
        best_action = self.determine_action(best_match_score)
        
        return DuplicateDetectionResult(
            reclamation_id=reclamation_id,
            is_duplicate=(best_action == DuplicateAction.AUTO_MARK_DUPLICATE),
            action=best_action,
            matched_id=best_match_id,
            similarity_score=best_match_score,
            message=f"Found {len(matches)} match(es), best matched with {best_match_id}"
        )'''

content = content.replace(old_section, new_section)

# Fix 7: Update logger.info in __init__ to include new params
content = content.replace(
    '''        logger.info(
            f"DuplicateDetector initialized: "
            f"auto_threshold={self.threshold_auto_duplicate}, "
            f"review_threshold={self.threshold_review}, "
            f"time_window={self.time_window_days} days"
        )''',
    '''        logger.info(
            f"DuplicateDetector initialized: "
            f"auto_threshold={self.threshold_auto_duplicate}, "
            f"review_threshold={self.threshold_review}, "
            f"time_window={self.time_window_days} days, "
            f"top_k_chunks={self.top_k_chunks}, "
            f"use_advanced_matching={self.use_advanced_matching}, "
            f"elasticity_factor={self.elasticity_factor}"
        )'''
)

# Write back
with open('src/duplicate_detector.py', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("Successfully updated duplicate_detector.py with chunked comparison support")
print("Changes:")
print("- Added MLConfig import and SimilarityMatchChunkResult import")
print("- Updated __init__ to include ML config and chunk parameters")
print("- Fixed number_of_tokens usage")
print("- Changed max_seq_tokens reference")
print("- Replaced chunked section to NOT return early")
print("- Changed find_similar limit from 1 to None")
print("- Updated to save ALL matches above threshold")
print("- Added detailed logging for each match")
print("- Added summary logging")