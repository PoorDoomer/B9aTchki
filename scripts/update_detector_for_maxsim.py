#!/usr/bin/env python
"""Script to update duplicate_detector.py for MaxSim compatibility"""

# Read the file
with open('src/duplicate_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Remove old parameter initializations in __init__
content = content.replace(
    '''        self.top_k_chunks = self._config.top_k_chunks
        self.use_advanced_matching = self._config.use_advanced_matching
        self.elasticity_factor = self._config.elasticity_factor
        
        logger.info(
            f"DuplicateDetector initialized: "
            f"auto_threshold={self.threshold_auto_duplicate}, "
            f"review_threshold={self.threshold_review}, "
            f"time_window={self.time_window_days} days, "
            f"top_k_chunks={self.top_k_chunks}, "
            f"use_advanced_matching={self.use_advanced_matching}, "
            f"elasticity_factor={self.elasticity_factor}"
        )''',
    '''        logger.info(
            f"DuplicateDetector initialized: "
            f"auto_threshold={self.threshold_auto_duplicate}, "
            f"review_threshold={self.threshold_review}, "
            f"time_window={self.time_window_days} days (using MaxSim)"
        )'''
)

# Fix 2: Update find_similar_exceeded call - remove old parameters
content = content.replace(
    '''            # Search for matches using chunk-aware comparison
            matches = self._embedding_repo.find_similar_exceeded(
                reclamation_id=reclamation_id,
                reclamant_id=reclamation.reclamant_id,
                exclude_id=reclamation_id,
                min_score=self.threshold_review,
                time_window_days=self.time_window_days,
                top_k_chunks=self.top_k_chunks,
                use_advanced_matching=self.use_advanced_matching,
                elasticity_factor=self.elasticity_factor
            )''',
    '''            # Search for matches using chunk-aware MaxSim comparison
            matches = self._embedding_repo.find_similar_exceeded(
                reclamation_id=reclamation_id,
                reclamant_id=reclamation.reclamant_id,
                exclude_id=reclamation_id,
                min_score=self.threshold_review,
                time_window_days=self.time_window_days
            )'''
)

# Fix 3: Update processing loop to use new MaxSim fields
content = content.replace(
    '''        # Process each match
        for match in matches:
            # Handle both SimilarityMatch and SimilarityMatchChunkResult
            if isinstance(match, SimilarityMatchChunkResult):
                matched_id = match.reclamation_id
                score = match.similarity_score
                chunk_info = f" (chunked: source={match.source_is_chunked}, target={match.target_is_chunked})"
            else:''',
    '''        # Process each match
        for match in matches:
            # Handle both SimilarityMatch and SimilarityMatchChunkResult
            if isinstance(match, SimilarityMatchChunkResult):
                matched_id = match.reclamation_id
                score = match.similarity_score
                chunk_info = f" (MaxSim: forward={match.forward_score:.3f}, backward={match.backward_score:.3f}, best={match.best_chunk_pair_score:.3f})"
            else:'''
)

# Write back
with open('src/duplicate_detector.py', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("Successfully updated duplicate_detector.py for MaxSim compatibility")
print("\nChanges:")
print("- Removed top_k_chunks, use_advanced_matching, elasticity_factor parameters")
print("- Updated initialization log message")
print("- Updated find_similar_exceeded call to use MaxSim")
print("- Updated processing loop to log forward/backward/best scores instead of chunked flags")