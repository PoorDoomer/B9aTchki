#!/usr/bin/env python
"""Script to add find_similar_exceeded method to database.py"""

# The complete method implementation
NEW_METHOD = '''
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

'''

# Read the file
with open('src/database.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "return results" in find_similar (end of find_similar method)
# The method ends around line 433 (index 432)
# We want to insert after line 434 (the blank line before delete_embedding)
insert_at = 434  # Line number (1-indexed), insert after this line

# Split the new method into lines
new_lines = NEW_METHOD.strip('\n').split('\n')
# Add proper newline at end of each line
new_lines = [line + '\n' for line in new_lines]

# Insert the method
lines = lines[:insert_at] + new_lines + lines[insert_at:]

# Write back
with open('src/database.py', 'w', encoding='utf-8', newline='') as f:
    f.writelines(lines)

print(f"Successfully inserted find_similar_exceeded method at line {insert_at + 1}")
print(f"Added {len(new_lines)} lines of code")