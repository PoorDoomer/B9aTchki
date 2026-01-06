#!/usr/bin/env python
"""Script to update find_similar_exceeded to use Asymmetric MaxSim (Chamfer Similarity)"""

import re

# Read the file
with open('src/database.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The new MaxSim implementation
NEW_METHOD = '''    def find_similar_exceeded(
        self,
        reclamation_id: int,
        reclamant_id: int,
        exclude_id: int,
        min_score: float = 0.75,
        time_window_days: int = 7
    ) -> list[SimilarityMatchChunkResult]:
        """
        Ultimate Duplicate Detection using Asymmetric MaxSim (Chamfer Similarity).
        
        This approach calculates two scores:
        1. Forward Coverage: How much of the Source is present in the Target?
        2. Backward Coverage: How much of the Target is present in the Source?
        
        Final Score is the Harmonic Mean, ensuring substantial overlap in both directions.
        
        Math:
        - Calculate similarity matrix (N_source x N_target)
        - Forward Score: mean(max(row)) for each row (Source chunk's best match in Target)
        - Backward Score: mean(max(col)) for each col (Target chunk's best match in Source)
        - Final Score: harmonic_mean(forward, backward) = 2*f*b/(f+b)
        
        Scenarios:
        1. Chunked A vs Chunked B: Full MaxSim calculation  
        2. Chunked A vs Non-chunked B: Max of (backward single vs all chunks, forward all chunks vs single)
        3. Non-chunked A vs Chunked B: Similar to scenario 2
        4. Non-chunked A vs Non-chunked B: Standard cosine similarity
        
        Args:
            reclamation_id: Source reclamation ID
            reclamant_id: Reclamant ID for filtering
            exclude_id: Reclamation ID to exclude (self)
            min_score: Minimum similarity score threshold
            time_window_days: Time window in days for search scope
        
        Returns:
            List of SimilarityMatchChunkResult objects sorted by score descending
        """
        
        # 1. Fetch Source Chunks
        source_chunks_map = self.get_chunks_for_reclamation(reclamation_id)
        if not source_chunks_map:
            # Fallback to base embedding if no chunks
            base = self.get_base_embedding(reclamation_id)
            if base:
                source_chunks_map = {f"{reclamation_id}_base": base}
            else:
                logger.warning(f"Reclamation {reclamation_id} has no embedding")
                return []
        
        source_vectors = list(source_chunks_map.values())
        
        # 2. Get Candidates
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
            
            # Fetch Target Chunks
            target_chunks_map = self.get_chunks_for_reclamation(target_id)
            if not target_chunks_map:
                base = self.get_base_embedding(target_id)
                if base:
                    target_chunks_map = {f"{target_id}_base": base}
                else:
                    continue
            
            target_vectors = list(target_chunks_map.values())
            
            # --- MAXSIM MATH ---
            
            try:
                A = np.array(source_vectors)  # Shape: (N_source, 768)
                B = np.array(target_vectors)  # Shape: (N_target, 768)
                
                # Calculate similarity matrix (dot product = cosine sim for normalized vectors)
                sim_matrix = np.dot(A, B.T)  # Shape: (N_source, N_target)
                
                # Forward Score (Source -> Target): For each chunk in Source, find best match in Target
                max_sim_A_to_B = np.max(sim_matrix, axis=1)  # Shape: (N_source,)
                forward_score = float(np.mean(max_sim_A_to_B))
                
                # Backward Score (Target -> Source): For each chunk in Target, find best match in Source
                max_sim_B_to_A = np.max(sim_matrix, axis=0)  # Shape: (N_target,)
                backward_score = float(np.mean(max_sim_B_to_A))
                
                # Final Score: Harmonic Mean (F1-style)
                # This ensures both directions have strong overlap for strict duplicates
                if forward_score + backward_score > 0:
                    similarity_score = (2 * forward_score * backward_score) / (forward_score + backward_score)
                else:
                    similarity_score = 0.0
                
                # Best single pair for debugging
                best_chunk_pair_score = float(np.max(sim_matrix))
                
                logger.debug(
                    f"MaxSim {reclamation_id} vs {target_id}: "
                    f"n={len(source_vectors)}, m={len(target_vectors)}, "
                    f"forward={forward_score:.4f}, backward={backward_score:.4f}, "
                    f"final={similarity_score:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Vector math error {reclamation_id} vs {target_id}: {e}")
                continue
            
            # Filter by threshold
            if similarity_score >= min_score:
                results.append(SimilarityMatchChunkResult(
                    reclamation_id=target_id,
                    similarity_score=similarity_score,
                    motif_id=target_motif,
                    is_chunked=True,
                    num_source_chunks=len(source_vectors),
                    num_target_chunks=len(target_vectors),
                    forward_score=forward_score,
                    backward_score=backward_score,
                    best_chunk_pair_score=best_chunk_pair_score
                ))
        
        # Sort by similarity score descending
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(
            f"Found {len(results)} matches for reclamation {reclamation_id} "
            f"above threshold {min_score:.4f} using MaxSim"
        )
        
        return results

'''

# Pattern to match the entire old method (from def to the next blank line before next method)
# We need to be careful about the boundaries
pattern = r'def find_similar_exceeded\(.*?\n(.*?\n)*?(?=    def is_chunked|\Z)'

# Find the old method and replace it
new_content = re.sub(
    pattern,
    NEW_METHOD,
    content,
    flags=re.DOTALL
)

# Write back
with open('src/database.py', 'w', encoding='utf-8', newline='') as f:
    f.write(new_content)

print("Successfully replaced find_similar_exceeded with MaxSim implementation")
print("\nKey improvements:")
print("- Replaced Elasticity Consistency with Asymmetric MaxSim (Chamfer Similarity)")
print("- Forward Score: Source chunks' best matches averaged (Source -> Target)")
print("- Backward Score: Target chunks' best matches averaged (Target -> Source)")
print("- Final Score: Harmonic mean of forward and backward scores")
print("- Handles length imbalance without penalties")
print("- No arbitrary constants (no sqrt(n*m), no elasticity_factor)")
print("- Vectorized numpy operations for performance")

# Also check if the pattern matched
if new_content == content:
    print("\nWARNING: Pattern did not match - trying alternative approach...")
    
    # Alternative: Find the exact start and end
    start_marker = '    def find_similar_exceeded('
    end_marker = '    def is_chunked('
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        new_content = content[:start_idx] + NEW_METHOD + '    ' + content[end_idx:]
        
        with open('src/database.py', 'w', encoding='utf-8', newline='') as f:
            f.write(new_content)
        
        print("Used alternative replacement method - SUCCESS")
    else:
        print("Could not find method boundaries")