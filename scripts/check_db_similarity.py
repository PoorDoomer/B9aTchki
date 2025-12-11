# -*- coding: utf-8 -*-
"""
Check similarity between actual stored embeddings in the database.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import DatabasePool


def main():
    print("=== DATABASE EMBEDDING SIMILARITY CHECK ===\n")
    
    db_pool = DatabasePool()
    
    with db_pool.get_connection() as conn:
        with conn.cursor() as cur:
            # Get all reclamations for reclamant_id=2 with embeddings
            cur.execute("""
                SELECT r.id, r.reference_reclamation, LEFT(r.description, 60) as desc_preview
                FROM reclamation.reclamation r
                JOIN public.reclamation_embeddings e ON r.id = e.reclamation_id
                WHERE r.reclamant_id = 2
                ORDER BY r.id
            """)
            reclamations = cur.fetchall()
            
            print(f"Found {len(reclamations)} reclamations with embeddings for reclamant_id=2:\n")
            for r in reclamations:
                ref = r[1] if r[1] else "N/A"
                print(f"  ID {r[0]}: {ref}")
            
            # Get the most recent reclamations (IDs 54-63)
            target_ids = [r[0] for r in reclamations if r[0] >= 54]
            
            if target_ids:
                print(f"\n\n=== PAIRWISE SIMILARITY FOR IDS {min(target_ids)}-{max(target_ids)} ===\n")
                
                # Compute similarity between all pairs
                for i, id1 in enumerate(target_ids):
                    for id2 in target_ids[i+1:]:
                        cur.execute("""
                            SELECT 
                                1 - (e1.embedding <=> e2.embedding) as similarity
                            FROM public.reclamation_embeddings e1
                            CROSS JOIN public.reclamation_embeddings e2
                            WHERE e1.reclamation_id = %s AND e2.reclamation_id = %s
                        """, (id1, id2))
                        row = cur.fetchone()
                        if row:
                            score = float(row[0])
                            action = "AUTO_DUP" if score >= 0.95 else ("REVIEW" if score >= 0.85 else "-")
                            print(f"  {id1} <-> {id2}: {score:.4f} ({action})")
            
            # Check the specific pairs that should be duplicates
            print("\n\n=== EXPECTED DUPLICATE PAIRS ===\n")
            
            # Check French-Arabic pairs (should be consecutive IDs)
            cur.execute("""
                SELECT 
                    r1.id as id1,
                    r1.reference_reclamation as ref1,
                    r2.id as id2,
                    r2.reference_reclamation as ref2,
                    1 - (e1.embedding <=> e2.embedding) as similarity
                FROM reclamation.reclamation r1
                JOIN reclamation.reclamation r2 ON r1.id = r2.id - 1
                JOIN public.reclamation_embeddings e1 ON r1.id = e1.reclamation_id
                JOIN public.reclamation_embeddings e2 ON r2.id = e2.reclamation_id
                WHERE r1.reclamant_id = 2 
                  AND r2.reclamant_id = 2
                  AND r1.id >= 54
                ORDER BY r1.id
            """)
            pairs = cur.fetchall()
            
            for pair in pairs:
                score = float(pair[4])
                action = "AUTO_DUP" if score >= 0.95 else ("REVIEW" if score >= 0.85 else "NO_ACTION")
                print(f"  {pair[0]} ({pair[1]}) <-> {pair[2]} ({pair[3]})")
                print(f"    Similarity: {score:.4f} -> {action}")
                print()
    
    print("DONE")


if __name__ == "__main__":
    main()
