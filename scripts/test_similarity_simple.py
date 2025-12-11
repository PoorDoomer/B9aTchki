# -*- coding: utf-8 -*-
"""
Simple debug script to test similarity scores.
Focuses on numerical output to avoid encoding issues.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import get_embedding_model
from src.preprocessing import normalize_text
from src.database import DatabasePool

# Sample texts from the SQL insert script
french_text_1 = "Lors de la souscription, il etait indique dans la fiche d'information que les consultations chez les specialistes conventionnes seraient remboursees au meme taux que celles chez le medecin generaliste."

arabic_text_1 = "عند الاكتتاب في عقد التأمين، تم التوضيح في ورقة المعلومات أن الاستشارات لدى الأطباء المتخصصين المتعاقدين سيتم تعويضها بنفس نسبة تعويض الاستشارات عند طبيب عام."

french_text_2 = "J'ai transmis un dossier complet de remboursement avec toutes les pieces justificatives demandees. Plus de 70 jours se sont ecoules sans qu'aucun paiement ne soit effectue."

arabic_text_2 = "قدمت ملفًا كاملاً لاسترجاع مصاريف الاستشارات والتحاليل والفحوصات بالأشعة مرفوقًا بجميع الوثائق المطلوبة. مرَّ أكثر من 70 يومًا دون أي تحويل مالي."

french_text_3 = "Lors de la signature de mon contrat, le conseiller m'avait confirme par ecrit que les soins preventifs seraient pris en charge a un taux preferentiel."


def main():
    print("LOADING MODEL...")
    model = get_embedding_model()
    _ = model.model
    print("MODEL LOADED")
    
    print("\n=== RAW SIMILARITY SCORES ===")
    
    # Test pairs
    pairs = [
        ("FR1 vs AR1 (same meaning)", french_text_1, arabic_text_1),
        ("FR2 vs AR2 (same meaning)", french_text_2, arabic_text_2),
        ("FR1 vs FR3 (paraphrase)", french_text_1, french_text_3),
        ("FR1 vs FR2 (different)", french_text_1, french_text_2),
    ]
    
    for name, t1, t2 in pairs:
        emb1 = model.encode_to_list(t1)
        emb2 = model.encode_to_list(t2)
        import numpy as np
        score = float(np.dot(emb1, emb2))
        action = "AUTO_DUPLICATE" if score >= 0.95 else ("REVIEW" if score >= 0.85 else "NO_ACTION")
        print(f"{name}: {score:.4f} -> {action}")
    
    print("\n=== WITH NORMALIZATION ===")
    
    for name, t1, t2 in pairs:
        n1 = normalize_text(t1)
        n2 = normalize_text(t2)
        emb1 = model.encode_to_list(n1)
        emb2 = model.encode_to_list(n2)
        import numpy as np
        score = float(np.dot(emb1, emb2))
        action = "AUTO_DUPLICATE" if score >= 0.95 else ("REVIEW" if score >= 0.85 else "NO_ACTION")
        print(f"{name}: {score:.4f} -> {action}")
    
    print("\n=== DATABASE CHECK ===")
    
    try:
        db_pool = DatabasePool()
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.reclamation_embeddings")
                count = cur.fetchone()[0]
                print(f"Total embeddings: {count}")
                
                cur.execute("""
                    SELECT r.id, e.reclamation_id is not null as has_embedding
                    FROM reclamation.reclamation r
                    LEFT JOIN public.reclamation_embeddings e ON r.id = e.reclamation_id
                    WHERE r.reclamant_id = 2
                    ORDER BY r.id DESC
                    LIMIT 20
                """)
                rows = cur.fetchall()
                print(f"Recent reclamations for user 2:")
                for row in rows:
                    print(f"  ID {row[0]}: has_embedding={row[1]}")
                    
    except Exception as e:
        print(f"DB Error: {e}")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
