"""
Debug script to test similarity scores between French and Arabic reclamations.
This helps diagnose why the duplicate detector is not finding matches.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import get_embedding_model, compute_similarity
from src.preprocessing import normalize_text
from src.database import DatabasePool, EmbeddingRepository

# Sample texts from the SQL insert script - First pair (Non respect des clauses contractuelles)
french_text_1 = """Lors de la souscription, il était indiqué dans la fiche d'information que les consultations chez les spécialistes conventionnés seraient remboursées au même taux que celles chez le médecin généraliste. Or, plusieurs consultations ont été prises en charge avec un taux inférieur, sans avenant ni notification préalable. J'estime que les conditions initialement présentées n'ont pas été respectées."""

arabic_text_1 = """عند الاكتتاب في عقد التأمين، تم التوضيح في ورقة المعلومات أن الاستشارات لدى الأطباء المتخصصين المتعاقدين سيتم تعويضها بنفس نسبة تعويض الاستشارات عند طبيب عام. إلا أن عدة استشارات تم تعويضها بنسبة أقل، دون أي ملحق أو إشعار مسبق. أعتبر أن الشروط المقدمة في البداية لم تُحترم."""

# Second pair (Retard de traitement)
french_text_2 = """J'ai transmis un dossier complet de remboursement (consultations, analyses et imagerie) avec toutes les pièces justificatives demandées. Plus de 70 jours se sont écoulés sans qu'aucun paiement ne soit effectué ni qu'une décision écrite ne me soit communiquée. Les relances téléphoniques restent sans effet concret, on me répète seulement que le dossier est "en cours de traitement"."""

arabic_text_2 = """قدمت ملفًا كاملاً لاسترجاع مصاريف الاستشارات والتحاليل والفحوصات بالأشعة مرفوقًا بجميع الوثائق المطلوبة. مرَّ أكثر من 70 يومًا دون أي تحويل مالي أو قرار مكتوب من شركة التأمين. الاتصالات المتكررة بمصلحة الزبناء لم تسفر عن أي نتيجة ملموسة، إذ يكتفون بالقول إن الملف "قيد المعالجة"."""

# French-only batch duplicate (should match first French text semantically)
french_text_3 = """Lors de la signature de mon contrat, le conseiller m'avait confirmé par écrit que les soins préventifs (bilan annuel, examens de routine) seraient pris en charge à un taux préférentiel. Aujourd'hui, ces actes sont remboursés sur la base du tarif minimum, sans aucune justification ni avenant. Il y a un écart manifeste entre ce qui m'a été promis et ce qui est appliqué."""


def main():
    print("=" * 80)
    print("DUPLICATE DETECTION SIMILARITY ANALYSIS")
    print("=" * 80)
    
    # Load the embedding model
    print("\nLoading LaBSE model...")
    model = get_embedding_model()
    _ = model.model  # Force model load
    print("Model loaded successfully!")
    
    # Test similarity between pairs
    pairs = [
        ("French 1 vs Arabic 1 (Same meaning)", french_text_1, arabic_text_1),
        ("French 2 vs Arabic 2 (Same meaning)", french_text_2, arabic_text_2),
        ("French 1 vs French 3 (Paraphrases)", french_text_1, french_text_3),
        ("French 1 vs French 2 (Different topics)", french_text_1, french_text_2),
        ("French 1 vs Arabic 2 (Different topics)", french_text_1, arabic_text_2),
    ]
    
    print("\n" + "-" * 80)
    print("RAW TEXT SIMILARITY (without preprocessing)")
    print("-" * 80)
    
    for name, text1, text2 in pairs:
        score = compute_similarity(text1, text2)
        print(f"\n{name}:")
        print(f"  Similarity Score: {score:.4f}")
        if score >= 0.95:
            print(f"  -> AUTO_MARK_DUPLICATE (>=0.95)")
        elif score >= 0.85:
            print(f"  -> FLAG_FOR_REVIEW (>=0.85)")
        else:
            print(f"  -> NO_ACTION (<0.85)")
    
    print("\n" + "-" * 80)
    print("NORMALIZED TEXT SIMILARITY (with preprocessing)")
    print("-" * 80)
    
    for name, text1, text2 in pairs:
        normalized1 = normalize_text(text1)
        normalized2 = normalize_text(text2)
        
        # Show normalized text preview
        print(f"\n{name}:")
        print(f"  Normalized Text 1 (first 100 chars): {normalized1[:100]}...")
        print(f"  Normalized Text 2 (first 100 chars): {normalized2[:100]}...")
        
        score = compute_similarity(normalized1, normalized2)
        print(f"  Similarity Score: {score:.4f}")
        if score >= 0.95:
            print(f"  -> AUTO_MARK_DUPLICATE (>=0.95)")
        elif score >= 0.85:
            print(f"  -> FLAG_FOR_REVIEW (>=0.85)")
        else:
            print(f"  -> NO_ACTION (<0.85)")
    
    # Now test against actual database embeddings
    print("\n" + "=" * 80)
    print("DATABASE EMBEDDING VERIFICATION")
    print("=" * 80)
    
    try:
        db_pool = DatabasePool()
        embedding_repo = EmbeddingRepository(db_pool)
        
        # Check if there are any embeddings stored
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.reclamation_embeddings")
                count = cur.fetchone()[0]
                print(f"\nTotal embeddings in database: {count}")
                
                # Get the recent reclamation IDs and their embeddings
                cur.execute("""
                    SELECT r.id, r.reference_reclamation, LEFT(r.description, 50) as desc_preview
                    FROM reclamation.reclamation r
                    JOIN public.reclamation_embeddings e ON r.id = e.reclamation_id
                    WHERE r.reclamant_id = 2
                    ORDER BY r.id DESC
                    LIMIT 20
                """)
                rows = cur.fetchall()
                
                if rows:
                    print("\nRecent embeddings for reclamant_id=2:")
                    for row in rows:
                        print(f"  ID {row[0]}: {row[1]} - {row[2]}...")
                else:
                    print("\nNo embeddings found for reclamant_id=2!")
                    print("This could be the issue - embeddings are not being stored properly.")
                    
    except Exception as e:
        print(f"\nDatabase connection error: {e}")
        print("Could not verify database embeddings.")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
