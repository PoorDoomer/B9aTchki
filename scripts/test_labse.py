# -*- coding: utf-8 -*-
"""
Test LaBSE with simple cross-lingual pairs to verify model behavior.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import get_embedding_model


def main():
    print("Loading LaBSE model...")
    model = get_embedding_model()
    _ = model.model
    print("Model loaded!\n")
    
    # Simple cross-lingual test cases (known translations)
    test_pairs = [
        # Direct translations - should be very high similarity
        ("I love you", "Je t'aime"),
        ("I love you", "احبك"),
        ("Hello", "Bonjour"),
        ("Hello", "مرحبا"),
        
        # Insurance domain - short and clear
        ("Insurance claim", "Reclamation d'assurance"),
        ("Insurance claim", "مطالبة التأمين"),
        ("Refund request", "Demande de remboursement"),
        ("Refund request", "طلب استرداد"),
        
        # Longer similar sentences
        ("The insurance company rejected my claim", "La compagnie d'assurance a rejete ma reclamation"),
        ("The insurance company rejected my claim", "رفضت شركة التأمين مطالبتي"),
        
        # Different meanings (should be low)
        ("I love you", "The weather is nice"),
        ("Insurance claim", "I want pizza"),
    ]
    
    print("=== CROSS-LINGUAL SIMILARITY TEST ===\n")
    
    for t1, t2 in test_pairs:
        score = model.similarity(t1, t2)
        action = "HIGH (>0.85)" if score >= 0.85 else ("MEDIUM" if score >= 0.5 else "LOW")
        print(f"{score:.4f} [{action}] : {t1[:30]:30} <-> {t2[:30]}")
    
    print("\n\n=== TESTING WITH ACTUAL RECLAMATION DESCRIPTIONS ===\n")
    
    # Your actual texts - but shorter versions
    fr = "Les consultations chez les specialistes conventionnes remboursees au meme taux"
    ar = "الاستشارات لدى الأطباء المتخصصين المتعاقدين تعويضها بنفس النسبة"
    
    score = model.similarity(fr, ar)
    print(f"FR vs AR (short): {score:.4f}")
    
    # Now with more context
    fr_full = "Lors de la souscription, il etait indique que les consultations chez les specialistes conventionnes seraient remboursees au meme taux que celles chez le medecin generaliste. Plusieurs consultations ont ete prises en charge avec un taux inferieur."
    ar_full = "عند الاكتتاب في عقد التأمين تم التوضيح في ورقة المعلومات ان الاستشارات لدى الاطباء المتخصصين المتعاقدين سيتم تعويضها بنفس نسبة تعويض الاستشارات عند طبيب عام. الا ان عدة استشارات تم تعويضها بنسبة اقل"
    
    score = model.similarity(fr_full, ar_full)
    print(f"FR vs AR (full): {score:.4f}")
    
    # Compare semantically similar French texts
    fr1 = "Les soins preventifs seraient pris en charge a un taux preferentiel"
    fr2 = "Les consultations seraient remboursees a un taux plus avantageux"
    score = model.similarity(fr1, fr2)
    print(f"FR vs FR (paraphrase): {score:.4f}")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
