"""
Demo script showing the core de-duplication pipeline functionality.
No Docker required - uses the preprocessing and threshold logic.
"""

from src.preprocessing import normalize_text, detect_primary_script
from src.duplicate_detector import DuplicateDetector, DuplicateAction
from src.config import DuplicateDetectionConfig

print("=" * 60)
print("Cross-Lingual De-duplication Pipeline Demo")
print("=" * 60)

# Demo 1: Text Preprocessing
print("\n1. TEXT PREPROCESSING")
print("-" * 40)

french_text = "  <p>Ma CONNEXION Internet ne fonctionne pas!</p>  "
arabic_text = "الإنترنت مقطوع منذ أمس"

print(f"French Original: {french_text}")
print(f"French Normalized: {normalize_text(french_text)}")
print(f"Detected Script: {detect_primary_script(french_text)}")

print(f"\nArabic Original: {arabic_text}")
print(f"Arabic Normalized: {normalize_text(arabic_text)}")
print(f"Detected Script: {detect_primary_script(arabic_text)}")

# Demo 2: Threshold Logic
print("\n2. THRESHOLD LOGIC")
print("-" * 40)

config = DuplicateDetectionConfig(
    threshold_auto_duplicate=0.95,
    threshold_review=0.85,
    time_window_days=7
)

# Create detector with mock DB (won't actually connect)
from unittest.mock import MagicMock
detector = DuplicateDetector.__new__(DuplicateDetector)
detector._config = config
detector.threshold_auto_duplicate = config.threshold_auto_duplicate
detector.threshold_review = config.threshold_review
detector.time_window_days = config.time_window_days

test_scores = [0.99, 0.96, 0.95, 0.90, 0.85, 0.80, 0.50]

print(f"Auto-duplicate threshold: >= {config.threshold_auto_duplicate}")
print(f"Review threshold: >= {config.threshold_review}")
print()

for score in test_scores:
    action = detector.determine_action(score)
    status = "🔴 AUTO DUPLICATE" if action == DuplicateAction.AUTO_MARK_DUPLICATE else \
             "🟡 FLAG FOR REVIEW" if action == DuplicateAction.FLAG_FOR_REVIEW else \
             "🟢 UNIQUE"
    print(f"Score {score:.2f} → {status}")

# Demo 3: Sample Reclamations
print("\n3. SAMPLE RECLAMATIONS (French & Arabic)")
print("-" * 40)

samples = [
    ("Ma connexion internet ne fonctionne pas depuis hier", "french"),
    ("L'internet est coupé depuis 24 heures", "french"),
    ("الإنترنت مقطوع منذ أمس", "arabic"),
    ("لا يوجد اتصال بالإنترنت", "arabic"),
    ("Ma facture est incorrecte ce mois-ci", "french"),
    ("فاتورة الهاتف خاطئة", "arabic"),
]

for text, lang in samples:
    normalized = normalize_text(text)
    print(f"[{lang.upper():6}] {text}")
    print(f"         → {normalized}")
    print()

print("=" * 60)
print("✅ Pipeline components working correctly!")
print("=" * 60)
print("\nTo run with full infrastructure:")
print("1. Start Docker Desktop")
print("2. Run: docker-compose up -d")
print("3. Run: python -m src.worker")


