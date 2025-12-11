# -*- coding: utf-8 -*-
"""
Re-run duplicate detection on existing reclamations with new thresholds.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Reset config singleton to pick up new thresholds
from src import config as cfg
cfg.reset_config()

from src.config import get_config
from src.duplicate_detector import DuplicateDetector


def main():
    config = get_config()
    print("=== NEW THRESHOLDS ===")
    print(f"AUTO_DUPLICATE: {config.detection.threshold_auto_duplicate}")
    print(f"REVIEW: {config.detection.threshold_review}")
    print()
    
    # Process the recent reclamations (54-63)
    detector = DuplicateDetector()
    
    print("=== RE-PROCESSING RECLAMATIONS 54-63 ===\n")
    
    for rec_id in range(54, 64):
        result = detector.process_reclamation(rec_id)
        status = "DUPLICATE" if result.is_duplicate else result.action.value
        if result.matched_id:
            print(f"ID {rec_id}: {status} -> matched with {result.matched_id} (score={result.similarity_score:.4f})")
        else:
            print(f"ID {rec_id}: {status}")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
