"""
Duplicate detection module for the De-duplication Pipeline.

Implements the core logic for detecting duplicate reclamations
using semantic similarity and business rules.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import get_config, DuplicateDetectionConfig
from src.database import (
    DatabasePool,
    ReclamationRepository,
    EmbeddingRepository,
    DuplicationLogRepository,
    ReclamationMatchRepository,
    Reclamation,
    SimilarityMatch,
)
from src.preprocessing import normalize_text
from src.embeddings import get_embedding_model


logger = logging.getLogger(__name__)


class DuplicateAction(Enum):
    """Actions that can be taken on potential duplicates."""
    AUTO_MARK_DUPLICATE = "AUTO_MARK_DUPLICATE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    NO_ACTION = "NO_ACTION"


class ReclamationStatus(Enum):
    """Status values for reclamation matches."""
    PENDING = "PENDING"
    CONFIRMED_DUPLICATE = "CONFIRMED_DUPLICATE"
    NOT_DUPLICATE = "NOT_DUPLICATE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection for a reclamation."""
    reclamation_id: int
    is_duplicate: bool
    action: DuplicateAction
    matched_id: Optional[int] = None
    similarity_score: Optional[float] = None
    message: str = ""


class DuplicateDetector:
    """
    Core duplicate detection engine.
    
    Combines text preprocessing, embedding generation, and similarity
    search to detect duplicate reclamations.
    """
    
    def __init__(
        self,
        config: Optional[DuplicateDetectionConfig] = None,
        db_pool: Optional[DatabasePool] = None
    ):
        """
        Initialize the duplicate detector.
        
        Args:
            config: Optional detection configuration override.
            db_pool: Optional database pool.
        """
        self._config = config or get_config().detection
        self._db_pool = db_pool
        
        # Initialize repositories
        self._reclamation_repo = ReclamationRepository(db_pool)
        self._embedding_repo = EmbeddingRepository(db_pool)
        self._log_repo = DuplicationLogRepository(db_pool)
        self._match_repo = ReclamationMatchRepository(db_pool)
        
        # Thresholds
        self.threshold_auto_duplicate = self._config.threshold_auto_duplicate
        self.threshold_review = self._config.threshold_review
        self.time_window_days = self._config.time_window_days
        
        logger.info(
            f"DuplicateDetector initialized: "
            f"auto_threshold={self.threshold_auto_duplicate}, "
            f"review_threshold={self.threshold_review}, "
            f"time_window={self.time_window_days} days"
        )
    
    def determine_action(self, score: float) -> DuplicateAction:
        """
        Determine the action based on similarity score.
        
        Args:
            score: Similarity score between 0 and 1.
        
        Returns:
            The appropriate DuplicateAction.
        """
        if score >= self.threshold_auto_duplicate:
            return DuplicateAction.AUTO_MARK_DUPLICATE
        elif score >= self.threshold_review:
            return DuplicateAction.FLAG_FOR_REVIEW
        else:
            return DuplicateAction.NO_ACTION
    
    def get_match_status_for_action(self, action: DuplicateAction) -> ReclamationStatus:
        """
        Get the match status for a given action.
        
        Args:
            action: The duplicate action.
        
        Returns:
            The appropriate ReclamationStatus for the match.
        """
        if action == DuplicateAction.AUTO_MARK_DUPLICATE:
            return ReclamationStatus.CONFIRMED_DUPLICATE
        elif action == DuplicateAction.FLAG_FOR_REVIEW:
            return ReclamationStatus.NEEDS_REVIEW
        else:
            return ReclamationStatus.PENDING
    
    def process_reclamation(self, reclamation_id: int) -> DuplicateDetectionResult:
        """
        Process a reclamation for duplicate detection.
        
        This is the main entry point that orchestrates:
        1. Fetching the reclamation
        2. Preprocessing the text
        3. Generating embeddings
        4. Storing embeddings
        5. Searching for similar reclamations
        6. Applying business rules
        7. Updating status and logging
        
        Args:
            reclamation_id: The ID of the reclamation to process.
        
        Returns:
            DuplicateDetectionResult with the outcome.
        """
        logger.info(f"Processing reclamation {reclamation_id}")
        
        # Step 1: Fetch the reclamation
        reclamation = self._reclamation_repo.get_by_id(reclamation_id)
        if not reclamation:
            logger.warning(f"Reclamation {reclamation_id} not found")
            return DuplicateDetectionResult(
                reclamation_id=reclamation_id,
                is_duplicate=False,
                action=DuplicateAction.NO_ACTION,
                message="Reclamation not found"
            )
        
        # Step 2: Preprocess the text
        # motif = self._reclamation_repo.get_motif_by_id(reclamation.motif_id) if reclamation.motif_id else None
        # if motif:
        #     normalized_text = normalize_text(motif.libelle)
        # else:
        #     return DuplicateDetectionResult(
        #         reclamation_id=reclamation_id,
        #         is_duplicate=False,
        #         action=DuplicateAction.NO_ACTION,
        #         message="Motif not found"
        #     )
        normalized_text = normalize_text(reclamation.description)
        if not normalized_text:
            logger.warning(f"Reclamation {reclamation_id} has empty text after normalization")
            return DuplicateDetectionResult(
                reclamation_id=reclamation_id,
                is_duplicate=False,
                action=DuplicateAction.NO_ACTION,
                message="Empty text after normalization"
            )
        
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
        )
    
    def check_similarity(
        self,
        text1: str,
        text2: str
    ) -> tuple[float, DuplicateAction]:
        """
        Check similarity between two texts.
        
        Utility method for ad-hoc similarity checks without database operations.
        
        Args:
            text1: First text.
            text2: Second text.
        
        Returns:
            Tuple of (similarity_score, action).
        """
        # Preprocess both texts
        normalized1 = normalize_text(text1)
        normalized2 = normalize_text(text2)
        
        if not normalized1 or not normalized2:
            return 0.0, DuplicateAction.NO_ACTION
        
        # Get embeddings and compute similarity
        model = get_embedding_model()
        score = model.similarity(normalized1, normalized2)
        action = self.determine_action(score)
        
        return score, action
    
    def batch_process(self, reclamation_ids: list[int]) -> list[DuplicateDetectionResult]:
        """
        Process multiple reclamations for duplicate detection.
        
        Args:
            reclamation_ids: List of reclamation IDs to process.
        
        Returns:
            List of DuplicateDetectionResult objects.
        """
        results = []
        for rec_id in reclamation_ids:
            try:
                result = self.process_reclamation(rec_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing reclamation {rec_id}: {e}")
                results.append(DuplicateDetectionResult(
                    reclamation_id=rec_id,
                    is_duplicate=False,
                    action=DuplicateAction.NO_ACTION,
                    message=f"Error: {str(e)}"
                ))
        return results


def get_duplicate_detector(
    config: Optional[DuplicateDetectionConfig] = None,
    db_pool: Optional[DatabasePool] = None
) -> DuplicateDetector:
    """
    Factory function to get a DuplicateDetector instance.
    
    Args:
        config: Optional configuration override.
        db_pool: Optional database pool.
    
    Returns:
        DuplicateDetector instance.
    """
    return DuplicateDetector(config, db_pool)


def detect_duplicates(reclamation_id: int) -> DuplicateDetectionResult:
    """
    Convenience function to detect duplicates for a single reclamation.
    
    Args:
        reclamation_id: The reclamation ID to check.
    
    Returns:
        DuplicateDetectionResult.
    """
    detector = get_duplicate_detector()
    return detector.process_reclamation(reclamation_id)


