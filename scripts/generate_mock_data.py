#!/usr/bin/env python
"""
Mock data generator for testing the de-duplication pipeline.

Generates French and Arabic reclamation data with known duplicates
for testing and validation purposes.
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0])

from src.config import get_config
from src.database import DatabasePool, ReclamationRepository, EmbeddingRepository
from src.preprocessing import normalize_text
from src.embeddings import get_embedding_model, EmbeddingModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# French reclamation templates
FRENCH_TEMPLATES = [
    # Internet issues
    ("Ma connexion internet ne fonctionne pas depuis hier", "internet"),
    ("L'internet est coupé depuis 24 heures", "internet"),
    ("Je n'ai plus d'accès à internet", "internet"),
    ("La connexion wifi ne marche plus", "internet"),
    ("Mon débit internet est très lent", "internet"),
    
    # Billing issues
    ("Ma facture est incorrecte ce mois-ci", "billing"),
    ("Il y a une erreur sur ma facture", "billing"),
    ("Je conteste le montant de ma facture", "billing"),
    ("On m'a facturé deux fois", "billing"),
    ("Le prélèvement automatique a échoué", "billing"),
    
    # Phone issues
    ("Mon téléphone ne capte plus de réseau", "phone"),
    ("Je ne peux plus passer d'appels", "phone"),
    ("Les SMS ne sont pas envoyés", "phone"),
    ("Ma ligne téléphonique est coupée", "phone"),
    
    # Account issues
    ("Je ne peux pas accéder à mon compte", "account"),
    ("Mon mot de passe ne fonctionne plus", "account"),
    ("Je veux changer mon forfait", "account"),
    ("Comment résilier mon abonnement", "account"),
]

# Arabic reclamation templates (semantically similar to French)
ARABIC_TEMPLATES = [
    # Internet issues
    ("الإنترنت مقطوع منذ أمس", "internet"),
    ("لا يوجد اتصال بالإنترنت", "internet"),
    ("خدمة الإنترنت لا تعمل", "internet"),
    ("الواي فاي لا يعمل", "internet"),
    ("سرعة الإنترنت بطيئة جدا", "internet"),
    
    # Billing issues
    ("فاتورة الهاتف خاطئة", "billing"),
    ("هناك خطأ في الفاتورة", "billing"),
    ("أعترض على مبلغ الفاتورة", "billing"),
    ("تم خصم المبلغ مرتين", "billing"),
    ("فشل الدفع التلقائي", "billing"),
    
    # Phone issues
    ("هاتفي لا يستقبل شبكة", "phone"),
    ("لا أستطيع إجراء مكالمات", "phone"),
    ("الرسائل لا ترسل", "phone"),
    ("خط الهاتف مقطوع", "phone"),
    
    # Account issues
    ("لا أستطيع الدخول إلى حسابي", "account"),
    ("كلمة المرور لا تعمل", "account"),
    ("أريد تغيير اشتراكي", "account"),
    ("كيف ألغي الاشتراك", "account"),
]


def generate_variation(text: str) -> str:
    """Generate a slight variation of the text."""
    variations = [
        lambda t: t + " depuis longtemps" if random.random() > 0.5 else t,
        lambda t: "Urgent: " + t if random.random() > 0.7 else t,
        lambda t: t + "." if not t.endswith(".") else t,
        lambda t: t + " SVP" if random.random() > 0.8 else t,
    ]
    
    for var in variations:
        text = var(text)
    
    return text


class MockDataGenerator:
    """Generator for mock reclamation data."""
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """Initialize the generator."""
        self._pool = db_pool or DatabasePool()
        self._reclamation_repo = ReclamationRepository(self._pool)
        self._embedding_repo = EmbeddingRepository(self._pool)
        self._model: Optional[EmbeddingModel] = None
    
    def _get_model(self) -> EmbeddingModel:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model
    
    def generate_user_reclamations(
        self,
        user_id: int,
        num_reclamations: int = 5,
        include_duplicates: bool = True,
        languages: list[str] = ["french", "arabic"]
    ) -> list[int]:
        """
        Generate reclamations for a single user.
        
        Args:
            user_id: The user ID to generate reclamations for.
            num_reclamations: Number of reclamations to generate.
            include_duplicates: Whether to include semantic duplicates.
            languages: Languages to use ("french", "arabic", or both).
        
        Returns:
            List of created reclamation IDs.
        """
        created_ids = []
        templates = []
        
        if "french" in languages:
            templates.extend([(t, "french", c) for t, c in FRENCH_TEMPLATES])
        if "arabic" in languages:
            templates.extend([(t, "arabic", c) for t, c in ARABIC_TEMPLATES])
        
        if include_duplicates and num_reclamations >= 2:
            # Create some duplicates by using same category
            # Pick a category to duplicate
            categories = list(set(c for _, _, c in templates))
            dup_category = random.choice(categories)
            
            # Get all templates in that category
            category_templates = [(t, l) for t, l, c in templates if c == dup_category]
            
            # Create 2-3 duplicates from same category
            num_dups = min(3, num_reclamations // 2)
            for i in range(num_dups):
                text, lang = random.choice(category_templates)
                text = generate_variation(text)
                
                rec_id = self._reclamation_repo.create(
                    user_id=user_id,
                    message_libre=text,
                    status="PENDING"
                )
                created_ids.append(rec_id)
                logger.info(f"Created duplicate ({lang}): {rec_id} - {text[:50]}...")
            
            num_reclamations -= num_dups
        
        # Create remaining unique reclamations
        used_categories = set()
        for _ in range(num_reclamations):
            # Try to use different categories
            available = [(t, l, c) for t, l, c in templates if c not in used_categories]
            if not available:
                available = templates
            
            text, lang, category = random.choice(available)
            used_categories.add(category)
            text = generate_variation(text)
            
            rec_id = self._reclamation_repo.create(
                user_id=user_id,
                message_libre=text,
                status="PENDING"
            )
            created_ids.append(rec_id)
            logger.info(f"Created unique ({lang}): {rec_id} - {text[:50]}...")
        
        return created_ids
    
    def generate_embeddings(self, reclamation_ids: list[int]) -> int:
        """
        Generate and store embeddings for reclamations.
        
        Args:
            reclamation_ids: List of reclamation IDs to process.
        
        Returns:
            Number of embeddings generated.
        """
        model = self._get_model()
        count = 0
        
        for rec_id in reclamation_ids:
            rec = self._reclamation_repo.get_by_id(rec_id)
            if rec:
                normalized = normalize_text(rec.message_libre)
                embedding = model.encode_to_list(normalized)
                self._embedding_repo.save_embedding(rec_id, embedding)
                count += 1
                logger.debug(f"Generated embedding for reclamation {rec_id}")
        
        logger.info(f"Generated {count} embeddings")
        return count
    
    def generate_dataset(
        self,
        num_users: int = 10,
        reclamations_per_user: int = 5,
        include_duplicates: bool = True,
        generate_embeddings: bool = True
    ) -> dict:
        """
        Generate a complete test dataset.
        
        Args:
            num_users: Number of users to generate.
            reclamations_per_user: Reclamations per user.
            include_duplicates: Whether to include duplicates.
            generate_embeddings: Whether to generate embeddings.
        
        Returns:
            Dictionary with statistics about generated data.
        """
        all_ids = []
        user_ids = list(range(1, num_users + 1))
        
        logger.info(f"Generating dataset: {num_users} users, {reclamations_per_user} reclamations each")
        
        for user_id in user_ids:
            ids = self.generate_user_reclamations(
                user_id=user_id,
                num_reclamations=reclamations_per_user,
                include_duplicates=include_duplicates
            )
            all_ids.extend(ids)
        
        embeddings_count = 0
        if generate_embeddings:
            embeddings_count = self.generate_embeddings(all_ids)
        
        return {
            "total_reclamations": len(all_ids),
            "num_users": num_users,
            "reclamation_ids": all_ids,
            "embeddings_generated": embeddings_count
        }


def main():
    """Main entry point for the mock data generator."""
    parser = argparse.ArgumentParser(
        description="Generate mock reclamation data for testing"
    )
    parser.add_argument(
        "--users", "-u",
        type=int,
        default=10,
        help="Number of users to generate (default: 10)"
    )
    parser.add_argument(
        "--reclamations", "-r",
        type=int,
        default=5,
        help="Reclamations per user (default: 5)"
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Don't include semantic duplicates"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Don't generate embeddings"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = MockDataGenerator()
        stats = generator.generate_dataset(
            num_users=args.users,
            reclamations_per_user=args.reclamations,
            include_duplicates=not args.no_duplicates,
            generate_embeddings=not args.no_embeddings
        )
        
        print("\n=== Mock Data Generation Complete ===")
        print(f"Total reclamations: {stats['total_reclamations']}")
        print(f"Number of users: {stats['num_users']}")
        print(f"Embeddings generated: {stats['embeddings_generated']}")
        print(f"Reclamation IDs: {stats['reclamation_ids'][:10]}...")
        
    except Exception as e:
        logger.error(f"Failed to generate mock data: {e}", exc_info=True)
        sys.exit(1)
    finally:
        DatabasePool.reset()


if __name__ == "__main__":
    main()


