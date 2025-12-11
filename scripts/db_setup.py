#!/usr/bin/env python
"""
Database Setup Script for the De-duplication Pipeline.

Checks and creates required database tables (reclamation_embeddings, 
duplication_logs) if they don't exist. Prompts user for confirmation
before making changes.

Usage:
    python scripts/db_setup.py              # Interactive mode
    python scripts/db_setup.py --check-only # Only check, don't create
    python scripts/db_setup.py --auto-approve # Skip confirmation prompts
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.database import DatabasePool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# SQL for table creation (extracted from migrations/001_initial_schema.sql)
CREATE_PGVECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS public.reclamation_embeddings (
    reclamation_id BIGINT PRIMARY KEY REFERENCES reclamation.reclamation(id) ON DELETE CASCADE,
    embedding vector(768)
);
"""

CREATE_EMBEDDINGS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON public.reclamation_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

CREATE_DUPLICATION_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS public.duplication_logs (
    id BIGSERIAL PRIMARY KEY,
    source_reclamation_id BIGINT REFERENCES reclamation.reclamation(id) ON DELETE SET NULL,
    matched_reclamation_id BIGINT REFERENCES reclamation.reclamation(id) ON DELETE SET NULL,
    similarity_score DECIMAL(5, 4),
    action VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_action CHECK (action IN ('AUTO_MARK_DUPLICATE', 'FLAG_FOR_REVIEW'))
);
"""

CREATE_DUPLICATION_LOGS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_duplication_logs_source ON public.duplication_logs(source_reclamation_id);
CREATE INDEX IF NOT EXISTS idx_duplication_logs_matched ON public.duplication_logs(matched_reclamation_id);
CREATE INDEX IF NOT EXISTS idx_duplication_logs_detected_at ON public.duplication_logs(detected_at DESC);
"""

CREATE_RECLAMATION_MATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS public.reclamation_matches (
    id BIGSERIAL PRIMARY KEY,
    reclamation_id BIGINT NOT NULL REFERENCES reclamation.reclamation(id) ON DELETE CASCADE,
    matched_reclamation_id BIGINT NOT NULL REFERENCES reclamation.reclamation(id) ON DELETE CASCADE,
    similarity_score DECIMAL(5, 4) NOT NULL,
    match_status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_match_status CHECK (match_status IN ('PENDING', 'CONFIRMED_DUPLICATE', 'NOT_DUPLICATE', 'NEEDS_REVIEW')),
    CONSTRAINT unique_match_pair UNIQUE (reclamation_id, matched_reclamation_id)
);
"""

CREATE_RECLAMATION_MATCHES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_reclamation_matches_reclamation ON public.reclamation_matches(reclamation_id);
CREATE INDEX IF NOT EXISTS idx_reclamation_matches_matched ON public.reclamation_matches(matched_reclamation_id);
CREATE INDEX IF NOT EXISTS idx_reclamation_matches_status ON public.reclamation_matches(match_status);
CREATE INDEX IF NOT EXISTS idx_reclamation_matches_score ON public.reclamation_matches(similarity_score DESC);
"""

CREATE_COSINE_FUNCTION = """
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


class DatabaseSetup:
    """Handles database schema verification and setup."""
    
    def __init__(self, auto_approve: bool = False, check_only: bool = False):
        self.auto_approve = auto_approve
        self.check_only = check_only
        self._pool = None
    
    def _get_pool(self) -> DatabasePool:
        """Get or create database pool."""
        if self._pool is None:
            self._pool = DatabasePool()
        return self._pool
    
    def check_connection(self) -> bool:
        """Test database connectivity."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is installed."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        );
                    """)
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to check pgvector extension: {e}")
            return False
    
    def check_table_exists(self, table_name: str, schema: str = "public") -> bool:
        """Check if a table exists in the database."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_schema = %s AND table_name = %s
                        );
                    """, (schema, table_name))
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to check table {schema}.{table_name}: {e}")
            return False
    
    def check_function_exists(self, function_name: str) -> bool:
        """Check if a function exists in the database."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_proc 
                            WHERE proname = %s
                        );
                    """, (function_name,))
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to check function {function_name}: {e}")
            return False
    
    def run_health_checks(self) -> dict:
        """Run all health checks and return status."""
        print("\n" + "=" * 60)
        print("DATABASE HEALTH CHECK")
        print("=" * 60)
        
        config = get_config().postgres
        print(f"\nTarget: {config.host}:{config.port}/{config.database}")
        
        checks = {}
        
        # Connection check
        print("\n[1] Testing database connection...", end=" ")
        checks["connection"] = self.check_connection()
        print("✓ OK" if checks["connection"] else "✗ FAILED")
        
        if not checks["connection"]:
            print("\n❌ Cannot proceed without database connection.")
            return checks
        
        # pgvector extension check  
        print("[2] Checking pgvector extension...", end=" ")
        checks["pgvector"] = self.check_pgvector_extension()
        print("✓ Installed" if checks["pgvector"] else "✗ Not installed")
        
        # reclamation table check (required, must exist in reclamation schema)
        print("[3] Checking reclamation.reclamation table...", end=" ")
        checks["reclamation"] = self.check_table_exists("reclamation", "reclamation")
        print("✓ Exists" if checks["reclamation"] else "✗ Missing")
        
        # reclamation_embeddings table check (in public schema)
        print("[4] Checking public.reclamation_embeddings table...", end=" ")
        checks["reclamation_embeddings"] = self.check_table_exists("reclamation_embeddings", "public")
        print("✓ Exists" if checks["reclamation_embeddings"] else "✗ Missing")
        
        # duplication_logs table check (in public schema)
        print("[5] Checking public.duplication_logs table...", end=" ")
        checks["duplication_logs"] = self.check_table_exists("duplication_logs", "public")
        print("✓ Exists" if checks["duplication_logs"] else "✗ Missing")
        
        # reclamation_matches table check (in public schema)
        print("[6] Checking public.reclamation_matches table...", end=" ")
        checks["reclamation_matches"] = self.check_table_exists("reclamation_matches", "public")
        print("✓ Exists" if checks["reclamation_matches"] else "✗ Missing")
        
        # cosine_similarity function check
        print("[7] Checking cosine_similarity function...", end=" ")
        checks["cosine_similarity"] = self.check_function_exists("cosine_similarity")
        print("✓ Exists" if checks["cosine_similarity"] else "✗ Missing")
        
        return checks
    
    def prompt_user(self, message: str) -> bool:
        """Prompt user for confirmation."""
        if self.auto_approve:
            print(f"{message} [auto-approved]")
            return True
        
        response = input(f"{message} (y/n): ").strip().lower()
        return response in ("y", "yes")
    
    def create_pgvector_extension(self) -> bool:
        """Create pgvector extension."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_PGVECTOR_EXTENSION)
                    conn.commit()
                    logger.info("Created pgvector extension")
                    return True
        except Exception as e:
            logger.error(f"Failed to create pgvector extension: {e}")
            return False
    
    def create_embeddings_table(self) -> bool:
        """Create reclamation_embeddings table and index."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_EMBEDDINGS_TABLE)
                    cur.execute(CREATE_EMBEDDINGS_INDEX)
                    conn.commit()
                    logger.info("Created reclamation_embeddings table with HNSW index")
                    return True
        except Exception as e:
            logger.error(f"Failed to create reclamation_embeddings table: {e}")
            return False
    
    def create_duplication_logs_table(self) -> bool:
        """Create duplication_logs table and indexes."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_DUPLICATION_LOGS_TABLE)
                    cur.execute(CREATE_DUPLICATION_LOGS_INDEXES)
                    conn.commit()
                    logger.info("Created duplication_logs table with indexes")
                    return True
        except Exception as e:
            logger.error(f"Failed to create duplication_logs table: {e}")
            return False
    
    def create_reclamation_matches_table(self) -> bool:
        """Create reclamation_matches table and indexes."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_RECLAMATION_MATCHES_TABLE)
                    cur.execute(CREATE_RECLAMATION_MATCHES_INDEXES)
                    conn.commit()
                    logger.info("Created reclamation_matches table with indexes")
                    return True
        except Exception as e:
            logger.error(f"Failed to create reclamation_matches table: {e}")
            return False
    
    def create_cosine_function(self) -> bool:
        """Create cosine_similarity helper function."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_COSINE_FUNCTION)
                    conn.commit()
                    logger.info("Created cosine_similarity function")
                    return True
        except Exception as e:
            logger.error(f"Failed to create cosine_similarity function: {e}")
            return False
    
    def setup(self) -> bool:
        """Run the complete setup process."""
        checks = self.run_health_checks()
        
        if not checks.get("connection"):
            return False
        
        if not checks.get("reclamation"):
            print("\n❌ The 'reclamation.reclamation' table must exist before running this script.")
            print("   Please ensure the base schema is set up first.")
            return False
        
        # Determine what needs to be created
        missing_items = []
        
        if not checks.get("pgvector"):
            missing_items.append(("pgvector extension", self.create_pgvector_extension))
        
        if not checks.get("reclamation_embeddings"):
            missing_items.append(("reclamation_embeddings table", self.create_embeddings_table))
        
        if not checks.get("duplication_logs"):
            missing_items.append(("duplication_logs table", self.create_duplication_logs_table))
        
        if not checks.get("reclamation_matches"):
            missing_items.append(("reclamation_matches table", self.create_reclamation_matches_table))
        
        if not checks.get("cosine_similarity"):
            missing_items.append(("cosine_similarity function", self.create_cosine_function))
        
        if not missing_items:
            print("\n✅ All required database objects exist. Nothing to do.")
            return True
        
        print("\n" + "-" * 60)
        print("MISSING DATABASE OBJECTS")
        print("-" * 60)
        for item, _ in missing_items:
            print(f"  • {item}")
        
        if self.check_only:
            print("\n[check-only mode] No changes will be made.")
            return False
        
        # Prompt for confirmation
        print()
        if not self.prompt_user("Do you want to create these missing objects?"):
            print("\nSetup cancelled by user.")
            return False
        
        # Create missing items
        print("\n" + "-" * 60)
        print("CREATING MISSING OBJECTS")
        print("-" * 60)
        
        success = True
        for item_name, create_func in missing_items:
            print(f"\nCreating {item_name}...", end=" ")
            if create_func():
                print("✓ Done")
            else:
                print("✗ Failed")
                success = False
        
        if success:
            print("\n✅ Database setup completed successfully!")
        else:
            print("\n⚠️ Some items failed to create. Check the logs above.")
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup database schema for the de-duplication pipeline"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check database state, don't create anything"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip confirmation prompts"
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
        setup = DatabaseSetup(
            auto_approve=args.auto_approve,
            check_only=args.check_only
        )
        success = setup.setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=args.verbose)
        sys.exit(1)
    finally:
        DatabasePool.reset()


if __name__ == "__main__":
    main()
