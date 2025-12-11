#!/usr/bin/env python
"""
Setup PostgreSQL NOTIFY Trigger for Reclamation Events.

Creates a trigger function and trigger on reclamation.reclamation table
that sends NOTIFY events when new rows are inserted. These events are
consumed by the pg_listener service to publish to RabbitMQ.

Usage:
    python scripts/setup_notify_trigger.py              # Interactive mode
    python scripts/setup_notify_trigger.py --auto-approve # Skip confirmation
    python scripts/setup_notify_trigger.py --check-only # Only check status
    python scripts/setup_notify_trigger.py --drop       # Remove trigger
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


# SQL for creating the notify function
CREATE_NOTIFY_FUNCTION = """
CREATE OR REPLACE FUNCTION notify_new_reclamation()
RETURNS TRIGGER AS $$
BEGIN
    -- Send notification with JSON payload containing reclamation details
    PERFORM pg_notify(
        'new_reclamation',
        json_build_object(
            'reclamation_id', NEW.id,
            'reclamant_id', NEW.reclamant_id,
            'operation', TG_OP
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

# SQL for creating the trigger
CREATE_NOTIFY_TRIGGER = """
DROP TRIGGER IF EXISTS reclamation_insert_notify ON reclamation.reclamation;

CREATE TRIGGER reclamation_insert_notify
    AFTER INSERT ON reclamation.reclamation
    FOR EACH ROW
    EXECUTE FUNCTION notify_new_reclamation();
"""

# SQL for dropping the trigger and function
DROP_TRIGGER = """
DROP TRIGGER IF EXISTS reclamation_insert_notify ON reclamation.reclamation;
"""

DROP_FUNCTION = """
DROP FUNCTION IF EXISTS notify_new_reclamation();
"""


class NotifyTriggerSetup:
    """Handles trigger setup and verification."""
    
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
    
    def check_function_exists(self) -> bool:
        """Check if notify_new_reclamation function exists."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_proc 
                            WHERE proname = 'notify_new_reclamation'
                        );
                    """)
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to check function: {e}")
            return False
    
    def check_trigger_exists(self) -> bool:
        """Check if trigger exists on reclamation.reclamation table."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_trigger t
                            JOIN pg_class c ON t.tgrelid = c.oid
                            JOIN pg_namespace n ON c.relnamespace = n.oid
                            WHERE t.tgname = 'reclamation_insert_notify'
                              AND n.nspname = 'reclamation'
                              AND c.relname = 'reclamation'
                        );
                    """)
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to check trigger: {e}")
            return False
    
    def check_table_exists(self) -> bool:
        """Check if reclamation.reclamation table exists."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_schema = 'reclamation' 
                              AND table_name = 'reclamation'
                        );
                    """)
                    result = cur.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to check table: {e}")
            return False
    
    def run_status_check(self) -> dict:
        """Run all status checks and display results."""
        print("\n" + "=" * 60)
        print("NOTIFY TRIGGER STATUS CHECK")
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
        
        # Table check
        print("[2] Checking reclamation.reclamation table...", end=" ")
        checks["table"] = self.check_table_exists()
        print("✓ Exists" if checks["table"] else "✗ Missing")
        
        # Function check
        print("[3] Checking notify_new_reclamation function...", end=" ")
        checks["function"] = self.check_function_exists()
        print("✓ Exists" if checks["function"] else "✗ Missing")
        
        # Trigger check
        print("[4] Checking reclamation_insert_notify trigger...", end=" ")
        checks["trigger"] = self.check_trigger_exists()
        print("✓ Exists" if checks["trigger"] else "✗ Missing")
        
        return checks
    
    def prompt_user(self, message: str) -> bool:
        """Prompt user for confirmation."""
        if self.auto_approve:
            print(f"{message} [auto-approved]")
            return True
        
        response = input(f"{message} (y/n): ").strip().lower()
        return response in ("y", "yes")
    
    def create_function(self) -> bool:
        """Create the notify function."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_NOTIFY_FUNCTION)
                    conn.commit()
                    logger.info("Created notify_new_reclamation function")
                    return True
        except Exception as e:
            logger.error(f"Failed to create function: {e}")
            return False
    
    def create_trigger(self) -> bool:
        """Create the trigger on reclamation.reclamation."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_NOTIFY_TRIGGER)
                    conn.commit()
                    logger.info("Created reclamation_insert_notify trigger")
                    return True
        except Exception as e:
            logger.error(f"Failed to create trigger: {e}")
            return False
    
    def drop_trigger_and_function(self) -> bool:
        """Drop the trigger and function."""
        try:
            pool = self._get_pool()
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(DROP_TRIGGER)
                    cur.execute(DROP_FUNCTION)
                    conn.commit()
                    logger.info("Dropped trigger and function")
                    return True
        except Exception as e:
            logger.error(f"Failed to drop trigger/function: {e}")
            return False
    
    def setup(self) -> bool:
        """Run the complete setup process."""
        checks = self.run_status_check()
        
        if not checks.get("connection"):
            return False
        
        if not checks.get("table"):
            print("\n❌ The 'reclamation.reclamation' table must exist first.")
            return False
        
        # Check if already fully set up
        if checks.get("function") and checks.get("trigger"):
            print("\n✅ NOTIFY trigger is already set up. Nothing to do.")
            return True
        
        if self.check_only:
            print("\n[check-only mode] No changes will be made.")
            missing = []
            if not checks.get("function"):
                missing.append("notify_new_reclamation function")
            if not checks.get("trigger"):
                missing.append("reclamation_insert_notify trigger")
            print(f"Missing: {', '.join(missing)}")
            return False
        
        # Show what will be created
        print("\n" + "-" * 60)
        print("ITEMS TO CREATE")
        print("-" * 60)
        
        items_to_create = []
        if not checks.get("function"):
            items_to_create.append(("notify_new_reclamation function", self.create_function))
            print("  • notify_new_reclamation() function")
        if not checks.get("trigger"):
            items_to_create.append(("reclamation_insert_notify trigger", self.create_trigger))
            print("  • reclamation_insert_notify trigger")
        
        print()
        if not self.prompt_user("Do you want to create these items?"):
            print("\nSetup cancelled by user.")
            return False
        
        # Create items
        print("\n" + "-" * 60)
        print("CREATING ITEMS")
        print("-" * 60)
        
        success = True
        for item_name, create_func in items_to_create:
            print(f"\nCreating {item_name}...", end=" ")
            if create_func():
                print("✓ Done")
            else:
                print("✗ Failed")
                success = False
        
        if success:
            print("\n" + "=" * 60)
            print("✅ NOTIFY TRIGGER SETUP COMPLETED!")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Start the listener: python -m src.pg_listener")
            print("  2. Start the worker:   python -m src.worker")
            print("  3. Insert a row to test the pipeline")
        else:
            print("\n⚠️ Some items failed to create. Check the logs above.")
        
        return success
    
    def drop(self) -> bool:
        """Drop the trigger and function."""
        checks = self.run_status_check()
        
        if not checks.get("connection"):
            return False
        
        if not checks.get("function") and not checks.get("trigger"):
            print("\n✅ Nothing to drop. Trigger and function don't exist.")
            return True
        
        print("\n" + "-" * 60)
        print("ITEMS TO DROP")
        print("-" * 60)
        if checks.get("trigger"):
            print("  • reclamation_insert_notify trigger")
        if checks.get("function"):
            print("  • notify_new_reclamation() function")
        
        print()
        if not self.prompt_user("Do you want to drop these items?"):
            print("\nDrop cancelled by user.")
            return False
        
        print("\nDropping trigger and function...", end=" ")
        if self.drop_trigger_and_function():
            print("✓ Done")
            print("\n✅ Trigger and function have been removed.")
            return True
        else:
            print("✗ Failed")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup PostgreSQL NOTIFY trigger for reclamation events"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check status, don't create anything"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the trigger and function instead of creating"
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
        setup = NotifyTriggerSetup(
            auto_approve=args.auto_approve,
            check_only=args.check_only
        )
        
        if args.drop:
            success = setup.drop()
        else:
            success = setup.setup()
        
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=args.verbose)
        sys.exit(1)
    finally:
        DatabasePool.reset()


if __name__ == "__main__":
    main()
