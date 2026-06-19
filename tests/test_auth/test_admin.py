"""Test the auth admin bootstrap module."""
import os
import sys
import unittest
from pathlib import Path


for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from app_auth.admin import (
    admin_credentials_configured,
    ensure_admin_exists,
    is_admin_account_created,
)
from app_auth.database import (
    create_user,
    delete_user,
    get_user_by_username,
    init_user_db,
    user_count,
)
import app_auth.database as db_module


class AdminBootstrapTestCase(unittest.TestCase):
    """Test case for admin bootstrap functions."""

    def setUp(self) -> None:
        self._original_env = os.environ.copy()
        init_user_db()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)
        users = db_module.list_users()
        for user in users:
            delete_user(sub=user["sub"])

    def test_admin_credentials_configured_returns_false_without_env(self) -> None:
        """Test that admin_credentials_configured returns False without env vars."""
        os.environ.pop("INITIAL_ADMIN_USERNAME", None)
        os.environ.pop("INITIAL_ADMIN_PASSWORD", None)
        
        result = admin_credentials_configured()
        self.assertFalse(result)

    def test_admin_credentials_configured_returns_false_without_username(self) -> None:
        """Test that admin_credentials_configured returns False without username."""
        os.environ.pop("INITIAL_ADMIN_USERNAME", None)
        os.environ["INITIAL_ADMIN_PASSWORD"] = "password123"
        
        result = admin_credentials_configured()
        self.assertFalse(result)

    def test_admin_credentials_configured_returns_false_without_password(self) -> None:
        """Test that admin_credentials_configured returns False without password."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ.pop("INITIAL_ADMIN_PASSWORD", None)
        
        result = admin_credentials_configured()
        self.assertFalse(result)

    def test_admin_credentials_configured_returns_true_with_both(self) -> None:
        """Test that admin_credentials_configured returns True with both env vars."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "password123"
        
        result = admin_credentials_configured()
        self.assertTrue(result)

    def test_is_admin_account_created_returns_false_when_empty(self) -> None:
        """Test that is_admin_account_created returns False when no users exist."""
        os.environ.pop("INITIAL_ADMIN_USERNAME", None)
        os.environ.pop("INITIAL_ADMIN_PASSWORD", None)
        
        result = is_admin_account_created()
        self.assertFalse(result)

    def test_is_admin_account_created_returns_true_when_users_exist(self) -> None:
        """Test that is_admin_account_created returns True when users exist."""
        create_user(username="user1", password_hash="hash1")
        
        result = is_admin_account_created()
        self.assertTrue(result)

    def test_ensure_admin_exists_does_nothing_without_credentials(self) -> None:
        """Test that ensure_admin_exists does nothing without admin credentials."""
        os.environ.pop("INITIAL_ADMIN_USERNAME", None)
        os.environ.pop("INITIAL_ADMIN_PASSWORD", None)
        
        ensure_admin_exists()
        
        self.assertEqual(user_count(), 0)

    def test_ensure_admin_exists_does_nothing_when_users_exist(self) -> None:
        """Test that ensure_admin_exists does nothing when users already exist."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        create_user(username="existing", password_hash="hash1")
        
        ensure_admin_exists()
        
        self.assertEqual(user_count(), 1)

    def test_ensure_admin_exists_creates_admin_when_empty(self) -> None:
        """Test that ensure_admin_exists creates admin account when empty."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        ensure_admin_exists()
        
        self.assertEqual(user_count(), 1)
        admin = get_user_by_username("admin")
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "admin")

    def test_ensure_admin_exists_with_custom_username(self) -> None:
        """Test that ensure_admin_exists uses custom username from env."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "superuser"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "superpass"
        
        ensure_admin_exists()
        
        admin = get_user_by_username("superuser")
        self.assertIsNotNone(admin)

    def test_ensure_admin_exists_sets_admin_role(self) -> None:
        """Test that ensure_admin_exists sets role to admin."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        ensure_admin_exists()
        
        admin = get_user_by_username("admin")
        self.assertEqual(admin["role"], "admin")

    def test_ensure_admin_exists_hashes_password(self) -> None:
        """Test that ensure_admin_exists hashes the admin password."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        ensure_admin_exists()
        
        admin = get_user_by_username("admin")
        self.assertIn("password_hash", admin)
        self.assertNotEqual(admin["password_hash"], "admin123")

    def test_ensure_admin_exists_creates_database_if_needed(self) -> None:
        """Test that ensure_admin_exists initializes database if it doesn't exist."""        
        db_module._USERS_DB.unlink(missing_ok=True)
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        ensure_admin_exists()
        
        self.assertTrue(db_module._USERS_DB.exists())
        self.assertEqual(user_count(), 1)

    def test_ensure_admin_exists_creates_admin_with_name(self) -> None:
        """Test that ensure_admin_exists sets the admin name."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        ensure_admin_exists()
        
        admin = get_user_by_username("admin")
        self.assertEqual(admin["name"], "Administrator")

    def test_ensure_admin_exists_multiple_calls(self) -> None:
        """Test that multiple calls to ensure_admin_exists don't create duplicates."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        ensure_admin_exists()
        ensure_admin_exists()
        ensure_admin_exists()
        
        self.assertEqual(user_count(), 1)

    def test_credentials_configured_after_ensure_admin(self) -> None:
        """Test that admin_credentials_configured works correctly after ensure_admin."""
        os.environ["INITIAL_ADMIN_USERNAME"] = "admin"
        os.environ["INITIAL_ADMIN_PASSWORD"] = "admin123"
        
        self.assertTrue(admin_credentials_configured())
        
        ensure_admin_exists()
        self.assertTrue(admin_credentials_configured())


if __name__ == "__main__":
    unittest.main()
