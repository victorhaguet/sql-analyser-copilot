"""Test the auth database module."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
import datetime
from typing import Any
from unittest.mock import patch

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from app_auth.database import (
    _get_connection,
    _ensure_users_dir,
    create_user,
    deactivate_user,
    delete_user,
    get_user_by_sub,
    get_user_by_username,
    init_user_db,
    list_users,
    update_user,
    user_count,
)
import app_auth.database as db_module


class AuthTestCase(unittest.TestCase):
    """Base test case for auth module tests."""

    def setUp(self) -> None:
        self._original_env = os.environ.copy()
        self._temp_dir = tempfile.TemporaryDirectory()
        self._temp_users_dir = Path(self._temp_dir.name)
        self._temp_db_path = self._temp_users_dir / "users.db"
        
        self._patcher_usrs_dir = patch('app_auth.database._USERS_DIR', self._temp_users_dir)
        self._patcher_usrs_db = patch('app_auth.database._USERS_DB', self._temp_db_path)
        self._patcher_usrs_dir.start()
        self._patcher_usrs_db.start()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)
        self._patcher_usrs_dir.stop()
        self._patcher_usrs_db.stop()
        self._temp_dir.cleanup()


class UserDatabaseTestCase(AuthTestCase):
    """Test case for the user database functions."""

    def test_init_user_db_creates_database(self) -> None:
        """Test that init_user_db creates the database file and tables."""
        init_user_db()
        self.assertTrue(self._temp_db_path.exists())

    def test_init_user_db_creates_users_table(self) -> None:
        """Test that init_user_db creates the users table with expected columns."""
        init_user_db()
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            )
            table = cursor.fetchone()
            self.assertIsNotNone(table)
        finally:
            conn.close()

    def test_create_user_returns_user_dict(self) -> None:
        """Test that create_user returns a dictionary with user information."""
        user = create_user(
            username="testuser",
            password_hash="hashed_password_123",
            name="Test User",
            role="readonly",
        )
        self.assertIsInstance(user, dict)
        self.assertEqual(user["username"], "testuser")
        self.assertEqual(user["name"], "Test User")
        self.assertEqual(user["role"], "readonly")
        self.assertTrue(user["is_active"])
        self.assertIn("sub", user)
        self.assertIn("created_at", user)

    def test_create_user_generates_unique_sub(self) -> None:
        """Test that create_user generates a unique sub for each user."""
        user1 = create_user(
            username="user1",
            password_hash="hash1",
        )
        user2 = create_user(
            username="user2",
            password_hash="hash2",
        )
        self.assertNotEqual(user1["sub"], user2["sub"])

    def test_create_user_raises_error_for_duplicate_username(self) -> None:
        """Test that create_user raises ValueError for duplicate usernames."""
        create_user(
            username="duplicate",
            password_hash="hash1",
        )
        with self.assertRaises(ValueError) as cm:
            create_user(
                username="duplicate",
                password_hash="hash2",
            )
        self.assertIn("already exists", str(cm.exception))

    def test_create_user_default_role_is_readonly(self) -> None:
        """Test that create_user defaults to readonly role when not specified."""
        user = create_user(
            username="default_role",
            password_hash="hash1",
        )
        self.assertEqual(user["role"], "readonly")

    def test_create_user_with_admin_role(self) -> None:
        """Test that create_user can create admin users."""
        user = create_user(
            username="admin_user",
            password_hash="hash1",
            role="admin",
        )
        self.assertEqual(user["role"], "admin")

    def test_get_user_by_username_returns_user(self) -> None:
        """Test that get_user_by_username returns user information."""
        created = create_user(
            username="findme",
            password_hash="hash1",
            name="Find Me",
        )
        user = get_user_by_username("findme")
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user["username"], "findme")
        self.assertEqual(user["name"], "Find Me")

    def test_get_user_by_username_returns_none_for_missing(self) -> None:
        """Test that get_user_by_username returns None for non-existent users."""
        user = get_user_by_username("nonexistent")
        self.assertIsNone(user)

    def test_get_user_by_sub_returns_user(self) -> None:
        """Test that get_user_by_sub returns user information."""
        created = create_user(
            username="byid",
            password_hash="hash1",
        )
        user = get_user_by_sub(created["sub"])
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user["sub"], created["sub"])

    def test_get_user_by_sub_returns_none_for_missing(self) -> None:
        """Test that get_user_by_sub returns None for non-existent users."""
        user = get_user_by_sub("nonexistent-sub")
        self.assertIsNone(user)

    def test_list_users_returns_all_users(self) -> None:
        """Test that list_users returns all users."""
        create_user(username="user1", password_hash="hash1")
        create_user(username="user2", password_hash="hash2")
        create_user(username="user3", password_hash="hash3")

        users = list_users()
        self.assertEqual(len(users), 3)
        usernames = {u["username"] for u in users}
        self.assertEqual(usernames, {"user1", "user2", "user3"})

    def test_list_users_excludes_password_hash_by_default(self) -> None:
        """Test that list_users excludes password_hash by default."""
        create_user(username="nosecret", password_hash="hash1")
        users = list_users()
        self.assertNotIn("password_hash", users[0])

    def test_list_users_includes_password_hash_when_requested(self) -> None:
        """Test that password hash can be included in user list."""
        created = create_user(username="withsecret", password_hash="hash1")
        
        conn = _get_connection()
        try:
            row = conn.execute("SELECT * FROM users WHERE username = ?", ("withsecret",)).fetchone()
            user = db_module._row_to_user(row, include_hash=True)
            self.assertIn("password_hash", user)
        finally:
            conn.close()

    def test_update_user_returns_updated_user(self) -> None:
        """Test that update_user returns the updated user information."""
        created = create_user(
            username="updater",
            password_hash="hash1",
            name="Original Name",
            role="readonly",
        )
        
        updated = update_user(
            sub=created["sub"],
            name="Updated Name",
            role="admin",
        )
        assert updated is not None
        
        self.assertEqual(updated["name"], "Updated Name")
        self.assertEqual(updated["role"], "admin")
        self.assertTrue(updated["is_active"])

    def test_update_user_preserves_unchanged_fields(self) -> None:
        """Test that update_user preserves fields not being updated."""
        created = create_user(
            username="partial",
            password_hash="hash1",
            name="Original",
        )
        sub = created["sub"]
        
        update_user(sub=sub, role="admin")
        
        user = get_user_by_sub(sub)
        assert user is not None
        self.assertEqual(user["name"], "Original")
        self.assertEqual(user["role"], "admin")

    def test_update_user_returns_none_for_missing(self) -> None:
        """Test that update_user returns None for non-existent users."""
        updated = update_user(sub="nonexistent-sub", name="Update")
        self.assertIsNone(updated)

    def test_deactivate_user_deactivates_account(self) -> None:
        """Test that deactivate_user sets is_active to False."""
        created = create_user(username="deact", password_hash="hash1")
        
        result = deactivate_user(sub=created["sub"])
        
        self.assertTrue(result)
        user = get_user_by_sub(created["sub"])
        assert user is not None
        self.assertFalse(user["is_active"])

    def test_deactivate_user_returns_false_for_missing(self) -> None:
        """Test that deactivate_user returns False for non-existent users."""
        result = deactivate_user(sub="nonexistent-sub")
        self.assertFalse(result)

    def test_delete_user_removes_user(self) -> None:
        """Test that delete_user removes the user from the database."""
        created = create_user(username="delete_me", password_hash="hash1")
        
        result = delete_user(sub=created["sub"])
        
        self.assertTrue(result)
        user = get_user_by_sub(created["sub"])
        self.assertIsNone(user)

    def test_delete_user_returns_false_for_missing(self) -> None:
        """Test that delete_user returns False for non-existent users."""
        result = delete_user(sub="nonexistent-sub")
        self.assertFalse(result)

    def test_user_count_returns_zero_for_empty_database(self) -> None:
        """Test that user_count returns 0 for an empty database."""
        init_user_db()
        count = user_count()
        self.assertEqual(count, 0)

    def test_user_count_returns_correct_count(self) -> None:
        """Test that user_count returns the correct number of users."""
        create_user(username="u1", password_hash="h1")
        create_user(username="u2", password_hash="h2")
        create_user(username="u3", password_hash="h3")
        
        count = user_count()
        self.assertEqual(count, 3)

    def test_user_count_updates_after_operations(self) -> None:
        """Test that user_count updates after create and delete operations."""
        temp_db_path = Path(self._temp_dir.name) / "count_test.db"
        
        with patch('app_auth.database._USERS_DB', temp_db_path):
            init_user_db()
            self.assertEqual(user_count(), 0)
            
            create_user(username="u1", password_hash="h1")
            self.assertEqual(user_count(), 1)
            
            user = get_user_by_username("u1")
            assert user is not None
            delete_user(sub=user["sub"])
            self.assertEqual(user_count(), 0)

    def test_users_directory_is_created(self) -> None:
        """Test that the users directory is created if it doesn't exist."""
        test_dir = self._temp_users_dir / "subdir" / "users"
        test_db_path = test_dir / "users.db"
        
        with patch('app_auth.database._USERS_DIR', test_dir), \
             patch('app_auth.database._USERS_DB', test_db_path):
            _ensure_users_dir()
        
        self.assertTrue(test_dir.exists())

    def test_created_at_timestamp_is_set(self) -> None:
        """Test that created_at timestamp is properly set."""        
        user = create_user(username="timestamp", password_hash="hash1")
        self.assertIn("created_at", user)
        created_at = datetime.datetime.fromisoformat(user["created_at"])
        self.assertIsInstance(created_at, datetime.datetime)

    def test_row_to_user_with_none_row(self) -> None:
        """Test that _row_to_user returns None for None input."""
        result = db_module._row_to_user(None)
        self.assertIsNone(result)
        
        result_with_hash = db_module._row_to_user(None, include_hash=True)
        self.assertIsNone(result_with_hash)


if __name__ == "__main__":
    unittest.main()
