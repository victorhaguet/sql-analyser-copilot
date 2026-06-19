"""Test the auth password module."""
import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from app_auth.password import hash_password, verify_password


class PasswordHashingTestCase(unittest.TestCase):
    """Test case for password hashing and verification functions."""

    def test_hash_password_returns_non_empty_string(self) -> None:
        """Test that hash_password returns a non-empty string."""
        hashed = hash_password("mypassword")
        self.assertIsInstance(hashed, str)
        self.assertGreater(len(hashed), 0)

    def test_hash_password_returns_different_hashes(self) -> None:
        """Test that hash_password returns different hashes for the same password."""
        password = "samepassword"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        self.assertNotEqual(hash1, hash2)

    def test_verify_password_returns_true_for_matching(self) -> None:
        """Test that verify_password returns True for matching password and hash."""
        password = "testpassword123"
        hashed = hash_password(password)
        result = verify_password(password, hashed)
        self.assertTrue(result)

    def test_verify_password_returns_false_for_non_matching(self) -> None:
        """Test that verify_password returns False for non-matching password and hash."""
        password = "testpassword123"
        hashed = hash_password(password)
        result = verify_password("wrongpassword", hashed)
        self.assertFalse(result)

    def test_verify_password_returns_false_for_empty_password(self) -> None:
        """Test that verify_password returns False for empty password."""
        password = ""
        hashed = hash_password(password)
        result = verify_password("nonempty", hashed)
        self.assertFalse(result)

    def test_verify_password_handles_special_characters(self) -> None:
        """Test that verification works with special characters in password."""
        password = "p@$$w0rd!#%^&*()"
        hashed = hash_password(password)
        result = verify_password(password, hashed)
        self.assertTrue(result)

    def test_hash_password_with_unicode(self) -> None:
        """Test that hashing works with unicode characters."""
        password = "密码123"
        hashed = hash_password(password)
        result = verify_password(password, hashed)
        self.assertTrue(result)

    def test_verify_password_with_long_password(self) -> None:
        """Test that verification works with a very long password."""
        password = "a" * 1000
        hashed = hash_password(password)
        result = verify_password(password, hashed)
        self.assertTrue(result)

    def test_verify_password_hash_is_argon2(self) -> None:
        """Test that the hash uses argon2 scheme."""
        password = "test"
        hashed = hash_password(password)
        self.assertTrue(hashed.startswith("$argon2"))


if __name__ == "__main__":
    unittest.main()
