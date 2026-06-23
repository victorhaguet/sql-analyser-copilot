"""Tests for the user management page module."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

import streamlit as st
from pages.user_management import (
    _fetch_users,
    _create_user,
    _delete_user,
    _update_user_active,
)


class UserManagementTestCase(unittest.TestCase):
    """Base test case for user management tests."""

    def setUp(self) -> None:
        self._original_session_state = dict(st.session_state)
        st.session_state.clear()

    def tearDown(self) -> None:
        st.session_state.clear()
        st.session_state.update(self._original_session_state)


class FetchUsersTestCase(UserManagementTestCase):
    """Test _fetch_users function."""

    def test_returns_empty_list_when_not_admin(self) -> None:
        """Test that _fetch_users returns empty list when not admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "readonly",
            "role": "readonly",
        }

        with patch("pages.auth.is_admin", return_value=False):
            with patch("streamlit.stop", side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    _fetch_users()

    @patch("pages.user_management.get_api_client")
    def test_returns_users_on_success(self, mock_get_client: MagicMock) -> None:
        """Test that _fetch_users returns users on success."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = [
            {"sub": "user1", "username": "user1", "role": "readonly", "is_active": True},
            {"sub": "user2", "username": "user2", "role": "editor", "is_active": True},
        ]
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = _fetch_users()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["username"], "user1")

    @patch("pages.user_management.get_api_client")
    def test_shows_error_on_api_failure(self, mock_get_client: MagicMock) -> None:
        """Test that _fetch_users shows error on API failure."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.text = "Unauthorized"
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        with patch.object(st, "error") as mock_error:
            result = _fetch_users()
            mock_error.assert_called_once_with("Failed to load users: Unauthorized")
            self.assertEqual(result, [])


class CreateUserTestCase(UserManagementTestCase):
    """Test _create_user function."""

    def test_returns_false_when_not_admin(self) -> None:
        """Test that _create_user returns False when not admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "user",
            "role": "user",
        }

        with patch("pages.auth.is_admin", return_value=False):
            with patch("streamlit.stop", side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    _create_user("test_user", "password", "Test User", "user")

    @patch("pages.user_management.get_api_client")
    def test_returns_true_on_success(self, mock_get_client: MagicMock) -> None:
        """Test that _create_user returns True on success."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = _create_user(
            "newuser", "pass123", "New User", "readonly"
        )

        self.assertTrue(result)
        mock_client.post.assert_called_once()

    @patch("pages.user_management.get_api_client")
    def test_returns_false_on_api_failure(self, mock_get_client: MagicMock) -> None:
        """Test that _create_user returns False on API failure."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.text = "Username already exists"
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        with patch.object(st, "error") as mock_error:
            result = _create_user(
                "newuser", "pass123", "New User", "readonly"
            )
            mock_error.assert_called_once_with(
                "Failed to create user: Username already exists"
            )
            self.assertFalse(result)


class UpdateUserRoleTestCase(UserManagementTestCase):
    """Test _update_user_role function."""

    def test_returns_false_when_not_admin(self) -> None:
        """Test that _delete_user returns False when not admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "readonly",
            "role": "readonly",
        }

        with patch("pages.auth.is_admin", return_value=False):
            with patch("streamlit.stop", side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    _delete_user("user-123")


class UpdateUserActiveTestCase(UserManagementTestCase):
    """Test _update_user_active function."""

    def test_returns_false_when_not_admin(self) -> None:
        """Test that _update_user_active returns False when not admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "readonly",
            "role": "readonly",
        }

        with patch("pages.auth.is_admin", return_value=False):
            with patch("streamlit.stop", side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    _update_user_active("user-123", True)

    @patch("pages.user_management.get_api_client")
    def test_returns_true_on_success(self, mock_get_client: MagicMock) -> None:
        """Test that _update_user_active returns True on success."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_client.put.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = _update_user_active("user-123", True)

        self.assertTrue(result)
        mock_client.put.assert_called_once()

    @patch("pages.user_management.get_api_client")
    def test_shows_error_on_api_failure(self, mock_get_client: MagicMock) -> None:
        """Test that _update_user_active shows error on API failure."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.text = "User not found"
        mock_client.put.return_value = mock_response
        mock_get_client.return_value = mock_client

        with patch.object(st, "error") as mock_error:
            result = _update_user_active("user-123", True)
            mock_error.assert_called_once_with("Failed to update status: User not found")
            self.assertFalse(result)


class DeleteUserTestCase(UserManagementTestCase):
    """Test _delete_user function."""

    def test_returns_false_when_not_admin(self) -> None:
        """Test that _delete_user returns False when not admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "readonly",
            "role": "readonly",
        }

        with patch("pages.auth.is_admin", return_value=False):
            with patch("streamlit.stop", side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    _delete_user("user-123")

    @patch("pages.user_management.get_api_client")
    def test_returns_true_on_success(self, mock_get_client: MagicMock) -> None:
        """Test that _delete_user returns True on success."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_client.delete.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = _delete_user("user-123")

        self.assertTrue(result)
        mock_client.delete.assert_called_once()

    @patch("pages.user_management.get_api_client")
    def test_shows_error_on_api_failure(self, mock_get_client: MagicMock) -> None:
        """Test that _delete_user shows error on API failure."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.text = "Cannot delete self"
        mock_client.delete.return_value = mock_response
        mock_get_client.return_value = mock_client

        with patch.object(st, "error") as mock_error:
            result = _delete_user("user-123")
            mock_error.assert_called_once_with("Failed to delete user: Cannot delete self")
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
