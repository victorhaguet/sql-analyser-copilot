"""Tests for the auth helpers module."""
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
from pages import auth


class AuthHelpersTestCase(unittest.TestCase):
    """Base test case for auth helpers."""

    def setUp(self) -> None:
        self._original_session_state = dict(st.session_state)
        st.session_state.clear()

    def tearDown(self) -> None:
        st.session_state.clear()
        st.session_state.update(self._original_session_state)


class IsAuthenticatedPageTestCase(AuthHelpersTestCase):
    """Test is_authenticated_page function."""

    def test_returns_false_when_not_authenticated(self) -> None:
        """Test that is_authenticated_page returns False when not logged in."""
        st.session_state["user_authenticated"] = False
        self.assertFalse(auth.is_authenticated_page())

    def test_returns_true_when_authenticated(self) -> None:
        """Test that is_authenticated_page returns True when logged in."""
        st.session_state["user_authenticated"] = True
        self.assertTrue(auth.is_authenticated_page())

    def test_returns_false_when_key_missing(self) -> None:
        """Test that is_authenticated_page returns False when key is missing."""
        self.assertFalse(auth.is_authenticated_page())


class IsAdminTestCase(AuthHelpersTestCase):
    """Test is_admin function."""

    def test_returns_false_when_no_user(self) -> None:
        """Test that is_admin returns False when no user in session state."""
        self.assertFalse(auth.is_admin())

    def test_returns_false_when_no_role(self) -> None:
        """Test that is_admin returns False when user has no role."""
        st.session_state["user"] = {"sub": "123", "username": "test"}
        self.assertFalse(auth.is_admin())

    def test_returns_false_for_non_admin_role(self) -> None:
        """Test that is_admin returns False for non-admin roles."""
        st.session_state["user"] = {
            "sub": "123",
            "username": "test",
            "role": "readonly",
        }
        self.assertFalse(auth.is_admin())

    def test_returns_true_for_admin_role(self) -> None:
        """Test that is_admin returns True for admin role."""
        st.session_state["user"] = {
            "sub": "123",
            "username": "test",
            "role": "admin",
        }
        self.assertTrue(auth.is_admin())


class RequireAuthTestCase(AuthHelpersTestCase):
    """Test require_auth function."""

    def test_does_not_stop_when_authenticated(self) -> None:
        """Test that require_auth does not stop when user is authenticated."""
        st.session_state["user_authenticated"] = True
        try:
            auth.require_auth()
        except Exception:
            self.fail("require_auth() raised exception unexpectedly")

    def test_shows_error_and_stops_when_not_authenticated(self) -> None:
        """Test that require_auth shows error and stops when not authenticated."""
        st.session_state["user_authenticated"] = False

        with patch.object(st, "error") as mock_error:
            with patch("pages.auth.st.stop") as mock_stop:
                auth.require_auth()
                mock_error.assert_called_once_with("Please log in first.")
                mock_stop.assert_called_once()


class RequireAdminTestCase(AuthHelpersTestCase):
    """Test require_admin function."""

    def test_does_not_stop_when_admin(self) -> None:
        """Test that require_admin does not stop when user is admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "admin",
            "role": "admin",
        }
        try:
            auth.require_admin()
        except Exception:
            self.fail("require_admin() raised exception unexpectedly")

    def test_shows_error_and_stops_when_not_admin(self) -> None:
        """Test that require_admin shows error and stops when not admin."""
        st.session_state["user_authenticated"] = True
        st.session_state["user"] = {
            "sub": "123",
            "username": "user",
            "role": "readonly",
        }

        with patch.object(st, "error") as mock_error:
            with patch("pages.auth.st.stop") as mock_stop:
                auth.require_admin()
                mock_error.assert_called_once_with(
                    "Your account is not authorized to access this page."
                )
                mock_stop.assert_called_once()

    def test_shows_error_and_stops_when_no_user(self) -> None:
        """Test that require_admin shows error and stops when no user."""
        st.session_state["user_authenticated"] = True

        with patch.object(st, "error") as mock_error:
            with patch("pages.auth.st.stop") as mock_stop:
                auth.require_admin()
                mock_error.assert_called_once_with(
                    "Your account is not authorized to access this page."
                )
                mock_stop.assert_called_once()


class GetApiClientTestCase(AuthHelpersTestCase):
    """Test get_api_client function."""

    def test_returns_httpx_client(self) -> None:
        """Test that get_api_client returns an httpx.Client."""
        client = auth.get_api_client()
        self.assertIsInstance(client, auth.httpx.Client)

    def test_client_has_base_url(self) -> None:
        """Test that client has correct base URL."""
        st.session_state["api_base_url"] = "http://localhost:8000"
        client = auth.get_api_client()
        self.assertEqual(str(client.base_url), "http://localhost:8000")

    def test_client_has_user_headers(self) -> None:
        """Test that client includes user headers."""
        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }
        client = auth.get_api_client()
        self.assertEqual(client.headers["X-User-Sub"], "user-123")
        self.assertEqual(client.headers["X-User-Role"], "admin")

    def test_client_has_default_timeout(self) -> None:
        """Test that client has default timeout."""
        client = auth.get_api_client()
        self.assertEqual(client.timeout.read, 30.0)


class RenderLogoutButtonTestCase(AuthHelpersTestCase):
    """Test render_logout_button function."""

    def test_shows_username(self) -> None:
        """Test that render_logout_button shows username."""
        st.session_state["user"] = {
            "username": "testuser",
            "sub": "123",
            "role": "admin",
        }

        with patch.object(st, "columns") as mock_columns:
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]

            with patch.object(st, "caption") as mock_caption:
                auth.render_logout_button()
                mock_caption.assert_called_with("Signed in as testuser")

    def test_shows_sign_out_button(self) -> None:
        """Test that render_logout_button shows sign out button."""
        st.session_state["user"] = {
            "username": "testuser",
            "sub": "123",
            "role": "admin",
        }

        with patch.object(st, "columns") as mock_columns:
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]

            with patch.object(st, "button") as mock_button:
                mock_button.return_value = False
                auth.render_logout_button()
                mock_button.assert_called_once()


class LogoutTestCase(AuthHelpersTestCase):
    """Test logout function."""

    def test_removes_user_from_session(self) -> None:
        """Test that logout removes user from session state."""
        st.session_state["user"] = {
            "sub": "123",
            "username": "test",
            "role": "admin",
        }
        st.session_state["user_authenticated"] = True

        auth.logout()

        self.assertNotIn("user", st.session_state)

    def test_sets_user_authenticated_false(self) -> None:
        """Test that logout sets user_authenticated to False."""
        st.session_state["user_authenticated"] = True

        auth.logout()

        self.assertFalse(st.session_state["user_authenticated"])

    def test_triggers_rerun(self) -> None:
        """Test that logout triggers st.rerun."""
        st.rerun = MagicMock()
        auth.logout()
        st.rerun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
