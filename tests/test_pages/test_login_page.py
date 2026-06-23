"""Tests for the login page module."""
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
from pages import login


class LoginPageTestCase(unittest.TestCase):
    """Base test case for login page tests."""

    def setUp(self) -> None:
        self._original_session_state = dict(st.session_state)
        st.session_state.clear()

    def tearDown(self) -> None:
        st.session_state.clear()
        st.session_state.update(self._original_session_state)


class ShowLoginPageTestCase(LoginPageTestCase):
    """Test show_login_page function."""

    def test_sets_default_api_base_url(self) -> None:
        """Test that show_login_page sets default API base URL."""
        st.session_state.clear()
        login.show_login_page()

        self.assertEqual(
            st.session_state["api_base_url"], login.DEFAULT_API_BASE_URL
        )

    def test_uses_provided_api_base_url(self) -> None:
        """Test that show_login_page uses provided API base URL."""
        st.session_state.clear()
        login.show_login_page("http://custom-url:9000")

        self.assertEqual(st.session_state["api_base_url"], "http://custom-url:9000")

    def test_overwrites_session_api_base_url(self) -> None:
        """Test that show_login_page overwrites session API base URL."""
        st.session_state["api_base_url"] = "http://old-url:8000"
        login.show_login_page("http://new-url:9000")

        self.assertEqual(st.session_state["api_base_url"], "http://new-url:9000")

    def test_uses_session_api_base_url_when_not_provided(self) -> None:
        """Test that show_login_page uses session API base URL when not provided."""
        st.session_state["api_base_url"] = "http://session-url:8000"
        login.show_login_page()

        self.assertEqual(st.session_state["api_base_url"], "http://session-url:8000")

    @patch("pages.login.httpx.post")
    def test_successful_login_sets_user(self, mock_post: MagicMock) -> None:
        """Test that successful login sets user in session state."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = {
            "sub": "user-123",
            "username": "testuser",
            "name": "Test User",
            "role": "admin",
            "is_active": True,
        }
        mock_post.return_value = mock_response

        st.session_state.clear()
        st.form_submit_button = MagicMock(return_value=True)

        with patch.object(st, "text_input", side_effect=["testuser", "password"]):
            with patch.object(st, "form"):
                login.show_login_page()

        self.assertIn("user", st.session_state)
        self.assertEqual(st.session_state["user"]["username"], "testuser")

    @patch("pages.login.httpx.post")
    def test_successful_login_sets_authenticated_flag(self, mock_post: MagicMock) -> None:
        """Test that successful login sets user_authenticated flag."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = {
            "sub": "user-123",
            "username": "testuser",
            "role": "admin",
            "is_active": True,
        }
        mock_post.return_value = mock_response

        st.session_state.clear()
        st.form_submit_button = MagicMock(return_value=True)

        with patch.object(st, "text_input", side_effect=["testuser", "password"]):
            with patch.object(st, "form"):
                login.show_login_page()

        self.assertTrue(st.session_state["user_authenticated"])

    @patch("pages.login.admin_credentials_configured")
    @patch("pages.login.is_admin_account_created")
    @patch("pages.login.httpx.post")
    def test_failed_login_shows_error(
        self, mock_post: MagicMock, mock_is_created: MagicMock, mock_configured: MagicMock
    ) -> None:
        """Test that failed login shows error message."""
        mock_is_created.return_value = True
        mock_configured.return_value = True
        
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_post.return_value = mock_response

        st.session_state.clear()
        st.form_submit_button = MagicMock(return_value=True)

        with patch.object(st, "text_input", side_effect=["testuser", "password"]):
            with patch.object(st, "form"):
                with patch.object(st, "error") as mock_error:
                    login.show_login_page()
                    mock_error.assert_called_once_with("Invalid username or password.")

    @patch("pages.login.admin_credentials_configured")
    @patch("pages.login.is_admin_account_created")
    def test_empty_credentials_shows_error(
        self, mock_is_created: MagicMock, mock_configured: MagicMock
    ) -> None:
        """Test that empty credentials show error message."""
        mock_is_created.return_value = True
        mock_configured.return_value = True
        
        st.session_state.clear()
        st.form_submit_button = MagicMock(return_value=True)

        with patch.object(st, "text_input", return_value=""):
            with patch.object(st, "form"):
                with patch.object(st, "error") as mock_error:
                    login.show_login_page()
                    mock_error.assert_called_once_with(
                        "Please enter both username and password."
                    )


class IsAuthenticatedPageTestCase(LoginPageTestCase):
    """Test is_authenticated_page function."""

    def test_returns_true_when_user_authenticated(self) -> None:
        """Test that is_authenticated_page returns True when user is authenticated."""
        st.session_state["user_authenticated"] = True
        self.assertTrue(login.is_authenticated_page())

    def test_returns_false_when_not_authenticated(self) -> None:
        """Test that is_authenticated_page returns False when not authenticated."""
        st.session_state["user_authenticated"] = False
        self.assertFalse(login.is_authenticated_page())

    def test_returns_false_when_key_missing(self) -> None:
        """Test that is_authenticated_page returns False when key is missing."""
        self.assertFalse(login.is_authenticated_page())


if __name__ == "__main__":
    unittest.main()
