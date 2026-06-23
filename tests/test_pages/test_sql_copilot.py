"""Tests for the SQL copilot page module."""
from __future__ import annotations

import httpx
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

import streamlit as st
from pages.sql_copilot import (
    _initialize_state,
    call_api,
    _sync_selected_databases,
    DEFAULT_QUESTION,
)


class SqlCopilotTestCase(unittest.TestCase):
    """Base test case for SQL copilot tests."""

    def setUp(self) -> None:
        self._original_session_state = dict(st.session_state)
        st.session_state.clear()

    def tearDown(self) -> None:
        st.session_state.clear()
        st.session_state.update(self._original_session_state)


class CallApiTestCase(SqlCopilotTestCase):
    """Test call_api function."""

    def test_returns_empty_state_with_missing_question(self) -> None:
        """Test that call_api returns empty state with missing question."""
        result = call_api(
            question="   ",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_databases=[],
        )

        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_message"], "Missing question")
        self.assertEqual(result["ai_answer"], "Enter a question before running the analysis.")

    @patch("pages.sql_copilot.httpx.post")
    def test_returns_error_on_http_exception(self, mock_post: MagicMock) -> None:
        """Test that call_api returns error on HTTP exception."""
        mock_post.side_effect = httpx.ReadError("Connection refused")

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        result = call_api(
            question="What is your name?",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_databases=[],
        )

        self.assertTrue(result["has_error"])
        self.assertIn("FastAPI request failed:", result["ai_answer"])

    @patch("pages.sql_copilot.httpx.post")
    def test_returns_error_on_http_error(self, mock_post: MagicMock) -> None:
        """Test that call_api returns error on HTTP error."""
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.json.return_value = {"detail": "Database not found"}
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        result = call_api(
            question="Which artists?",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_databases=["music"],
        )

        self.assertTrue(result["has_error"])
        self.assertIn("Database not found", result["ai_answer"])

    @patch("pages.sql_copilot.httpx.post")
    def test_returns_parsed_response_on_success(self, mock_post: MagicMock) -> None:
        """Test that call_api returns parsed response on success."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = {
            "question": "Which artists?",
            "validated_sql": "SELECT * FROM Artist",
            "analysis": "Here are the artists",
            "query_result": {
                "columns": ["Id", "Name"],
                "rows": [{"Id": 1, "Name": "AC/DC"}],
                "row_count": 1,
            },
        }
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        result = call_api(
            question="Which artists?",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_databases=["music"],
        )

        self.assertFalse(result["has_error"])
        self.assertEqual(result["question"], "Which artists?")
        self.assertEqual(result["sql_query"], "SELECT * FROM Artist")
        self.assertEqual(result["ai_answer"], "Here are the artists")

    @patch("pages.sql_copilot.httpx.post")
    def test_includes_user_headers_in_request(self, mock_post: MagicMock) -> None:
        """Test that call_api includes user headers in request."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        call_api(
            question="Test",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_databases=[],
        )

        call_kwargs = mock_post.call_args.kwargs
        headers = call_kwargs["headers"]
        self.assertEqual(headers["X-User-Sub"], "user-123")
        self.assertEqual(headers["X-User-Role"], "admin")


class InitializeStateTestCase(SqlCopilotTestCase):
    """Test _initialize_state function."""

    def test_initializes_question_input(self) -> None:
        """Test that _initialize_state sets question input."""
        _initialize_state()
        self.assertEqual(st.session_state["question_input"], DEFAULT_QUESTION)

    def test_initializes_result_state(self) -> None:
        """Test that _initialize_state sets result state."""
        _initialize_state()
        self.assertIn("has_error", st.session_state["result_state"])

    def test_initializes_selected_databases(self) -> None:
        """Test that _initialize_state sets selected databases."""
        _initialize_state()
        self.assertEqual(st.session_state["selected_databases"], [])


class SyncSelectedDatabasesTestCase(SqlCopilotTestCase):
    """Test _sync_selected_databases function."""

    def test_returns_empty_list_when_no_databases(self) -> None:
        """Test that _sync_selected_databases returns empty list when no databases."""
        databases = []
        result = _sync_selected_databases(databases)
        self.assertEqual(result, [])

    def test_returns_all_databases_when_none_selected(self) -> None:
        """Test that _sync_selected_databases returns all databases when none selected."""
        databases = [
            MagicMock(name="db1"),
            MagicMock(name="db2"),
        ]
        databases[0].name = "db1"
        databases[1].name = "db2"

        result = _sync_selected_databases(databases)
        self.assertEqual(result, ["db1", "db2"])

    def test_returns_only_available_databases(self) -> None:
        """Test that _sync_selected_databases filters out unavailable databases."""
        st.session_state["selected_databases"] = ["db1", "db3"]
        databases = [
            MagicMock(name="db1"),
            MagicMock(name="db2"),
        ]
        databases[0].name = "db1"
        databases[1].name = "db2"

        result = _sync_selected_databases(databases)
        self.assertEqual(result, ["db1"])


if __name__ == "__main__":
    unittest.main()
