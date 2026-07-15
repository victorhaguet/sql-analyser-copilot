"""Tests for the SQL copilot page module."""
from __future__ import annotations

from contextlib import nullcontext
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
    call_confirmation_api,
    render_sql_copilot_page,
    _sync_selected_database,
    _render_header,
    _load_database_catalog,
    _render_database_catalog,
    _render_question_panel,
    _render_result_summary,
    _render_results,
    _render_approval_dialog,
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


class CallConfirmationApiTestCase(SqlCopilotTestCase):
    """Test call_confirmation_api function."""

    def test_returns_error_on_http_exception(self) -> None:
        """Test that call_confirmation_api returns error on HTTP exception."""
        with patch("pages.sql_copilot.httpx.post") as mock_post:
            mock_post.side_effect = httpx.ReadError("Connection refused")

            st.session_state["user"] = {
                "sub": "user-123",
                "role": "admin",
            }

            result = call_confirmation_api(
                thread_id="thread-123",
                decision="approve",
                api_base_url="http://localhost:8000",
            )

            self.assertTrue(result["has_error"])
            self.assertIn("FastAPI request failed:", result["ai_answer"])

    @patch("pages.sql_copilot.httpx.post")
    def test_returns_error_on_http_error(self, mock_post: MagicMock) -> None:
        """Test that call_confirmation_api returns error on HTTP error."""
        mock_response = MagicMock()
        mock_response.is_error = True
        mock_response.json.return_value = {"detail": "Thread not found"}
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        result = call_confirmation_api(
            thread_id="thread-123",
            decision="reject",
            api_base_url="http://localhost:8000",
        )

        self.assertTrue(result["has_error"])
        self.assertIn("Thread not found", result["ai_answer"])

    @patch("pages.sql_copilot.httpx.post")
    def test_handles_invalid_json_response(self, mock_post: MagicMock) -> None:
        """Test that call_confirmation_api handles invalid JSON response."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        result = call_confirmation_api(
            thread_id="thread-123",
            decision="approve",
            api_base_url="http://localhost:8000",
        )

        self.assertFalse(result["has_error"])

    @patch("pages.sql_copilot.httpx.post")
    def test_returns_parsed_response_on_success(self, mock_post: MagicMock) -> None:
        """Test that call_confirmation_api returns parsed response on success."""
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

        result = call_confirmation_api(
            thread_id="thread-123",
            decision="approve",
            api_base_url="http://localhost:8000",
        )

        self.assertFalse(result["has_error"])
        self.assertEqual(result["question"], "Which artists?")
        self.assertEqual(result["sql_query"], "SELECT * FROM Artist")

    @patch("pages.sql_copilot.httpx.post")
    def test_includes_user_headers_in_request(self, mock_post: MagicMock) -> None:
        """Test that call_confirmation_api includes user headers in request."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "editor",
        }

        call_confirmation_api(
            thread_id="thread-456",
            decision="reject",
            api_base_url="http://localhost:8000",
        )

        call_kwargs = mock_post.call_args.kwargs
        headers = call_kwargs["headers"]
        self.assertEqual(headers["X-User-Sub"], "user-123")
        self.assertEqual(headers["X-User-Role"], "editor")


class CallApiTestCase(SqlCopilotTestCase):
    """Test call_api function."""

    def test_returns_empty_state_with_missing_question(self) -> None:
        """Test that call_api returns empty state with missing question."""
        result = call_api(
            question="   ",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_database=None,
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
            selected_database=None,
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
            selected_database="music",
        )

        self.assertTrue(result["has_error"])
        self.assertIn("Database not found", result["ai_answer"])

    @patch("pages.sql_copilot.httpx.post")
    def test_handles_invalid_json_response(self, mock_post: MagicMock) -> None:
        """Test that call_api handles invalid JSON response."""
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        st.session_state["user"] = {
            "sub": "user-123",
            "role": "admin",
        }

        result = call_api(
            question="Test question",
            execution_limit=200,
            api_base_url="http://localhost:8000",
            selected_database="music",
        )

        self.assertFalse(result["has_error"])

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
            selected_database="music",
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
            selected_database=None,
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

    def test_initializes_selected_database(self) -> None:
        """Test that _initialize_state sets selected database."""
        _initialize_state()
        self.assertIsNone(st.session_state["selected_database"])


class RenderHeaderTestCase(SqlCopilotTestCase):
    """Test _render_header function."""

    @patch("pages.sql_copilot.st.container", return_value=nullcontext())
    @patch("pages.sql_copilot.st.title")
    @patch("pages.sql_copilot.st.write")
    def test_renders_header_content(self, mock_write: MagicMock, mock_title: MagicMock, _mock_container: MagicMock) -> None:
        """Test that _render_header renders correct content."""
        _render_header()

        mock_title.assert_called_once_with("Natural language to SQL copilot")
        self.assertTrue(mock_write.called)


class RenderDatabaseCatalogTestCase(SqlCopilotTestCase):
    """Test _render_database_catalog function."""

    @patch("pages.sql_copilot._load_database_catalog", return_value=([], "Config error"))
    @patch("pages.sql_copilot.st.container", return_value=nullcontext())
    @patch("pages.sql_copilot.st.text")
    @patch("pages.sql_copilot.st.error")
    def test_shows_error_on_catalog_error(self, mock_error: MagicMock, _mock_text: MagicMock, _mock_container: MagicMock, _mock_load: MagicMock) -> None:
        """Test that _render_database_catalog shows error on catalog error."""
        result = _render_database_catalog()

        self.assertIsNone(result)
        mock_error.assert_called_once()

    @patch("pages.sql_copilot._load_database_catalog", return_value=([], None))
    @patch("pages.sql_copilot.st.container", return_value=nullcontext())
    @patch("pages.sql_copilot.st.text")
    @patch("pages.sql_copilot.st.error")
    def test_shows_error_on_no_databases(self, mock_error: MagicMock, _mock_text: MagicMock, _mock_container: MagicMock, _mock_load: MagicMock) -> None:
        """Test that _render_database_catalog shows error when no databases."""
        result = _render_database_catalog()

        self.assertIsNone(result)
        mock_error.assert_called_once()

    @patch("pages.sql_copilot._load_database_catalog", return_value=([], None))
    @patch("pages.sql_copilot.st.container", return_value=nullcontext())
    @patch("pages.sql_copilot.st.text")
    @patch("pages.sql_copilot.st.error")
    def test_shows_error_on_no_databases_available(self, mock_error: MagicMock, _mock_text: MagicMock, _mock_container: MagicMock, _mock_load: MagicMock) -> None:
        """Test that _render_database_catalog shows error when no databases available."""
        result = _render_database_catalog()

        self.assertIsNone(result)
        mock_error.assert_called()

    @patch("pages.sql_copilot._load_database_catalog")
    @patch("pages.sql_copilot._sync_selected_database", return_value="newdb")
    @patch("pages.sql_copilot.st.container", return_value=nullcontext())
    @patch("pages.sql_copilot.st.text")
    @patch("pages.sql_copilot.st.selectbox", return_value="newdb")
    @patch("pages.sql_copilot.st.expander", return_value=nullcontext())
    @patch("pages.sql_copilot.st.write")
    @patch("pages.sql_copilot.st.caption")
    @patch("pages.sql_copilot.st.code")
    def test_handles_database_change(self, mock_code: MagicMock, mock_caption: MagicMock, mock_write: MagicMock, _mock_expander: MagicMock, mock_selectbox: MagicMock, _mock_text: MagicMock, _mock_container: MagicMock, _mock_sync: MagicMock, mock_load: MagicMock) -> None:
        """Test that _render_database_catalog handles database change."""
        mock_db = MagicMock()
        mock_db.name = "newdb"
        mock_db.description = "New database"
        mock_db.database.describe.return_value = {
            "tables": ["Table1"],
            "database_path": "/path/to/db.sqlite",
        }
        mock_load.return_value = ([mock_db], None)

        result = _render_database_catalog()

        self.assertEqual(result, "newdb")

    @patch("pages.sql_copilot._load_database_catalog")
    @patch("pages.sql_copilot._sync_selected_database", return_value="music")
    @patch("pages.sql_copilot.st.container", return_value=nullcontext())
    @patch("pages.sql_copilot.st.text")
    @patch("pages.sql_copilot.st.selectbox", return_value="music")
    @patch("pages.sql_copilot.st.expander", return_value=nullcontext())
    @patch("pages.sql_copilot.st.write")
    @patch("pages.sql_copilot.st.caption")
    @patch("pages.sql_copilot.st.code")
    def test_renders_database_options(
        self,
        mock_code: MagicMock,
        mock_caption: MagicMock,
        mock_write: MagicMock,
        _mock_expander: MagicMock,
        _mock_selectbox: MagicMock,
        _mock_text: MagicMock,
        _mock_container: MagicMock,
        _mock_sync: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Test that _render_database_catalog renders database options."""
        mock_db = MagicMock()
        mock_db.name = "music"
        mock_db.description = "Music database"
        mock_db.database.describe.return_value = {
            "tables": ["Artist", "Album"],
            "database_path": "/path/to/db.sqlite",
        }
        mock_load.return_value = ([mock_db], None)

        result = _render_database_catalog()

        self.assertEqual(result, "music")


class RenderQuestionPanelTestCase(SqlCopilotTestCase):
    """Test _render_question_panel function."""

    @patch("pages.sql_copilot.st.form", return_value=nullcontext())
    @patch("pages.sql_copilot.st.text_area")
    @patch("pages.sql_copilot.st.slider", return_value=200)
    @patch("pages.sql_copilot.st.caption")
    @patch("pages.sql_copilot.st.columns", return_value=MagicMock(__getitem__=lambda s, k: MagicMock()))
    @patch("pages.sql_copilot.st.form_submit_button", return_value=True)
    def test_renders_question_form(
        self,
        _mock_submit: MagicMock,
        _mock_columns: MagicMock,
        _mock_caption: MagicMock,
        _mock_slider: MagicMock,
        _mock_text_area: MagicMock,
        _mock_form: MagicMock,
    ) -> None:
        """Test that _render_question_panel renders form."""
        result = _render_question_panel()

        self.assertTrue(result)


class RenderResultSummaryTestCase(SqlCopilotTestCase):
    """Test _render_result_summary function."""

    @patch("pages.sql_copilot.st.columns")
    @patch("pages.sql_copilot.st.metric")
    @patch("pages.sql_copilot.st.info")
    def test_renders_metrics_without_truncation(
        self,
        mock_info: MagicMock,
        mock_metric: MagicMock,
        mock_columns: MagicMock,
    ) -> None:
        """Test that _render_result_summary renders metrics without truncation."""
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_col4 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]
        result_state = {
            "selected_database": "music",
            "query_result": {
                "row_count": 10,
                "truncated": False,
            },
            "has_error": False,
        }
        st.session_state["execution_limit"] = 200

        _render_result_summary(result_state)

        mock_info.assert_not_called()

    @patch("pages.sql_copilot.st.columns", return_value=[MagicMock()] * 4)
    @patch("pages.sql_copilot.st.metric")
    @patch("pages.sql_copilot.st.info")
    def test_shows_truncated_info(
        self,
        mock_info: MagicMock,
        _mock_metric: MagicMock,
        _mock_columns: MagicMock,
    ) -> None:
        """Test that _render_result_summary shows truncated info."""
        result_state = {
            "selected_database": "music",
            "query_result": {
                "row_count": 250,
                "truncated": True,
            },
            "has_error": False,
        }
        st.session_state["execution_limit"] = 200

        _render_result_summary(result_state)

        mock_info.assert_called_once()


class RenderResultsTestCase(SqlCopilotTestCase):
    """Test _render_results function."""

    @patch("pages.sql_copilot._render_result_summary")
    @patch("pages.sql_copilot.st.tabs", return_value=[MagicMock(), MagicMock(), MagicMock()])
    @patch("pages.sql_copilot.st.write")
    @patch("pages.sql_copilot.st.code")
    @patch("pages.sql_copilot.st.dataframe")
    def test_renders_successful_result(
        self,
        mock_dataframe: MagicMock,
        _mock_code: MagicMock,
        _mock_write: MagicMock,
        _mock_tabs: MagicMock,
        _mock_summary: MagicMock,
    ) -> None:
        """Test that _render_results renders successful result."""
        result_state = {
            "question": "Test question",
            "sql_query": "SELECT * FROM test",
            "ai_answer": "Here is the answer",
            "query_result": {
                "rows": [{"id": 1, "name": "test"}],
            },
            "has_error": False,
            "intent": "query",
        }

        _render_results(result_state)

        mock_dataframe.assert_called_once()

    @patch("pages.sql_copilot._render_result_summary")
    @patch("pages.sql_copilot.st.tabs", return_value=[MagicMock(), MagicMock(), MagicMock()])
    @patch("pages.sql_copilot.st.error")
    @patch("pages.sql_copilot.st.code")
    def test_renders_error_result(
        self,
        _mock_code: MagicMock,
        mock_error: MagicMock,
        _mock_tabs: MagicMock,
        _mock_summary: MagicMock,
    ) -> None:
        """Test that _render_results renders error result."""
        result_state = {
            "question": "Test question",
            "sql_query": "SELECT * FROM test",
            "ai_answer": "Error occurred",
            "has_error": True,
            "intent": "query",
        }

        _render_results(result_state)

        mock_error.assert_called_once()

    @patch("pages.sql_copilot._render_result_summary")
    @patch("pages.sql_copilot.st.tabs", return_value=[MagicMock(), MagicMock(), MagicMock()])
    @patch("pages.sql_copilot.st.warning")
    @patch("pages.sql_copilot.st.code")
    def test_renders_rejected_modification(
        self,
        _mock_code: MagicMock,
        mock_warning: MagicMock,
        _mock_tabs: MagicMock,
        _mock_summary: MagicMock,
    ) -> None:
        """Test that _render_results renders rejected modification."""
        result_state = {
            "question": "Test modification",
            "sql_query": "UPDATE test SET x=1",
            "ai_answer": "Modification draft",
            "has_error": False,
            "intent": "modification",
            "execution_confirmed": False,
            "execution_requested": False,
        }

        _render_results(result_state)

        mock_warning.assert_called_once()


class SyncSelectedDatabaseTestCase(SqlCopilotTestCase):
    """Test _sync_selected_database function."""

    def test_returns_none_when_no_databases(self) -> None:
        """Test that _sync_selected_database returns None when no databases."""
        databases = []
        result = _sync_selected_database(databases)
        self.assertIsNone(result)

    def test_returns_first_database_when_none_selected(self) -> None:
        """Test that _sync_selected_database returns first database when none selected."""
        databases = [
            MagicMock(name="db1"),
            MagicMock(name="db2"),
        ]
        databases[0].name = "db1"
        databases[1].name = "db2"

        result = _sync_selected_database(databases)
        self.assertEqual(result, "db1")

    def test_returns_only_available_database(self) -> None:
        """Test that _sync_selected_database filters out unavailable databases."""
        st.session_state["selected_database"] = "db1"
        databases = [
            MagicMock(name="db1"),
            MagicMock(name="db2"),
        ]
        databases[0].name = "db1"
        databases[1].name = "db2"

        result = _sync_selected_database(databases)
        self.assertEqual(result, "db1")


class RenderSqlCopilotPageTestCase(SqlCopilotTestCase):
    """Test render_sql_copilot_page behavior."""

    @patch("pages.sql_copilot.st.container", side_effect=lambda **_: nullcontext())
    @patch("pages.sql_copilot.st.spinner", side_effect=lambda *_args, **_kwargs: nullcontext())
    @patch("pages.sql_copilot._render_results")
    @patch("pages.sql_copilot.call_api")
    @patch("pages.sql_copilot._render_question_panel", return_value=True)
    @patch("pages.sql_copilot._render_database_catalog", return_value="music")
    @patch("pages.sql_copilot._render_header")
    @patch("pages.sql_copilot.render_logout_button")
    @patch("pages.sql_copilot.is_authenticated_page", return_value=True)
    def test_renders_current_run_result(
        self,
        _mock_is_authenticated: MagicMock,
        _mock_render_logout_button: MagicMock,
        _mock_render_header: MagicMock,
        _mock_render_database_catalog: MagicMock,
        _mock_render_question_panel: MagicMock,
        mock_call_api: MagicMock,
        mock_render_results: MagicMock,
        _mock_spinner: MagicMock,
        _mock_container: MagicMock,
    ) -> None:
        """Test that the page renders the latest API result in the same run."""
        st.session_state["question_input"] = "Current question"
        st.session_state["execution_limit"] = 123
        st.session_state["result_state"] = {
            "question": "Previous question",
            "sql_query": "SELECT old",
            "ai_answer": "Previous analysis",
        }
        latest_result = {
            "question": "Current question",
            "sql_query": "SELECT new",
            "ai_answer": "Current analysis",
        }
        mock_call_api.return_value = latest_result

        render_sql_copilot_page()

        self.assertEqual(st.session_state["result_state"], latest_result)
        mock_render_results.assert_called_once_with(latest_result)



    @patch("pages.sql_copilot.st.error")
    @patch("pages.sql_copilot.st.container", side_effect=lambda **_: nullcontext())
    @patch("pages.sql_copilot.st.spinner", side_effect=lambda *_args, **_kwargs: nullcontext())
    @patch("pages.sql_copilot._render_question_panel", return_value=True)
    @patch("pages.sql_copilot._render_database_catalog", return_value=None)
    @patch("pages.sql_copilot._render_header")
    @patch("pages.sql_copilot.render_logout_button")
    @patch("pages.sql_copilot.is_authenticated_page", return_value=True)
    def test_shows_error_when_no_database_selected(
        self,
        _mock_is_authenticated: MagicMock,
        _mock_render_logout_button: MagicMock,
        _mock_render_header: MagicMock,
        _mock_render_database_catalog: MagicMock,
        _mock_render_question_panel: MagicMock,
        _mock_spinner: MagicMock,
        _mock_container: MagicMock,
        mock_error: MagicMock,
    ) -> None:
        """Test that the page shows error when no database selected."""
        st.session_state["question_input"] = "Test question"
        st.session_state["execution_limit"] = 200
        st.session_state["result_state"] = {}

        render_sql_copilot_page()

        mock_error.assert_called_once()

    @patch("pages.sql_copilot.st.error")
    @patch("pages.sql_copilot.st.container", side_effect=lambda **_: nullcontext())
    @patch("pages.sql_copilot._render_results")
    @patch("pages.sql_copilot.call_api")
    @patch("pages.sql_copilot._render_question_panel", return_value=True)
    @patch("pages.sql_copilot._render_database_catalog", return_value="music")
    @patch("pages.sql_copilot._render_header")
    @patch("pages.sql_copilot.render_logout_button")
    @patch("pages.sql_copilot.is_authenticated_page", return_value=True)
    def test_shows_authorization_error(
        self,
        _mock_is_authenticated: MagicMock,
        _mock_render_logout_button: MagicMock,
        _mock_render_header: MagicMock,
        _mock_render_database_catalog: MagicMock,
        _mock_render_question_panel: MagicMock,
        mock_call_api: MagicMock,
        _mock_render_results: MagicMock,
        _mock_container: MagicMock,
        mock_error: MagicMock,
    ) -> None:
        """Test that the page shows authorization error."""
        st.session_state["question_input"] = "Test question"
        st.session_state["execution_limit"] = 200
        st.session_state["result_state"] = {}
        mock_call_api.return_value = {
            "authorization_error": "User not authorized",
        }

        render_sql_copilot_page()

        mock_error.assert_called_once()

    @patch("pages.sql_copilot.st.container", side_effect=lambda **_: nullcontext())
    @patch("pages.sql_copilot.st.spinner", side_effect=lambda *_args, **_kwargs: nullcontext())
    @patch("pages.sql_copilot._render_question_panel", return_value=False)
    @patch("pages.sql_copilot._render_database_catalog", return_value="music")
    @patch("pages.sql_copilot._render_header")
    @patch("pages.sql_copilot.render_logout_button")
    @patch("pages.sql_copilot.is_authenticated_page", return_value=True)
    def test_does_not_call_api_when_not_submitted(
        self,
        _mock_is_authenticated: MagicMock,
        _mock_render_logout_button: MagicMock,
        _mock_render_header: MagicMock,
        _mock_render_database_catalog: MagicMock,
        _mock_render_question_panel: MagicMock,
        _mock_spinner: MagicMock,
        _mock_container: MagicMock,
    ) -> None:
        """Test that the page does not call API when form not submitted."""
        st.session_state["question_input"] = "Test question"
        st.session_state["execution_limit"] = 200
        st.session_state["result_state"] = {}

        with patch("pages.sql_copilot.call_api") as mock_call_api:
            render_sql_copilot_page()
            mock_call_api.assert_not_called()



    @patch("pages.sql_copilot.show_login_page")
    @patch("pages.sql_copilot.is_authenticated_page", return_value=False)
    def test_shows_login_when_not_authenticated(
        self,
        _mock_is_authenticated: MagicMock,
        mock_show_login: MagicMock,
    ) -> None:
        """Test that the page shows login when not authenticated."""
        render_sql_copilot_page()

        mock_show_login.assert_called_once()


if __name__ == "__main__":
    unittest.main()
