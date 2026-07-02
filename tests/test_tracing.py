"""Tests for trace logging utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from tracing import (
    extract_trace_steps_from_content,
    format_trace_header,
    format_trace_payload,
    format_trace_step,
    build_trace_log_content,
    write_trace_log,
    TRACE_LOG_DIR
)
from tools.database import QueryResult


def build_trace_step(
    node: str,
    update: dict,
    *,
    outcome: str = "success",
    state: dict | None = None,
) -> dict:
    """Build a normalized trace step matching the runtime trace schema."""
    return {
        "node": node,
        "update": update,
        "state": state or update.copy(),
        "outcome": outcome,
    }


class TracingTestCase(unittest.TestCase):
    """Test trace logging and formatting utilities."""

    def test_format_trace_header(self) -> None:
        """format_trace_header should center title with = padding."""
        
        result = format_trace_header("Test Title")
        
        self.assertIn("Test Title", result)
        self.assertEqual(result[0], "=")
        self.assertEqual(result[-1], "=")
        self.assertEqual(len(result), 80)

    def test_format_trace_payload_none(self) -> None:
        """format_trace_payload should return (none) for None."""
        
        result = format_trace_payload(None)
        self.assertEqual(result, "(none)")

    def test_format_trace_payload_string(self) -> None:
        """format_trace_payload should return string as-is."""
        
        result = format_trace_payload("test string")
        self.assertEqual(result, "test string")

    def test_format_trace_payload_query_result(self) -> None:
        """format_trace_payload should format QueryResult."""
        
        
        result = format_trace_payload(
            QueryResult(
                columns=["id", "name"],
                rows=[{"id": 1, "name": "Test"}],
                row_count=1,
                truncated=False,
            )
        )
        
        self.assertIn("Columns: id, name", result)
        self.assertIn("Row Count: 1", result)
        self.assertIn("Truncated: False", result)

    def test_format_trace_step_database_selector(self) -> None:
        """format_trace_step should format database_selector node."""
        
        step = build_trace_step(
            "database_selector",
            {
                "metadata": {
                    "selected_database": "test_db",
                    "candidate_databases": ["db1", "db2"],
                    "database_selection_reason": "Matches question",
                }
            },
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Node: database_selector", result)
        self.assertIn("Outcome: success", result)
        self.assertIn("Selected Database: test_db", result)
        self.assertIn("Candidate Databases: db1, db2", result)
        self.assertIn("Reason: Matches question", result)

    def test_format_trace_step_database_selector_with_error(self) -> None:
        """database_selector should include execution errors in the trace."""
        result = format_trace_step(
            build_trace_step(
                "database_selector",
                {"metadata": {}, "execution_error": "No database matched"},
                outcome="database_selection_failed",
            )
        )
        self.assertIn("Error: No database matched", result)

    def test_format_trace_step_intent_classifier_with_confirmation(self) -> None:
        """intent_classifier should note when confirmation is required."""
        result = format_trace_step(
            build_trace_step(
                "intent_classifier",
                {"intent": "modification", "needs_confirmation": True},
                outcome="Intention classified",
            )
        )
        self.assertIn("User intent: modification", result)
        self.assertIn("Needs user's confirmation.", result)

    def test_format_trace_step_intent_classifier_with_error(self) -> None:
        """intent_classifier should include classification errors."""
        result = format_trace_step(
            build_trace_step(
                "intent_classifier",
                {"intent_error": "Model did not return valid JSON"},
                outcome="intent_failed",
            )
        )
        self.assertIn("Error: Model did not return valid JSON", result)

    def test_format_trace_step_role_authorizer(self) -> None:
        """role_authorizer should render the current user role and authorization error."""
        result = format_trace_step(
            build_trace_step(
                "role_authorizer",
                {"user_role": "readonly", "authorization_error": "Writes are forbidden"},
                outcome="authorization_failed",
            )
        )
        self.assertIn("User Role: readonly", result)
        self.assertIn("Error: Writes are forbidden", result)

    def test_format_trace_step_sql_generator(self) -> None:
        """format_trace_step should format sql_generator node."""
        
        step = build_trace_step(
            "sql_generator",
            {
                "generated_sql": "SELECT * FROM test",
            },
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Node: sql_generator", result)
        self.assertIn("Outcome: success", result)
        self.assertIn("SELECT * FROM test", result)

    def test_format_trace_step_sql_validator_success(self) -> None:
        """format_trace_step should format sql_validator with validated SQL."""
        
        step = build_trace_step(
            "sql_validator",
            {
                "validated_sql": "SELECT * FROM test WHERE id > 0",
            },
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Name: sql_validator", result)
        self.assertIn("SELECT * FROM test WHERE id > 0", result)

    def test_format_trace_step_sql_validator_error(self) -> None:
        """format_trace_step should format sql_validator with error."""
        
        step = build_trace_step(
            "sql_validator",
            {
                "sql_validation_error": "Invalid syntax",
            },
            outcome="error",
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Name: sql_validator", result)
        self.assertIn("Invalid syntax", result)

    def test_format_trace_step_interrupt(self) -> None:
        """interrupt steps should include the request and options."""
        result = format_trace_step(
            build_trace_step(
                "__interrupt__",
                {
                    "interrupt": {
                        "request": "Approve this SQL change?",
                        "options": ["approve", "reject"],
                    }
                },
                outcome="execution_pending_approval",
            )
        )
        self.assertIn("Request: Approve this SQL change?", result)
        self.assertIn("Options: approve/reject", result)

    def test_format_trace_step_sql_modification_validator_approved(self) -> None:
        """modification validator should render approved execution."""
        result = format_trace_step(
            build_trace_step(
                "sql_modification_validator",
                {"execution_confirmed": True},
            )
        )
        self.assertIn("Name: sql_modification_validator", result)
        self.assertIn("Request was approved.", result)

    def test_format_trace_step_sql_modification_validator_without_confirmation(self) -> None:
        """modification validator should render validated SQL before confirmation."""
        result = format_trace_step(
            build_trace_step(
                "sql_modification_validator",
                {"validated_sql": "DELETE FROM artist WHERE ArtistId = 1"},
            )
        )
        self.assertIn("DELETE FROM artist WHERE ArtistId = 1", result)

    def test_format_trace_step_sql_executor_success(self) -> None:
        """format_trace_step should format sql_executor with result."""

        step = build_trace_step(
            "sql_executor",
            {
                "query_result": QueryResult(
                    columns=["id"],
                    rows=[{"id": 1}],
                    row_count=1,
                    truncated=False,
                ),
            },
            state={
                "intent": "query",
                "validated_sql": "SELECT * FROM test",
            },
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Outcome: success", result)
        self.assertIn("SQL:", result)
        self.assertIn("SELECT * FROM test", result)
        self.assertIn("Columns: id", result)

    def test_format_trace_step_sql_executor_modification_success(self) -> None:
        """sql_executor should render successful modification execution."""
        result = format_trace_step(
            build_trace_step(
                "sql_executor",
                {},
                state={
                    "intent": "modification",
                    "generated_sql": "UPDATE artist SET Name = 'A' WHERE ArtistId = 1",
                },
            )
        )
        self.assertIn("UPDATE artist SET Name = 'A' WHERE ArtistId = 1", result)
        self.assertIn("The SQL request was properly executed.", result)

    def test_format_trace_step_sql_executor_error(self) -> None:
        """format_trace_step should format sql_executor with error."""
        
        step = build_trace_step(
            "sql_executor",
            {
                "execution_error": "Connection failed",
            },
            state={
                "intent": "query",
                "validated_sql": "SELECT * FROM test",
            },
            outcome="error",
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Connection failed", result)

    def test_format_trace_step_analyst(self) -> None:
        """format_trace_step should format analyst node."""
        
        step = build_trace_step(
            "result_analyst",
            {
                "analysis": "The data shows a trend.",
            },
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Graph step", result)
        self.assertIn("The data shows a trend.", result)

    def test_extract_trace_steps_from_content_skips_human_message_headers(self) -> None:
        """Existing trace extraction should strip human-message headers and questions."""
        content = "\n".join(
            [
                format_trace_header("Human Message"),
                "First question",
                "step one",
                "step two",
                format_trace_header("Human Message"),
                "Second question",
                "step three",
            ]
        )
        result = extract_trace_steps_from_content(content)
        self.assertEqual(result, ["step one", "step two", "step three"])

    def test_build_trace_log_content(self) -> None:
        """build_trace_log_content should build complete trace log."""

        trace = [
            build_trace_step("database_selector", {"metadata": {}}),
        ]
        
        result = build_trace_log_content("Test question", trace)
        
        self.assertIn("Human Message", result)
        self.assertIn("Test question", result)
        self.assertIn("Graph step", result)

    def test_write_trace_log_creates_file(self) -> None:
        """write_trace_log should create log file."""
        
        trace = [
            build_trace_step("database_selector", {"metadata": {}}),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = write_trace_log("Test question", trace, temp_dir)
            
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.parent, Path(temp_dir))
            
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("Test question", content)

    def test_write_trace_log_appends_steps_to_existing_file_from_dict_path(self) -> None:
        """write_trace_log should append new steps to an existing trace file."""
        old_trace = [build_trace_step("sql_generator", {"generated_sql": "SELECT 1"})]
        new_trace = [build_trace_step("result_analyst", {"analysis": "Returned one row."})]

        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = write_trace_log("Original question", old_trace, temp_dir)

            returned_path = write_trace_log(
                "Ignored new question",
                new_trace,
                temp_dir,
                {"path": str(existing_path)},
            )

            self.assertEqual(returned_path, existing_path)
            content = existing_path.read_text(encoding="utf-8")
            self.assertIn("Original question", content)
            self.assertIn("SELECT 1", content)
            self.assertIn("Returned one row.", content)
            self.assertNotIn("Ignored new question", content)

    def test_write_trace_log_appends_steps_to_existing_file_from_trace_log_path(self) -> None:
        """write_trace_log should accept trace_log_path dict inputs for existing files."""
        trace = [build_trace_step("result_analyst", {"analysis": "Second pass."})]

        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = write_trace_log(
                "Original question",
                [build_trace_step("sql_generator", {"generated_sql": "SELECT 1"})],
                temp_dir,
            )

            returned_path = write_trace_log(
                "Another question",
                trace,
                temp_dir,
                {"trace_log_path": str(existing_path)},
            )

            self.assertEqual(returned_path, existing_path)
            self.assertIn("Second pass.", existing_path.read_text(encoding="utf-8"))

    def test_write_trace_log_uses_default_dir(self) -> None:
        """write_trace_log should use default TRACE_LOG_DIR when not specified."""
        
        trace = [
            build_trace_step("database_selector", {"metadata": {}}),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tracing.TRACE_LOG_DIR", Path(temp_dir)):
                log_path = write_trace_log("Test question", trace, None)
                self.assertTrue(log_path.exists())
                self.assertTrue(str(log_path).startswith(str(Path(temp_dir))))


if __name__ == "__main__":
    unittest.main()
