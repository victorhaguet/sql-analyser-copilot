"""Tests for trace logging utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from tracing import (
    format_trace_header,
    format_trace_payload,
    format_trace_step,
    build_trace_log_content,
    write_trace_log,
    TRACE_LOG_DIR
)
from tools.database import QueryResult

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
        
        step = {
            "node": "database_selector",
            "update": {
                "metadata": {
                    "selected_database": "test_db",
                    "candidate_databases": ["db1", "db2"],
                    "database_selection_reason": "Matches question",
                }
            },
            "outcome": "success",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Node: database_selector", result)
        self.assertIn("Outcome: success", result)
        self.assertIn("Selected Database: test_db", result)
        self.assertIn("Candidate Databases: db1, db2", result)
        self.assertIn("Reason: Matches question", result)

    def test_format_trace_step_sql_generator(self) -> None:
        """format_trace_step should format sql_generator node."""
        
        step = {
            "node": "sql_generator",
            "update": {
                "generated_sql": "SELECT * FROM test",
            },
            "outcome": "success",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Node: sql_generator", result)
        self.assertIn("Outcome: success", result)
        self.assertIn("SELECT * FROM test", result)

    def test_format_trace_step_sql_validator_success(self) -> None:
        """format_trace_step should format sql_validator with validated SQL."""
        
        step = {
            "node": "sql_validator",
            "update": {
                "validated_sql": "SELECT * FROM test WHERE id > 0",
            },
            "outcome": "success",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Name: sql_validator", result)
        self.assertIn("SELECT * FROM test WHERE id > 0", result)

    def test_format_trace_step_sql_validator_error(self) -> None:
        """format_trace_step should format sql_validator with error."""
        
        step = {
            "node": "sql_validator",
            "update": {
                "sql_validation_error": "Invalid syntax",
            },
            "outcome": "error",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Name: sql_validator", result)
        self.assertIn("Invalid syntax", result)

    def test_format_trace_step_sql_executor_success(self) -> None:
        """format_trace_step should format sql_executor with result."""

        step = {
            "node": "sql_executor",
            "update": {
                "query_result": QueryResult(
                    columns=["id"],
                    rows=[{"id": 1}],
                    row_count=1,
                    truncated=False,
                ),
            },
            "state": {
                "validated_sql": "SELECT * FROM test",
            },
            "outcome": "success",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Outcome: success", result)
        self.assertIn("SQL:", result)
        self.assertIn("SELECT * FROM test", result)
        self.assertIn("Columns: id", result)

    def test_format_trace_step_sql_executor_error(self) -> None:
        """format_trace_step should format sql_executor with error."""
        
        step = {
            "node": "sql_executor",
            "update": {
                "execution_error": "Connection failed",
            },
            "state": {
                "validated_sql": "SELECT * FROM test",
            },
            "outcome": "error",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Connection failed", result)

    def test_format_trace_step_analyst(self) -> None:
        """format_trace_step should format analyst node."""
        
        step = {
            "node": "result_analyst",
            "update": {
                "analysis": "The data shows a trend.",
            },
            "outcome": "success",
        }
        
        result = format_trace_step(step)
        
        self.assertIn("Ai Message", result)
        self.assertIn("The data shows a trend.", result)

    def test_build_trace_log_content(self) -> None:
        """build_trace_log_content should build complete trace log."""

        trace = [
            {
                "node": "database_selector",
                "update": {"metadata": {}},
                "outcome": "success",
            },
        ]
        
        result = build_trace_log_content("Test question", trace)
        
        self.assertIn("Human Message", result)
        self.assertIn("Test question", result)
        self.assertIn("Ai Message", result)

    def test_write_trace_log_creates_file(self) -> None:
        """write_trace_log should create log file."""
        
        trace = [
            {
                "node": "database_selector",
                "update": {"metadata": {}},
                "outcome": "success",
            },
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = write_trace_log("Test question", trace, temp_dir)
            
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.parent, Path(temp_dir))
            
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("Test question", content)

    def test_write_trace_log_uses_default_dir(self) -> None:
        """write_trace_log should use default TRACE_LOG_DIR when not specified."""
        
        trace = [
            {
                "node": "database_selector",
                "update": {"metadata": {}},
                "outcome": "success",
            },
        ]
        
        log_path = write_trace_log("Test question", trace, None)
        
        self.assertTrue(log_path.exists())
        self.assertTrue(str(log_path).startswith(str(TRACE_LOG_DIR)))


if __name__ == "__main__":
    unittest.main()
