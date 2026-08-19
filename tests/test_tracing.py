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

from langchain_core.messages import AIMessage, ToolMessage

from tracing import (
    extract_trace_steps_from_content,
    format_trace_header,
    format_trace_payload,
    format_trace_step,
    truncate_trace_lines,
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

    def test_truncate_trace_lines_leaves_short_text_untouched(self) -> None:
        """Text at or under the line cap should pass through unchanged."""
        text = "\n".join(f"line {i}" for i in range(10))
        self.assertEqual(truncate_trace_lines(text), text)

    def test_truncate_trace_lines_caps_long_text(self) -> None:
        """Text over the line cap should be cut with a note of how much was removed."""
        text = "\n".join(f"line {i}" for i in range(30))
        result = truncate_trace_lines(text, max_lines=20)
        self.assertEqual(len(result.splitlines()), 21)  # 20 kept + 1 truncation note
        self.assertIn("[10 more line(s) truncated]", result)
        self.assertIn("line 19", result)
        self.assertNotIn("line 20", result)

    def test_format_trace_step_sql_agent_llm_lists_multiple_tool_calls_in_order(self) -> None:
        """A mixed-batch turn should list every requested tool call, in the order requested."""
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c1"},
                {"name": "run_readonly_probe", "args": {"sql": "SELECT 1"}, "id": "c2"},
            ],
        )
        result = format_trace_step(
            build_trace_step("sql_agent_llm", {"messages": [message], "agent_iterations": 2})
        )

        schema_index = result.index("inspect_schema")
        probe_index = result.index("run_readonly_probe")
        self.assertLess(schema_index, probe_index)

    def test_format_trace_step_database_checker(self) -> None:
        """format_trace_step should format database_checker node."""
        
        step = build_trace_step(
            "database_checker",
            {
                "metadata": {
                    "selected_database": "test_db",
                    "database_selection_reason": "Matches question",
                }
            },
        )
        
        result = format_trace_step(step)
        
        self.assertIn("Node: database_checker", result)
        self.assertIn("Outcome: success", result)
        self.assertIn("Selected Database: test_db", result)
        self.assertIn("Reason: Matches question", result)

    def test_format_trace_step_database_checker_with_error(self) -> None:
        """database_checker should include execution errors in the trace."""
        result = format_trace_step(
            build_trace_step(
                "database_checker",
                {"metadata": {}, "execution_error": "Question does not relate to database"},
                outcome="database_selection_failed",
            )
        )
        self.assertIn("Error: Question does not relate to database", result)

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

    def test_format_trace_step_sql_agent_llm_shows_iteration_and_tool_calls(self) -> None:
        """sql_agent_llm traces should show the iteration number and requested tool calls."""
        message = AIMessage(
            content="",
            tool_calls=[{"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c1"}],
        )
        result = format_trace_step(
            build_trace_step(
                "sql_agent_llm",
                {"messages": [message], "agent_iterations": 2},
            )
        )

        self.assertIn("Iteration: 2", result)
        self.assertIn("Tool calls requested (1):", result)
        self.assertIn("inspect_schema(tables=['Artist'])", result)

    def test_format_trace_step_sql_agent_llm_shows_model_text_and_no_tool_calls(self) -> None:
        """A finalize-ready turn (no tool calls) should be labeled distinctly, with model text shown."""
        message = AIMessage(content="SELECT * FROM Artist", tool_calls=[])
        result = format_trace_step(
            build_trace_step(
                "sql_agent_llm",
                {"messages": [message], "agent_iterations": 3},
            )
        )

        self.assertIn("Model text: SELECT * FROM Artist", result)
        self.assertIn("Tool calls requested: (none - ready to finalize)", result)

    def test_format_trace_step_sql_agent_tools_summarizes_each_result(self) -> None:
        """sql_agent_tools traces should show one line per tool result, by name."""
        tool_message = ToolMessage(
            content="8 columns, 2 incoming FKs", name="inspect_schema", tool_call_id="c1"
        )
        result = format_trace_step(
            build_trace_step("sql_agent_tools", {"messages": [tool_message], "probe_count": 0})
        )

        self.assertIn("- inspect_schema: 8 columns, 2 incoming FKs", result)

    def test_format_trace_step_sql_agent_tools_truncates_long_probe_results(self) -> None:
        """A probe result over ~20 lines should be truncated, not dumped in full."""
        long_result = "25 row(s)\n" + "\n".join(f"row {i}" for i in range(24))
        tool_message = ToolMessage(content=long_result, name="run_readonly_probe", tool_call_id="c1")
        result = format_trace_step(
            build_trace_step("sql_agent_tools", {"messages": [tool_message], "probe_count": 1})
        )

        self.assertIn("more line(s) truncated", result)
        self.assertNotIn("row 23", result)

    def test_format_trace_step_sql_agent_clarify_shows_answers(self) -> None:
        """sql_agent_clarify traces should show the round number and the answers received."""
        result = format_trace_step(
            build_trace_step(
                "sql_agent_clarify",
                {
                    "clarification_rounds": 1,
                    "clarification_answers": [
                        {"key": "metric", "question": "Which metric?", "answer": "total sales"}
                    ],
                },
            )
        )

        self.assertIn("round 1", result)
        self.assertIn("Which metric? -> total sales", result)

    def test_format_trace_step_sql_agent_clarify_shows_cancellation(self) -> None:
        """A cancelled clarification should be traced distinctly from an answered one."""
        result = format_trace_step(
            build_trace_step(
                "sql_agent_clarify",
                {"agent_status": "cancelled", "execution_error": "The user cancelled."},
            )
        )

        self.assertIn("cancelled by user", result)
        self.assertIn("The user cancelled.", result)

    def test_format_trace_step_sql_agent_finalize_shows_sql_and_rationale(self) -> None:
        """A successful finalize should show the final SQL and rationale."""
        result = format_trace_step(
            build_trace_step(
                "sql_agent_finalize",
                {
                    "generated_sql": "SELECT * FROM Artist",
                    "agent_status": "final",
                    "agent_rationale": "Lists every artist.",
                },
            )
        )

        self.assertIn("Final SQL:\nSELECT * FROM Artist", result)
        self.assertIn("Rationale: Lists every artist.", result)

    def test_format_trace_step_sql_agent_finalize_carries_over_repair_info(self) -> None:
        """A repair-pass finalize should carry the previous failure info (was sql_fallback_regenerator's job)."""
        result = format_trace_step(
            build_trace_step(
                "sql_agent_finalize",
                {
                    "generated_sql": "SELECT Name FROM Artist",
                    "previous_sql": "SELECT Name FROM Artists",
                    "agent_status": "final",
                    "agent_rationale": "Fixed the table name.",
                },
                state={
                    "retry_count": 1,
                    "max_retries": 3,
                    "last_execution_error": "no such table: Artists",
                },
            )
        )

        self.assertIn("Repair attempt: 1 of 3", result)
        self.assertIn("Previous Error: no such table: Artists", result)
        self.assertIn("Previous (failed) SQL:\nSELECT Name FROM Artists", result)
        self.assertIn("Final SQL:\nSELECT Name FROM Artist", result)

    def test_format_trace_step_sql_agent_finalize_failure(self) -> None:
        """A finalize that failed to produce SQL should be traced clearly."""
        result = format_trace_step(
            build_trace_step(
                "sql_agent_finalize",
                {"agent_status": "failed", "agent_error": "no SQL statement returned"},
            )
        )

        self.assertIn("failed to produce a valid SQL statement", result)
        self.assertIn("no SQL statement returned", result)

    def test_format_trace_step_sql_agent_budget_exhausted(self) -> None:
        """The budget-exhausted node's explanation should appear verbatim in the trace."""
        result = format_trace_step(
            build_trace_step(
                "sql_agent_budget_exhausted",
                {
                    "agent_status": "budget_exhausted",
                    "analysis": "I found the Artist table but ran out of iterations.",
                },
            )
        )

        self.assertIn("iteration budget exhausted", result)
        self.assertIn("I found the Artist table but ran out of iterations.", result)

    def test_format_trace_step_interrupt_clarification_shows_questions(self) -> None:
        """A clarification interrupt should show the actual questions asked."""
        result = format_trace_step(
            build_trace_step(
                "__interrupt__",
                {
                    "interrupt": {
                        "kind": "clarification",
                        "questions": [
                            {
                                "key": "metric",
                                "question": "Which metric?",
                                "why": "Ambiguous request.",
                                "suggested_default": "total sales",
                            }
                        ],
                    }
                },
                outcome="awaiting_clarification",
            )
        )

        self.assertIn("Questions asked (1):", result)
        self.assertIn("[metric] Which metric?", result)
        self.assertIn("why: Ambiguous request.", result)
        self.assertIn("suggested default: total sales", result)

    def test_format_trace_step_sql_executor_notes_repair_handoff(self) -> None:
        """A failed execution routed back to the agent should say so in the trace."""
        result = format_trace_step(
            build_trace_step(
                "sql_executor",
                {
                    "execution_error": "FOREIGN KEY constraint failed",
                    "agent_status": "repairing",
                    "retry_count": 1,
                },
                state={
                    "intent": "modification",
                    "generated_sql": "INSERT INTO Album (Title, ArtistId) VALUES ('X', 999)",
                    "max_retries": 3,
                },
            )
        )

        self.assertIn("Handed back to the agent for repair (attempt 1 of 3).", result)

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
            build_trace_step("database_checker", {"metadata": {}}),
        ]
        
        result = build_trace_log_content("Test question", trace)
        
        self.assertIn("Human Message", result)
        self.assertIn("Test question", result)
        self.assertIn("Graph step", result)

    def test_write_trace_log_creates_file(self) -> None:
        """write_trace_log should create log file."""
        
        trace = [
            build_trace_step("database_checker", {"metadata": {}}),
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
            build_trace_step("database_checker", {"metadata": {}}),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tracing.TRACE_LOG_DIR", Path(temp_dir)):
                log_path = write_trace_log("Test question", trace, None)
                self.assertTrue(log_path.exists())
                self.assertTrue(str(log_path).startswith(str(Path(temp_dir))))


if __name__ == "__main__":
    unittest.main()
