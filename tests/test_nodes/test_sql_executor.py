"""Tests for the SQL execution node."""

from __future__ import annotations

import unittest

from nodes.sql_executor import SQLExecutorNode
from tests.test_db.helpers import fixture_database


class SQLExecutorNodeTestCase(unittest.TestCase):
    """Test SQL execution behavior."""

    def test_executes_validated_sql(self) -> None:
        result = SQLExecutorNode(database=fixture_database())(
            {"validated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2"}
        )

        self.assertIsNone(result["execution_error"])
        self.assertEqual(result["query_result"].row_count, 2)

    def test_reports_database_errors(self) -> None:
        result = SQLExecutorNode(database=fixture_database())(
            {"validated_sql": "SELECT MissingColumn FROM Artist"}
        )

        self.assertIn("no such column", result["execution_error"].lower())
        self.assertIn("SQL execution failed", result["analysis"])

    def test_executes_generated_sql_for_modification_intent(self) -> None:
        result = SQLExecutorNode(database=fixture_database())(
            {
                "intent": "modification",
                "generated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 1",
            }
        )

        self.assertIsNone(result["execution_error"])
        self.assertEqual(result["query_result"].row_count, 1)

    def test_error_path_appends_repair_message_and_increments_retry_count(self) -> None:
        """D6: a failed execution must feed the transcript for sql_agent_llm to repair,
        since sql_fallback_regenerator no longer exists."""
        result = SQLExecutorNode(database=fixture_database())(
            {
                "intent": "modification",
                "generated_sql": "INSERT INTO Artists (Name) VALUES ('Mandyspie')",
                "retry_count": 0,
            }
        )

        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["agent_status"], "repairing")
        self.assertEqual(
            result["previous_sql"], "INSERT INTO Artists (Name) VALUES ('Mandyspie')"
        )
        self.assertEqual(len(result["messages"]), 1)
        message = result["messages"][0]
        self.assertEqual(type(message).__name__, "HumanMessage")
        self.assertIn("INSERT INTO Artists (Name) VALUES ('Mandyspie')", message.content)
        self.assertIn("no such table", message.content.lower())

    def test_error_path_resets_agent_iterations_for_the_repair_attempt(self) -> None:
        """Step 10 (D5): a repair re-entry must get its own iteration allowance,
        so it isn't starved by iterations already spent on the first attempt."""
        result = SQLExecutorNode(database=fixture_database())(
            {
                "intent": "modification",
                "generated_sql": "INSERT INTO Artists (Name) VALUES ('Mandyspie')",
                "retry_count": 0,
                "agent_iterations": 8,
            }
        )

        self.assertEqual(result["agent_iterations"], 0)

    def test_error_path_increments_retry_count_from_existing_value(self) -> None:
        """Repeated failures must keep incrementing, matching max_retries semantics."""
        result = SQLExecutorNode(database=fixture_database())(
            {
                "intent": "modification",
                "generated_sql": "INSERT INTO Artists (Name) VALUES ('Mandyspie')",
                "retry_count": 2,
            }
        )

        self.assertEqual(result["retry_count"], 3)

    def test_success_path_does_not_touch_messages(self) -> None:
        """A successful execution should not append a repair message."""
        result = SQLExecutorNode(database=fixture_database())(
            {"validated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2"}
        )

        self.assertNotIn("messages", result)
        self.assertNotIn("agent_status", result)


if __name__ == "__main__":
    unittest.main()
