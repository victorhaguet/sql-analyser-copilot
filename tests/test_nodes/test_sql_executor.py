"""Tests for the SQL execution node."""

from __future__ import annotations

import unittest
from pathlib import Path

from nodes.sql_executor import SQLExecutorNode
from tools.database import SQLiteDatabase

FIXTURE_DB_PATH = Path(__file__).resolve().parents[1] / "test_db" / "Chinook_Sqlite.sqlite"


class SQLExecutorNodeTestCase(unittest.TestCase):
    """Test SQL execution behavior."""

    def test_executes_validated_sql(self) -> None:
        result = SQLExecutorNode(database=SQLiteDatabase(FIXTURE_DB_PATH))(
            {"validated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2"}
        )

        self.assertIsNone(result["execution_error"])
        self.assertEqual(result["query_result"].row_count, 2)

    def test_reports_database_errors(self) -> None:
        result = SQLExecutorNode(database=SQLiteDatabase(FIXTURE_DB_PATH))(
            {"validated_sql": "SELECT MissingColumn FROM Artist"}
        )

        self.assertIn("no such column", result["execution_error"].lower())
        self.assertIn("SQL execution failed", result["analysis"])

    def test_executes_generated_sql_for_modification_intent(self) -> None:
        result = SQLExecutorNode(database=SQLiteDatabase(FIXTURE_DB_PATH))(
            {
                "intent": "modification",
                "generated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 1",
            }
        )

        self.assertIsNone(result["execution_error"])
        self.assertEqual(result["query_result"].row_count, 1)


if __name__ == "__main__":
    unittest.main()
