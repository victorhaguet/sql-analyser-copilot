"""Tests for the SQL execution node."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.nodes.sql_executor import SQLExecutorNode
from sql_copilot.tools.database import SQLiteDatabase


class SQLExecutorNodeTestCase(unittest.TestCase):
    """Test SQL execution behavior."""

    def test_sql_executor_runs_validated_sql(self) -> None:
        """Test that the SQLExecutorNode can execute validated SQL and return results."""
        result = SQLExecutorNode(database=SQLiteDatabase())(
            {
                "validated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2",
            }
        )
        self.assertIsNone(result["execution_error"])
        self.assertEqual(result["query_result"].row_count, 2)

    def test_sql_executor_reports_database_errors(self) -> None:
        """Test that the SQLExecutorNode reports database errors in the analysis."""
        result = SQLExecutorNode(database=SQLiteDatabase())(
            {
                "validated_sql": "SELECT MissingColumn FROM Artist",
            }
        )
        self.assertIn("no such column", result["execution_error"].lower())
        self.assertIn("SQL execution failed", result["analysis"])


if __name__ == "__main__":
    unittest.main()
