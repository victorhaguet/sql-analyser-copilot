"""Tests for the result analysis node."""

from __future__ import annotations

import unittest
from pathlib import Path

from nodes.result_analyst import ResultAnalystNode
from tools.database import SQLiteDatabase

FIXTURE_DB_PATH = Path(__file__).resolve().parents[1] / "test_db" / "Chinook_Sqlite.sqlite"


class ResultAnalystNodeTestCase(unittest.TestCase):
    """Test result analysis behavior."""

    def test_has_non_llm_fallback(self) -> None:
        database = SQLiteDatabase(FIXTURE_DB_PATH)
        query_result = database.execute_command(
            "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 1"
        )
        result = ResultAnalystNode()(
            {
                "question": "Show one artist",
                "validated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 1",
                "query_result": query_result,
            }
        )

        self.assertIn("Returned 1 row(s)", result["analysis"])
        self.assertIn("AC/DC", result["analysis"])


if __name__ == "__main__":
    unittest.main()
