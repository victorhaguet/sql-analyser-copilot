"""Tests for the result analysis node."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from nodes.result_analyst import ResultAnalystNode
from tools.database import SQLiteDatabase


class ResultAnalystNodeTestCase(unittest.TestCase):
    """Test result analysis behavior."""

    def test_result_analyst_has_non_llm_fallback(self) -> None:
        """Test that the ResultAnalystNode can analyze results without an LLM."""
        database = SQLiteDatabase()
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
