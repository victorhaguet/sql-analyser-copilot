"""Tests for the result analysis node."""

from __future__ import annotations

import unittest

from nodes.result_analyst import ResultAnalystNode
from tests.test_db.helpers import fixture_database


class ResultAnalystNodeTestCase(unittest.TestCase):
    """Test result analysis behavior."""

    def test_has_non_llm_fallback(self) -> None:
        database = fixture_database()
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
