"""Tests for the SQL validation node."""

from __future__ import annotations

import unittest

from nodes.sql_validator import SQLValidatorNode


class SQLValidatorNodeTestCase(unittest.TestCase):
    """Test SQL validation behavior."""

    def test_rejects_mutating_sql(self) -> None:
        result = SQLValidatorNode()({"generated_sql": "DELETE FROM Artist"})

        self.assertEqual(result["validated_sql"], "")
        self.assertIn("Only SELECT queries are allowed", result["sql_validation_error"])
        self.assertIn("SQL validation failed", result["analysis"])


if __name__ == "__main__":
    unittest.main()
