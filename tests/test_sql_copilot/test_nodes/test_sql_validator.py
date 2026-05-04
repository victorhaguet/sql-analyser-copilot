"""Tests for the SQL validation node."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.nodes.sql_validator import SQLValidatorNode


class SQLValidatorNodeTestCase(unittest.TestCase):
    """Test SQL validation behavior."""

    def test_sql_validator_rejects_mutating_sql(self) -> None:
        """Test that the SQLValidatorNode rejects non-SELECT queries and provides appropriate error messages."""
        result = SQLValidatorNode()({"generated_sql": "DELETE FROM Artist"})
        self.assertEqual(result["validated_sql"], "")
        self.assertIn("Only SELECT queries are allowed", result["sql_validation_error"])
        self.assertIn("SQL validation failed", result["analysis"])


if __name__ == "__main__":
    unittest.main()
