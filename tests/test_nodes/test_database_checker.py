"""Tests for the database checker node."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from nodes.database_checker import DatabaseCheckerNode
from tools.database import SQLiteDatabase


class DatabaseCheckerNodeTestCase(unittest.TestCase):
    """Test database validation behavior."""

    def test_requires_model(self) -> None:
        """Test that node requires a model to be provided."""
        node = DatabaseCheckerNode(SQLiteDatabase())
        result = node({"question": "Show artists"})

        self.assertIn("Please initialize the llm checker", result["execution_error"])
        self.assertTrue(result["metadata"]["database_selection_failed"])

    def test_validates_single_selected_database_with_model(self) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content='{"match": true, "database": "chinook", "reason": "Matches question"}')
        
        node = DatabaseCheckerNode(
            SQLiteDatabase(),
            model=mock_model
        )
        result = node({"question": "Show the first artist", "selected_database": "chinook"})

        self.assertIn("Artist(", result["schema_overview"])
        self.assertEqual(result["selected_database"], "default")


if __name__ == "__main__":
    unittest.main()
