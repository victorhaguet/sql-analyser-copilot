"""Tests for the database selector node."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.nodes.database_selector import DatabaseSelectorNode
from sql_copilot.tools.database import SQLiteDatabase, register_database


class FakeResponse:
    """Simple model response object with a content field."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """Minimal test double for invoke-based text models."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> FakeResponse:
        del prompt
        return FakeResponse(self.response)


class DatabaseSelectorNodeTestCase(unittest.TestCase):
    """Test database routing behavior."""

    def test_single_database_is_selected_without_model(self) -> None:
        """Single-database mode should preserve the current workflow."""
        node = DatabaseSelectorNode(
            [register_database(SQLiteDatabase(), name="chinook", description="Music data")]
        )
        result = node({"question": "Show the first artist"})
        self.assertEqual(result["metadata"]["selected_database"], "chinook")
        self.assertIn("Artist(", result["schema_overview"])
        self.assertIsInstance(result["selected_database"], SQLiteDatabase)

    def test_selector_returns_error_when_no_database_matches(self) -> None:
        """Unmatched questions should stop the graph before SQL generation."""
        node = DatabaseSelectorNode(
            [
                register_database(SQLiteDatabase(), name="music", description="Music store data"),
                register_database(SQLiteDatabase(), name="billing", description="Invoice data"),
            ],
            model=FakeModel('{"match": false, "database": "", "reason": "No database fits this weather question."}'),
        )
        result = node({"question": "What will the weather be tomorrow?"})
        self.assertIn("No database fits this weather question.", result["execution_error"])
        self.assertTrue(result["metadata"]["database_selection_failed"])

    def test_single_database_can_still_reject_irrelevant_question(self) -> None:
        """Single-database mode should validate relevance when a selector model is available."""
        node = DatabaseSelectorNode(
            [register_database(SQLiteDatabase(), name="music", description="Music store data")],
            model=FakeModel('{"match": false, "database": "", "reason": "This question is unrelated to the catalog."}'),
        )
        result = node({"question": "What will the weather be tomorrow?"})
        self.assertIn("This question is unrelated to the catalog.", result["execution_error"])


if __name__ == "__main__":
    unittest.main()
