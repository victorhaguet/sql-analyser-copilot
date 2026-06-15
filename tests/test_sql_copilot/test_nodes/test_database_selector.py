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
            model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "No database fits this weather question."}'
            ),
        )
        result = node({"question": "What will the weather be tomorrow?"})
        self.assertIn("No database fits this weather question.", result["execution_error"])
        self.assertTrue(result["metadata"]["database_selection_failed"])

    def test_selector_returns_ambiguity_error_when_multiple_databases_match(self) -> None:
        """Ambiguous questions should stop before SQL generation and surface candidates."""
        node = DatabaseSelectorNode(
            [
                register_database(SQLiteDatabase(), name="music", description="Music store data"),
                register_database(SQLiteDatabase(), name="billing", description="Invoice data"),
            ],
            model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": ["music", "billing"], "reason": "The question could be answered from either catalog."}'
            ),
        )
        result = node({"question": "Show me the latest invoices"})
        self.assertIn("matches multiple databases: music, billing", result["execution_error"])
        self.assertTrue(result["metadata"]["database_selection_ambiguous"])
        self.assertEqual(result["metadata"]["candidate_databases"], ["music", "billing"])

    def test_single_database_can_still_reject_irrelevant_question(self) -> None:
        """Single-database mode should validate relevance when a selector model is available."""
        node = DatabaseSelectorNode(
            [register_database(SQLiteDatabase(), name="music", description="Music store data")],
            model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "This question is unrelated to the catalog."}'
            ),
        )
        result = node({"question": "What will the weather be tomorrow?"})
        self.assertIn("This question is unrelated to the catalog.", result["execution_error"])

    def test_selector_handles_json_code_block_format(self) -> None:
        """Selector should parse JSON wrapped in markdown code blocks."""
        node = DatabaseSelectorNode(
            [register_database(SQLiteDatabase(), name="music", description="Music data")],
            model=FakeModel(
                '```json\n{"match": true, "database": "music", "candidate_databases": ["music"], "reason": "Match found"}\n```'
            ),
        )
        result = node({"question": "Show artists"})
        self.assertEqual(result["metadata"]["selected_database"], "music")

    def test_selector_handles_json_with_leading_label(self) -> None:
        """Selector should strip 'json' prefix from code blocks."""
        node = DatabaseSelectorNode(
            [register_database(SQLiteDatabase(), name="music", description="Music data")],
            model=FakeModel(
                '```json\n{"match": true, "database": "music", "candidate_databases": ["music"], "reason": "Match"}\n```'
            ),
        )
        result = node({"question": "Show artists"})
        self.assertEqual(result["metadata"]["selected_database"], "music")

    def test_selector_returns_error_for_invalid_json(self) -> None:
        """Selector should handle invalid JSON response gracefully."""
        node = DatabaseSelectorNode(
            [register_database(SQLiteDatabase(), name="music", description="Music data")],
            model=FakeModel("this is not valid json"),
        )
        result = node({"question": "Show artists"})
        self.assertIn("invalid selection payload", result["execution_error"])


if __name__ == "__main__":
    unittest.main()
