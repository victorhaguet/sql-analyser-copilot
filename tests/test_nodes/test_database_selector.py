"""Tests for the database selector node."""

from __future__ import annotations

import unittest

from nodes.database_selector import DatabaseSelectorNode
from tests.test_db.helpers import fixture_registered_database


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> FakeResponse:
        del prompt
        return FakeResponse(self.response)


class DatabaseSelectorNodeTestCase(unittest.TestCase):
    """Test database routing behavior."""

    def test_single_database_selected_without_model(self) -> None:
        node = DatabaseSelectorNode(
            [fixture_registered_database(name="chinook", description="Music data")]
        )
        result = node({"question": "Show the first artist"})

        self.assertEqual(result["metadata"]["selected_database"], "chinook")
        self.assertIn("Artist(", result["schema_overview"])
        self.assertEqual(result["selected_database"], "chinook")

    def test_returns_error_when_no_database_matches(self) -> None:
        node = DatabaseSelectorNode(
            [
                fixture_registered_database(name="music", description="Music store data"),
                fixture_registered_database(name="billing", description="Invoice data"),
            ],
            model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "No database fits this weather question."}'
            ),
        )
        result = node({"question": "What will the weather be tomorrow?"})

        self.assertIn("No database fits this weather question.", result["execution_error"])
        self.assertTrue(result["metadata"]["database_selection_failed"])

    def test_returns_ambiguity_error_when_multiple_databases_match(self) -> None:
        node = DatabaseSelectorNode(
            [
                fixture_registered_database(name="music", description="Music store data"),
                fixture_registered_database(name="billing", description="Invoice data"),
            ],
            model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": ["music", "billing"], "reason": "The question could be answered from either catalog."}'
            ),
        )
        result = node({"question": "Show me the latest invoices"})

        self.assertIn("matches multiple databases: music, billing", result["execution_error"])
        self.assertTrue(result["metadata"]["database_selection_ambiguous"])
        self.assertEqual(result["metadata"]["candidate_databases"], ["music", "billing"])

    def test_single_database_can_reject_irrelevant_question(self) -> None:
        node = DatabaseSelectorNode(
            [fixture_registered_database(name="music", description="Music store data")],
            model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "This question is unrelated to the catalog."}'
            ),
        )
        result = node({"question": "What will the weather be tomorrow?"})

        self.assertIn("This question is unrelated to the catalog.", result["execution_error"])

    def test_handles_json_code_block_format(self) -> None:
        node = DatabaseSelectorNode(
            [fixture_registered_database(name="music", description="Music data")],
            model=FakeModel(
                '```json\n{"match": true, "database": "music", "candidate_databases": ["music"], "reason": "Match found"}\n```'
            ),
        )
        result = node({"question": "Show artists"})

        self.assertEqual(result["metadata"]["selected_database"], "music")

    def test_handles_json_with_leading_label(self) -> None:
        node = DatabaseSelectorNode(
            [fixture_registered_database(name="music", description="Music data")],
            model=FakeModel(
                '```json\n{"match": true, "database": "music", "candidate_databases": ["music"], "reason": "Match"}\n```'
            ),
        )
        result = node({"question": "Show artists"})

        self.assertEqual(result["metadata"]["selected_database"], "music")

    def test_returns_error_for_invalid_json(self) -> None:
        node = DatabaseSelectorNode(
            [fixture_registered_database(name="music", description="Music data")],
            model=FakeModel("this is not valid json"),
        )
        result = node({"question": "Show artists"})

        self.assertIn("invalid selection payload", result["execution_error"])


if __name__ == "__main__":
    unittest.main()
