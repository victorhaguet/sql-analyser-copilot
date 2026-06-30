"""Tests for the intent classifier node."""

from __future__ import annotations

import unittest

from nodes.intent_classifier import IntentClassifierNode


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> FakeResponse:
        del prompt
        return FakeResponse(self.response)


class IntentClassifierNodeTestCase(unittest.TestCase):
    """Test intent classification behavior."""

    def test_classifies_query_intent(self) -> None:
        """Test that a query question is classified as 'query'."""
        node = IntentClassifierNode(FakeModel('{"intent": "query"}'))
        result = node({"question": "Which artists have the most albums?"})

        self.assertEqual(result["intent"], "query")
        self.assertIsNone(result["intent_error"])

    def test_classifies_modification_intent(self) -> None:
        """Test that a modification question is flagged for confirmation."""
        node = IntentClassifierNode(FakeModel('{"intent": "modification"}'))
        result = node({"question": "Delete all artists"})

        self.assertEqual(result["intent"], "modification")
        self.assertIsNone(result["intent_error"])
        self.assertTrue(result["needs_confirmation"])
        self.assertNotIn("analysis", result)

    def test_classifies_insert_as_modification(self) -> None:
        """Test that INSERT requests are flagged for confirmation."""
        node = IntentClassifierNode(FakeModel('{"intent": "modification"}'))
        result = node({"question": "Add a new artist"})

        self.assertEqual(result["intent"], "modification")
        self.assertTrue(result["needs_confirmation"])
        self.assertIsNone(result["intent_error"])

    def test_classifies_update_as_modification(self) -> None:
        """Test that UPDATE requests are flagged for confirmation."""
        node = IntentClassifierNode(FakeModel('{"intent": "modification"}'))
        result = node({"question": "Update artist names"})

        self.assertEqual(result["intent"], "modification")
        self.assertTrue(result["needs_confirmation"])
        self.assertIsNone(result["intent_error"])

    def test_classifies_delete_as_modification(self) -> None:
        """Test that DELETE requests are flagged for confirmation."""
        node = IntentClassifierNode(FakeModel('{"intent": "modification"}'))
        result = node({"question": "Delete old records"})

        self.assertEqual(result["intent"], "modification")
        self.assertTrue(result["needs_confirmation"])
        self.assertIsNone(result["intent_error"])

    def test_handles_unparseable_json(self) -> None:
        """Test that unparseable JSON returns an error."""
        node = IntentClassifierNode(FakeModel("this is not json"))
        result = node({"question": "Show artists"})

        self.assertIsNone(result["intent"])
        self.assertIn("Could not classify", result["intent_error"])
        self.assertIn("Could not classify", result["analysis"])

    def test_handles_invalid_intent_value(self) -> None:
        """Test that an invalid intent value returns an error."""
        node = IntentClassifierNode(FakeModel('{"intent": "invalid"}'))
        result = node({"question": "Show artists"})

        self.assertIsNone(result["intent"])
        self.assertIn("Could not classify", result["intent_error"])
        self.assertIn("Could not classify", result["analysis"])

    def test_handles_empty_intent(self) -> None:
        """Test that an empty intent value returns an error."""
        node = IntentClassifierNode(FakeModel('{"intent": ""}'))
        result = node({"question": "Show artists"})

        self.assertIsNone(result["intent"])
        self.assertIn("Could not classify", result["intent_error"])
        self.assertIn("Could not classify", result["analysis"])

    def test_error_message_shown_in_ui(self) -> None:
        """Modification classification should not surface a classifier error."""
        node = IntentClassifierNode(FakeModel('{"intent": "modification"}'))
        result = node({"question": "DROP TABLE Artist"})

        self.assertIsNone(result["intent_error"])
        self.assertTrue(result["needs_confirmation"])
        self.assertNotIn("analysis", result)
        self.assertNotIn("execution_error", result)
