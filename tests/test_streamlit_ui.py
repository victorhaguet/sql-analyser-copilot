"""Tests for the Streamlit UI helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from streamlit_ui import (
    build_display_context,
    build_display_payload,
    load_stylesheet,
)


class StreamlitUITestCase(unittest.TestCase):
    """Test the UI payload formatting helpers."""

    def test_build_display_context_keeps_result_metadata(self) -> None:
        """The UI context should preserve structured API fields for rendering."""
        context = build_display_context(
            {
                "question": "Top customers",
                "validated_sql": "SELECT * FROM Customer LIMIT 2",
                "analysis": "Two customers are shown.",
                "query_result": {
                    "columns": ["CustomerId", "FirstName"],
                    "rows": [
                        {"CustomerId": 1, "FirstName": "Luiz"},
                        {"CustomerId": 2, "FirstName": "Leonie"},
                    ],
                    "row_count": 2,
                    "truncated": False,
                },
                "schema_overview": "Customer(CustomerId, FirstName)",
                "metadata": {"duration_ms": 84},
            }
        )

        self.assertEqual(context["question"], "Top customers")
        self.assertEqual(context["sql_query"], "SELECT * FROM Customer LIMIT 2")
        self.assertEqual(context["ai_answer"], "Two customers are shown.")
        self.assertEqual(context["query_result"]["row_count"], 2)
        self.assertEqual(context["schema_overview"], "Customer(CustomerId, FirstName)")
        self.assertEqual(context["metadata"], {"duration_ms": 84})
        self.assertFalse(context["has_error"])

    def test_build_display_payload_prefers_validated_sql_and_analysis(self) -> None:
        """The UI should prefer validated SQL and the final analysis text."""
        original_query, sql_query, ai_answer = build_display_payload(
            {
                "question": "List the first 3 artists",
                "generated_sql": "SELECT * FROM Artist LIMIT 3",
                "validated_sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 3",
                "analysis": "The first three artists are AC/DC, Accept, and Aerosmith.",
            }
        )

        self.assertEqual(original_query, "List the first 3 artists")
        self.assertEqual(sql_query, "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 3")
        self.assertEqual(
            ai_answer,
            "The first three artists are AC/DC, Accept, and Aerosmith.",
        )

    def test_build_display_payload_falls_back_to_errors(self) -> None:
        """The UI should surface API errors when SQL or analysis are unavailable."""
        original_query, sql_query, ai_answer = build_display_payload(
            {
                "question": "Drop all tables",
                "sql_validation_error": "Unsafe SQL detected.",
            }
        )

        self.assertEqual(original_query, "Drop all tables")
        self.assertEqual(sql_query, "Unsafe SQL detected.")
        self.assertEqual(ai_answer, "Unsafe SQL detected.")

        context = build_display_context(
            {
                "question": "Drop all tables",
                "sql_validation_error": "Unsafe SQL detected.",
            }
        )
        self.assertTrue(context["has_error"])

    def test_build_display_payload_defaults_analysis_message(self) -> None:
        """The UI should expose a fallback message when analysis is missing."""
        original_query, sql_query, ai_answer = build_display_payload(
            {
                "question": "Show me 1 artist",
                "generated_sql": "SELECT Name FROM Artist LIMIT 1",
            }
        )

        self.assertEqual(original_query, "Show me 1 artist")
        self.assertEqual(sql_query, "SELECT Name FROM Artist LIMIT 1")
        self.assertEqual(ai_answer, "No analysis returned.")

    def test_build_display_context_keeps_retry_approval_state(self) -> None:
        """The UI should retain retry metadata required by the approval dialog."""
        context = build_display_context(
            {
                "intent": "modification",
                "execution_requested": True,
                "retry_count": 1,
                "max_retries": 3,
                "last_execution_error": "no such table: Artists",
                "previous_sql": "SELECT * FROM Artists",
                "regeneration_explanation": "Artists was replaced with Artist.",
            }
        )

        self.assertEqual(context["intent"], "modification")
        self.assertEqual(context["retry_count"], 1)
        self.assertEqual(context["last_execution_error"], "no such table: Artists")
        self.assertEqual(context["previous_sql"], "SELECT * FROM Artists")
        self.assertEqual(
            context["regeneration_explanation"],
            "Artists was replaced with Artist.",
        )

    def test_load_stylesheet_returns_content(self) -> None:
        """Test that the stylesheet is loaded and has content."""
        content = load_stylesheet()
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)


if __name__ == "__main__":
    unittest.main()
