"""Tests for SQL fallback regeneration."""

from __future__ import annotations

import unittest

from nodes.sql_fallback_regenerator import SQLFallbackRegeneratorNode


class RecordingModel:
    """Return a configured response and retain the rendered prompt."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.prompt = ""

    def invoke(self, prompt: str) -> object:
        self.prompt = prompt
        return self.response


def build_failed_state() -> dict[str, object]:
    """Build representative graph state after a failed SQL execution."""

    return {
        "question": "Create the artist Mandyspie",
        "intent": "modification",
        "schema_overview": "Artist(ArtistId INTEGER PRIMARY KEY, Name NVARCHAR)",
        "generated_sql": "INSERT INTO Artists (Name) VALUES ('Mandyspie')",
        "execution_error": "no such table: Artists",
        "last_execution_error": "no such table: Artists",
        "retry_count": 0,
        "max_retries": 3,
        "execution_confirmed": True,
    }


class SQLFallbackRegeneratorTestCase(unittest.TestCase):
    """Test context-aware SQL repair and failure handling."""

    def test_passes_failure_context_and_schema_to_model(self) -> None:
        model = RecordingModel("INSERT INTO Artist (Name) VALUES ('Mandyspie')")

        SQLFallbackRegeneratorNode(model)(build_failed_state())

        self.assertIn(
            "Artist(ArtistId INTEGER PRIMARY KEY, Name NVARCHAR)", model.prompt
        )
        self.assertIn("INSERT INTO Artists (Name) VALUES ('Mandyspie')", model.prompt)
        self.assertIn("no such table: Artists", model.prompt)

    def test_prioritizes_user_approved_sql_over_original_request(self) -> None:
        state = build_failed_state()
        state["question"] = "Create the artist Disiz"
        state["generated_sql"] = "INSERT INTO Artists (Name) VALUES ('Suikon Blaz AD')"
        model = RecordingModel(
            "INSERT INTO Artist (Name) VALUES ('Suikon Blaz AD')"
        )

        result = SQLFallbackRegeneratorNode(model)(state)

        self.assertIn("authoritative repair target", model.prompt)
        self.assertIn("If they conflict, the failed SQL wins", model.prompt)
        self.assertEqual(
            result["generated_sql"],
            "INSERT INTO Artist (Name) VALUES ('Suikon Blaz AD')",
        )

    def test_accepts_string_responses_and_strips_code_fences(self) -> None:
        model = RecordingModel(
            "```sql\nINSERT INTO Artist (Name) VALUES ('Mandyspie');\n```"
        )

        result = SQLFallbackRegeneratorNode(model)(build_failed_state())

        self.assertEqual(
            result["generated_sql"],
            "INSERT INTO Artist (Name) VALUES ('Mandyspie');",
        )

    def test_preserves_previous_error_for_tracing(self) -> None:
        model = RecordingModel("INSERT INTO Artist (Name) VALUES ('Mandyspie')")

        result = SQLFallbackRegeneratorNode(model)(build_failed_state())

        self.assertEqual(result["last_execution_error"], "no such table: Artists")

    def test_exposes_failed_sql_and_explanation_for_approval(self) -> None:
        model = RecordingModel("INSERT INTO Artist (Name) VALUES ('Mandyspie')")

        result = SQLFallbackRegeneratorNode(model)(build_failed_state())

        self.assertEqual(
            result["previous_sql"],
            "INSERT INTO Artists (Name) VALUES ('Mandyspie')",
        )
        self.assertIn("no such table: Artists", result["regeneration_explanation"])

    def test_clears_confirmation_before_modified_sql_is_reapproved(self) -> None:
        model = RecordingModel("INSERT INTO Artist (Name) VALUES ('Mandyspie')")

        result = SQLFallbackRegeneratorNode(model)(build_failed_state())

        self.assertFalse(result["execution_confirmed"])

    def test_fails_closed_when_model_returns_no_text(self) -> None:
        result = SQLFallbackRegeneratorNode(RecordingModel(""))(build_failed_state())

        self.assertTrue(result["regeneration_error"])
        self.assertEqual(result["execution_error"], result["regeneration_error"])


if __name__ == "__main__":
    unittest.main()
