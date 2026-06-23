"""Tests for the LLM utilities."""

from __future__ import annotations

import unittest

from utils.llm import extract_text_from_response, strip_code_fences


class LlmTestCase(unittest.TestCase):
    """Test LLM utility behavior."""

    def test_extract_text_from_string(self) -> None:
        """Should extract and strip text from a string."""
        response = "  hello world  "

        result = extract_text_from_response(response)

        self.assertEqual(result, "hello world")

    def test_extract_text_from_object_with_content(self) -> None:
        """Should extract text from object with content attribute."""
        class Response:
            content = "  response content  "

        result = extract_text_from_response(Response())

        self.assertEqual(result, "response content")

    def test_extract_text_from_list_of_dicts(self) -> None:
        """Should extract text from list containing dicts."""
        class Response:
            content = [
                {"text": "first"},
                {"text": "second"},
            ]

        result = extract_text_from_response(Response())

        self.assertEqual(result, "first\nsecond")

    def test_extract_text_from_mixed_list(self) -> None:
        """Should handle list with both dicts and strings."""
        class Response:
            content = [
                {"text": "first"},
                "second",
            ]

        result = extract_text_from_response(Response())

        self.assertEqual(result, "first\nsecond")

    def test_extract_text_raises_error_for_empty_response(self) -> None:
        """Should raise RuntimeError when no content can be extracted."""
        class Response:
            content = []

        with self.assertRaises(RuntimeError):
            extract_text_from_response(Response())

    def test_extract_text_raises_error_for_missing_content(self) -> None:
        """Should raise RuntimeError when content attribute is missing."""
        with self.assertRaises(RuntimeError):
            extract_text_from_response(object())

    def test_strip_code_fences_removes_markdown_blocks(self) -> None:
        """Should remove markdown code fences."""
        text = "```\nSELECT * FROM users\n```"

        result = strip_code_fences(text)

        self.assertEqual(result, "SELECT * FROM users")

    def test_strip_code_fences_with_language_prefix(self) -> None:
        """Should remove language prefix when specified."""
        text = "```sql\nSELECT * FROM users\n```"

        result = strip_code_fences(text, language_prefix="sql")

        self.assertEqual(result, "SELECT * FROM users")

    def test_strip_code_fences_preserves_content_without_fences(self) -> None:
        """Should return text unchanged when no fences present."""
        text = "plain text"

        result = strip_code_fences(text)

        self.assertEqual(result, "plain text")

    def test_strip_code_fences_case_insensitive_language(self) -> None:
        """Should handle language prefix case-insensitively."""
        text = "```SQL\nSELECT * FROM users\n```"

        result = strip_code_fences(text, language_prefix="sql")

        self.assertEqual(result, "SELECT * FROM users")

    def test_strip_code_fences_only_sql_prefix(self) -> None:
        """Should remove only the language prefix, not content."""
        text = "```sql\nSELECT * FROM users\n```"

        result = strip_code_fences(text)

        self.assertEqual(result, "SELECT * FROM users")

    def test_strip_code_fences_handles_multiline(self) -> None:
        """Should handle multiline code blocks."""
        text = "```\nline1\nline2\nline3\n```"

        result = strip_code_fences(text)

        self.assertEqual(result, "line1\nline2\nline3")


if __name__ == "__main__":
    unittest.main()
