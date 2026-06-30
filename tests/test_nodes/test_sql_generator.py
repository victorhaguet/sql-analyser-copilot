"""Tests for the SQL generation node."""

from __future__ import annotations

import unittest

from nodes.sql_generator import SQLGeneratorNode
from tests.test_db.helpers import fixture_database


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response)


class SQLGeneratorNodeTestCase(unittest.TestCase):
    """Test SQL generation behavior."""

    def test_uses_schema_and_strips_code_fence(self) -> None:
        model = FakeModel("```sql\nSELECT Name FROM Artist LIMIT 1\n```")
        node = SQLGeneratorNode(model=model, database=fixture_database())
        result = node({"question": "Show one artist"})

        self.assertEqual(result["generated_sql"], "SELECT Name FROM Artist LIMIT 1")
        self.assertIn("Artist(", result["schema_overview"])


if __name__ == "__main__":
    unittest.main()
