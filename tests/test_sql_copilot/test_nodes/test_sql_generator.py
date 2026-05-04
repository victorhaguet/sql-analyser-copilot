"""Tests for the SQL generation node."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.nodes.sql_generator import SQLGeneratorNode
from sql_copilot.tools.database import SQLiteDatabase


class FakeResponse:
    """Simple model response object with a content field."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """Minimal test double for invoke-based text models."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response)


class SQLGeneratorNodeTestCase(unittest.TestCase):
    """Test SQL generation behavior."""

    def test_sql_generator_uses_schema_and_strips_code_fence(self) -> None:
        """Test that the SQLGeneratorNode includes schema information in the prompt and correctly extracts SQL from code fences."""
        model = FakeModel("```sql\nSELECT Name FROM Artist LIMIT 1\n```")
        node = SQLGeneratorNode(model=model, database=SQLiteDatabase())
        result = node({"question": "Show one artist"})
        self.assertEqual(result["generated_sql"], "SELECT Name FROM Artist LIMIT 1")
        self.assertIn("Artist(", result["schema_overview"])


if __name__ == "__main__":
    unittest.main()
