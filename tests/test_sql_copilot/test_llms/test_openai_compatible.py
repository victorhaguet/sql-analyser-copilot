"""Tests for the LangChain OpenAI adapter."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.llms.openai_compatible import OpenAICompatibleResponsesModel #import-error:ignore
from sql_copilot.main import load_models_from_env

class FakeClient:
    """Simple fake LangChain client."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[object] = []

    def invoke(self, payload: object) -> object:
        self.calls.append(payload)
        return self.response


class OpenAICompatibleResponsesModelTestCase(unittest.TestCase):
    """Test OpenAI-compatible model behavior."""

    def test_openai_compatible_model_with_fake_client(self) -> None:
        """Test OpenAICompatibleResponsesModel with a fake client."""

        original_key = os.getenv("OPENAI_API_KEY")
        original_model = os.getenv("SQL_COPILOT_MODEL")
        try:
            os.environ["OPENAI_API_KEY"] = "test-key"
            os.environ["SQL_COPILOT_MODEL"] = "alias-fast"
            
            generator, analyst = load_models_from_env()
            self.assertIsNotNone(generator)
            self.assertIsNotNone(analyst)
            self.assertEqual(generator.model, "alias-fast")
            self.assertEqual(analyst.model, "alias-fast")
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if original_model is not None:
                os.environ["SQL_COPILOT_MODEL"] = original_model
            else:
                os.environ.pop("SQL_COPILOT_MODEL", None)

    def test_invoke_uses_chat_model_and_returns_message_content(self) -> None:
        """The wrapper should call the chat model and expose message content."""
        client = FakeClient(type("FakeMessage", (), {"content": "SELECT 1"})())
        model = OpenAICompatibleResponsesModel(
            model="demo-model",
            api_key="secret",
            client=client,
        )

        result = model.invoke("Return SQL")

        self.assertEqual(result, "SELECT 1")
        self.assertEqual(client.calls[0], "Return SQL")

    def test_invoke_supports_system_instructions(self) -> None:
        """The wrapper should pass instructions as a system message."""
        client = FakeClient(type("FakeMessage", (), {"content": "SELECT 1"})())
        model = OpenAICompatibleResponsesModel(
            model="demo-model",
            api_key="secret",
            instructions="Return SQL only.",
            client=client,
        )

        model.invoke("List artists")

        self.assertEqual(
            client.calls[0],
            [("system", "Return SQL only."), ("human", "List artists")],
        )

    def test_openai_compatible_model_creation(self) -> None:
        """Test that OpenAICompatibleResponsesModel can be instantiated."""
        original_key = os.getenv("OPENAI_API_KEY")
        original_generator = os.getenv("SQL_COPILOT_MODEL")
        original_analyst = os.getenv("SQL_COPILOT_ANALYST_MODEL")
        try:
            os.environ["OPENAI_API_KEY"] = "test-key"
            os.environ["SQL_COPILOT_MODEL"] = "test-model"
            os.environ["SQL_COPILOT_ANALYST_MODEL"] = "test-analyst"
            
            generator, analyst = load_models_from_env()
            self.assertIsNotNone(generator)
            self.assertIsNotNone(analyst)
            self.assertEqual(generator.model, "test-model")
            self.assertEqual(analyst.model, "test-analyst")
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if original_generator is not None:
                os.environ["SQL_COPILOT_MODEL"] = original_generator
            else:
                os.environ.pop("SQL_COPILOT_MODEL", None)
            if original_analyst is not None:
                os.environ["SQL_COPILOT_ANALYST_MODEL"] = original_analyst
            else:
                os.environ.pop("SQL_COPILOT_ANALYST_MODEL", None)
