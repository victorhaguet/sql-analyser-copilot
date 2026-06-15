"""Tests for the LangChain OpenAI adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.llms.openai_compatible import OpenAICompatibleResponsesModel, load_models_from_env

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

    def test_load_models_from_env_returns_none_when_no_api_key(self) -> None:
        """When OPENAI_API_KEY is not set, models should not be created."""
        import os
        original_key = os.getenv("OPENAI_API_KEY")
        try:
            os.environ.pop("OPENAI_API_KEY", None)
            generator, analyst = load_models_from_env()
            self.assertIsNone(generator)
            self.assertIsNone(analyst)
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
