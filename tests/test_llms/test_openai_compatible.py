"""Tests for the LangChain OpenAI adapter."""

from __future__ import annotations

import unittest

from llms.openai_compatible import OpenAICompatibleResponsesModel, build_tool_calling_chat_model

class FakeClient:
    """Simple fake LangChain client."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[object] = []

    def invoke(self, payload: object) -> object:
        self.calls.append(payload)
        return self.response


class FakeToolCallingClient(FakeClient):
    """Fake client that also exposes bind_tools, for chat_model tests."""

    def bind_tools(self, tools: object) -> "FakeToolCallingClient":
        self.bound_tools = tools
        return self


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

    def test_chat_model_property_exposes_underlying_client(self) -> None:
        """chat_model should expose the raw client, not the invoke(str) wrapper."""
        client = FakeToolCallingClient(type("FakeMessage", (), {"content": "SELECT 1"})())
        model = OpenAICompatibleResponsesModel(
            model="demo-model",
            api_key="secret",
            client=client,
        )

        self.assertIs(model.chat_model, client)
        self.assertTrue(hasattr(model.chat_model, "bind_tools"))

    def test_build_tool_calling_chat_model_exposes_bind_tools(self) -> None:
        """build_tool_calling_chat_model should return a bind_tools-capable client."""
        chat_model = build_tool_calling_chat_model(model="demo-model", api_key="secret")

        self.assertTrue(hasattr(chat_model, "bind_tools"))
        self.assertEqual(chat_model.model_name, "demo-model")

    def test_build_tool_calling_chat_model_does_not_use_responses_api(self) -> None:
        """The agent model must not set use_responses_api, unlike the invoke(str) wrapper."""
        chat_model = build_tool_calling_chat_model(model="demo-model", api_key="secret")

        self.assertFalse(getattr(chat_model, "use_responses_api", False))


