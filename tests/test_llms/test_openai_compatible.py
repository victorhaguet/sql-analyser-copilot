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

    def test_build_tool_calling_chat_model_disables_reasoning_effort_for_reasoning_models(
        self,
    ) -> None:
        """Reasoning-tier models must get reasoning_effort='none' to allow tool calling."""
        for model_name in ("gpt-5.6-terra", "gpt-5", "gpt-5-mini", "o1", "o3-mini", "o4-mini"):
            with self.subTest(model=model_name):
                chat_model = build_tool_calling_chat_model(model=model_name, api_key="secret")

                self.assertEqual(chat_model.reasoning_effort, "none")

    def test_build_tool_calling_chat_model_leaves_reasoning_effort_unset_for_other_models(
        self,
    ) -> None:
        """Non-reasoning models must not receive reasoning_effort, which they reject outright."""
        for model_name in ("gpt-4o", "gpt-4.1", "gpt-3.5-turbo", "gpt-5-chat", "demo-model"):
            with self.subTest(model=model_name):
                chat_model = build_tool_calling_chat_model(model=model_name, api_key="secret")

                self.assertIsNone(chat_model.reasoning_effort)

    def test_build_tool_calling_chat_model_explicit_reasoning_effort_overrides_heuristic(
        self,
    ) -> None:
        """An explicit reasoning_effort must apply even for an unrecognized model alias."""
        chat_model = build_tool_calling_chat_model(
            model="terra-large", # gateway alias the name heuristic can't recognize
            api_key="secret",
            reasoning_effort="none",
        )

        self.assertEqual(chat_model.reasoning_effort, "none")

    def test_build_tool_calling_chat_model_explicit_reasoning_effort_wins_over_default(
        self,
    ) -> None:
        """An explicit reasoning_effort must take priority over the name-based default."""
        chat_model = build_tool_calling_chat_model(
            model="gpt-5.6-terra",
            api_key="secret",
            reasoning_effort="low",
        )

        self.assertEqual(chat_model.reasoning_effort, "low")


