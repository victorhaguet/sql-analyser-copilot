"""OpenAI model adapter built on langchain-openai."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from utils.llm import extract_text_from_response
from utils.nodes import LLM


class OpenAICompatibleResponsesModel(LLM):
    """Minimal invoke-based wrapper around langchain-openai ChatOpenAI."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        instructions: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the model wrapper."""
        self.model: str = model
        self.instructions: str | None = instructions
        self.client: Any = client or self._build_client(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    def invoke(self, prompt: str) -> str:
        """
        Generate a text response for the given prompt.
        
        Args:
            prompt: The input prompt to generate a response for.

        Returns:
            The generated text response from the model.
        """
        if self.instructions:
            message_input: Any = [
                ("system", self.instructions),
                ("human", prompt),
            ]
        else:
            message_input = prompt
        return extract_text_from_response(self.client.invoke(message_input))

    @property
    def chat_model(self) -> Any:
        """
        Expose the underlying LangChain chat model.

        The wrapper only implements `invoke(prompt: str) -> str`, which cannot
        express a tool call. This property reaches the raw client (e.g. `ChatOpenAI`)
        so callers that need `bind_tools` are not forced to build a second client.

        Returns:
            The underlying LangChain chat model client.
        """
        return self.client

    @staticmethod
    def _build_client(
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> Any:
        """
        Create a langchain-openai client lazily so tests do not require the package.
        
        Args:
            model: The name of the OpenAI model to use.
            api_key: The OpenAI API key for authentication.
            base_url: Optional base URL for the OpenAI API (useful for compatible services).

        Returns:
            An instance of the langchain-openai ChatOpenAI client configured with the provided parameters.
        """
        return ChatOpenAI(
            model=model,
            api_key=api_key, # type: ignore
            base_url=base_url,
            use_responses_api=True,
        )


def _is_reasoning_model(model: str) -> bool:
    """
    Detect OpenAI reasoning-tier models (o-series, gpt-5 non-chat variants).

    These models reject `reasoning_effort` combined with function/tool calling
    on the Chat Completions endpoint unless `reasoning_effort` is explicitly
    set to `'none'`. Non-reasoning models reject the `reasoning_effort` field
    outright, so it must only be sent for models detected here.

    Args:
        model: The model name to check.

    Returns:
        True if the model is a reasoning-tier model, False otherwise.
    """
    name = model.lower()
    if name.startswith(("o1", "o3", "o4")):
        return True
    return name.startswith("gpt-5") and "chat" not in name


def build_tool_calling_chat_model(
    model: str,
    api_key: str,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """
    Build a plain langchain-openai `ChatOpenAI` client for tool-calling use cases.

    Deliberately built on standard chat-completions rather than the Responses API:
    `OpenAICompatibleResponsesModel` always sets `use_responses_api=True`, and
    `bind_tools` behaviour under that API is unverified. Standard chat-completions
    tool calling is the well-trodden path.

    Reasoning-tier models reject `reasoning_effort` combined with tool calling on
    this endpoint unless it's explicitly set to `'none'`, while non-reasoning
    models reject the field outright if it's sent at all. `_is_reasoning_model`
    detects known OpenAI naming schemes (o-series, gpt-5 non-chat) as a default,
    but a gateway/proxy can alias a reasoning model under a name the heuristic
    won't recognize (this is what happened with the alias `gpt-5.6-terra`) —
    pass `reasoning_effort` explicitly to override the heuristic for those cases
    without touching this function again.

    Args:
        model: The name of the OpenAI (or compatible) model to use.
        api_key: The API key for authentication.
        base_url: Optional base URL for OpenAI-compatible services.
        reasoning_effort: Explicit value to send for the `reasoning_effort` field
            (e.g. `'none'`, `'low'`). Overrides `_is_reasoning_model`'s guess.
            Leave unset to use the name-based default.

    Returns:
        A `ChatOpenAI` instance exposing `bind_tools`.
    """
    if reasoning_effort is None and _is_reasoning_model(model):
        reasoning_effort = "none"

    reasoning_kwargs: dict[str, Any] = (
        {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key, # type: ignore
        base_url=base_url,
        **reasoning_kwargs,
    )
