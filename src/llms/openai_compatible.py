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


def build_tool_calling_chat_model(
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> ChatOpenAI:
    """
    Build a plain langchain-openai `ChatOpenAI` client for tool-calling use cases.

    Deliberately built on standard chat-completions rather than the Responses API:
    `OpenAICompatibleResponsesModel` always sets `use_responses_api=True`, and
    `bind_tools` behaviour under that API is unverified. Standard chat-completions
    tool calling is the well-trodden path.

    Args:
        model: The name of the OpenAI (or compatible) model to use.
        api_key: The API key for authentication.
        base_url: Optional base URL for OpenAI-compatible services.

    Returns:
        A `ChatOpenAI` instance exposing `bind_tools`.
    """
    return ChatOpenAI(
        model=model,
        api_key=api_key, # type: ignore
        base_url=base_url,
    )
