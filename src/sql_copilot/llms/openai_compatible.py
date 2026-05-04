"""OpenAI model adapter built on langchain-openai."""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI




class OpenAICompatibleResponsesModel:
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
        return _message_to_text(self.client.invoke(message_input))

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


def load_models_from_env() -> tuple[OpenAICompatibleResponsesModel | None, OpenAICompatibleResponsesModel | None]:
    """
    Create default SQL generator and analyst models from environment variables.
    
    Returns:
        A tuple containing the generator model and analyst model instances, or (None, None) if the API key is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    generator_model_name: str | None = os.getenv("SQL_COPILOT_MODEL")
    analyst_model_name: str | None = os.getenv("SQL_COPILOT_ANALYST_MODEL") or generator_model_name

    return (
        OpenAICompatibleResponsesModel(
            model=generator_model_name, # type: ignore
            api_key=api_key, # type: ignore
            base_url=base_url,
        ),
        OpenAICompatibleResponsesModel( 
            model=analyst_model_name, # type: ignore
            api_key=api_key,
            base_url=base_url,
        ),
    )


def _message_to_text(message: Any) -> str:
    """
    Extract text from a LangChain AIMessage-like object.
    
    Args:
        message: The message object returned by the model's invoke method.

    Returns:
        The extracted text content from the message.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
        if text_parts:
            return "\n".join(text_parts)

    raise RuntimeError("The model returned no text output.")
