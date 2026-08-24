"""Shared utilities for SQL copilot nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from jinja2 import Environment, StrictUndefined


PROMPT_ENVIRONMENT = Environment(
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


def load_prompt(path: Path, fallback: str) -> str:
    """Load a prompt template from a jinja file, falling back to a default string if the file does not exist.

    Args:
        path (Path): Path of the jinja prompt file
        fallback (str): fallback prompt

    Returns:
        str: final prompt
    """    
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    return text or fallback


def render_prompt(template: str, **context: Any) -> str:
    """Render a Jinja prompt template with the given context.

    Args:
        template (str): Jinja template

    Returns:
        str: rendered prompt
    """    
    return PROMPT_ENVIRONMENT.from_string(template).render(**context).strip()


class LLM(Protocol):
    """Minimal protocol for language models used across all nodes."""

    def invoke(self, prompt: str) -> Any:
        """
        Return model output for the given prompt.

        Args:
            prompt: The input prompt string to generate a response for.

        Returns:
            The raw output from the model
        """


class ToolCallingChatModel(Protocol):
    """Protocol for LangChain chat models capable of tool-calling.

    Unlike `LLM`, `invoke` here takes message-shaped input (e.g. a list of
    `BaseMessage`) rather than a plain string, and `bind_tools` is what the SQL
    generation agent loop needs to let the model request tool calls.
    """

    def bind_tools(self, tools: Sequence[Any]) -> Any:
        """
        Return a runnable bound to the given tools, so the model may request
        tool calls instead of (or alongside) a text answer.

        Args:
            tools: The tools the model may call.

        Returns:
            A runnable exposing the same `invoke` interface.
        """

    def invoke(self, input_: Any) -> Any:
        """
        Return model output for the given message input.

        Args:
            input_: The message(s) to send to the model.

        Returns:
            The raw output from the model (typically an `AIMessage`).
        """
