"""LLM response processing utilities."""

from __future__ import annotations

from typing import Any


def extract_text_from_response(response: Any) -> str:
    """
    Extract text from a model response object, handling various formats.

    Args:
        response: The response object (string, dict, or object with 'content' attribute).

    Returns:
        The extracted text content, stripped of whitespace.

    Raises:
        RuntimeError: If no text content can be extracted.
    """
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", None)
    if content is None:
        raise RuntimeError("The model returned no text output.")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
            elif isinstance(part, str) and part.strip():
                text_parts.append(part.strip())

        if text_parts:
            return "\n".join(text_parts)

    raise RuntimeError("The model returned no text output.")


def strip_code_fences(text: str, language_prefix: str | None = None) -> str:
    """
    Remove Markdown code fences and optional language prefix from text.

    Args:
        text: The text to clean.
        language_prefix: Optional language prefix to remove (e.g., 'sql', 'json').

    Returns:
        The cleaned text without code fences and prefix.
    """
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    if language_prefix:
        prefix = f"{language_prefix}\n"
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix) :].strip()

    return stripped.strip()
