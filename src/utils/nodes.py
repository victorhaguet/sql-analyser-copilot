"""Shared utilities for SQL copilot nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
