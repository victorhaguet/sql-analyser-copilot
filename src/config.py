"""Application configuration and environment variable loading."""

from __future__ import annotations

import os
from typing import Any, Tuple
from fastapi import FastAPI

from dotenv import load_dotenv

from llms.openai_compatible import OpenAICompatibleResponsesModel, build_tool_calling_chat_model
from app_auth import ensure_admin_exists

load_dotenv()

def load_models_from_env() -> Tuple[
    OpenAICompatibleResponsesModel | None,
    OpenAICompatibleResponsesModel | None,
    OpenAICompatibleResponsesModel | None,
]:
    """Create default SQL generator, analyst, and intent models from environment variables.

    Returns:
        A tuple containing the generator model, analyst model, and intent model instances,
        or (None, None, None) if the API key is not set or the generator model name is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None, None

    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    generator_model_name: str | None = os.getenv("SQL_COPILOT_MODEL")

    if not generator_model_name:
        return None, None, None

    analyst_model_name: str = os.getenv("SQL_COPILOT_ANALYST_MODEL") or generator_model_name
    intent_model_name: str = os.getenv("SQL_COPILOT_INTENT_MODEL") or generator_model_name

    return (
        OpenAICompatibleResponsesModel(
            model=generator_model_name,
            api_key=api_key,
            base_url=base_url,
        ),
        OpenAICompatibleResponsesModel(
            model=analyst_model_name,
            api_key=api_key,
            base_url=base_url,
        ),
        OpenAICompatibleResponsesModel(
            model=intent_model_name,
            api_key=api_key,
            base_url=base_url,
        ),
    )


def load_sql_agent_model_from_env() -> Any | None:
    """Create the tool-calling chat model used by the SQL generation agent loop.

    Reads `SQL_COPILOT_AGENT_MODEL`, falling back to `SQL_COPILOT_MODEL` when unset,
    so the agent works out of the box for anyone who already configured a generator
    model. Unlike `load_models_from_env`, the returned client is a plain
    `ChatOpenAI` (not wrapped in `OpenAICompatibleResponsesModel`) so `bind_tools`
    is reachable directly.

    Returns:
        A `ChatOpenAI` instance, or `None` if the API key or a model name is not
        configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    agent_model_name = os.getenv("SQL_COPILOT_AGENT_MODEL") or os.getenv("SQL_COPILOT_MODEL")
    if not agent_model_name:
        return None

    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    return build_tool_calling_chat_model(
        model=agent_model_name,
        api_key=api_key,
        base_url=base_url,
    )


def load_trace_config() -> tuple[bool, str]:
    """Load tracing configuration from environment variables.

    Returns:
        A tuple containing (enable_trace, trace_log_dir) where:
        - enable_trace: Whether to enable trace logging
        - trace_log_dir: Directory path for trace logs
    """
    enable_trace = os.getenv("ENABLE_TRACE_LOGGING", "false").lower() in ("true", "1", "yes")
    trace_log_dir = os.getenv("TRACE_LOG_DIR", "logs")
    return enable_trace, trace_log_dir


def create_app_from_env():
    """Create the FastAPI app using OpenAI-compatible environment variables when present."""

    sql_generator_model, analyst_model, intent_model = load_models_from_env()
    sql_agent_model = load_sql_agent_model_from_env()
    enable_trace, trace_log_dir = load_trace_config()
    return create_app(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        selector_model=sql_generator_model,
        intent_model=intent_model,
        sql_agent_model=sql_agent_model,
        enable_trace=enable_trace,
        trace_log_dir=trace_log_dir,
    )


def create_app(
    sql_generator_model=None,
    analyst_model=None,
    selector_model=None,
    intent_model=None,
    sql_agent_model=None,
    validator=None,
    execution_limit: int = 200,
    enable_trace: bool = False,
    trace_log_dir: str = "logs",
):
    """Build the FastAPI application around the SQL copilot core.

    The models are injected at startup time so the HTTP layer stays independent
    from the concrete LLM provider implementation.

    Args:
        sql_generator_model: The language model to use for SQL generation (optional).
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        intent_model: The language model to use for intent classification (optional).
        sql_agent_model: The bind_tools-capable chat model for the SQL agent loop (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        A FastAPI application instance ready to serve SQL copilot requests.
    """
    ensure_admin_exists()

    fastapi_app = FastAPI(title="SQL Analyser Copilot", version="0.1.0")
    fastapi_app.state.sql_generator_model = sql_generator_model
    fastapi_app.state.analyst_model = analyst_model
    fastapi_app.state.selector_model = selector_model
    fastapi_app.state.intent_model = intent_model
    fastapi_app.state.sql_agent_model = sql_agent_model
    fastapi_app.state.validator = validator
    fastapi_app.state.execution_limit = execution_limit
    fastapi_app.state.enable_trace = enable_trace
    fastapi_app.state.trace_log_dir = trace_log_dir
    fastapi_app.state.pending_approval_sessions = {}

    return fastapi_app
