"""Application configuration and environment variable loading."""

from __future__ import annotations

import os
from typing import Tuple
from fastapi import FastAPI

from dotenv import load_dotenv

from llms.openai_compatible import OpenAICompatibleResponsesModel
from app_auth import ensure_admin_exists

load_dotenv()

def load_models_from_env() -> Tuple[OpenAICompatibleResponsesModel | None, OpenAICompatibleResponsesModel | None]:
    """Create default SQL generator and analyst models from environment variables.

    Returns:
        A tuple containing the generator model and analyst model instances, 
        or (None, None) if the API key is not set or the generator model name is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    generator_model_name: str | None = os.getenv("SQL_COPILOT_MODEL")

    if not generator_model_name:
        return None, None
    
    analyst_model_name: str = os.getenv("SQL_COPILOT_ANALYST_MODEL") or generator_model_name

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
    )


def create_app_from_env():
    """Create the FastAPI app using OpenAI-compatible environment variables when present."""

    sql_generator_model, analyst_model = load_models_from_env()
    return create_app(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        selector_model=sql_generator_model,
        databases=None,
    )


def create_app(
    sql_generator_model=None,
    analyst_model=None,
    selector_model=None,
    databases=None,
    validator=None,
    execution_limit: int = 200,
):
    """Build the FastAPI application around the SQL copilot core.

    The models are injected at startup time so the HTTP layer stays independent
    from the concrete LLM provider implementation.

    Args:
        sql_generator_model: The language model to use for SQL generation (optional).
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        databases: The registered databases available for routing (optional).
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
    fastapi_app.state.databases = databases
    fastapi_app.state.validator = validator
    fastapi_app.state.execution_limit = execution_limit

    return fastapi_app
