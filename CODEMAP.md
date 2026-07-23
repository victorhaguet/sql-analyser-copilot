# SQL Copilot - Code Map

## General Overview

This project is a SQL Copilot application that uses LangGraph to orchestrate an AI agent for generating, validating, and executing SQL queries. The codebase is organized into two main directories:

- **`src/`** - Contains the main application source code including the FastAPI backend, Streamlit frontend, core business logic, and agent orchestration
- **`tests/`** - Contains unit tests that mirror the structure of the `src/` directory for comprehensive test coverage

Additional directories at the root:
- **`data/`** - Data files for the application
- **`logs/`** - Log files generated during execution

---

## Source Code Structure (`src/`)

### Root Level Files

| File | Description |
|------|-------------|
| `main.py` | FastAPI application entrypoint, API endpoints for authentication, user management, and query processing |
| `core.py` | Core business logic for the SQL copilot, including graph execution and state serialization |
| `graph.py` | LangGraph orchestration defining the SQL agent workflow with 9 nodes and conditional routing |
| `config.py` | Application configuration, environment variable loading, and FastAPI app creation |
| `models.py` | Pydantic models for API request/response schemas |
| `state.py` | State definitions for the LangGraph workflow |
| `tracing.py` | Trace logging utilities for debugging graph execution |
| `streamlit_app.py` | Streamlit application entrypoint and page navigation |
| `streamlit_ui.py` | Streamlit UI components and stylesheet loading |

### `app_auth/` - Authentication Module

Handles user authentication and authorization.

| File | Description |
|------|-------------|
| `database.py` | User database operations (create, read, update, delete users) |
| `password.py` | Password hashing and verification utilities |
| `admin.py` | Admin-specific authentication functions |

### `nodes/` - Graph Nodes

Individual processing nodes in the LangGraph workflow with conditional routing.

| File | Description |
|------|-------------|
| `database_selector.py` | Selects the appropriate database based on user input |
| `intent_classifier.py` | Classifies user intent as query or modification |
| `role_authorizer.py` | Checks user role authorization for modification operations |
| `sql_generator.py` | Generates SQL queries from natural language using LLM |
| `sql_validator.py` | Validates SQL for safety (SELECT-only for queries) |
| `sql_modification_validator.py` | Manages modification confirmation workflow |
| `sql_executor.py` | Executes validated SQL queries against the database |
| `sql_fallback_regenerator.py` | Repairs SQL after execution errors using the failed statement, database error, and schema |
| `result_analyst.py` | Analyzes query results and generates insights |

### `tools/` - Utility Tools

Reusable tools and helpers for database operations and SQL safety.

| File | Description |
|------|-------------|
| `database.py` | Database connection and query execution utilities |
| `sql_safety.py` | SQL safety validation rules and checks |
| `_helpers.py` | Internal helper functions |
| `exceptions.py` | Custom exception classes |

### `utils/` - General Utilities

General-purpose utility functions.

| File | Description |
|------|-------------|
| `llm.py` | LLM utility functions |
| `nodes.py` | Node-related utilities |
| `paths.py` | Path handling utilities |

### `llms/` - LLM Integrations

Language model integration interfaces.

| File | Description |
|------|-------------|
| `openai_compatible.py` | OpenAI-compatible LLM interface implementation |

### `prompts/` - Prompt Templates

Jinja2 prompt templates for LLM interactions.

| File | Description |
|------|-------------|
| `database_selector.j2` | Prompt template for database selection |
| `sql_generator.j2` | Prompt template for SQL generation |
| `result_analyst.j2` | Prompt template for result analysis |

### `pages/` - Streamlit Pages

Streamlit page components for the UI.

| File | Description |
|------|-------------|
| `sql_copilot.py` | Main SQL Copilot interface page |
| `user_management.py` | User administration page (admin only) |
| `login.py` | Login page |
| `auth.py` | Authentication utilities for Streamlit |
| `config.py` | Page configuration |

### `sql_copilot/` - Internal Package

Internal package structure (currently contains pycache directories).

---

## Test Structure (`tests/`)

The test directory mirrors the source structure:

| Test Directory/File | Tests |
|---------------------|-------|
| `test_auth/` | Authentication-related tests |
| `test_nodes/` | Graph node tests |
| `test_llms/` | LLM integration tests |
| `test_pages/` | Streamlit page tests |
| `test_tools/` | Tool and utility tests |
| `test_utils/` | General utility tests |
| `test_sql_copilot/` | SQL Copilot core tests |
| `test_config.py` | Configuration tests |
| `test_core.py` | Core logic tests |
| `test_core_serialization.py` | State serialization tests |
| `test_graph.py` | Graph orchestration tests |
| `test_main.py` | API endpoint tests |
| `test_models.py` | Pydantic model tests |
| `test_state.py` | State management tests |
| `test_streamlit_ui.py` | Streamlit UI component tests |
| `test_tracing.py` | Tracing utility tests |

---

## Execution Flow

1. **User Request** → FastAPI `/query` endpoint (`main.py:query`)
2. **Database Selection** → `DatabaseSelectorNode` (`nodes/database_selector.py`) or abort
3. **Intent Classification** → `IntentClassifierNode` (`nodes/intent_classifier.py`)
4. **Role Authorization** → `RoleAuthorizerNode` (`nodes/role_authorizer.py`) for modifications, or skip for queries
5. **SQL Generation** → `SQLGeneratorNode` (`nodes/sql_generator.py`)
6. **SQL Validation** → `SQLValidatorNode` (`nodes/sql_validator.py`) for queries, or `SQLModificationValidatorNode` (`nodes/sql_modification_validator.py`) for modifications
7. **Query Execution** → `SQLExecutorNode` (`nodes/sql_executor.py`)
8. **Fallback Regeneration** → `SQLFallbackRegeneratorNode` (`nodes/sql_fallback_regenerator.py`) after execution errors, bounded to three retries
9. **Result Analysis** → `ResultAnalystNode` (`nodes/result_analyst.py`) for successful queries
10. **Response** → Serialized state returned to client

### Conditional Routing

The graph uses conditional edges to route execution based on state:

- Database selection fails → abort
- Intent is modification → role_authorizer → sql_generator
- Intent is query → sql_generator directly
- Authorization fails → abort
- SQL is invalid → abort
- Modification not confirmed → abort
- Execution error below retry limit → regenerate → validate or re-approve → execute again
- Execution error at retry limit, or regeneration failure → abort
- Success → result_analyst → end
