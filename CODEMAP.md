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
| `core.py` | Core business logic for the SQL copilot, including graph execution, state serialization, and the env-tunable guardrail constants (`SQL_AGENT_RECURSION_LIMIT`, ...) |
| `graph.py` | LangGraph orchestration defining the SQL agent workflow with 12 nodes and conditional routing, including the agent loop's iteration budgets |
| `config.py` | Application configuration, environment variable loading, and FastAPI app creation |
| `models.py` | Pydantic models for API request/response schemas |
| `state.py` | State definitions for the LangGraph workflow, including the agent loop's `messages` transcript (the only reducer field) and its budget/status keys |
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
| `database_checker.py` | Verifies the user's question matches the selected database's schema |
| `intent_classifier.py` | Classifies user intent as query or modification |
| `role_authorizer.py` | Checks user role authorization for modification operations |
| `sql_agent.py` | The SQL generation agent loop — four nodes (`SQLAgentLLMNode`, `SQLAgentToolsNode`, `SQLAgentClarifyNode`, `SQLAgentFinalizeNode`) plus `SQLAgentBudgetExhaustedNode`, wired into `graph.py` in place of `sql_generator.py` and the retired `sql_fallback_regenerator.py`. The model decides on its own how many times to inspect the schema, ask the user for business details, and probe the database read-only before committing to a final statement |
| `sql_generator.py` | Single-shot NL→SQL node. No longer wired into the graph (replaced by `sql_agent.py`); kept in the repo but unused |
| `sql_validator.py` | Validates SQL for safety (SELECT-only for queries) |
| `sql_modification_validator.py` | Manages modification confirmation workflow |
| `sql_executor.py` | Executes validated SQL queries against the database; on failure, feeds the error back into the agent's `messages` transcript and increments `retry_count` so `sql_agent_llm` can repair it in-loop instead of via a dedicated regeneration node |
| `result_analyst.py` | Analyzes query results and generates insights |

### `tools/` - Utility Tools

Reusable tools and helpers for database operations and SQL safety.

| File | Description |
|------|-------------|
| `database.py` | Database connection and query execution utilities |
| `sql_safety.py` | SQL safety validation rules and checks |
| `_helpers.py` | Internal helper functions |
| `exceptions.py` | Custom exception classes |
| `agent_tools.py` | The three `@tool` callables the agent loop calls — `inspect_schema`, `ask_user`, `run_readonly_probe` — built by `build_agent_tools(database, validator, limits)` |

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
| `database_checker.j2` | Prompt template for the database-checker node |
| `intent_classifier.j2` | Prompt template for intent classification |
| `sql_agent.j2` | System prompt seeding the SQL generation agent loop — the three tools, probe-before-finalize, `ask_user`-alone, and INSERT/UPDATE/DELETE pre-conditions. No hardcoded fallback: the node raises if this file is missing or empty |
| `sql_generator.j2` | Prompt template for the retired single-shot `sql_generator.py` node; unused by the graph |
| `result_analyst_query.j2` | Prompt template for analyzing SELECT results |
| `result_analyst_modification.j2` | Prompt template for summarizing a completed modification |

### `pages/` - Streamlit Pages

Streamlit page components for the UI.

| File | Description |
|------|-------------|
| `sql_copilot.py` | Main SQL Copilot interface page |
| `user_management.py` | User administration page (admin only) |
| `login.py` | Login page |
| `auth.py` | Authentication utilities for Streamlit |
| `config.py` | Page configuration |

### `assets/` - Static Assets

| File | Description |
|------|-------------|
| `streamlit_styles.html` | Stylesheet injected into the Streamlit UI |

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
| `test_graph.py` | Graph orchestration tests, including full end-to-end agent-loop runs (clarification, approval, D6 repair) |
| `test_env_tunable_config.py` | Verifies every guardrail constant (iteration/probe/clarification budgets, recursion limit, schema-char cap) actually reads from its documented env var |
| `test_main.py` | API endpoint tests |
| `test_models.py` | Pydantic model tests |
| `test_state.py` | State management tests |
| `test_streamlit_ui.py` | Streamlit UI component tests |
| `test_tracing.py` | Tracing utility tests |

---

## Execution Flow

1. **User Request** → FastAPI `/query` endpoint (`main.py:query`)
2. **Database Check** → `DatabaseCheckerNode` (`nodes/database_checker.py`) or abort
3. **Intent Classification** → `IntentClassifierNode` (`nodes/intent_classifier.py`)
4. **Role Authorization** → `RoleAuthorizerNode` (`nodes/role_authorizer.py`) for modifications, or skip for queries
5. **SQL Generation Agent Loop** → `nodes/sql_agent.py`'s four nodes, driven by a
   tool-calling model that decides on its own how much schema inspection,
   clarification, and read-only probing it needs:
   - **sql_agent_llm** — seeds/continues the transcript, calls the model
   - **sql_agent_tools** — dispatches `inspect_schema` / `run_readonly_probe`
     calls, then loops back to sql_agent_llm
   - **sql_agent_clarify** — interrupts for user input when the model's sole
     tool call is `ask_user`, then loops back to sql_agent_llm
   - **sql_agent_finalize** — extracts the final SQL + rationale once the
     model stops requesting tools
   - **sql_agent_budget_exhausted** — reached only if tools are still
     requested past the iteration cap; asks the (unbound) model to explain
     its progress, then ends
6. **SQL Validation** → `SQLValidatorNode` (`nodes/sql_validator.py`) for queries, or `SQLModificationValidatorNode` (`nodes/sql_modification_validator.py`) for modifications
7. **Query Execution** → `SQLExecutorNode` (`nodes/sql_executor.py`)
8. **In-Loop Repair** → on an execution error, the error is fed back into the
   agent's `messages` transcript and routed straight back to `sql_agent_llm`,
   bounded by `retry_count` / `max_retries` — there is no dedicated
   regeneration node (the retired `sql_fallback_regenerator.py`'s job moved
   into the loop itself)
9. **Result Analysis** → `ResultAnalystNode` (`nodes/result_analyst.py`) for successful queries
10. **Response** → Serialized state returned to client

### Conditional Routing

The graph uses conditional edges to route execution based on state:

- Database check fails → abort
- Intent is modification → role_authorizer → sql_agent_llm
- Intent is query → sql_agent_llm directly
- Authorization fails → abort
- sql_agent_llm with no tool calls → sql_agent_finalize
- sql_agent_llm with a lone `ask_user` call → sql_agent_clarify (interrupt)
- sql_agent_llm with any other tool call(s) → sql_agent_tools, then loop back
- sql_agent_llm past the iteration budget (tools still requested) →
  sql_agent_budget_exhausted → end
- sql_agent_clarify cancelled → abort; answered → loop back to sql_agent_llm
- sql_agent_finalize with no SQL extracted → abort
- SQL is invalid → abort
- Modification not confirmed (reject) → abort
- Execution error below `max_retries` → loop back to sql_agent_llm for repair,
  then re-validate/re-approve and execute again
- Execution error at `max_retries` → abort
- Success → result_analyst → end
