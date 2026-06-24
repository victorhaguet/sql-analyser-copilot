# src/AGENTS_CONTEXT.md

## Purpose
Root folder for the SQL Copilot application - FastAPI backend and Streamlit UI entrypoints.

## Key Files

- **main.py**: FastAPI application with endpoints (/health, /auth/*, /query)
- **core.py**: Core business logic - graph execution and state serialization
- **graph.py**: LangGraph state machine definition (nodes, edges, routing)
- **state.py**: SQLAgentState TypedDict for node communication
- **models.py**: Pydantic request/response models
- **config.py**: Environment variable loading and FastAPI app wiring
- **tracing.py**: Trace logging utilities and formatting
- **streamlit_ui.py**: Streamlit UI helper functions

## Important Exports

- FastAPI app instance (from main.py)
- answer_question() - Run the full SQL copilot graph
- build_sql_agent_graph() - Build the LangGraph graph
- SQLAgentState - State schema for node communication

## Dependencies

- src/nodes/ - SQL copilot graph nodes
- src/tools/ - Database and SQL safety
- src/llms/ - LLM provider integrations
- src/utils/ - Shared utilities
- src/app_auth/ - Authentication

## When to Read Files

- **main.py**: When modifying API endpoints or authentication flow
- **core.py**: When changing graph execution or state serialization
- **graph.py**: When modifying LangGraph structure or node routing
- **state.py**: When extending node communication schema
- **config.py**: When adding environment configuration or model loading
- **models.py**: When updating API request/response schemas

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - LangGraph node implementations
- **src/tools/AGENTS_CONTEXT.md** - Database operations and SQL safety
- **src/llms/AGENTS_CONTEXT.md** - LLM provider integrations
- **src/utils/AGENTS_CONTEXT.md** - Shared utilities
- **src/app_auth/AGENTS_CONTEXT.md** - Authentication module
- **src/pages/AGENTS_CONTEXT.md** - Streamlit page components
- **src/prompts/AGENTS_CONTEXT.md** - Jinja2 prompt templates
- **src/sql_copilot/AGENTS_CONTEXT.md** - Unused reorganization placeholder
