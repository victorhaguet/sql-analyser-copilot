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
- **streamlit_app.py**: Streamlit application entrypoint and page navigation
- **streamlit_ui.py**: Streamlit UI helper functions and stylesheet loading

## Architecture

The SQL Copilot uses a 9-node LangGraph workflow with conditional routing and interruption support:

### Nodes

1. **Database Selector** - Routes questions to the most relevant database
2. **Intent Classifier** - Determines if user request is a query or modification
3. **Role Authorizer** - Checks if user has permission for modification operations
4. **SQL Generator** - Converts natural language to SQL using LLM
5. **SQL Validator** - Ensures query safety for SELECT operations (no destructive operations)
6. **Modification Validator** - Manages interruption/confirmation workflow for modifications
7. **SQL Executor** - Runs the validated query against the database
8. **Result Analyst** - Analyzes results and generates insights

### Edge Routing

- **database_selector** → intent_classifier (abort if database selection fails)
- **intent_classifier** → role_authorizer (for modifications) or sql_generator (for queries)
- **role_authorizer** → sql_generator (authorized) or abort (unauthorized)
- **sql_generator** → sql_validator (queries) or modification_validator (modifications)
- **sql_validator** → sql_executor (valid) or abort (invalid)
- **modification_validator** → sql_executor (confirmed) or abort (cancelled)
- **sql_executor** → result_analyst (success) or abort (error)
- **result_analyst** → end

### Interruption Workflow

The **Modification Validator** node uses LangGraph's interruption feature to pause execution and wait for user confirmation before executing modifications:

1. User makes a modification request (INSERT/UPDATE/DELETE)
2. Intent classifier routes to modification path
3. Role authorizer checks permissions
4. SQL generator creates the modification query
5. **Modification validator interrupts** - pauses execution and waits for user confirmation
6. User reviews and confirms/cancels the modification
7. If confirmed → SQL executor runs the query
8. If cancelled → graph aborts with error

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
- **streamlit_app.py**: When modifying Streamlit page navigation

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - LangGraph node implementations
- **src/tools/AGENTS_CONTEXT.md** - Database operations and SQL safety
- **src/llms/AGENTS_CONTEXT.md** - LLM provider integrations
- **src/utils/AGENTS_CONTEXT.md** - Shared utilities
- **src/app_auth/AGENTS_CONTEXT.md** - Authentication module
- **src/pages/AGENTS_CONTEXT.md** - Streamlit page components
- **src/prompts/AGENTS_CONTEXT.md** - Jinja2 prompt templates
