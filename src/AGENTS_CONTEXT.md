# src/AGENTS_CONTEXT.md

## Purpose
Root folder for the SQL Copilot application - FastAPI backend and Streamlit UI entrypoints.

## Key Files

- **main.py**: FastAPI application with endpoints (/health, /auth/*, /query)
- **core.py**: Core business logic - graph execution and state serialization
- **graph.py**: LangGraph state machine definition (nodes, edges, routing)
- **state.py**: SQLAgentState TypedDict for node communication. Also carries the
  SQL generation agent loop's `messages` transcript (the only reducer field,
  `Annotated[list[AnyMessage], add_messages]` — appends instead of overwriting)
  plus its budget/status keys (`agent_status`, `agent_iterations`, `probe_count`,
  `clarification_rounds`, ...).
- **models.py**: Pydantic request/response models
- **config.py**: Environment variable loading and FastAPI app wiring
- **tracing.py**: Trace logging utilities and formatting
- **streamlit_app.py**: Streamlit application entrypoint and page navigation
- **streamlit_ui.py**: Streamlit UI helper functions and stylesheet loading

## Architecture

The SQL Copilot uses an 11-node LangGraph workflow with conditional routing and
interruption support. SQL generation is no longer a single-shot node: a bounded
agent loop (`sql_agent_llm` ⇄ `sql_agent_tools`, with a separate interrupting
node for user clarification) decides on its own how many times to inspect the
schema, ask the user for business details, and probe the database read-only
before committing to a final statement. See
[AGENTIC_SQL_GENERATION_PLAN.md](../AGENTIC_SQL_GENERATION_PLAN.md) for the
full design rationale.

### Nodes

1. **Database Checker** - Verifies the question matches the selected database
2. **Intent Classifier** - Determines if user request is a query or modification
3. **Role Authorizer** - Checks if user has permission for modification operations
4. **SQL Agent LLM** - Seeds/continues the agent transcript and calls the
   tool-calling model (`bind_tools`)
5. **SQL Agent Tools** - Dispatches `inspect_schema`/`run_readonly_probe` calls;
   rejects a mixed batch containing `ask_user`; enforces the probe budget
6. **SQL Agent Clarify** - Interrupts for user input when `ask_user` is the
   sole requested tool call
7. **SQL Agent Finalize** - Extracts the final SQL + rationale from the
   transcript, or resets state for a repair pass (D6)
8. **SQL Validator** - Ensures query safety for SELECT operations (no destructive operations)
9. **Modification Validator** - Manages interruption/confirmation workflow for modifications
10. **SQL Executor** - Runs the validated query against the database
11. **Result Analyst** - Analyzes results and generates insights

### Edge Routing

- **database_checker** → intent_classifier (abort if database selection fails)
- **intent_classifier** → role_authorizer (for modifications) or sql_agent_llm (for queries)
- **role_authorizer** → sql_agent_llm (authorized) or abort (unauthorized)
- **sql_agent_llm** → sql_agent_finalize (no tool calls), sql_agent_clarify (lone
  `ask_user` call), sql_agent_tools (other tool calls), or abort (iteration
  budget exceeded, intent-scoped)
- **sql_agent_tools** → sql_agent_llm (loop back)
- **sql_agent_clarify** → sql_agent_llm (answered) or abort (cancelled)
- **sql_agent_finalize** → sql_validator (queries), modification_validator
  (modifications), or abort (no valid SQL extracted)
- **sql_validator** → sql_executor (valid) or abort (invalid)
- **modification_validator** → sql_executor (confirmed) or abort (cancelled)
- **sql_executor** → result_analyst (success), sql_agent_llm (execution error,
  retries left — D6 repair), or abort (retries exhausted)
- **result_analyst** → end

### Interruption Workflow

Two nodes pause execution via LangGraph's interrupt feature, sharing the same
resume/session plumbing:

- **SQL Agent Clarify** interrupts when the model calls `ask_user` alone,
  surfacing its questions; resuming appends the answers to the transcript as a
  `ToolMessage` and the loop continues.
- **Modification Validator** interrupts before executing an INSERT/UPDATE/DELETE,
  surfacing the draft SQL (plus, on a repair pass, `previous_sql` /
  `regeneration_explanation`) for user approval; resuming with `approve` routes
  to SQL Executor, `reject` aborts.

A failed execution (query or modification) is fed back into the agent
transcript by SQL Executor and routed back to SQL Agent LLM for repair,
bounded by `retry_count` / `max_retries` — there is no separate regeneration
node.

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
