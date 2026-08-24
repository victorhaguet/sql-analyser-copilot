# sql-analyser-copilot

A LangGraph-based SQL Copilot that translates natural language questions into SQL queries, validates them for safety, executes them against SQLite databases, and analyzes the results. Built with FastAPI, Streamlit, and LangGraph.

## Architecture

The SQL Copilot uses a 12-node LangGraph workflow with conditional routing and interruption support. SQL generation is not a single-shot LLM call: a bounded agent loop decides on its own how many times to inspect the schema, ask the user for business details, and probe the database read-only before committing to a final statement.

![Architecture](docs/graph.png)

### Nodes

1. **Database Checker** - Validates if the user's question matches the configured database (e.g., if the question asks about artists and the database is about music, it's a fit; if it asks about World Cup winners but the database is about war, it isn't)
2. **Intent Classifier** - Determines if user request is a query or modification
3. **Role Authorizer** - Checks if user has permission for modification operations
4. **SQL Agent LLM** - Seeds/continues the agent's transcript and calls the tool-calling model
5. **SQL Agent Tools** - Dispatches `inspect_schema` / `run_readonly_probe` calls the model requested, then loops back to SQL Agent LLM
6. **SQL Agent Clarify** - Pauses for user input when the model's sole requested tool call is `ask_user`
7. **SQL Agent Finalize** - Extracts the final SQL statement (+ rationale) once the model stops requesting tools
8. **SQL Agent Budget Exhausted** - Reached only if the model still wants to call tools past its iteration cap; asks the model (now unable to call tools) to explain what it learned and why it couldn't finish
9. **SQL Validator** - Ensures query safety for SELECT operations (no destructive operations)
10. **Modification Validator** - Manages interruption/confirmation workflow for modifications
11. **SQL Executor** - Runs the validated query against the database
12. **Result Analyst** - Analyzes results and generates insights

### Edge Routing

- **database_checker** → intent_classifier (abort if query doesn't match database)
- **intent_classifier** → role_authorizer (for modifications) or sql_agent_llm (for queries)
- **role_authorizer** → sql_agent_llm (authorized) or abort (unauthorized)
- **sql_agent_llm** → sql_agent_finalize (no tool calls), sql_agent_clarify (a lone `ask_user` call), sql_agent_tools (any other tool call), or sql_agent_budget_exhausted (iteration cap hit while tools are still requested — a clean finish is never blocked)
- **sql_agent_tools** → sql_agent_llm (loop back)
- **sql_agent_clarify** → sql_agent_llm (answered) or abort (cancelled)
- **sql_agent_budget_exhausted** → end
- **sql_agent_finalize** → sql_validator (queries) or modification_validator (modifications), or abort (no valid SQL extracted)
- **sql_validator** → sql_executor (valid) or abort (invalid)
- **modification_validator** → sql_executor (confirmed) or abort (rejected)
- **sql_executor** → result_analyst (success), sql_agent_llm (execution error, retries left — repaired in-loop), or abort (retries exhausted)
- **result_analyst** → end

### How the agent reasons

Say an admin asks: *"Add a new artist called Mandyspie."*

1. **sql_agent_llm** gets the question plus the full schema (already in its system prompt) and decides it needs more detail before generating an `INSERT`. It calls `inspect_schema(tables=["Artist", "Album"])`.
2. **sql_agent_tools** runs the call and returns both tables' column detail, including that `Album.ArtistId` references `Artist.ArtistId` — so adding an artist might imply adding albums too.
3. Back at **sql_agent_llm**, the model can't derive "should I also create albums?" from the schema or the question, so it calls `ask_user` alone with that question.
4. **sql_agent_clarify** pauses the graph (a LangGraph interrupt) and surfaces the question to the UI.
5. The admin answers "No, just the artist." The answer is appended to the transcript as a tool result, and the loop resumes at sql_agent_llm.
6. The model now has everything it needs and returns `INSERT INTO Artist (Name) VALUES ('Mandyspie')` with no further tool calls.
7. **sql_agent_finalize** extracts that statement. Since the intent is a modification, the graph routes to **modification_validator**, which interrupts again — this time for approval, showing the exact SQL before anything runs.
8. Only after the admin approves does **sql_executor** actually run the `INSERT`.

If that `INSERT` had instead failed (say, a constraint violation), **sql_executor** would feed the database error back into the transcript and route straight back to **sql_agent_llm** to repair it — bounded by `max_retries` — going through a fresh **modification_validator** approval before executing again. There is no separate "fallback" node for this: repair is just another turn of the same loop, using the same probe-before-finalize discipline as the first attempt.

The loop is bounded on every axis that could otherwise run away: `SQL_AGENT_MAX_ITERATIONS_QUERY` / `_MODIFICATION` caps total sql_agent_llm turns (scoped by intent, since reads should stay fast), `SQL_AGENT_MAX_PROBES` caps `run_readonly_probe` calls, `SQL_AGENT_MAX_CLARIFICATIONS` caps `ask_user` rounds, and `SQL_AGENT_RECURSION_LIMIT` is LangGraph's own hard step ceiling — a backstop in case the others fail. See `.env.example` for all of these, and [AGENTIC_SQL_GENERATION_PLAN.md](AGENTIC_SQL_GENERATION_PLAN.md) for the full design rationale.

### Interruption Workflow

Two nodes pause execution via LangGraph's interrupt feature:

- **SQL Agent Clarify** interrupts when the model calls `ask_user` alone, surfacing its questions to the UI. Resuming with answers appends them to the transcript and the loop continues; resuming with a cancellation aborts the run cleanly.
- **Modification Validator** interrupts before executing any INSERT/UPDATE/DELETE, surfacing the draft SQL (and, on a repair pass, the previous attempt's error and the model's explanation of what changed) for user approval:

1. User makes a modification request
2. Intent classifier routes to modification path
3. Role authorizer checks permissions
4. The agent loop (see "How the agent reasons" above) proposes a statement, inspecting the schema and asking clarifying questions along the way
5. **Modification validator interrupts** - waits for user confirmation
6. User reviews and confirms/cancels
7. If confirmed → SQL executor runs the query
8. If cancelled → graph aborts

This interruption mechanism is a key safety feature of the SQL Copilot.

## Authentication System

The application includes a user authentication system with role-based access control:

- **Roles**: `admin`, `editor`, `readonly`
- **Password hashing**: Argon2 via passlib
- **User management**: Admin-only CRUD operations

### Endpoints

| Endpoint | Method | Description | Role Required |
|----------|--------|-------------|---------------|
| `/auth/login` | POST | Authenticate user | Any |
| `/auth/me` | GET | Get current user info | Any |
| `/auth/users` | POST | Create user | admin |
| `/auth/users` | GET | List all users | admin |
| `/auth/users/{sub}` | PUT | Update user | admin |
| `/auth/users/{sub}` | DELETE | Delete user | admin |

## Run with an OpenAI model via LangChain

Install dependencies:

```bash
pip install -e .
```

Create your local environment file:

```bash
cp .env.example .env
```

Then set your provider settings in `.env`:

```bash
OPENAI_API_KEY="your_api_key_here"
SQL_COPILOT_MODEL="gpt-5.2"
```

Optional:

```bash
OPENAI_BASE_URL="https://your-provider.example/v1"
SQL_COPILOT_ANALYST_MODEL="gpt-5.2"
SQL_COPILOT_DATABASES='[
  {
    "name": "chinook",
    "path": "data/Chinook_Sqlite.sqlite",
    "description": "Music store database with artists, albums, tracks, customers, invoices, and employees."
  }
]'
```

`SQL_COPILOT_DATABASES` configures the database for the application. The database checker validates if the user's question matches the database content before generating SQL. If the question doesn't match (e.g., asking about sports in a music database), the app returns an error.

Start the API:

```bash
make api
```

Try it:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -d '{"question":"Which 5 artists have the most albums?"}'
```

Start the Streamlit UI in a separate process:

```bash
make ui
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Execute SQL query (requires authentication) |

**Authentication**: Include the user's `sub` (subject ID) in the `x_user_sub` header.

Defaults:

- FastAPI API: `http://127.0.0.1:8000`
- Streamlit UI: `http://127.0.0.1:8501`

Optional UI configuration:

```bash
SQL_COPILOT_API_BASE_URL="http://127.0.0.1:8000"
```

Note: `langchain-openai` targets the official OpenAI API. A custom `OPENAI_BASE_URL` can work for compatible endpoints, but some third-party providers may require a provider-specific LangChain package.

## Continuous Integration

The CI pipeline (`.github/workflows/ci.yaml`) runs on every push to `main` and on all pull requests. It performs two jobs:

- **Python Tests**: Installs dependencies via `make install` and runs `make coverage` to execute tests with a coverage report.
- **Docker Build**: Builds the Docker image using `docker compose -f docker-compose.yml build`.

## Makefile Commands

A Makefile file has been created to easily control and use the repo. All of the following commands can be used. 

```bash
make env        # Create venv and install dependencies
make install    # Reinstall dependencies in existing venv
make test       # Run unit tests
make coverage   # Run tests with coverage report
make api        # Start FastAPI (port 8000)
make ui         # Start Streamlit UI (port 8501)
make docker-up  # Build and start Docker containers
make docker-down # Stop Docker containers
```

## Run with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

### Configuration

Copy the environment file and set your API keys:

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and other settings
```

### Quick start

```bash
make docker-up
```

Open the Streamlit UI at [http://localhost:8501](http://localhost:8501).

### Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8501 | Streamlit UI |
| Backend | http://localhost:8000 | FastAPI REST API |
| Health | http://localhost:8000/health | Health check endpoint |

### Updating database files

Database files are mounted from the host's `data/` directory at runtime. To use different databases, place SQLite files in `data/` and update `SQL_COPILOT_DATABASES` in your `.env` file.

### Stopping

```bash
make docker-down
```
