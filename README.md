# sql-analyser-copilot

## Try It Now

Live demo: https://sql-analyser-copilot-ywij9amvcnpzon78fqkt3z.streamlit.app/

SQL Analyser Copilot is an AI-powered data assistant that translates natural language queries into SQL, executes them on a relational database, and returns structured results along with business-level insights.

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

`SQL_COPILOT_DATABASES` lets the app route a question to the most relevant database before generating SQL. If none of the configured databases match the question, or if the question could match more than one configured database, the app returns an error asking the user to reformulate the request more specifically.

Start the API:

```bash
python3 src/sql_copilot/main.py
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
streamlit run src/sql_copilot/streamlit_app.py
```

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

### Quick start

```bash
docker compose up --build
```

Open the Streamlit UI at [http://localhost:8501](http://localhost:8501).

### Configuration

Copy the environment file and set your API keys:

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and other settings
```

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
docker-compose down
```
