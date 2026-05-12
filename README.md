# sql-analyser-copilot
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

`SQL_COPILOT_DATABASES` lets the app route a question to the most relevant database before generating SQL. If none of the configured databases match the question, the app returns an error asking the user to reformulate the request more specifically.

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
