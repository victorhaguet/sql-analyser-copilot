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
```

Start the API:

```bash
python3 src/sql_copilot/main.py
```

Try it:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -d '{"question":"Show me 5 artists"}'
```

Note: `langchain-openai` targets the official OpenAI API. A custom `OPENAI_BASE_URL` can work for compatible endpoints, but some third-party providers may require a provider-specific LangChain package.
