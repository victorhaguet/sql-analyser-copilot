# AGENTS.md

You are a code maintainer.

## Modification Workflow

For each new modification, follow these steps:

1. Read the CODEMAP.md to understand the project structure
2. Identify the code part where modification/addition of code must be done
3. Read the AGENT_CONTEXT.md of those places, and only read files if it is strictly necessary
4. If you are in plan mode, provide a plan. If you are in build mode, build the modifications

## Repository Specificities

### Virtual Environment

This project uses a Python virtual environment located at `venv/`. Activate it before running commands:

```bash
source venv/bin/activate
```

### Available Make Commands

- `make env` - Create a virtual environment and install dependencies
- `make install` - Install dependencies in the existing virtual environment
- `make test` - Run unit tests
- `make coverage` - Run tests with coverage and generate a report
- `make api` - Start the FastAPI backend server
- `make ui` - Start the Streamlit frontend server
- `make docker-up` - Build and start the Docker containers
- `make docker-down` - Stop and remove the Docker containers
