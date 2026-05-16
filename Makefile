# Makefile for SQL Copilot project
# Here are the available commands:
# - env: Create a virtual environment and install dependencies
# - install: Install dependencies in the existing virtual environment
# - test: Run unit tests
# - coverage: Run tests with coverage and generate a report
# - api: Start the FastAPI backend server
# - ui: Start the Streamlit frontend server
# - docker-up: Build and start the Docker containers
# - docker-down: Stop and remove the Docker containers
.PHONY: env install test coverage api ui docker-up docker-down


env install:
	python3 -m venv venv
	./venv/bin/pip install -e .

test:
	./venv/bin/coverage run -m unittest discover -s tests

coverage:
	./venv/bin/pip install coverage -q
	./venv/bin/coverage run -m unittest discover -s tests
	./venv/bin/coverage report --show-missing
	./venv/bin/coverage html --dir .coverage_html

api:
	./venv/bin/python3 src/sql_copilot/main.py

ui:
	./venv/bin/streamlit run src/sql_copilot/streamlit_app.py --server.port 8501

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down