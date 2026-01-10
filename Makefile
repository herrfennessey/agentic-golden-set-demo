.PHONY: install test lint format clean weaviate-up weaviate-down load-data run dashboard pre-commit

# Install dependencies and pre-commit hooks
install:
	poetry install
	poetry run pre-commit install

# Run tests
test:
	poetry run pytest tests/ -v

# Run tests with coverage
test-cov:
	poetry run pytest tests/ -v --cov=src/goldendemo --cov-report=term-missing

# Lint code
lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/

# Format code
format:
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

# Run pre-commit on all files
pre-commit:
	poetry run pre-commit run --all-files

# Clean up
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Start Weaviate
weaviate-up:
	docker compose up -d weaviate

# Stop Weaviate
weaviate-down:
	docker compose down

# Download WANDS dataset
download-wands:
	poetry run python scripts/download_wands.py

# Load data into Weaviate
load-data:
	poetry run python scripts/load_weaviate.py

# Run the agent on a query
run-agent:
	@echo "Usage: make run-agent QUERY='your search query'"
	@test -n "$(QUERY)" || (echo "Error: QUERY is required" && exit 1)
	poetry run python scripts/run_agent.py "$(QUERY)"

# Run Streamlit app
run:
	poetry run streamlit run app.py

# Run evaluation dashboard
dashboard:
	poetry run streamlit run scripts/dashboard.py

# Full setup: install, download data, start weaviate, load data
setup: install download-wands weaviate-up
	@echo "Waiting for Weaviate to start..."
	sleep 10
	$(MAKE) load-data
	@echo "Setup complete! Run 'make run' to start the app."
