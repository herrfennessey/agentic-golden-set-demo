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

# Test that search is working
test-search:
	poetry run python scripts/check_search.py

# Run Streamlit app
run:
	poetry run streamlit run app.py

# Run evaluation dashboard
dashboard:
	poetry run streamlit run scripts/dashboard.py

# Setup: install dependencies and download data (run weaviate-up and load-data separately)
setup: install download-wands
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "  1. make weaviate-up    # Start Weaviate (Docker)"
	@echo "  2. make load-data      # Load products (~\$$0.50 for embeddings)"
	@echo "  3. make run-agent QUERY='leather dining chairs'"
