# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an agentic golden set generation demo for search relevance evaluation. It uses an AI agent to generate relevance judgments for product search queries, compared against the WANDS dataset (Wayfair's 233K human judgments across 42K products).

## Common Commands

```bash
# Install dependencies and pre-commit hooks
make install

# Run all tests
make test

# Run a single test file
poetry run pytest tests/test_models.py -v

# Run a single test
poetry run pytest tests/test_models.py::TestProduct::test_categories -v

# Run tests excluding integration tests (no external services needed)
poetry run pytest -m "not integration"

# Lint and type check
make lint

# Format code
make format

# Run pre-commit hooks on all files
make pre-commit

# Start Weaviate (Docker required)
make weaviate-up

# Load WANDS data into Weaviate (~$0.50 for OpenAI embeddings)
poetry run python scripts/load_weaviate.py

# Run Streamlit app
make run
```

## Architecture

**Data Flow**: WANDS CSV files → `WANDSLoader` → Pydantic models → `WeaviateClient` → Weaviate vector DB

**Key Components**:
- `src/goldendemo/config.py` - Central settings via pydantic-settings, reads from `.env`
- `src/goldendemo/data/models.py` - Core Pydantic models (`Product`, `Query`, `RelevanceLabel`, `AgentJudgment`)
- `src/goldendemo/data/wands_loader.py` - Loads WANDS TSV files (note: files use tab delimiter despite .csv extension)
- `src/goldendemo/clients/weaviate_client.py` - Hybrid search (vector + BM25) over products

**Relevance Scale** (WANDS format):
- `Exact (2)` - Product exactly matches query intent
- `Partial (1)` - Product is somewhat relevant
- `Irrelevant (0)` - Product does not match

## Testing

Tests are in `tests/`. Integration tests requiring Weaviate are marked with `@pytest.mark.integration`.

The test fixtures create mock WANDS data using tab-separated format to match the actual dataset.

## Configuration

Environment variables in `.env`:
- `OPENAI_API_KEY` - Required for embeddings and LLM
- `WEAVIATE_URL` - Defaults to `http://localhost:8080`
