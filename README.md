# Agentic Golden Set Demo

## Welcome

Welcome to my demo! I created this to showcase how, using a Python framework and a frontier LLM model (gpt-5-nano), we
can generate high quality training data that took Wayfair hundreds of hours to create manually.

This POC is meant to accompany a talk I'm giving in Berlin, and it is not production-ready. If you see any glaring
issues, or weird malfunctions, please let me know by opening up a github issue, and I'd be happy to take a look!

## Overview

An AI agent that autonomously generates search relevance **golden sets** - curated lists of products with human-quality
relevance judgments for a given search query. These are used to evaluate and improve search ranking algorithms.

This project evaluates the agent against the [WANDS dataset](https://github.com/wayfair/WANDS) (233K human judgments
from Wayfair).

tl;dr we will:

* Download the WANDS dataset (available for free on Wayfair's public GitHub repo)
* Load products into a vector database (Weaviate) for fast hybrid search
* Run the agent on a set of queries to generate golden sets
* Compare the agent's results to the ground truth to evaluate the quality of the generated golden sets
* Visualize the results in a Streamlit dashboard

## Quick Start

### Prerequisites

- Python 3.11+
- Docker
- OpenAI API key
- Poetry

### Setup

```bash
# Clone and install
git clone https://github.com/herrfennessey/agentic-golden-set-demo.git
cd agentic-golden-set-demo
make install

# Configure (add your OpenAI API key)
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

# Download WANDS dataset
make download-wands

# Start Weaviate (requires Docker)
make weaviate-up

# Load products into Weaviate (with text-embedding-3-small, you can expect to pay ~$0.50 to embed the entire catalog)
make load-data
```

### Run the Agent

```bash
# Run on a single query
make run-agent QUERY="leather dining chairs"

# Or use the script directly for more options
poetry run python scripts/run_agent.py "podium with locking cabinet" --max-iterations 20
```

### View Results

```bash
# Start the Streamlit dashboard
make dashboard
```

### Verify Search Works

```bash
make test-search
```

## How It Works

The agent uses a **three-phase execution model**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DISCOVERY                           │
│                                                                 │
│  Agent explores the catalog and creates an execution plan       │
│                                                                 │
│  Tools: list_categories, search_products, submit_plan           │
│  Output: Plan with search steps + category steps                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: EXECUTION                           │
│                                                                 │
│  Search steps: Auto-execute, fetch full data, judge products    │
│  Category steps: Browse entire category, judge products         │
│                                                                 │
│  Tools: browse_category, finish_judgments                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: VALIDATION                          │
│                                                                 │
│  Separate subagent reviews ALL judgments with fresh context     │
│  Can: KEEP correct, ADJUST scores, REMOVE incorrect products    │
│                                                                 │
│  Catches: Size mismatches, wrong relevance levels, hallucinations│
└─────────────────────────────────────────────────────────────────┘
```

**Discovery Phase**: The agent calls `list_categories()` and `search_products()` to understand what's in the catalog.
Discovery searches return slim data (no judgments) for fast exploration. The agent then submits a plan with both search
steps and category steps.

**Execution Phase**:

- **Search steps** execute automatically - they re-fetch products with full data and judge each one in parallel
- **Category steps** require the agent to call `browse_category()`, which fetches all products and judges them in
  parallel

**Validation Phase**: When the agent calls `finish_judgments()`, a separate validation subagent reviews all judgments
with fresh context. It can keep correct judgments, adjust relevance scores (upgrade Partial→Exact or downgrade
Exact→Partial), or remove products that don't belong. This catches size/measurement mismatches, incorrect relevance
levels, and hallucinated products before the golden set is saved.

### Subagent Architecture

Both Execution and Validation phases use **isolated subagents** - lightweight LLM calls with fresh context (not part of
the main agent's conversation). This keeps context windows small and enables parallel processing. Each subagent has its
own configuration settings (model, reasoning effort, chunk size, workers, retries).

### Relevance Scale (WANDS Methodology)

The agent judges products on a 2-level scale:

- **Exact (2)**: Product fully matches the query - exactly what the user is looking for
- **Partial (1)**: Product is related but doesn't fully match (wrong size, color, style, etc.)

Products that are completely unrelated are not included in the golden set.

**Key principle**: Be generous with Partial judgments. If a customer searching for X might find the product useful
(even if not exactly X), it's Partial.

## Configuration

All settings are configured via environment variables in `.env`:

### Core Settings

| Variable                 | Default                  | Description                   |
|--------------------------|--------------------------|-------------------------------|
| `OPENAI_API_KEY`         | required                 | OpenAI API key                |
| `WEAVIATE_URL`           | `http://localhost:8080`  | Weaviate server URL           |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for Weaviate embeddings |

### Agent Settings

| Variable                  | Default      | Description                                   |
|---------------------------|--------------|-----------------------------------------------|
| `AGENT_MODEL`             | `gpt-5-nano` | Model for main agent                          |
| `AGENT_REASONING_EFFORT`  | `medium`     | Reasoning effort (low/medium/high)            |
| `AGENT_REASONING_SUMMARY` | `true`       | Enable reasoning summaries                    |
| `AGENT_MAX_ITERATIONS`    | `20`         | Max iterations before agentic flow errors out |

### Judgment Subagent Settings (search / browse judgements)

| Variable                 | Default      | Description                        |
|--------------------------|--------------|------------------------------------|
| `JUDGE_MODEL`            | `gpt-5-nano` | Model for judging products         |
| `JUDGE_REASONING_EFFORT` | `low`        | Reasoning effort (low recommended) |
| `JUDGE_CHUNK_SIZE`       | `25`         | Products per judgment batch        |
| `JUDGE_MAX_WORKERS`      | `5`          | Parallel judgment workers          |
| `JUDGE_MAX_RETRIES`      | `2`          | Retry attempts on failure          |

### Validation Subagent Settings

| Variable                    | Default      | Description                              |
|-----------------------------|--------------|------------------------------------------|
| `VALIDATE_MODEL`            | `gpt-5-nano` | Model for validating judgments           |
| `VALIDATE_REASONING_EFFORT` | `medium`     | Higher effort to catch errors            |
| `VALIDATE_CHUNK_SIZE`       | `20`         | Judgments per validation batch           |
| `VALIDATE_MAX_WORKERS`      | `3`          | Parallel validation workers              |
| `VALIDATE_MAX_RETRIES`      | `2`          | Retry attempts on failure                |

### Quality Thresholds

| Variable               | Default | Description                          |
|------------------------|---------|--------------------------------------|
| `MIN_EXACT_JUDGMENTS`  | `2`     | Minimum Exact (2) judgments required |
| `MIN_TOTAL_JUDGMENTS`  | `50`    | Minimum total judgments required     |
| `BROWSE_PRODUCT_LIMIT` | `2000`  | Max products per category browse     |

### Key Entry Points

| If you want to...         | Start here                                            |
|---------------------------|-------------------------------------------------------|
| Understand the agent flow | `src/goldendemo/agent/agent.py` (read the docstring!) |
| Modify tool behavior      | `src/goldendemo/agent/tools/`                         |
| Change prompts            | `src/goldendemo/agent/prompts.py`                     |
| Adjust guardrails         | `src/goldendemo/agent/guardrails/`                    |
| Debug tool dispatch       | `src/goldendemo/agent/runtime.py`                     |

### Running the Agent Programmatically

```python
from goldendemo.agent import GoldenSetAgent
from goldendemo.clients.weaviate_client import WeaviateClient

# Connect to Weaviate
client = WeaviateClient()
with client.connect():
    # Create agent
    agent = GoldenSetAgent(client, max_iterations=20)

    # Run with streaming events
    for event in agent.run_streaming("leather dining chairs"):
        print(f"{event.type}: {event.data}")

    # Or run blocking
    result = agent.run("leather dining chairs")
    print(f"Found {len(result.products)} relevant products")
```
