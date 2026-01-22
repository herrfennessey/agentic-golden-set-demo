# Agentic Golden Set Demo

An AI agent that autonomously generates search relevance golden sets, evaluated against the [WANDS dataset](https://github.com/wayfair/WANDS) (233K human judgments from Wayfair).

## Quick Start

### Prerequisites

- Python 3.11+
- Docker
- OpenAI API key
- Poetry

### Setup

```bash
# Clone and install
git clone https://github.com/your-username/agentic-golden-set-demo.git
cd agentic-golden-set-demo
make install

# Configure (add your OpenAI API key)
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

# Download WANDS dataset
make download-wands

# Start Weaviate (requires Docker)
make weaviate-up

# Load products into Weaviate (~$0.50 for embeddings)
make load-data
```

### Run the Agent

```bash
# Run on a single query
make run-agent QUERY="blue velvet sofa"

# Or use the script directly for more options
poetry run python scripts/run_agent.py "modern coffee table" --max-iterations 20
```

### View Results

```bash
# Start the Streamlit dashboard
make run
# Open http://localhost:8501
```

## Data Setup Details

More details on each setup step:

### 1. Download WANDS Dataset

```bash
make download-wands
# Or: poetry run python scripts/download_wands.py
```

This downloads the WANDS dataset (~42K products, 480 queries, 233K judgments) to `data/wands/`.

### 2. Start Weaviate

```bash
make weaviate-up
# Or: docker compose up -d weaviate
```

Weaviate runs on `http://localhost:8080`. Check it's ready:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

### 3. Load Products into Weaviate

```bash
make load-data
# Or: poetry run python scripts/load_weaviate.py
```

This creates embeddings for all 42K products using OpenAI's embedding API (~$0.50 one-time cost) and loads them into Weaviate for hybrid search.

### 4. Verify Search Works

```bash
make test-search
# Or: poetry run python scripts/check_search.py
```

## How It Works

The agent uses a **two-phase execution model**:

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
```

**Discovery Phase**: The agent calls `list_categories()` and `search_products()` to understand what's in the catalog. Discovery searches return slim data (no judgments) for fast exploration. The agent then submits a plan with both search steps and category steps.

**Execution Phase**:
- **Search steps** execute automatically - they re-fetch products with full data and judge each one in parallel
- **Category steps** require the agent to call `browse_category()`, which fetches all products and judges them in parallel

When all steps are complete, the agent calls `finish_judgments()` to save the golden set.

## Project Structure

```
agentic-golden-set-demo/
├── src/goldendemo/
│   ├── agent/
│   │   ├── agent.py          # Main orchestrator (start here!)
│   │   ├── runtime.py        # OpenAI API calls, tool dispatch
│   │   ├── judge.py          # Judgment subagent (parallel evaluation)
│   │   ├── state.py          # AgentState, PlanStep tracking
│   │   ├── events.py         # Streaming event types
│   │   ├── prompts.py        # System prompts for each phase
│   │   ├── utils.py          # Shared utilities
│   │   ├── tools/
│   │   │   ├── search.py     # search_products (discovery)
│   │   │   ├── browse.py     # list_categories, browse_category
│   │   │   ├── plan.py       # submit_plan, complete_step
│   │   │   └── finish.py     # finish_judgments
│   │   └── guardrails/       # Validation (iteration, exploration, distribution)
│   ├── clients/
│   │   └── weaviate_client.py  # Vector DB (hybrid search)
│   ├── data/
│   │   ├── models.py         # Pydantic models
│   │   └── wands_loader.py   # Dataset loader
│   ├── evaluation/
│   │   └── comparator.py     # Compare agent vs ground truth
│   └── config.py             # Settings from .env
├── scripts/
│   ├── download_wands.py     # Download WANDS dataset
│   ├── load_weaviate.py      # Load products into Weaviate
│   ├── run_agent.py          # Run agent on queries
│   ├── check_search.py       # Verify search works
│   └── dashboard.py          # Evaluation dashboard
├── tests/                    # Unit tests
├── data/
│   ├── wands/                # WANDS dataset (gitignored)
│   └── golden_sets/          # Generated golden sets
└── app.py                    # Streamlit entry point
```

## Configuration

All settings are configured via environment variables in `.env`:

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate server URL |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for Weaviate embeddings |

### Agent Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `gpt-5-nano` | Model for main agent |
| `AGENT_REASONING_EFFORT` | `medium` | Reasoning effort (low/medium/high) |
| `AGENT_REASONING_SUMMARY` | `true` | Enable reasoning summaries |
| `AGENT_MAX_ITERATIONS` | `20` | Max iterations before timeout |

### Judgment Subagent Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGE_MODEL` | `gpt-5-nano` | Model for judging products |
| `JUDGE_REASONING_EFFORT` | `low` | Reasoning effort (low recommended) |
| `JUDGE_CHUNK_SIZE` | `25` | Products per judgment batch |
| `JUDGE_MAX_WORKERS` | `5` | Parallel judgment workers |
| `JUDGE_MAX_RETRIES` | `2` | Retry attempts on failure |

### Quality Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_EXACT_JUDGMENTS` | `5` | Minimum Exact (2) judgments required |
| `MIN_TOTAL_JUDGMENTS` | `50` | Minimum total judgments required |
| `BROWSE_PRODUCT_LIMIT` | `2000` | Max products per category browse |

## Development

```bash
# Run tests
make test

# Run tests excluding integration tests (no Weaviate needed)
poetry run pytest -m "not integration"

# Lint and type check
make lint

# Format code
make format

# Run pre-commit hooks
make pre-commit
```

### Key Entry Points

| If you want to... | Start here |
|-------------------|------------|
| Understand the agent flow | `src/goldendemo/agent/agent.py` (read the docstring!) |
| Modify tool behavior | `src/goldendemo/agent/tools/` |
| Change prompts | `src/goldendemo/agent/prompts.py` |
| Adjust guardrails | `src/goldendemo/agent/guardrails/` |
| Debug tool dispatch | `src/goldendemo/agent/runtime.py` |

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
    for event in agent.run_streaming("blue velvet sofa"):
        print(f"{event.type}: {event.data}")

    # Or run blocking
    result = agent.run("blue velvet sofa")
    print(f"Found {len(result.products)} relevant products")
```

## WANDS Dataset

The [WANDS dataset](https://github.com/wayfair/WANDS) contains:
- **42,994 products** from Wayfair's home goods catalog
- **480 search queries** with relevance judgments
- **233,448 human judgments** on a 3-level scale (Exact, Partial, Irrelevant)

The agent generates Exact (2) and Partial (1) judgments. Irrelevant products are not included in golden sets.

## License

MIT License - see [LICENSE](LICENSE) for details.

```bibtex
@InProceedings{wands,
  title = {WANDS: Dataset for Product Search Relevance Assessment},
  author = {Chen, Yan and Liu, Shujian and Liu, Zheng and Sun, Weiyi and Baltrunas, Linas and Schroeder, Benjamin},
  booktitle = {Proceedings of the 44th European Conference on Information Retrieval},
  year = {2022}
}
```
