# Agentic Golden Set Demo

An AI agent that autonomously generates search relevance golden sets, evaluated against the [WANDS dataset](https://github.com/wayfair/WANDS) (233K human judgments from Wayfair).

## Quick Start (5 minutes)

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker
- OpenAI API key

### Setup

```bash
# Clone and install
git clone https://github.com/your-username/agentic-golden-set-demo.git
cd agentic-golden-set-demo
make install

# Configure (add your OpenAI API key)
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

# Download data, start Weaviate, load products (~$0.50 for embeddings)
make setup
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

## How It Works

The agent uses a **two-phase execution model**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DISCOVERY                           │
│                                                                 │
│  Agent explores the catalog and creates an exploration plan    │
│                                                                 │
│  Tools: list_categories, search_products, submit_plan          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: EXECUTION                           │
│                                                                 │
│  Agent browses each planned category, judges products          │
│                                                                 │
│  Tools: browse_category, complete_step, finish_judgments       │
└─────────────────────────────────────────────────────────────────┘
```

**Discovery Phase**: The agent calls `list_categories()` and `search_products()` to understand what's in the catalog, then submits a plan of categories to explore.

**Execution Phase**: For each category in the plan, the agent calls `browse_category()` which:
1. Fetches all products in the category
2. Runs a parallel judgment subagent to score each product (Exact=2, Partial=1)
3. Saves judgments automatically

When all categories are browsed, the agent calls `finish_judgments()` to save the golden set.

## Project Structure

```
agentic-golden-set-demo/
├── src/goldendemo/
│   ├── agent/
│   │   ├── agent.py          # Main agent orchestrator (start here!)
│   │   ├── runtime.py        # OpenAI API calls, tool dispatch
│   │   ├── judge.py          # Judgment subagent (parallel evaluation)
│   │   ├── state.py          # AgentState tracking
│   │   ├── prompts.py        # System prompts for each phase
│   │   ├── tools/            # Tool implementations
│   │   │   ├── search.py     # search_products
│   │   │   ├── browse.py     # list_categories, browse_category
│   │   │   ├── plan.py       # submit_plan
│   │   │   ├── finish.py     # complete_step, finish_judgments
│   │   │   └── base.py       # BaseTool, ToolResult
│   │   └── guardrails/       # Validation rules
│   ├── clients/
│   │   └── weaviate_client.py  # Vector DB client
│   ├── data/
│   │   ├── models.py         # Pydantic models
│   │   └── wands_loader.py   # Dataset loader
│   └── config.py             # Settings from .env
├── scripts/
│   ├── run_agent.py          # CLI to run agent on queries
│   ├── load_weaviate.py      # Load products into vector DB
│   ├── download_wands.py     # Download WANDS dataset
│   └── check_search.py       # Test search is working
├── tests/                    # Unit tests
├── data/
│   ├── wands/                # WANDS dataset (gitignored)
│   └── golden_sets/          # Generated golden sets
└── app.py                    # Streamlit entry point
```

## Configuration

Environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate URL |
| `AGENT_MODEL` | `o3-mini` | Model for main agent |
| `AGENT_REASONING_EFFORT` | `medium` | Reasoning effort (low/medium/high) |
| `AGENT_MAX_ITERATIONS` | `30` | Max iterations before timeout |
| `JUDGE_MODEL` | `gpt-4o-mini` | Model for judgment subagent |
| `MIN_EXACT_JUDGMENTS` | `3` | Guardrail: minimum exact matches |
| `MIN_PARTIAL_JUDGMENTS` | `5` | Guardrail: minimum partial matches |

## Development

```bash
# Run tests
make test

# Run tests excluding integration tests (no Weaviate needed)
poetry run pytest -m "not integration"

# Run a specific test
poetry run pytest tests/test_agent_tools.py::test_search_tool -v

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
client.connect()

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

## Citation

If you use this project or the WANDS dataset:

```bibtex
@InProceedings{wands,
  title = {WANDS: Dataset for Product Search Relevance Assessment},
  author = {Chen, Yan and Liu, Shujian and Liu, Zheng and Sun, Weiyi and Baltrunas, Linas and Schroeder, Benjamin},
  booktitle = {Proceedings of the 44th European Conference on Information Retrieval},
  year = {2022}
}
```
