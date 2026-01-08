# Agentic Golden Set Demo

A public demonstration of using AI agents to generate search relevance golden sets, evaluated against the [WANDS dataset](https://github.com/wayfair/WANDS) from Wayfair.

## Overview

This project demonstrates how an AI agent can autonomously generate search relevance judgments ("golden sets") for e-commerce product search. The agent explores a product catalog, applies semantic reasoning, and assigns relevance scores to products for given search queries.

The generated golden sets are then compared against WANDS (Wayfair ANnotation Dataset) - the largest public e-commerce search relevance dataset with 233,448 human judgments across 480 queries and 42,994 products.

## Features

- **Agentic Golden Set Generation**: AI agent with tool access to search, browse categories, and submit relevance judgments
- **WANDS Dataset Integration**: Full support for the WANDS product catalog and queries
- **Evaluation Framework**: Compare agent judgments against human ground truth using nDCG, precision, and recall
- **Streamlit UI**: Interactive visualization of agent results and comparison with WANDS

## Quick Start

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker (for Weaviate)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-golden-set-demo.git
cd agentic-golden-set-demo

# Install dependencies
make install

# Copy environment file and add your OpenAI API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Download WANDS dataset
make download-wands

# Start Weaviate and load data
make setup
```

### Loading Data into Weaviate

After downloading the WANDS dataset, load it into Weaviate:

```bash
# Start Weaviate
make weaviate-up

# Load products (creates embeddings via OpenAI API)
poetry run python scripts/load_weaviate.py
```

**Cost Note**: Embedding all 42,994 products using OpenAI's `text-embedding-3-small` costs approximately **$0.50 USD**. The script supports incremental loading - if interrupted, simply re-run and it will resume from where it left off.

```bash
# Check loading progress / resume interrupted load
poetry run python scripts/load_weaviate.py

# Full reset and reload
poetry run python scripts/load_weaviate.py --reset
```

### Running the Demo

```bash
# Start the Streamlit app
make run
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
agentic-golden-set-demo/
├── app.py                      # Streamlit entry point
├── src/goldendemo/
│   ├── agent/                  # Agent implementation
│   │   ├── agent.py           # Main agent loop
│   │   ├── tools.py           # Tool definitions
│   │   ├── guardrails.py      # Safety guardrails
│   │   └── prompts.py         # System prompts
│   ├── clients/
│   │   └── weaviate_client.py # Weaviate search client
│   ├── data/
│   │   ├── models.py          # Data models
│   │   └── wands_loader.py    # WANDS dataset loader
│   ├── evaluation/
│   │   ├── metrics.py         # nDCG, precision, recall
│   │   └── comparator.py      # Agent vs WANDS comparison
│   └── pages/                  # Streamlit pages
├── scripts/
│   ├── download_wands.py      # Download WANDS dataset
│   └── load_weaviate.py       # Load data into Weaviate
├── data/
│   ├── wands/                  # WANDS CSV files (gitignored)
│   └── golden_sets/            # Generated golden sets
└── tests/                      # Unit tests
```

## WANDS Dataset

The [WANDS dataset](https://github.com/wayfair/WANDS) contains:

- **42,994 products** from Wayfair's home goods catalog
- **480 search queries** with product class labels
- **233,448 relevance judgments** using a 3-level scale:
  - **Exact (2)**: Product is exactly what the user wants
  - **Partial (1)**: Product is somewhat relevant
  - **Irrelevant (0)**: Product does not match user intent

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  Golden Set     │────▶│   Weaviate DB   │
│   (Visualize)   │     │  Agent (OpenAI) │     │   (42K products)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         ▼                                               │
┌─────────────────┐     ┌─────────────────┐             │
│  Evaluation     │◀────│  WANDS Labels   │◀────────────┘
│  (nDCG, etc.)   │     │  (233K judgments)│
└─────────────────┘     └─────────────────┘
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint code
make lint

# Format code
make format
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [WANDS Dataset](https://github.com/wayfair/WANDS) by Wayfair
- [Weaviate](https://weaviate.io/) for vector search
- [OpenAI](https://openai.com/) for LLM capabilities

## Citation

If you use this project or the WANDS dataset, please cite:

```bibtex
@InProceedings{wands,
  title = {WANDS: Dataset for Product Search Relevance Assessment},
  author = {Chen, Yan and Liu, Shujian and Liu, Zheng and Sun, Weiyi and Baltrunas, Linas and Schroeder, Benjamin},
  booktitle = {Proceedings of the 44th European Conference on Information Retrieval},
  year = {2022}
}
```
