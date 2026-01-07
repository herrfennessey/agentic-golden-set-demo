#!/usr/bin/env python3
"""
Quick script to test Weaviate search functionality.

Usage:
    poetry run python scripts/test_search.py
    poetry run python scripts/test_search.py "your custom query"
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from goldendemo.clients.weaviate_client import WeaviateClient


def test_search(query: str, limit: int = 10) -> float:
    """Run a search query and display results. Returns elapsed time in ms."""
    print(f"\nSearching for: '{query}'")
    print("-" * 60)

    client = WeaviateClient()
    with client.connect():
        if not client.is_ready():
            print("ERROR: Weaviate is not ready. Is it running?")
            print("Run 'make weaviate-up' to start Weaviate.")
            return 0.0

        start = time.perf_counter()
        results = client.hybrid_search(query, limit=limit)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"Search completed in {elapsed_ms:.1f}ms ({len(results)} results)")

        if not results:
            print("No results found.")
            return elapsed_ms

        for i, product in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"   Product ID: {product.product_id}")
            print(f"   Product Name:{product.product_name}")
            print(f"   Class: {product.product_class}")
            print(f"   Category: {product.category_hierarchy}")
            if product.description_snippet:
                snippet = product.description_snippet
                print(f"   Description: {snippet}")

        return elapsed_ms


def main() -> None:
    print("=" * 60)
    print("Weaviate Search Test")
    print("=" * 60)

    client = WeaviateClient()
    with client.connect():
        if not client.is_ready():
            print("ERROR: Weaviate is not ready. Is it running?")
            sys.exit(1)

        count = client.get_collection_count()
        print(f"Collection contains {count:,} products")

        if count == 0:
            print("No data loaded. Run 'poetry run python scripts/load_weaviate.py' first.")
            sys.exit(1)

    # If custom query provided, just run that
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        test_search(query)
        return

    # Otherwise run sample queries
    sample_queries = [
        "blue velvet sofa",
        "wooden coffee table",
        "modern dining chairs",
        "outdoor patio furniture",
        "king size bed frame",
    ]

    for query in sample_queries:
        test_search(query, limit=3)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
