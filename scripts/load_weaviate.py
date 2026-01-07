#!/usr/bin/env python3
"""
Load WANDS products into Weaviate.

This script:
1. Connects to Weaviate
2. Creates the WandsProduct collection (if needed)
3. Loads products from the WANDS dataset (incrementally or full reset)
4. Reports statistics

Usage:
    poetry run python scripts/load_weaviate.py           # Incremental (add missing)
    poetry run python scripts/load_weaviate.py --reset   # Full reset and reload
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from goldendemo.clients.weaviate_client import WeaviateClient
from goldendemo.data.wands_loader import WANDSLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load WANDS products into Weaviate")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and reload all data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for insertions (default: 100)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("WANDS → Weaviate Loader")
    print("=" * 60)
    print()

    # Load WANDS data
    print("Loading WANDS dataset...")
    loader = WANDSLoader()
    try:
        products = loader.get_all_products()
        stats = loader.get_stats()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'make download-wands' first to download the dataset.")
        sys.exit(1)

    source_ids = {p.product_id for p in products}
    print(f"  Products in source: {stats['products']:,}")
    print(f"  Product classes: {stats['product_classes']}")
    print()

    # Connect to Weaviate
    print("Connecting to Weaviate...")
    client = WeaviateClient()

    with client.connect():
        if not client.is_ready():
            print("ERROR: Weaviate is not ready. Is it running?")
            print("Run 'make weaviate-up' to start Weaviate.")
            sys.exit(1)

        print("  Connected successfully")
        print()

        # Handle reset mode
        if args.reset:
            print("Reset mode: deleting existing collection...")
            client.create_collection(delete_existing=True)
            existing_ids: set[str] = set()
        else:
            # Create collection if it doesn't exist
            client.create_collection(delete_existing=False)

            # Get existing product IDs
            print("Fetching existing product IDs...")
            existing_ids = client.get_all_product_ids()
            print(f"  Products in Weaviate: {len(existing_ids):,}")

        # Calculate diff
        to_add = source_ids - existing_ids
        to_remove = existing_ids - source_ids

        print()
        print("Sync plan:")
        print(f"  To add: {len(to_add):,}")
        print(f"  To remove: {len(to_remove):,}")
        print(f"  Already synced: {len(existing_ids & source_ids):,}")
        print()

        # Remove products no longer in source
        if to_remove:
            print(f"Removing {len(to_remove):,} products no longer in source...")
            removed = client.delete_products(list(to_remove))
            print(f"  Removed {removed:,} products")
            print()

        # Add missing products
        if to_add:
            products_to_insert = [p for p in products if p.product_id in to_add]
            print(f"Inserting {len(products_to_insert):,} products (batch size: {args.batch_size})...")
            print("This may take a few minutes (embedding via OpenAI API)...")
            print()

            start_time = time.time()
            inserted = client.insert_products(products_to_insert, batch_size=args.batch_size)
            elapsed = time.time() - start_time

            print()
            print(f"Inserted {inserted:,} products in {elapsed:.1f}s")
            print(f"  Rate: {inserted / elapsed:.1f} products/sec")
        else:
            print("No products to add.")

        print()

        # Verify
        final_count = client.get_collection_count()
        print(f"Final collection count: {final_count:,}")

        # Show sample classes
        print()
        print("Sample product classes:")
        classes = client.get_all_classes()[:10]
        for cls in classes:
            print(f"  - {cls.product_class}: {cls.count} products")

    print()
    print("=" * 60)
    print("Load complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
