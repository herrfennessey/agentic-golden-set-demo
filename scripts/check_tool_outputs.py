#!/usr/bin/env python3
"""Debug tool to inspect product data sent to judgment subagent."""

from goldendemo.agent.judge import _format_products_for_judgment
from goldendemo.agent.utils import parse_product_features
from goldendemo.clients.weaviate_client import WeaviateClient
from goldendemo.config import settings


def main():
    client = WeaviateClient()

    with client.connect():
        query = input("Query [dinosaur]: ").strip() or "dinosaur"
        category = input("Category [Garden Statues]: ").strip() or "Garden Statues"

        print(f"\nFetching '{category}'...")
        products = client.get_by_class(product_class=category, limit=settings.browse_product_limit)
        print(f"Found {len(products)} products.\n")

        # Format exactly as browse_category does
        products_data = [
            {
                "product_id": p.product_id,
                "product_name": p.product_name,
                "category": p.product_class,
                "category_hierarchy": p.category_hierarchy,
                "description": p.product_description or "",
                "attributes": parse_product_features(p.product_features),
            }
            for p in products
        ]

        # Show formatted output (what subagent sees)
        chunk = products_data[:20]  # Show first 20
        formatted = _format_products_for_judgment(chunk)

        print("=" * 70)
        print(f"FORMATTED PRODUCT DATA (query: '{query}')")
        print("=" * 70)
        print(formatted)


if __name__ == "__main__":
    main()
