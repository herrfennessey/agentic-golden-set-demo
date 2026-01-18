#!/usr/bin/env python3
"""Test tool outputs to see what the agent receives."""

import json

from goldendemo.agent.state import AgentState
from goldendemo.agent.tools.browse import ListCategoriesTool
from goldendemo.agent.tools.search import SearchProductsTool
from goldendemo.clients.weaviate_client import WeaviateClient


def main():
    print("🔧 Testing Tool Outputs\n")
    print("=" * 60)

    # Connect to Weaviate
    client = WeaviateClient()

    with client.connect():
        # Create state
        state = AgentState(query="driftwood mirror")

        # Test 1: list_categories
        print("\n1️⃣  Testing list_categories")
        print("-" * 60)
        list_tool = ListCategoriesTool(weaviate_client=client)
        result = list_tool.execute(state)

        print(f"Success: {result.success}")

        if not result.success:
            print(f"❌ Error: {result.error}")
            print("This likely means the collection doesn't exist or is empty.")
            print("\nRun: poetry run python scripts/load_weaviate.py --reset")
            return

        print(f"Total categories: {result.metadata.get('total_categories', 0)}")
        print("\nFirst 10 categories:")
        for i, cat in enumerate(result.data[:10], 1):
            print(f"  {i}. {cat}")

        print("\nLooking for pipe characters in categories...")
        pipe_cats = [cat for cat in result.data if "|" in cat.get("product_class", "")]
        if pipe_cats:
            print(f"  ❌ Found {len(pipe_cats)} categories with pipes:")
            for cat in pipe_cats[:5]:
                print(f"     - {cat}")
        else:
            print("  ✅ No pipe characters found")

        # Test 2: search_products
        print("\n\n2️⃣  Testing search_products")
        print("-" * 60)
        search_tool = SearchProductsTool(weaviate_client=client)
        result = search_tool.execute(state, query="driftwood mirror", limit=50)

        print(f"Success: {result.success}")

        if result.success:
            print(f"Result count: {result.metadata.get('result_count', 0)}")
            print(f"Message: {result.metadata.get('message', '')}")
            print("\nFirst 5 products:")
            for i, prod in enumerate(result.data[:5], 1):
                print(f"  {i}. {prod['product_name'][:50]}")
                print(f"     product_class: {prod.get('product_class', 'N/A')}")
        else:
            print(f"❌ Error: {result.error}")
            print("This likely means the collection doesn't exist or is empty.")
            print("Run: poetry run python scripts/load_weaviate.py --reset")
            return

        # Test 3: Check specific category
        category_to_check = input("\n\n3️⃣  Enter a category name to check (or press Enter to skip): ").strip()
        if category_to_check:
            print(f"\nChecking if '{category_to_check}' exists...")
            available_names = {c["product_class"] for c in list_tool.execute(state).data}
            if category_to_check in available_names:
                print("  ✅ Category exists!")
            else:
                print("  ❌ Category NOT found")
                print("\n  Similar categories:")
                similar = [c for c in available_names if category_to_check.lower() in c.lower()]
                for s in similar[:10]:
                    print(f"     - {s}")

        # Test 4: Export all categories to file
        print("\n\n4️⃣  Exporting all categories to file...")
        all_cats = list_tool.execute(state)
        output_file = "/tmp/all_categories.json"
        with open(output_file, "w") as f:
            json.dump(all_cats.data, f, indent=2)
        print(f"  ✅ Saved {len(all_cats.data)} categories to {output_file}")

        print("\n" + "=" * 60)
        print("✅ Testing complete!")


if __name__ == "__main__":
    main()
