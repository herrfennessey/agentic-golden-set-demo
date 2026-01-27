#!/usr/bin/env python3
"""Run the golden set agent on a query."""

import argparse
import logging

from goldendemo.agent.agent import GoldenSetAgent
from goldendemo.clients.weaviate_client import WeaviateClient


def main():
    parser = argparse.ArgumentParser(description="Run the golden set agent")
    parser.add_argument("query", help="Search query to generate golden set for")
    parser.add_argument("--max-iterations", type=int, help="Max iterations (default: from .env)")
    parser.add_argument("--model", help="Model to use (default: from .env)")
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], help="Reasoning effort level (default: from .env)"
    )
    parser.add_argument(
        "--no-reasoning-summary", action="store_true", help="Disable reasoning summaries (for unverified OpenAI orgs)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s [%(name)s] %(message)s",
        )
    elif args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s [%(name)s] %(message)s",
        )
    else:
        # Only show warnings and errors
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    print(f'🔍 Generating golden set for: "{args.query}"')
    print("-" * 50)

    # Connect to Weaviate using context manager
    client = WeaviateClient()

    with client.connect():
        # Build agent kwargs - only pass if explicitly set
        agent_kwargs = {"weaviate_client": client}
        if args.max_iterations is not None:
            agent_kwargs["max_iterations"] = args.max_iterations
        if args.model is not None:
            agent_kwargs["model"] = args.model
        if args.reasoning_effort is not None:
            agent_kwargs["reasoning_effort"] = args.reasoning_effort
        if args.no_reasoning_summary:
            agent_kwargs["reasoning_summary"] = False

        agent = GoldenSetAgent(**agent_kwargs)

        # Run with streaming to see progress
        for event in agent.run_streaming(args.query):
            if args.verbose:
                print(f"[{event.type.value}] {event.data}")
            else:
                # Show key events only
                if event.type.value == "iteration_start":
                    print(f"\n📍 Iteration {event.data['iteration']}/{event.data['max_iterations']}")
                elif event.type.value == "phase_change":
                    from_phase = event.data.get("from_phase", "")
                    to_phase = event.data.get("to_phase", "")
                    plan_steps = event.data.get("plan_steps", 0)
                    print(f"\n{'=' * 50}")
                    print(f"Phase: {from_phase} -> {to_phase}")
                    if plan_steps:
                        print(f"Plan submitted with {plan_steps} categories to explore")
                    print(f"{'=' * 50}")
                elif event.type.value == "step_completed":
                    category = event.data.get("category", "")
                    summary = event.data.get("summary", "")
                    has_more = event.data.get("has_more_steps", False)
                    print(f"\n  ✓ Completed: {category}")
                    if summary:
                        print(f"    {summary}")
                    if not has_more:
                        print("    All plan steps complete!")
                elif event.type.value == "tool_call":
                    print(f"  🔧 {event.data['tool_name']}({_format_args(event.data.get('arguments', {}))})")
                elif event.type.value == "tool_result":
                    result = event.data.get("result", {})
                    tool_name = event.data.get("tool_name", "")
                    if event.data.get("success"):
                        # Show useful summary based on tool type
                        # Note: ToolResult.to_dict() returns {success, data, error, metadata}
                        data = result.get("data", {})
                        metadata = result.get("metadata", {})
                        if tool_name == "search_products":
                            count = metadata.get("result_count", len(data) if isinstance(data, list) else 0)
                            print(f"     ✅ Found {count} products")
                        elif tool_name == "browse_category":
                            # Check if already browsed (re-browsing won't help)
                            if isinstance(data, dict) and data.get("already_browsed"):
                                category = metadata.get("product_class", "category")
                                print(f"     ⏭️  Already browsed '{category}' - skipped (products already judged)")
                            else:
                                judgments_added = data.get("judgments_added", 0) if isinstance(data, dict) else 0
                                exact_count = data.get("exact_count", 0) if isinstance(data, dict) else 0
                                partial_count = data.get("partial_count", 0) if isinstance(data, dict) else 0
                                total = metadata.get("total_in_category", 0)
                                print(
                                    f"     ✅ Judged all {total} products, added {judgments_added} (Exact: {exact_count}, Partial: {partial_count})"
                                )
                        elif tool_name == "list_categories":
                            count = metadata.get("total_categories", len(data) if isinstance(data, list) else 0)
                            print(f"     ✅ Found {count} categories")
                        elif tool_name == "get_product_details":
                            found = metadata.get("found_count", len(data) if isinstance(data, list) else 0)
                            requested = metadata.get("requested_count", found)
                            print(f"     ✅ Retrieved {found}/{requested} product details")
                        elif tool_name == "submit_judgments":
                            added = data.get("added", 0) if isinstance(data, dict) else 0
                            total_j = data.get("total", 0) if isinstance(data, dict) else 0
                            print(f"     ✅ Added {added} judgments (total: {total_j})")
                        elif tool_name == "finish_judgments":
                            count = data.get("judgments_count", 0) if isinstance(data, dict) else 0
                            print(f"     ✅ Finished with {count} judgments")
                        elif tool_name == "submit_plan":
                            search_steps = data.get("search_steps", 0) if isinstance(data, dict) else 0
                            category_steps = data.get("category_steps", 0) if isinstance(data, dict) else 0
                            print(
                                f"     ✅ Plan submitted: 🔍 {search_steps} searches + 📂 {category_steps} categories"
                            )
                        elif tool_name == "search_step":
                            query = metadata.get("query", "")
                            products_found = metadata.get("products_found", 0)
                            judgments_added = data.get("judgments_added", 0) if isinstance(data, dict) else 0
                            exact_count = data.get("exact_count", 0) if isinstance(data, dict) else 0
                            partial_count = data.get("partial_count", 0) if isinstance(data, dict) else 0
                            print(
                                f"     ✅ Searched '{query}': {products_found} products, added {judgments_added} (Exact: {exact_count}, Partial: {partial_count})"
                            )
                        elif tool_name == "complete_step":
                            completed = data.get("completed_step", "") if isinstance(data, dict) else ""
                            progress = data.get("progress", "") if isinstance(data, dict) else ""
                            print(f"     ✅ Completed '{completed}' ({progress})")
                        else:
                            print("     ✅ Success")
                    else:
                        error = event.data.get("error", result.get("error", "Unknown"))
                        print(f"     ❌ {error}")
                elif event.type.value == "reasoning":
                    summary = event.data.get("summary", "")
                    if summary:
                        print(f"  💭 {summary}")
                elif event.type.value == "guardrail_warning":
                    message = event.data.get("message", event.data.get("warning", "Warning"))
                    print(f"  ⚠️  {message}")
                elif event.type.value == "guardrail_failure":
                    message = event.data.get("message", event.data.get("error", "Failure"))
                    print(f"  🚫 {message}")
                elif event.type.value == "validation_phase":
                    print("\n📋 Validation Phase")
                    total_reviewed = event.data.get("total_reviewed", 0)
                    total_removed = event.data.get("total_removed", 0)
                    exact_removed = event.data.get("exact_removed", 0)
                    partial_removed = event.data.get("partial_removed", 0)
                    total_adjusted = event.data.get("total_adjusted", 0)
                    upgrades = event.data.get("upgrades", 0)
                    downgrades = event.data.get("downgrades", 0)
                    exact_remaining = event.data.get("exact_remaining", 0)
                    partial_remaining = event.data.get("partial_remaining", 0)

                    print(f"  🔍 Reviewed {total_reviewed} judgments")

                    # Show adjustments
                    if total_adjusted > 0:
                        print(f"  🔄 Adjusted {total_adjusted} (↑{upgrades} upgrades, ↓{downgrades} downgrades)")
                        adjustment_details = event.data.get("adjustment_details", [])
                        details_to_show = adjustment_details[:5] if len(adjustment_details) > 5 else adjustment_details
                        for adj in details_to_show:
                            old_rel = adj.get("old_relevance", 0)
                            new_rel = adj.get("new_relevance", 0)
                            old_label = "Exact" if old_rel == 2 else "Partial"
                            new_label = "Exact" if new_rel == 2 else "Partial"
                            reason = adj.get("reason", "")[:40]
                            print(f"     • {adj.get('product_id', '?')}: {old_label}→{new_label} ({reason})")
                        if len(adjustment_details) > 5:
                            print(f"     • ... and {len(adjustment_details) - 5} more")

                    # Show removals
                    if total_removed > 0:
                        print(f"  ❌ Removed {total_removed} (Exact: {exact_removed}, Partial: {partial_removed})")
                        removal_details = event.data.get("removal_details", [])
                        details_to_show = removal_details[:5] if len(removal_details) > 5 else removal_details
                        for removal in details_to_show:
                            relevance = removal.get("relevance", 0)
                            label = "Exact" if relevance == 2 else "Partial"
                            reason = removal.get("reason", "Unknown")[:40]
                            print(f"     • {removal.get('product_id', '?')} ({label}): {reason}")
                        if len(removal_details) > 5:
                            print(f"     • ... and {len(removal_details) - 5} more")

                    if total_adjusted == 0 and total_removed == 0:
                        print("  ✅ All judgments verified correct")

                    print(f"  📊 Final: {exact_remaining} Exact, {partial_remaining} Partial")

                elif event.type.value == "completed":
                    print("-" * 50)
                    print(f"✅ Completed: {event.data['judgments_count']} judgments")

                    # Display token usage if available
                    if token_usage := event.data.get("token_usage"):
                        print("\n💰 Token Usage:")
                        print(f"   Input tokens:     {token_usage['input_tokens']:,}")
                        print(f"   Output tokens:    {token_usage['output_tokens']:,}")
                        if token_usage.get("reasoning_tokens", 0) > 0:
                            print(f"   Reasoning tokens: {token_usage['reasoning_tokens']:,}")
                        if token_usage.get("cached_tokens", 0) > 0:
                            print(f"   Cached tokens:    {token_usage['cached_tokens']:,}")
                        print(f"   Total tokens:     {token_usage['total_tokens']:,}")

                    if event.data.get("warnings"):
                        print()  # Add blank line before warnings
                        for w in event.data["warnings"]:
                            print(f"   ⚠️  {w}")
                elif event.type.value == "error":
                    print(f"❌ Error: {event.data['error']}")


def _format_args(args: dict) -> str:
    """Format arguments for display."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 30:
            v = v[:30] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
