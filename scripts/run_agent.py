#!/usr/bin/env python3
"""Run the golden set agent on a query."""

import argparse

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
    args = parser.parse_args()

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
                        if tool_name == "search_products":
                            count = result.get("result_count", len(result.get("data", [])))
                            print(f"     ✅ Found {count} products")
                        elif tool_name == "browse_category":
                            count = result.get("result_count", len(result.get("data", [])))
                            total = result.get("total_in_category")
                            if total:
                                print(f"     ✅ Retrieved {count} of {total} products in category")
                            else:
                                print(f"     ✅ Found {count} products in category")
                        elif tool_name == "list_categories":
                            count = result.get("total_categories", len(result.get("data", [])))
                            print(f"     ✅ Found {count} categories")
                        elif tool_name == "get_product_details":
                            found = result.get("found_count", len(result.get("data", [])))
                            requested = result.get("requested_count", found)
                            print(f"     ✅ Retrieved {found}/{requested} product details")
                        elif tool_name == "submit_judgments":
                            data = result.get("data", {})
                            added = data.get("added", 0)
                            total = data.get("total", 0)
                            print(f"     ✅ Added {added} judgments (total: {total})")
                        elif tool_name == "finish_judgments":
                            count = result.get("data", {}).get("judgments_count", 0)
                            print(f"     ✅ Finished with {count} judgments")
                        elif tool_name == "submit_plan":
                            data = result.get("data", {})
                            steps = data.get("total_steps", 0)
                            print(f"     ✅ Plan submitted with {steps} categories")
                        elif tool_name == "complete_step":
                            data = result.get("data", {})
                            completed = data.get("completed_step", "")
                            progress = data.get("progress", "")
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
                    print(f"  ⚠️  {event.data['message']}")
                elif event.type.value == "guardrail_failure":
                    print(f"  🚫 {event.data['message']}")
                elif event.type.value == "completed":
                    print("-" * 50)
                    print(f"✅ Completed: {event.data['judgments_count']} judgments")
                    if event.data.get("warnings"):
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
