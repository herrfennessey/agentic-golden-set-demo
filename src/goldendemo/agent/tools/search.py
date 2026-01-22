"""Search products tool."""

import logging
from typing import TYPE_CHECKING, Any

from goldendemo.agent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState

logger = logging.getLogger(__name__)


class SearchProductsTool(BaseTool):
    """Search for products using hybrid vector + keyword search.

    Discovery phase tool - returns slim data for planning, NO judgment.
    Judgment happens later during execution phase via search steps in agent.py.
    """

    def __init__(self, weaviate_client: Any):
        """Initialize search products tool.

        Args:
            weaviate_client: Weaviate client instance.
        """
        super().__init__(weaviate_client)

    @property
    def name(self) -> str:
        return "search_products"

    @property
    def description(self) -> str:
        return (
            "Search for products using natural language query. Returns slim data for planning. "
            "Uses hybrid search combining semantic (vector) and keyword (BM25) matching. "
            "Use this for discovery/exploration - NO judgment happens here. "
            "To get judgments, include the query as a search step in your plan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (e.g., 'blue velvet sofa', 'wooden coffee table')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 200)",
                    "default": 200,
                    "minimum": 1,
                    "maximum": 500,
                },
                "alpha": {
                    "type": "number",
                    "description": (
                        "Balance between vector (1.0) and keyword (0.0) search. "
                        "Default 0.5 for balanced hybrid search. "
                        "Use higher values for semantic similarity, lower for exact keyword matching."
                    ),
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["query"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute product search for discovery/planning.

        Returns slim data for planning purposes - NO judgment happens here.
        To get judgments, include the query as a search step in your plan.

        Args:
            state: Current agent state.
            query: Search query string.
            limit: Max results (default 200).
            alpha: Vector/keyword balance (default 0.5).

        Returns:
            ToolResult with slim product data and category summary for planning.
        """
        query = kwargs.get("query", "")
        limit = min(kwargs.get("limit", 200), 500)
        alpha = kwargs.get("alpha", 0.5)

        if not query:
            return ToolResult.fail("Query string is required")

        # Check for duplicate query (discovery phase only)
        normalized_query = query.lower().strip()
        existing_queries = {s.query.lower().strip() for s in state.search_history}
        if normalized_query in existing_queries:
            return ToolResult.fail(
                f"You already searched for '{query}'. Duplicate searches don't count toward exploration. "
                f"Try a different query: synonyms, related terms, or different product attributes. "
                f"Previous searches: {[s.query for s in state.search_history]}"
            )

        try:
            # Perform the search
            results = self.weaviate_client.hybrid_search(query=query, limit=limit, alpha=alpha)

            # Update state (tracking only, no judgment)
            state.add_seen_products(results)
            state.record_search(query, len(results), alpha)
            state.record_tool_call(self.name)

            # Convert to SLIM serializable format (for planning - no full descriptions)
            products_data = [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "product_class": p.product_class,
                    "description": p.product_description[:200] if p.product_description else "",
                }
                for p in results
            ]

            # Aggregate categories found in results (for planning)
            category_counts: dict[str, int] = {}
            for p in results:
                if p.product_class:
                    for cls in p.product_class:
                        if cls:
                            category_counts[cls] = category_counts.get(cls, 0) + 1

            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Build message (no judgment info - discovery only)
            message = (
                f"Found {len(results)} products across {len(category_counts)} unique categories. "
                f"Top categories: {', '.join([f'{cat} ({count})' for cat, count in top_categories[:5]])}. "
                f"NOTE: This is discovery data only - no judgments made. "
                f"To judge these products, add this query as a search step in your plan."
            )

            return ToolResult.ok(
                products_data,
                query=query,
                result_count=len(results),
                alpha=alpha,
                top_categories=top_categories,
                message=message,
            )

        except Exception as e:
            return ToolResult.fail(f"Search failed: {e}")
