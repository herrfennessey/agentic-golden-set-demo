"""Get product details tool."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class GetProductDetailsTool(BaseTool):
    """Get full product details for specific products."""

    @property
    def name(self) -> str:
        return "get_product_details"

    @property
    def description(self) -> str:
        return (
            "Get full details for specific products by their IDs. "
            "Returns complete product information including description and features. "
            "Use this before making final judgments to ensure you have all relevant information."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of product IDs to retrieve (max 50)",
                    "maxItems": 50,
                },
            },
            "required": ["product_ids"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute product details retrieval.

        Args:
            state: Current agent state.
            product_ids: List of product IDs.

        Returns:
            ToolResult with list of full Product objects.
        """
        product_ids = kwargs.get("product_ids", [])

        if not product_ids:
            return ToolResult.fail("product_ids list is required")

        if len(product_ids) > 50:
            product_ids = product_ids[:50]

        try:
            products = self.weaviate_client.get_by_ids(product_ids, limit=50)

            # Update state
            state.exploration_metrics.product_details_retrieved += len(products)
            state.record_tool_call(self.name)

            # Cache full products for reference during judgment
            for product in products:
                state.candidate_products[product.product_id] = product

            # Convert to serializable format with full details
            products_data = [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "product_class": p.product_class,
                    "category_hierarchy": p.category_hierarchy,
                    "product_description": p.product_description,
                    "product_features": p.product_features,
                    "average_rating": p.average_rating,
                    "rating_count": p.rating_count,
                    "review_count": p.review_count,
                }
                for p in products
            ]

            return ToolResult.ok(
                products_data,
                requested_count=len(product_ids),
                found_count=len(products),
            )

        except Exception as e:
            return ToolResult.fail(f"Failed to get product details: {e}")
