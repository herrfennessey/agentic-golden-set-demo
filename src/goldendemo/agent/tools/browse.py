"""Browse category tools."""

import unicodedata
from typing import TYPE_CHECKING, Any

from goldendemo.agent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


def _normalize_text(text: str) -> str:
    """Normalize text for comparison - handle Unicode issues from LLM output."""
    # Remove null bytes and control characters
    text = "".join(c for c in text if ord(c) >= 32 or c in "\t\n")
    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def _find_matching_category(query: str, categories: list) -> str | None:
    """Find category matching the query, handling Unicode normalization."""
    query_normalized = _normalize_text(query).lower()

    # Try exact match first (normalized)
    for cat in categories:
        if _normalize_text(cat.product_class).lower() == query_normalized:
            return str(cat.product_class)

    # Try ASCII-only comparison (strip accents)
    query_ascii = unicodedata.normalize("NFKD", query_normalized).encode("ascii", "ignore").decode()
    for cat in categories:
        cat_ascii = (
            unicodedata.normalize("NFKD", _normalize_text(cat.product_class).lower()).encode("ascii", "ignore").decode()
        )
        if cat_ascii == query_ascii:
            return str(cat.product_class)

    return None


def _parse_product_features(features: str) -> dict[str, list[str]]:
    """Parse pipe-delimited features, grouping empty keys with previous key.

    Handles multi-value attributes like:
        dswoodtone : red wood| : barn brown| : cottonwood
    Which becomes:
        {"dswoodtone": ["red wood", "barn brown", "cottonwood"]}
    """
    result: dict[str, list[str]] = {}
    current_key: str | None = None

    if not features:
        return result

    for pair in features.split("|"):
        if ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key, value = key.strip(), value.strip()

        if key:  # New key
            current_key = key
            if key not in result:
                result[key] = []
            if value:
                result[key].append(value)
        elif current_key and value:  # Empty key = append to previous
            result[current_key].append(value)

    return result


class ListCategoriesTool(BaseTool):
    """List all available product categories."""

    @property
    def name(self) -> str:
        return "list_categories"

    @property
    def description(self) -> str:
        return (
            "List all available product classes/categories with their product counts. "
            "Use this to understand the catalog structure and find relevant categories to explore."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute category listing.

        Returns:
            ToolResult with list of CategoryInfo objects.
        """
        try:
            categories = self.weaviate_client.get_all_classes()
            state.record_tool_call(self.name)

            # Cache in state for later reference
            state.available_classes = categories

            # Convert to serializable format (exclude count to prevent LLM gaming)
            categories_data = [
                {
                    "product_class": c.product_class,
                    "category_hierarchy": c.category_hierarchy,
                }
                for c in categories
            ]

            return ToolResult.ok(
                categories_data,
                total_categories=len(categories),
            )

        except Exception as e:
            return ToolResult.fail(f"Failed to list categories: {e}")


class BrowseCategoryTool(BaseTool):
    """Browse all products in a specific category."""

    @property
    def name(self) -> str:
        return "browse_category"

    @property
    def description(self) -> str:
        return (
            "Browse products in a category. Returns up to 100 products per call. "
            "Use offset to paginate through large categories. "
            "IMPORTANT: Only use category names from list_categories, NOT product names from search results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_class": {
                    "type": "string",
                    "description": "The exact category name from list_categories (e.g., 'Sofas', 'Coffee Tables')",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting position for pagination (default 0). Use to get more results.",
                },
            },
            "required": ["product_class"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute category browsing.

        Args:
            state: Current agent state.
            product_class: The product class to browse.
            offset: Starting position for pagination.

        Returns:
            ToolResult with list of ProductSummary objects.
        """
        product_class_input = kwargs.get("product_class", "")
        offset = kwargs.get("offset", 0) or 0
        page_size = 100  # Fixed page size to prevent context overflow

        if not product_class_input:
            return ToolResult.fail("product_class is required")

        # Try to match category name (handles Unicode issues from LLM)
        product_class = product_class_input

        if state.available_classes:
            matched = _find_matching_category(product_class_input, state.available_classes)
            if matched:
                product_class = matched

        try:
            # Fetch more than needed to handle offset (Weaviate doesn't support offset directly)
            all_results = self.weaviate_client.get_by_class(
                product_class=product_class,
                limit=2000,  # Get all, then slice
            )

            # Apply pagination
            results = all_results[offset : offset + page_size]
            has_more = (offset + page_size) < len(all_results)

            # Update state
            state.add_seen_products(results)
            state.record_category_browse(product_class)
            state.record_tool_call(self.name)

            # Convert to serializable format with full description and parsed attributes
            products_data = [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.product_class,
                    "category_hierarchy": p.category_hierarchy,
                    "description": p.product_description or "",
                    "attributes": _parse_product_features(p.product_features),
                }
                for p in results
            ]

            return ToolResult.ok(
                products_data,
                product_class=product_class,
                result_count=len(results),
                total_in_category=len(all_results),
                offset=offset,
                has_more=has_more,
                next_offset=offset + page_size if has_more else None,
                message=f"Retrieved {len(results)} products (offset {offset}, {len(all_results)} total in '{product_class}')"
                + (f" - use offset={offset + page_size} for more" if has_more else " - no more results"),
            )

        except Exception as e:
            return ToolResult.fail(f"Failed to browse category: {e}")
