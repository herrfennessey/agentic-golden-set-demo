"""Browse category tools."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from goldendemo.agent.tools.base import BaseTool, ToolResult
from goldendemo.agent.utils import find_matching_category, parse_product_features
from goldendemo.config import settings

if TYPE_CHECKING:
    from goldendemo.agent.judge import JudgmentSubagent
    from goldendemo.agent.state import AgentState

logger = logging.getLogger(__name__)


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

            # Convert to serializable format
            # Note: Since product_class is now stored as an array in Weaviate,
            # each category is returned as a clean individual value (no pipes)
            categories_data = [
                {
                    "product_class": c.product_class,
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

    def __init__(self, weaviate_client: Any, judgment_subagent: "JudgmentSubagent | None" = None):
        """Initialize browse category tool.

        Args:
            weaviate_client: Weaviate client instance.
            judgment_subagent: Optional judgment subagent for automatic evaluation.
        """
        super().__init__(weaviate_client)
        self.judgment_subagent = judgment_subagent

    @property
    def name(self) -> str:
        return "browse_category"

    @property
    def description(self) -> str:
        return (
            "Browse ALL products in a category and automatically judge their relevance. "
            "Processes the entire category in parallel for speed. "
            "Returns a summary of all judgments added. "
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
            },
            "required": ["product_class"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute category browsing with parallel judgment.

        Args:
            state: Current agent state.
            product_class: The product class to browse.

        Returns:
            ToolResult with judgment summary for entire category.
        """
        product_class_input = kwargs.get("product_class", "")

        if not product_class_input:
            return ToolResult.fail("product_class is required")

        # Try to match category name (handles Unicode issues from LLM)
        product_class = product_class_input

        if state.available_classes:
            matched = find_matching_category(product_class_input, state.available_classes)
            if matched:
                product_class = matched

        # Check if this category was already browsed (step auto-completes now)
        if product_class in state.browsed_categories:
            return ToolResult.ok(
                {"already_browsed": True},
                product_class=product_class,
                message=f"Category '{product_class}' was already browsed and auto-completed. "
                f"Move to the next step or call finish_judgments().",
            )

        try:
            # Fetch ALL products in category at once
            logger.info(f"Fetching all products from category '{product_class}'")
            all_results = self.weaviate_client.get_by_class(
                product_class=product_class,
                limit=settings.browse_product_limit,
            )

            total_products = len(all_results)
            logger.info(f"Retrieved {total_products} total products from '{product_class}'")

            if total_products == 0:
                return ToolResult.ok(
                    {"judgments_added": 0},
                    product_class=product_class,
                    result_count=0,
                    total_in_category=0,
                    message=f"No products found in category '{product_class}'.",
                )

            # Update state
            state.add_seen_products(all_results)
            state.record_category_browse(product_class)
            state.record_tool_call(self.name)
            # Update step progress for plan tracking
            state.update_step_progress(products_count=total_products)

            # Call judgment subagent to automatically judge products
            if not self.judgment_subagent:
                return ToolResult.fail("BrowseCategoryTool requires judgment_subagent but it was not provided")

            # Convert to serializable format
            all_products_data = [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.product_class,
                    "category_hierarchy": p.category_hierarchy,
                    "description": p.product_description or "",
                    "attributes": parse_product_features(p.product_features),
                }
                for p in all_results
            ]

            # Split into chunks and judge in parallel
            chunk_size = settings.judge_chunk_size
            chunks = [all_products_data[i : i + chunk_size] for i in range(0, len(all_products_data), chunk_size)]

            logger.info(f"Judging {total_products} products in {len(chunks)} parallel chunks")

            total_judgments_added = 0
            total_hallucinated = 0
            batch_exact = 0
            batch_partial = 0

            # Judge all chunks in parallel
            with ThreadPoolExecutor(max_workers=settings.judge_max_workers) as executor:
                # Submit all judgment tasks
                futures = [
                    executor.submit(self._judge_chunk, state, state.query, chunk, idx + 1, len(chunks))
                    for idx, chunk in enumerate(chunks)
                ]

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        valid_judgments, hallucinated_count = future.result()
                        # Count Exact (2) and Partial (1) in this batch before adding
                        for j in valid_judgments:
                            if j.get("relevance") == 2:
                                batch_exact += 1
                            elif j.get("relevance") == 1:
                                batch_partial += 1
                        judgments_added = state.add_judgments_from_dicts(valid_judgments)
                        total_judgments_added += judgments_added
                        total_hallucinated += hallucinated_count
                    except Exception as e:
                        logger.error(f"Chunk judgment failed: {e}", exc_info=True)
                        return ToolResult.fail(f"Judgment subagent error: {e}")

            if total_hallucinated > 0:
                logger.warning(f"Total hallucinated judgments across all chunks: {total_hallucinated}")

            # Update step with judgments count
            state.update_step_progress(products_count=0, judgments_count=total_judgments_added)

            # Auto-complete the step
            summary = (
                f"Browsed {total_products} products, "
                f"added {total_judgments_added} judgments (Exact: {batch_exact}, Partial: {batch_partial})"
            )
            has_more_steps = state.complete_current_step(summary)

            # Determine next action message
            if has_more_steps:
                next_step = state.get_current_step()
                if next_step:
                    next_action = f"Next: {next_step.step_type.value} '{next_step.target}'"
                else:
                    next_action = "Call finish_judgments() to finalize."
            else:
                next_action = "All steps complete! Call finish_judgments() to finalize."

            # Return summary WITHOUT product data to keep context small
            return ToolResult.ok(
                {
                    "judgments_added": total_judgments_added,
                    "exact_count": batch_exact,
                    "partial_count": batch_partial,
                    "step_completed": True,
                },
                product_class=product_class,
                result_count=total_products,
                total_in_category=total_products,
                message=f"Browsed all {total_products} products in '{product_class}'. "
                f"Added {total_judgments_added} judgments "
                f"(Exact: {batch_exact}, Partial: {batch_partial}). "
                f"Step auto-completed. {next_action}",
            )

        except Exception as e:
            return ToolResult.fail(f"Failed to browse category: {e}")

    def _judge_chunk(
        self, state: "AgentState", query: str, products_data: list[dict], chunk_num: int, total_chunks: int
    ) -> tuple[list[dict], int]:
        """Judge a single chunk of products (called in parallel).

        Args:
            state: Agent state (for token tracking).
            query: Search query.
            products_data: Products to judge (up to 100).
            chunk_num: Which chunk this is (for logging).
            total_chunks: Total number of chunks.

        Returns:
            Tuple of (valid_judgments, hallucinated_count).
        """
        logger.info(f"[Chunk {chunk_num}/{total_chunks}] Judging {len(products_data)} products")

        # Call judgment subagent (pass state for token tracking)
        # Note: judgment_subagent is checked in execute() before _judge_chunk is called
        assert self.judgment_subagent is not None
        judgments = self.judgment_subagent.judge_products(query, products_data, state)

        # Validate judgments
        valid_product_ids = {p["product_id"] for p in products_data}
        valid_judgments = []
        hallucinated_ids = []

        for j in judgments:
            if j["product_id"] in valid_product_ids:
                valid_judgments.append(j)
            else:
                hallucinated_ids.append(j["product_id"])

        if hallucinated_ids:
            logger.error(
                f"[Chunk {chunk_num}/{total_chunks}] Hallucinated {len(hallucinated_ids)}/{len(judgments)} IDs: "
                f"{hallucinated_ids[:5]}..."
            )

        logger.info(f"[Chunk {chunk_num}/{total_chunks}] Added {len(valid_judgments)} valid judgments")

        return valid_judgments, len(hallucinated_ids)
