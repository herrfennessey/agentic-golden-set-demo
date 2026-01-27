"""Finish judgments tool - signals completion and runs final validation."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails import (
    CategoryBrowsingGuardrail,
    GuardrailResult,
    IterationBudgetGuardrail,
    MinimumExplorationGuardrail,
    ScoreDistributionGuardrail,
)
from goldendemo.agent.tools.base import BaseTool, ToolResult
from goldendemo.config import settings
from goldendemo.data.models import AgentJudgment, GoldenSetConfig, ProductRelevance

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState
    from goldendemo.agent.validator import ValidationSubagent
    from goldendemo.clients.weaviate_client import WeaviateClient

logger = logging.getLogger(__name__)


class FinishJudgmentsTool(BaseTool):
    """Signal completion of judgment collection and trigger final validation."""

    def __init__(
        self,
        weaviate_client: "WeaviateClient",
        guardrails: dict[str, Any] | None = None,
        validator: "ValidationSubagent | None" = None,
    ):
        """Initialize with Weaviate client, guardrails, and optional validator.

        Args:
            weaviate_client: Connected Weaviate client.
            guardrails: Optional dict of guardrails to run on finish.
            validator: Optional ValidationSubagent for reviewing judgments.
        """
        super().__init__(weaviate_client)
        self.guardrails = guardrails or self._default_guardrails()
        self.validator = validator

    def _default_guardrails(self) -> dict[str, Any]:
        """Create default guardrails."""
        return {
            "iteration_budget": IterationBudgetGuardrail(),
            "minimum_exploration": MinimumExplorationGuardrail(),
            "score_distribution": ScoreDistributionGuardrail(
                min_exact=settings.min_exact_judgments,
                min_total=settings.min_total_judgments,
            ),
            "category_browsing": CategoryBrowsingGuardrail(),
        }

    @property
    def name(self) -> str:
        return "finish_judgments"

    @property
    def description(self) -> str:
        return (
            "Signal that you have finished exploring all relevant categories and are done collecting judgments. "
            "Call this when you have thoroughly explored all relevant categories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "overall_reasoning": {
                    "type": "string",
                    "description": "Brief summary of your exploration approach and key findings",
                },
            },
            "required": ["overall_reasoning"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute finish - runs guardrails, validation, and saves golden set.

        Args:
            state: Current agent state with accumulated judgments.
            overall_reasoning: Summary of exploration approach.

        Returns:
            ToolResult indicating success or guardrail failures.
        """
        overall_reasoning = kwargs.get("overall_reasoning", "")
        state.final_reasoning = overall_reasoning
        state.record_tool_call(self.name)

        # Run initial guardrails on judgments
        guardrail_failures, guardrail_warnings = self._run_guardrails(state)

        # If any blocking failures, reject finish
        if guardrail_failures:
            failure_details = "; ".join(guardrail_failures)
            # Build helpful context about what's been explored
            explored_categories = list(state.browsed_categories)[:5]
            explored_searches = [s.query for s in state.search_history][:5]
            return ToolResult(
                success=False,
                error=f"Cannot finish: {failure_details}",
                data={
                    "guardrail_failures": guardrail_failures,
                    "already_explored": {
                        "categories": explored_categories,
                        "searches": explored_searches,
                    },
                    "message": "Submit a new plan with DIFFERENT search queries and UNEXPLORED categories. "
                    "Re-browsing the same categories won't help since those products are already judged.",
                },
            )

        # Capture pre-validation counts
        pre_validation_counts = state.judgment_counts_by_level()
        pre_exact = pre_validation_counts.get(2, 0)
        pre_partial = pre_validation_counts.get(1, 0)

        # Run validation phase if validator is configured
        validation_result: dict[str, Any] = {
            "total_reviewed": 0,
            "total_removed": 0,
            "exact_removed": 0,
            "partial_removed": 0,
            "removal_details": [],
        }

        if self.validator and state.judgments:
            try:
                validation_result = self._run_validation(state)
                if validation_result["total_removed"] > 0:
                    logger.info(
                        f"Validation removed {validation_result['total_removed']} products "
                        f"(Exact: {validation_result['exact_removed']}, Partial: {validation_result['partial_removed']})"
                    )

                    # Re-run guardrails after removals
                    guardrail_failures, post_validation_warnings = self._run_guardrails(state)
                    guardrail_warnings.extend(post_validation_warnings)

                    if guardrail_failures:
                        failure_details = "; ".join(guardrail_failures)
                        return ToolResult(
                            success=False,
                            error=f"Cannot finish after validation removals: {failure_details}",
                            data={
                                "guardrail_failures": guardrail_failures,
                                "validation_result": validation_result,
                                "message": "Validation removed products and thresholds no longer met. Submit a new plan with additional search and category steps.",
                            },
                        )
            except Exception as e:
                logger.error(f"Validation failed: {e}", exc_info=True)
                # Continue without validation - better to save than fail completely
                guardrail_warnings.append(f"Validation skipped due to error: {e}")

        # Save the golden set (all judgments are Exact or Partial)
        try:
            self._save_golden_set(state, state.judgments, overall_reasoning)
        except Exception as e:
            return ToolResult.fail(f"Failed to save golden set: {e}")

        # Capture post-validation counts
        post_validation_counts = state.judgment_counts_by_level()
        post_exact = post_validation_counts.get(2, 0)
        post_partial = post_validation_counts.get(1, 0)

        # Add remaining counts to validation result
        validation_result["exact_remaining"] = post_exact
        validation_result["partial_remaining"] = post_partial
        validation_result["pre_exact"] = pre_exact
        validation_result["pre_partial"] = pre_partial

        return ToolResult.ok(
            {
                "status": "complete",
                "judgments_count": len(state.judgments),
                "by_level": post_validation_counts,
                "warnings": guardrail_warnings,
                "validation_result": validation_result,
            },
            message=f"Golden set saved with {len(state.judgments)} judgments (Exact: {post_exact}, Partial: {post_partial}).",
        )

    def _run_guardrails(self, state: "AgentState") -> tuple[list[str], list[str]]:
        """Run all guardrails on the current state.

        Returns:
            Tuple of (failures list, warnings list).
        """
        failures: list[str] = []
        warnings: list[str] = []

        for name, guardrail in self.guardrails.items():
            if name == "iteration_budget":
                result: GuardrailResult = guardrail.check(state, is_submission=True)
            else:
                result = guardrail.check(state, judgments=state.judgments)

            if not result.passed:
                failures.append(f"[{name}] {result.message}")
            elif result.warning:
                warnings.append(f"[{name}] {result.warning}")

        return failures, warnings

    def _run_validation(self, state: "AgentState") -> dict[str, Any]:
        """Run validation phase on all judgments.

        Fetches full product data and validates judgments in chunks.
        Can adjust relevance scores or remove products entirely.

        Args:
            state: Agent state with judgments to validate.

        Returns:
            Dict with validation results:
                - total_reviewed: Number of judgments reviewed
                - total_removed: Number of judgments removed
                - exact_removed: Number of Exact (2) judgments removed
                - partial_removed: Number of Partial (1) judgments removed
                - removal_details: List of {product_id, reason, relevance}
                - total_adjusted: Number of judgments with relevance changed
                - upgrades: Number upgraded (Partial→Exact)
                - downgrades: Number downgraded (Exact→Partial)
                - adjustment_details: List of {product_id, old_relevance, new_relevance, reason}
        """
        result: dict[str, Any] = {
            "total_reviewed": 0,
            "total_removed": 0,
            "exact_removed": 0,
            "partial_removed": 0,
            "removal_details": [],
            "total_adjusted": 0,
            "upgrades": 0,
            "downgrades": 0,
            "adjustment_details": [],
        }

        if not self.validator:
            return result

        # Build a lookup of judgment relevance levels for tracking
        judgment_relevance: dict[str, int] = {j.product_id: j.relevance for j in state.judgments}

        # Get all product IDs from judgments
        product_ids = [j.product_id for j in state.judgments]
        if not product_ids:
            return result

        result["total_reviewed"] = len(product_ids)
        logger.info(f"Validation phase: reviewing {len(product_ids)} judgments")

        # Fetch full product data for all judgments
        # Process in batches to avoid overwhelming Weaviate
        all_products: dict[str, Any] = {}
        batch_size = 100
        for i in range(0, len(product_ids), batch_size):
            batch_ids = product_ids[i : i + batch_size]
            products = self.weaviate_client.get_by_ids(batch_ids, limit=len(batch_ids))
            for p in products:
                all_products[p.product_id] = p

        # Build judgment-product pairs
        judgments_with_products: list[dict] = []
        for j in state.judgments:
            product = all_products.get(j.product_id)
            if product:
                judgments_with_products.append(
                    {
                        "judgment": {
                            "product_id": j.product_id,
                            "relevance": j.relevance,
                            "reasoning": j.reasoning,
                        },
                        "product": {
                            "product_id": product.product_id,
                            "product_name": product.product_name,
                            "product_class": product.product_class,
                            "product_description": product.product_description,
                            "product_features": product.product_features,
                        },
                    }
                )
            else:
                # Product not found in Weaviate - definitely remove
                logger.warning(f"Product {j.product_id} not found in Weaviate - will be removed")

        # Process validation in chunks (parallel)
        chunk_size = settings.validate_chunk_size
        chunks = [
            judgments_with_products[i : i + chunk_size] for i in range(0, len(judgments_with_products), chunk_size)
        ]

        logger.info(f"Validating {len(judgments_with_products)} judgments in {len(chunks)} chunks")

        all_removal_details: list[dict] = []
        all_adjustment_details: list[dict] = []

        with ThreadPoolExecutor(max_workers=settings.validate_max_workers) as executor:
            futures = [
                executor.submit(self._validate_chunk, state, chunk, idx + 1, len(chunks))
                for idx, chunk in enumerate(chunks)
            ]

            for future in as_completed(futures):
                try:
                    chunk_result = future.result()
                    all_removal_details.extend(chunk_result["removals"])
                    all_adjustment_details.extend(chunk_result["adjustments"])
                except Exception as e:
                    logger.error(f"Validation chunk failed: {e}", exc_info=True)

        # Also remove products that weren't found in Weaviate
        found_ids = set(all_products.keys())
        for j in state.judgments:
            if j.product_id not in found_ids:
                already_flagged = any(r["product_id"] == j.product_id for r in all_removal_details)
                if not already_flagged:
                    all_removal_details.append(
                        {
                            "product_id": j.product_id,
                            "reason": "Not found in Weaviate",
                            "relevance": j.relevance,
                        }
                    )
                    logger.info(f"Removing {j.product_id}: not found in Weaviate")

        # Apply adjustments to state
        upgrades = 0
        downgrades = 0
        for adj in all_adjustment_details:
            product_id = adj["product_id"]
            new_relevance = adj["new_relevance"]
            old_relevance = adj["old_relevance"]
            reason = adj.get("reason", "")

            if state.update_judgment(product_id, new_relevance, f"Validation: {reason}"):
                if new_relevance > old_relevance:
                    upgrades += 1
                else:
                    downgrades += 1

        result["total_adjusted"] = len(all_adjustment_details)
        result["upgrades"] = upgrades
        result["downgrades"] = downgrades
        result["adjustment_details"] = all_adjustment_details

        if all_adjustment_details:
            logger.info(
                f"Validation adjusted {len(all_adjustment_details)} judgments "
                f"(Upgrades: {upgrades}, Downgrades: {downgrades})"
            )

        # Remove flagged products from state and track exact vs partial
        if all_removal_details:
            removal_ids = [r["product_id"] for r in all_removal_details]
            removed_count = state.remove_judgments(removal_ids)

            # Count exact vs partial removals
            exact_removed = 0
            partial_removed = 0
            for removal in all_removal_details:
                relevance = removal.get("relevance") or judgment_relevance.get(removal["product_id"], 0)
                removal["relevance"] = relevance  # Ensure it's set
                if relevance == 2:
                    exact_removed += 1
                elif relevance == 1:
                    partial_removed += 1

            result["total_removed"] = removed_count
            result["exact_removed"] = exact_removed
            result["partial_removed"] = partial_removed
            result["removal_details"] = all_removal_details

            logger.info(
                f"Validation removed {removed_count} judgments (Exact: {exact_removed}, Partial: {partial_removed})"
            )

        return result

    def _validate_chunk(
        self,
        state: "AgentState",
        chunk: list[dict],
        chunk_num: int,
        total_chunks: int,
    ) -> dict[str, list[dict]]:
        """Validate a single chunk of judgments.

        Args:
            state: Agent state for token tracking.
            chunk: Judgments with product data to validate.
            chunk_num: Which chunk this is (for logging).
            total_chunks: Total number of chunks.

        Returns:
            Dict with:
                - removals: List of {product_id, reason, relevance}
                - adjustments: List of {product_id, old_relevance, new_relevance, reason}
        """
        logger.info(f"[Validation Chunk {chunk_num}/{total_chunks}] Reviewing {len(chunk)} judgments")

        # Build a lookup for relevance from the chunk
        chunk_relevance = {item["judgment"]["product_id"]: item["judgment"]["relevance"] for item in chunk}

        assert self.validator is not None
        result = self.validator.validate_judgments(state.query, chunk, state)

        removal_details = []
        for removal in result.get("remove", []):
            product_id = removal.get("product_id")
            reason = removal.get("reason", "No reason provided")
            if product_id:
                relevance = chunk_relevance.get(product_id, 0)
                removal_details.append(
                    {
                        "product_id": product_id,
                        "reason": reason,
                        "relevance": relevance,
                    }
                )
                relevance_label = "Exact" if relevance == 2 else "Partial"
                logger.info(f"[Validation] Removing {product_id} ({relevance_label}): {reason}")

        adjustment_details = []
        for adj in result.get("adjust", []):
            product_id = adj.get("product_id")
            new_relevance = adj.get("new_relevance")
            reason = adj.get("reason", "No reason provided")
            if product_id and new_relevance in (1, 2):
                old_relevance = chunk_relevance.get(product_id, 0)
                if old_relevance != new_relevance:  # Only if actually changing
                    adjustment_details.append(
                        {
                            "product_id": product_id,
                            "old_relevance": old_relevance,
                            "new_relevance": new_relevance,
                            "reason": reason,
                        }
                    )
                    old_label = "Exact" if old_relevance == 2 else "Partial"
                    new_label = "Exact" if new_relevance == 2 else "Partial"
                    logger.info(f"[Validation] Adjusting {product_id}: {old_label}→{new_label}: {reason}")

        logger.info(
            f"[Validation Chunk {chunk_num}/{total_chunks}] "
            f"Keeping {len(result.get('keep', []))}, adjusting {len(adjustment_details)}, removing {len(removal_details)}"
        )

        return {"removals": removal_details, "adjustments": adjustment_details}

    def _save_golden_set(
        self,
        state: "AgentState",
        judgments: list[AgentJudgment],
        reasoning: str,
    ) -> None:
        """Save the golden set to file."""
        products = [ProductRelevance(product_id=j.product_id, relevance=j.relevance) for j in judgments]

        config = GoldenSetConfig(
            query_id=state.query_id or state.query,
            query=state.query,
            products=products,
            reasoning=reasoning,
            created_by="agent",
            created_at=datetime.now(UTC),
        )

        # Ensure directory exists
        golden_sets_dir = Path(settings.golden_sets_dir)
        golden_sets_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        filename = f"{state.query.replace(' ', '_').lower()}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = golden_sets_dir / filename

        with open(filepath, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2, default=str)
