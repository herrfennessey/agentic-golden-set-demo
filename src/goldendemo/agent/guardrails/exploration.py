"""Minimum exploration guardrail."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class MinimumExplorationGuardrail(Guardrail):
    """Validates that agent has adequately explored the catalog before submission.

    Without behavioral data (clicks, carts), we need evidence that the agent
    actually explored the catalog rather than just judging the first few results.
    """

    def __init__(
        self,
        min_products_seen: int = 30,
        min_search_queries: int = 2,
        min_categories_explored: int = 1,
    ):
        """Initialize the minimum exploration guardrail.

        Args:
            min_products_seen: Minimum unique products to have seen.
            min_search_queries: Minimum number of search queries executed.
            min_categories_explored: Minimum categories browsed.
        """
        self.min_products_seen = min_products_seen
        self.min_search_queries = min_search_queries
        self.min_categories_explored = min_categories_explored

    @property
    def name(self) -> str:
        return "minimum_exploration"

    def check(self, state: "AgentState", **kwargs: Any) -> GuardrailResult:
        """Check exploration metrics.

        Args:
            state: Current agent state.

        Returns:
            GuardrailResult with pass/fail status and specific feedback.
        """
        metrics = state.exploration_metrics
        issues: list[str] = []

        # Check products seen
        if metrics.unique_products_seen < self.min_products_seen:
            issues.append(
                f"Only seen {metrics.unique_products_seen} products "
                f"(minimum {self.min_products_seen}). "
                "Use search_products or browse_category to explore more."
            )

        # Check search queries
        if metrics.search_queries_executed < self.min_search_queries:
            issues.append(
                f"Only executed {metrics.search_queries_executed} search(es) "
                f"(minimum {self.min_search_queries}). "
                "Try alternative phrasings or related terms."
            )

        # Check category exploration
        if metrics.categories_explored < self.min_categories_explored:
            issues.append(
                f"Haven't browsed any product categories yet "
                f"(minimum {self.min_categories_explored}). "
                "Use list_categories to find relevant classes, then browse_category to explore."
            )

        if issues:
            return GuardrailResult.failure(
                "Insufficient exploration before submission:\n• " + "\n• ".join(issues),
                action=GuardrailAction.BLOCK,
            )

        return GuardrailResult.success()
