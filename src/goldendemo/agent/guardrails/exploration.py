"""Minimum exploration guardrail."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState, ExplorationMetrics


@dataclass
class ExplorationRule:
    """A single exploration requirement."""

    name: str
    minimum: int
    get_value: Callable[["ExplorationMetrics"], int]
    suggestion: str

    def check(self, metrics: "ExplorationMetrics") -> str | None:
        """Return issue message if rule fails, None if passes."""
        value = self.get_value(metrics)
        if value < self.minimum:
            return f"{self.name}: {value}/{self.minimum}. {self.suggestion}"
        return None


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
        self.rules = [
            ExplorationRule(
                name="Products seen",
                minimum=min_products_seen,
                get_value=lambda m: m.unique_products_seen,
                suggestion="Use search_products or browse_category to explore more.",
            ),
            ExplorationRule(
                name="Unique searches",
                minimum=min_search_queries,
                get_value=lambda m: m.search_queries_executed,
                suggestion="Run diverse searches: synonyms, related terms, specific attributes. Duplicates don't count.",
            ),
            ExplorationRule(
                name="Categories browsed",
                minimum=min_categories_explored,
                get_value=lambda m: m.categories_explored,
                suggestion="Use list_categories then browse_category.",
            ),
        ]

    @property
    def name(self) -> str:
        return "minimum_exploration"

    def check(self, state: "AgentState", *, is_submission: bool = False, **kwargs: Any) -> GuardrailResult:
        """Check exploration metrics.

        Args:
            state: Current agent state.
            is_submission: Whether this check is for a final submission.

        Returns:
            GuardrailResult with pass/fail status and specific feedback.
        """
        issues = [msg for rule in self.rules if (msg := rule.check(state.exploration_metrics))]

        if issues:
            return GuardrailResult.failure(
                "Insufficient exploration:\n• " + "\n• ".join(issues),
                action=GuardrailAction.BLOCK,
            )
        return GuardrailResult.success()
