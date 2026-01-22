"""Category browsing guardrail."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class CategoryBrowsingGuardrail(Guardrail):
    """Validates that agent browsed at least one category before submission.

    This prevents the agent from shortcutting by only using search results
    without exploring the actual category structure.
    """

    def __init__(self, min_categories_browsed: int = 1):
        """Initialize the category browsing guardrail.

        Args:
            min_categories_browsed: Minimum categories that must be browsed.
        """
        self.min_categories_browsed = min_categories_browsed

    @property
    def name(self) -> str:
        return "category_browsing"

    def check(self, state: "AgentState", *, is_submission: bool = False, **kwargs: Any) -> GuardrailResult:
        """Check that categories were browsed before submission.

        Args:
            state: Current agent state.
            is_submission: Whether this check is for a final submission.

        Returns:
            GuardrailResult with pass/fail status.
        """
        browsed = len(state.browsed_categories)

        if browsed < self.min_categories_browsed:
            return GuardrailResult.failure(
                f"You must browse at least {self.min_categories_browsed} category before submitting. "
                f"Use list_categories to find the PRIMARY category that matches the query, "
                f"then use browse_category to explore ALL products in that category.",
                action=GuardrailAction.BLOCK,
            )

        # Provide helpful feedback about what was browsed
        if browsed == 1:
            return GuardrailResult.success()

        return GuardrailResult.success()
