"""Iteration budget guardrail."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class IterationBudgetGuardrail(Guardrail):
    """Enforces iteration limits and minimum exploration before submission.

    Prevents:
    - Premature submissions (before adequate exploration)
    - Runaway loops (exceeding max iterations)
    """

    def __init__(
        self,
        max_iterations: int = 20,
        min_iterations_before_submit: int = 3,
        warn_at_iteration: int | None = None,
    ):
        """Initialize the iteration budget guardrail.

        Args:
            max_iterations: Maximum allowed iterations.
            min_iterations_before_submit: Minimum iterations before submission allowed.
            warn_at_iteration: Optional iteration number to start warning. Defaults to max_iterations - 5.
        """
        self.max_iterations = max_iterations
        self.min_iterations_before_submit = min_iterations_before_submit
        self.warn_at_iteration = warn_at_iteration if warn_at_iteration is not None else max_iterations - 5

    @property
    def name(self) -> str:
        return "iteration_budget"

    def check(self, state: "AgentState", is_submission: bool = False, **kwargs: Any) -> GuardrailResult:
        """Check iteration budget.

        Args:
            state: Current agent state.
            is_submission: True if this check is for a submission attempt.

        Returns:
            GuardrailResult with pass/fail status.
        """
        iteration = state.iteration

        # Block submission if too early
        if is_submission and iteration < self.min_iterations_before_submit:
            return GuardrailResult.failure(
                f"Cannot submit before exploring. You've only completed {iteration} iterations. "
                f"Use search_products and browse_category to explore the catalog first. "
                f"Minimum {self.min_iterations_before_submit} iterations required.",
                action=GuardrailAction.BLOCK,
            )

        # Force submission at max iterations
        if iteration >= self.max_iterations:
            return GuardrailResult.failure(
                f"Maximum iterations ({self.max_iterations}) reached. "
                "You must submit your judgments now using submit_judgments.",
                action=GuardrailAction.BLOCK,
            )

        # Warn when approaching limit
        if iteration >= self.warn_at_iteration:
            remaining = self.max_iterations - iteration
            return GuardrailResult.with_warning(
                f"You've used {iteration}/{self.max_iterations} iterations. "
                f"Only {remaining} remaining. Start converging toward submission."
            )

        return GuardrailResult.success()
