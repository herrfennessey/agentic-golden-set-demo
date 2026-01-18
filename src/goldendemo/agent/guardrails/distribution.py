"""Score distribution guardrail."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class ScoreDistributionGuardrail(Guardrail):
    """Validates judgment quality and minimum thresholds.

    Requires:
    - Minimum Exact judgments (ensures quality matches)
    - Minimum Partial judgments (ensures boundary exploration)
    - Minimum total judgments

    Golden sets only contain Exact and Partial judgments.
    """

    def __init__(
        self,
        min_exact: int = 5,
        min_partial: int = 3,
        min_total: int = 50,
    ):
        """Initialize the score distribution guardrail.

        Args:
            min_exact: Minimum Exact (2) judgments required.
            min_partial: Minimum Partial (1) judgments required.
            min_total: Minimum total judgments required.
        """
        self.min_exact = min_exact
        self.min_partial = min_partial
        self.min_total = min_total

    @property
    def name(self) -> str:
        return "score_distribution"

    def check(self, state: "AgentState", **kwargs: Any) -> GuardrailResult:
        """Check judgment quality and thresholds."""
        judgments = kwargs.get("judgments", state.judgments)

        # Check minimum total
        if len(judgments) < self.min_total:
            return GuardrailResult.failure(
                f"Too few judgments ({len(judgments)}). Need at least {self.min_total}.",
                action=GuardrailAction.BLOCK,
            )

        # Count by level
        counts = {1: 0, 2: 0}
        for j in judgments:
            counts[j.relevance] = counts.get(j.relevance, 0) + 1

        # Check minimum thresholds
        failures = []
        if counts[2] < self.min_exact:
            failures.append(f"Need {self.min_exact} Exact, have {counts[2]}")
        if counts[1] < self.min_partial:
            failures.append(f"Need {self.min_partial} Partial, have {counts[1]}")

        if failures:
            return GuardrailResult.failure(
                f"Insufficient diversity: {'; '.join(failures)}. Continue exploring.",
                action=GuardrailAction.BLOCK,
            )

        return GuardrailResult.success()
