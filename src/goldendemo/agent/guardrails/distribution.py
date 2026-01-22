"""Score distribution guardrail."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class ScoreDistributionGuardrail(Guardrail):
    """Validates judgment quality and minimum thresholds.

    Requires:
    - Minimum Exact judgments (ensures quality matches)
    - Minimum total judgments

    Golden sets only contain Exact and Partial judgments.
    Note: We don't require Partial judgments since some queries may not
    have products that match the "right entity, wrong modifier" criteria.
    """

    def __init__(
        self,
        min_exact: int = 5,
        min_total: int = 50,
    ):
        """Initialize the score distribution guardrail.

        Args:
            min_exact: Minimum Exact (2) judgments required.
            min_total: Minimum total judgments required.
        """
        self.min_exact = min_exact
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

        # Count Exact judgments
        exact_count = sum(1 for j in judgments if j.relevance == 2)

        if exact_count < self.min_exact:
            return GuardrailResult.failure(
                f"Need {self.min_exact} Exact judgments, have {exact_count}. Continue exploring.",
                action=GuardrailAction.BLOCK,
            )

        return GuardrailResult.success()
