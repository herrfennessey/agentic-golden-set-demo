"""Score distribution guardrail."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.guardrails.base import Guardrail, GuardrailAction, GuardrailResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class ScoreDistributionGuardrail(Guardrail):
    """Validates judgment count and score diversity.

    Requires:
    - Minimum number of judgments
    - All three relevance levels present (Exact, Partial, Irrelevant)

    Note: We do NOT enforce percentage distribution - if 90% are Exact,
    that's fine as long as all 3 categories are represented.
    """

    def __init__(self, min_judgments: int = 50):
        """Initialize the score distribution guardrail.

        Args:
            min_judgments: Minimum number of judgments required.
        """
        self.min_judgments = min_judgments

    @property
    def name(self) -> str:
        return "score_distribution"

    def check(self, state: "AgentState", **kwargs: Any) -> GuardrailResult:
        """Check judgment count and score diversity.

        Args:
            state: Current agent state.

        Returns:
            GuardrailResult with pass/fail status.
        """
        judgments = state.judgments

        # Check minimum judgments
        if len(judgments) < self.min_judgments:
            return GuardrailResult.failure(
                f"Too few judgments ({len(judgments)}). Submit at least {self.min_judgments} product judgments.",
                action=GuardrailAction.BLOCK,
            )

        # Check that all 3 relevance levels are present
        scores = {j.relevance for j in judgments}
        missing = {0, 1, 2} - scores
        if missing:
            missing_names = {0: "Irrelevant (0)", 1: "Partial (1)", 2: "Exact (2)"}
            missing_str = ", ".join(missing_names[s] for s in sorted(missing))
            return GuardrailResult.failure(
                f"Golden set is missing {missing_str} judgments. "
                "A valid golden set requires ALL THREE relevance levels.",
                action=GuardrailAction.BLOCK,
            )

        return GuardrailResult.success()
