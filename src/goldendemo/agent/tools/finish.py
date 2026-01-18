"""Finish judgments tool - signals completion and runs final validation."""

import json
import logging
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

logger = logging.getLogger(__name__)


class FinishJudgmentsTool(BaseTool):
    """Signal completion of judgment collection and trigger final validation."""

    def __init__(self, weaviate_client: Any, guardrails: dict[str, Any] | None = None):
        """Initialize with Weaviate client and guardrails.

        Args:
            weaviate_client: Connected Weaviate client.
            guardrails: Optional dict of guardrails to run on finish.
        """
        super().__init__(weaviate_client)
        self.guardrails = guardrails or self._default_guardrails()

    def _default_guardrails(self) -> dict[str, Any]:
        """Create default guardrails."""
        return {
            "iteration_budget": IterationBudgetGuardrail(),
            "minimum_exploration": MinimumExplorationGuardrail(),
            "score_distribution": ScoreDistributionGuardrail(
                min_exact=settings.min_exact_judgments,
                min_partial=settings.min_partial_judgments,
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
        """Execute finish - runs guardrails and saves golden set.

        Args:
            state: Current agent state with accumulated judgments.
            overall_reasoning: Summary of exploration approach.

        Returns:
            ToolResult indicating success or guardrail failures.
        """
        overall_reasoning = kwargs.get("overall_reasoning", "")
        state.final_reasoning = overall_reasoning
        state.record_tool_call(self.name)

        # Run guardrails on judgments
        guardrail_failures: list[str] = []
        guardrail_warnings: list[str] = []

        for name, guardrail in self.guardrails.items():
            if name == "iteration_budget":
                result: GuardrailResult = guardrail.check(state, is_submission=True)
            else:
                result = guardrail.check(state, judgments=state.judgments)

            if not result.passed:
                guardrail_failures.append(f"[{name}] {result.message}")
            elif result.warning:
                guardrail_warnings.append(f"[{name}] {result.warning}")

        # If any blocking failures, reject finish
        if guardrail_failures:
            failure_details = "; ".join(guardrail_failures)
            return ToolResult(
                success=False,
                error=f"Cannot finish: {failure_details}",
                data={
                    "guardrail_failures": guardrail_failures,
                    "message": "Continue exploring to meet requirements.",
                },
            )

        # Save the golden set (all judgments are Exact or Partial)
        try:
            self._save_golden_set(state, state.judgments, overall_reasoning)
        except Exception as e:
            return ToolResult.fail(f"Failed to save golden set: {e}")

        counts = state.judgment_counts_by_level()
        return ToolResult.ok(
            {
                "status": "complete",
                "judgments_count": len(state.judgments),
                "by_level": counts,
                "warnings": guardrail_warnings,
            },
            message=f"Golden set saved with {len(state.judgments)} judgments (Exact: {counts.get(2, 0)}, Partial: {counts.get(1, 0)}).",
        )

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
