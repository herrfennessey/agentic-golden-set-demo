"""Submit judgments tool."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.tools.base import BaseTool, ToolResult
from goldendemo.data.models import AgentJudgment

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class SubmitJudgmentsTool(BaseTool):
    """Submit relevance judgments for products (additive - accumulates across calls)."""

    @property
    def name(self) -> str:
        return "submit_judgments"

    @property
    def description(self) -> str:
        return (
            "Submit relevance judgments for products you've reviewed. "
            "Call this after browsing each page to submit judgments for those products. "
            "Each judgment must include product_id, relevance score (0, 1, or 2), and reasoning."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "judgments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "Product ID being judged",
                            },
                            "relevance": {
                                "type": "integer",
                                "enum": [0, 1, 2],
                                "description": "Relevance score: 0=Irrelevant, 1=Partial, 2=Exact",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for this relevance judgment",
                            },
                        },
                        "required": ["product_id", "relevance", "reasoning"],
                    },
                    "description": "List of relevance judgments for products you just reviewed",
                },
            },
            "required": ["judgments"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute judgment submission (additive - accumulates across calls).

        Args:
            state: Current agent state.
            judgments: List of judgment dicts.

        Returns:
            ToolResult with count of accumulated judgments.
        """
        judgments_data = kwargs.get("judgments", [])

        if not judgments_data:
            return ToolResult.fail("judgments list is required")

        # Parse and add judgments (deduplicates by product_id)
        added_count = 0
        try:
            for j in judgments_data:
                judgment = AgentJudgment(
                    product_id=str(j["product_id"]),
                    relevance=j["relevance"],
                    reasoning=j["reasoning"],
                    confidence=j.get("confidence", 1.0),
                )
                state.add_judgment(judgment)
                added_count += 1
        except (KeyError, ValueError) as e:
            return ToolResult.fail(f"Invalid judgment format: {e}")

        state.record_tool_call(self.name)

        # Return accumulated totals
        counts = state.judgment_counts_by_level()
        return ToolResult.ok(
            {
                "added": added_count,
                "total": len(state.judgments),
                "by_level": counts,
            },
            message=f"Added {added_count} judgments. Total: {len(state.judgments)} (Exact: {counts.get(2, 0)}, Partial: {counts.get(1, 0)}, Irrelevant: {counts.get(0, 0)})",
        )
