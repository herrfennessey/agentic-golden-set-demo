"""Planning tools for agent execution."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class SubmitPlanTool(BaseTool):
    """Submit an execution plan after discovery phase."""

    @property
    def name(self) -> str:
        return "submit_plan"

    @property
    def description(self) -> str:
        return (
            "Submit your exploration plan after running list_categories and search_products. "
            "Include all categories you plan to browse, ordered by relevance (primary matches first). "
            "Once submitted, you will execute each step sequentially."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "List of categories to explore, in order of relevance",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Exact category name from list_categories",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this category is relevant (e.g., 'Primary match - search returned 15 products')",
                            },
                        },
                        "required": ["category", "reason"],
                    },
                },
            },
            "required": ["steps"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute plan submission.

        Args:
            state: Current agent state.
            steps: List of category steps to explore.

        Returns:
            ToolResult confirming plan submission.
        """
        steps = kwargs.get("steps", [])

        if not steps:
            return ToolResult.fail("Plan must include at least one category to explore")

        if state.plan_submitted:
            return ToolResult.fail("Plan already submitted. Use complete_step to progress.")

        # Validate categories exist
        if state.available_classes:
            available_names = {c.product_class for c in state.available_classes}
            invalid = [s["category"] for s in steps if s["category"] not in available_names]
            if invalid:
                return ToolResult.fail(
                    f"Invalid categories: {invalid}. Use exact product_class names from list_categories."
                )

        # Set the plan in state
        state.set_plan(steps)
        state.record_tool_call(self.name)

        return ToolResult.ok(
            {
                "status": "plan_submitted",
                "total_steps": len(steps),
                "steps": [{"category": s["category"], "reason": s["reason"]} for s in steps],
            },
            message=f"Plan submitted with {len(steps)} steps. Starting with: {steps[0]['category']}",
        )


class CompleteStepTool(BaseTool):
    """Mark the current plan step as complete."""

    @property
    def name(self) -> str:
        return "complete_step"

    @property
    def description(self) -> str:
        return (
            "Mark the current plan step as complete after exhausting all products in the category. "
            "Call this when browse_category returns has_more=false. "
            "Include a summary of what you found in this category."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of findings (e.g., 'Browsed 245 products, found 12 exact, 8 partial, 20 irrelevant')",
                },
            },
            "required": ["summary"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute step completion.

        Args:
            state: Current agent state.
            summary: Summary of findings for this step.

        Returns:
            ToolResult indicating next step or plan completion.
        """
        summary = kwargs.get("summary", "")

        if not state.plan_submitted:
            return ToolResult.fail("No plan submitted yet. Call submit_plan first.")

        if not summary:
            return ToolResult.fail("Summary is required when completing a step.")

        current_step = state.get_current_step()
        if not current_step:
            return ToolResult.fail("No current step to complete. Plan may already be finished.")

        # Complete the step and advance
        has_more_steps = state.complete_current_step(summary)
        state.record_tool_call(self.name)

        completed_count = sum(1 for s in state.plan if s.status == "complete")

        if has_more_steps:
            next_step = state.get_current_step()
            return ToolResult.ok(
                {
                    "status": "step_completed",
                    "completed_step": current_step.category,
                    "next_step": next_step.category if next_step else None,
                    "progress": f"{completed_count}/{len(state.plan)} steps complete",
                },
                message=f"Completed '{current_step.category}'. Next: '{next_step.category if next_step else 'None'}'",
            )
        else:
            return ToolResult.ok(
                {
                    "status": "plan_complete",
                    "completed_step": current_step.category,
                    "progress": f"{completed_count}/{len(state.plan)} steps complete",
                    "ready_to_finish": True,
                },
                message=f"All {len(state.plan)} steps complete! Call finish_judgments to submit.",
            )
