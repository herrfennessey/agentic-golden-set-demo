"""Planning tools for agent execution."""

from typing import TYPE_CHECKING, Any

from goldendemo.agent.state import StepType
from goldendemo.agent.tools.base import BaseTool, ToolResult
from goldendemo.agent.utils import find_matching_category

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
            "Submit your exploration plan with search and category steps. "
            "REQUIRED: At least 1 search step AND at least 2 category steps. "
            "Search steps execute first (auto-complete with judgments). "
            "Category steps auto-complete after browse_category. "
            "Max 10 total steps."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "List of search and category steps to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["search", "category"],
                                "description": "Step type: 'search' for query execution, 'category' for browsing",
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query (required if type=search)",
                            },
                            "category": {
                                "type": "string",
                                "description": "Exact category name from list_categories (required if type=category)",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this step is relevant",
                            },
                        },
                        "required": ["type", "reason"],
                    },
                },
            },
            "required": ["steps"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute plan submission.

        Args:
            state: Current agent state.
            steps: List of search and category steps to execute.

        Returns:
            ToolResult confirming plan submission.
        """
        steps = kwargs.get("steps", [])

        if not steps:
            return ToolResult.fail("Plan must include at least one step")

        if len(steps) > 10:
            return ToolResult.fail(
                f"Plan has {len(steps)} steps - maximum is 10. Focus on PRIMARY categories and key searches only."
            )

        if state.plan_submitted and not state.is_plan_complete():
            return ToolResult.fail("Plan already submitted and in progress. Use complete_step to progress.")

        # If resubmitting after a complete plan, reset for new exploration
        if state.plan_submitted and state.is_plan_complete():
            state.plan = []
            state.plan_submitted = False
            state.current_step_index = 0

        # Validate step types and required fields
        search_steps = []
        category_steps = []
        invalid_categories = []

        for step in steps:
            step_type = step.get("type", "category")

            if step_type == "search":
                if not step.get("query"):
                    return ToolResult.fail("Search steps require a 'query' field")
                search_steps.append(step)
            elif step_type == "category":
                category = step.get("category", "")
                if not category:
                    return ToolResult.fail("Category steps require a 'category' field")

                # Validate and normalize category names
                if state.available_classes:
                    matched = find_matching_category(category, state.available_classes)
                    if matched:
                        step = {**step, "category": matched}
                        category_steps.append(step)
                    else:
                        invalid_categories.append(category)
                else:
                    category_steps.append(step)
            else:
                return ToolResult.fail(f"Invalid step type: {step_type}. Use 'search' or 'category'.")

        if invalid_categories:
            return ToolResult.fail(
                f"Invalid categories: {invalid_categories}. Use exact product_class names from list_categories."
            )

        # Validate plan has required mix of step types
        if len(search_steps) < 1:
            return ToolResult.fail(
                "Plan must include at least 1 search step. "
                "Search steps capture products using different query phrasings/synonyms."
            )

        if len(category_steps) < 2:
            return ToolResult.fail(
                f"Plan must include at least 2 category steps (found {len(category_steps)}). "
                "Category steps ensure systematic coverage of primary product types."
            )

        # Combine validated steps (search first, then categories - reordering happens in set_plan)
        validated_steps = search_steps + category_steps

        # Set the plan in state (will reorder search steps first)
        state.set_plan(validated_steps)
        state.record_tool_call(self.name)

        # Format response showing the order
        search_count = len(search_steps)
        category_count = len(category_steps)
        first_step = state.plan[0] if state.plan else None
        first_step_desc = (
            f'Search "{first_step.target}"'
            if first_step and first_step.step_type.value == "search"
            else f'Browse "{first_step.target}"'
            if first_step
            else "None"
        )

        return ToolResult.ok(
            {
                "status": "plan_submitted",
                "total_steps": len(validated_steps),
                "search_steps": search_count,
                "category_steps": category_count,
                "execution_order": "Search steps execute first (auto-complete), then category steps",
            },
            message=f"Plan submitted with {search_count} search + {category_count} category steps. Starting with: {first_step_desc}",
        )


class CompleteStepTool(BaseTool):
    """Manually complete a step (fallback - steps normally auto-complete)."""

    @property
    def name(self) -> str:
        return "complete_step"

    @property
    def description(self) -> str:
        return (
            "FALLBACK: Manually complete a step if auto-completion failed. "
            "Normally you don't need this - browse_category auto-completes steps. "
            "Only use if a step is stuck."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of findings",
                },
            },
            "required": ["summary"],
        }

    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute step completion (fallback).

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

        # Search steps auto-complete - reject manual completion
        if current_step.step_type == StepType.SEARCH:
            return ToolResult.fail(
                f"Search steps auto-complete. The search for '{current_step.target}' will complete automatically."
            )

        # Complete the step and advance
        has_more_steps = state.complete_current_step(summary)
        state.record_tool_call(self.name)

        completed_count = sum(1 for s in state.plan if s.status == "complete")

        if has_more_steps:
            next_step = state.get_current_step()
            next_step_desc = (
                f'Search "{next_step.target}"'
                if next_step and next_step.step_type == StepType.SEARCH
                else f'Browse "{next_step.target}"'
                if next_step
                else "None"
            )
            return ToolResult.ok(
                {
                    "status": "plan_step_completed",
                    "completed_step": current_step.target,
                    "next_step": next_step.target if next_step else None,
                    "next_step_type": next_step.step_type.value if next_step else None,
                    "progress": f"{completed_count}/{len(state.plan)} steps complete",
                },
                message=f"Manually completed '{current_step.target}'. Next: {next_step_desc}",
            )
        else:
            return ToolResult.ok(
                {
                    "status": "plan_complete",
                    "completed_step": current_step.target,
                    "progress": f"{completed_count}/{len(state.plan)} steps complete",
                    "ready_to_finish": True,
                },
                message=f"All {len(state.plan)} steps complete! Call finish_judgments to submit.",
            )
