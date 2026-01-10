"""Golden set generation agent."""

import json
import logging
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from goldendemo.agent.errors import FatalToolError, is_fatal_error
from goldendemo.agent.events import (
    AgentEvent,
    completed_event,
    error_event,
    guardrail_warning_event,
    iteration_start_event,
    phase_change_event,
    reasoning_event,
    started_event,
    step_completed_event,
    tool_call_event,
    tool_result_event,
)
from goldendemo.agent.guardrails import (
    CategoryBrowsingGuardrail,
    IterationBudgetGuardrail,
    MinimumExplorationGuardrail,
    ScoreDistributionGuardrail,
)
from goldendemo.agent.prompts import format_system_prompt
from goldendemo.agent.state import AgentState
from goldendemo.agent.tools import (
    BaseTool,
    BrowseCategoryTool,
    CompleteStepTool,
    FinishJudgmentsTool,
    ListCategoriesTool,
    SearchProductsTool,
    SubmitJudgmentsTool,
    SubmitPlanTool,
)
from goldendemo.clients.weaviate_client import WeaviateClient
from goldendemo.config import settings
from goldendemo.data.models import AgentResult

logger = logging.getLogger(__name__)


def _convert_tools_to_responses_format(tool_definitions: list[dict]) -> list[dict]:
    """Convert Chat Completions tool definitions to Responses API format.

    The Responses API function tool format:
    {
        "type": "function",
        "name": "...",
        "description": "...",
        "parameters": {...},
        "strict": false
    }
    """
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": False,  # Our schemas have dynamic objects
        }
        for tool in tool_definitions
    ]


class GoldenSetAgent:
    """Agent that generates relevance judgments for search queries.

    Uses OpenAI's Responses API with reasoning to autonomously explore
    a product catalog and generate golden sets.
    """

    def __init__(
        self,
        weaviate_client: WeaviateClient,
        openai_client: OpenAI | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: bool | None = None,
        max_iterations: int | None = None,
    ):
        """Initialize the agent.

        Args:
            weaviate_client: Connected Weaviate client.
            openai_client: OpenAI client. Creates one if not provided.
            model: Model to use (defaults to AGENT_MODEL from .env).
            reasoning_effort: Reasoning effort level (defaults to AGENT_REASONING_EFFORT from .env).
            reasoning_summary: Enable reasoning summaries (defaults to AGENT_REASONING_SUMMARY from .env).
            max_iterations: Maximum iterations (defaults to AGENT_MAX_ITERATIONS from .env).
        """
        self.weaviate_client = weaviate_client
        self.openai_client = openai_client or OpenAI(api_key=settings.openai_api_key)
        self.model = model if model is not None else settings.agent_model
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else settings.agent_reasoning_effort
        self.reasoning_summary = (
            reasoning_summary if reasoning_summary is not None else settings.agent_reasoning_summary
        )
        self.max_iterations = max_iterations if max_iterations is not None else settings.agent_max_iterations

        # Initialize tools
        self.tools = self._init_tools()
        tool_definitions = [tool.to_openai_schema() for tool in self.tools.values()]
        self.responses_tools = _convert_tools_to_responses_format(tool_definitions)

        # Initialize guardrails
        self.guardrails = self._init_guardrails()

    def _init_tools(self) -> dict[str, BaseTool]:
        """Initialize agent tools."""
        return {
            "search_products": SearchProductsTool(self.weaviate_client),
            "list_categories": ListCategoriesTool(self.weaviate_client),
            "browse_category": BrowseCategoryTool(self.weaviate_client),
            "submit_judgments": SubmitJudgmentsTool(self.weaviate_client),
            "submit_plan": SubmitPlanTool(self.weaviate_client),
            "complete_step": CompleteStepTool(self.weaviate_client),
            "finish_judgments": FinishJudgmentsTool(
                self.weaviate_client,
                guardrails=self._init_guardrails(),
            ),
        }

    def _init_guardrails(self) -> dict[str, Any]:
        """Initialize guardrails."""
        return {
            "iteration_budget": IterationBudgetGuardrail(
                max_iterations=self.max_iterations,
            ),
            "minimum_exploration": MinimumExplorationGuardrail(),
            "score_distribution": ScoreDistributionGuardrail(),
            "category_browsing": CategoryBrowsingGuardrail(),
        }

    def run(self, query: str, query_id: str | None = None) -> AgentResult:
        """Generate golden set for a query (blocking).

        Args:
            query: Search query to generate golden set for.
            query_id: Optional query ID for tracking.

        Returns:
            AgentResult with status and judgments.
        """
        # Consume the streaming generator to get final result
        result = None
        for event in self.run_streaming(query, query_id):
            if event.type.value == "completed":
                result = AgentResult(
                    query=query,
                    status=event.data.get("status", "success"),
                    products=[j for j in self._get_state().judgments] if hasattr(self, "_state") else None,
                    reasoning=self._get_state().final_reasoning if hasattr(self, "_state") else None,
                    tool_call_count=event.data.get("tool_calls", 0),
                )
            elif event.type.value == "error":
                result = AgentResult(
                    query=query,
                    status="failed",
                    error=event.data.get("error"),
                )

        return result or AgentResult(query=query, status="failed", error="No result")

    def _get_state(self) -> AgentState:
        """Get current state (for internal use)."""
        return getattr(self, "_state", AgentState(query=""))

    def run_streaming(
        self,
        query: str,
        query_id: str | None = None,
    ) -> Generator[AgentEvent, None, None]:
        """Generate golden set with streaming events using two-phase execution.

        Phase 1 (Discovery): Agent explores catalog and creates a plan.
        Phase 2 (Execution): Agent executes plan with context reset each iteration.

        Args:
            query: Search query to generate golden set for.
            query_id: Optional query ID for tracking.

        Yields:
            AgentEvent objects for UI updates.
        """
        # Initialize state
        state = AgentState(
            query=query,
            query_id=query_id,
            max_iterations=self.max_iterations,
        )
        self._state = state  # Store for run() access

        yield started_event(query, query_id)

        try:
            # ==================== PHASE 1: DISCOVERY ====================
            # Normal context accumulation until plan is submitted
            input_items: list[dict[str, Any]] = [
                {
                    "type": "message",
                    "role": "system",
                    "content": format_system_prompt(state),
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": f'Generate a golden set for the search query: "{query}"',
                },
            ]

            while not state.plan_submitted and state.iteration < self.max_iterations:
                state.iteration += 1
                yield iteration_start_event(state.iteration, self.max_iterations)

                # Call model and process response
                response = yield from self._call_model_iteration(state, input_items)
                if response is None:
                    return  # Error occurred

                # Process response and get function calls
                function_calls, plan_submitted = yield from self._process_response(
                    state, response, input_items, accumulate_context=True
                )
                if function_calls is None:
                    return  # Error or completion occurred

                # Check if plan was just submitted
                if state.plan_submitted:
                    yield phase_change_event(
                        from_phase="discovery",
                        to_phase="execution",
                        plan_steps=len(state.plan),
                    )
                    break

                # Handle no function calls
                if not function_calls:
                    yield from self._handle_no_tool_call(state, response, input_items)
                    continue

                state.clear_guardrail_feedback()

            # ==================== PHASE 2: EXECUTION ====================
            # Context resets only when starting a NEW step, not every iteration
            # This allows: browse → judge → submit → browse next page within same context
            current_step_at_start = state.current_step_index
            input_items = self._build_execution_context(state)

            while state.iteration < self.max_iterations:
                state.iteration += 1
                yield iteration_start_event(state.iteration, self.max_iterations)

                # Check if we moved to a new step - if so, reset context
                if state.current_step_index != current_step_at_start:
                    current_step_at_start = state.current_step_index
                    input_items = self._build_execution_context(state)

                # Call model and process response
                response = yield from self._call_model_iteration(state, input_items)
                if response is None:
                    return  # Error occurred

                # Process response - ACCUMULATE context within same step
                function_calls, finished = yield from self._process_response(
                    state, response, input_items, accumulate_context=True
                )
                if function_calls is None:
                    return  # Error or completion occurred

                if finished:
                    return  # Successfully completed

                # Handle no function calls
                if not function_calls:
                    yield from self._handle_no_tool_call(state, response, input_items)
                    continue

                state.clear_guardrail_feedback()

            # Max iterations reached
            yield error_event(
                f"Maximum iterations ({self.max_iterations}) reached without successful submission",
                recoverable=False,
            )

        except Exception as e:
            logger.exception("Agent execution failed")
            yield error_event(str(e), recoverable=False)

    def _build_execution_context(self, state: AgentState) -> list[dict[str, Any]]:
        """Build fresh context for execution phase (called at step boundaries)."""
        return [
            {
                "type": "message",
                "role": "system",
                "content": format_system_prompt(state),  # Includes plan summary
            },
            {
                "type": "message",
                "role": "user",
                "content": self._get_execution_user_message(state),
            },
        ]

    def _get_execution_user_message(self, state: AgentState) -> str:
        """Generate the user message for execution phase iterations."""
        step = state.get_current_step()
        if not step:
            return "All plan steps are complete. Call finish_judgments() to finalize your golden set."

        if step.products_browsed == 0:
            return f'Start browsing category "{step.category}". Call browse_category(product_class="{step.category}", offset=0).'
        else:
            return (
                f'Continue browsing category "{step.category}". '
                f"You have browsed {step.products_browsed} products so far. "
                f'Call browse_category(product_class="{step.category}", offset={step.current_offset}).'
            )

    def _call_model_iteration(
        self,
        state: AgentState,
        input_items: list[dict[str, Any]],
    ) -> Generator[AgentEvent, None, Any]:
        """Call the model for one iteration.

        Returns the response object or None if an error occurred.
        """
        # Check iteration budget guardrail
        budget_check = self.guardrails["iteration_budget"].check(state)
        if budget_check.warning:
            yield guardrail_warning_event("iteration_budget", budget_check.warning)
            input_items.append(
                {
                    "type": "message",
                    "role": "system",
                    "content": f"WARNING: {budget_check.warning}",
                }
            )

        try:
            # Build reasoning config
            reasoning_config: dict[str, Any] = {"effort": self.reasoning_effort}
            include_params: list[str] = []

            if self.reasoning_summary:
                reasoning_config["summary"] = "auto"
                include_params.append("reasoning.encrypted_content")

            response = self.openai_client.responses.create(
                model=self.model,
                input=input_items,
                tools=self.responses_tools,
                reasoning=reasoning_config,
                include=include_params if include_params else None,
                max_output_tokens=25000,
            )
            return response

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model call failed: {e}")

            if "verified to generate reasoning summaries" in error_msg:
                yield error_event(
                    "Reasoning summaries require a verified OpenAI organization.\n"
                    "Options:\n"
                    "  1. Verify your org: https://platform.openai.com/settings/organization/general\n"
                    "  2. Run with --no-reasoning-summary flag to disable summaries",
                    recoverable=False,
                )
            else:
                yield error_event(error_msg, recoverable=False)
            return None

    def _process_response(
        self,
        state: AgentState,
        response: Any,
        input_items: list[dict[str, Any]],
        accumulate_context: bool,
    ) -> Generator[AgentEvent, None, tuple[list | None, bool]]:
        """Process model response and execute tool calls.

        Args:
            state: Current agent state.
            response: Model response object.
            input_items: Current conversation context.
            accumulate_context: Whether to add results to input_items.

        Returns:
            Tuple of (function_calls list, finished bool).
            Returns (None, False) if an error occurred.
        """
        # Check for incomplete response
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            logger.warning(f"Response incomplete: {reason}")

        function_calls = []
        finished = False

        for item in response.output:
            # Handle reasoning items
            if item.type == "reasoning":
                if item.summary:
                    summary_texts = [s.text for s in item.summary if hasattr(s, "text") and s.text]
                    if summary_texts:
                        yield reasoning_event(" ".join(summary_texts), item.id)

                if accumulate_context:
                    input_items.append(
                        {
                            "type": "reasoning",
                            "id": item.id,
                            "summary": [
                                {"type": s.type, "text": s.text}
                                for s in (item.summary or [])
                                if hasattr(s, "type") and hasattr(s, "text")
                            ],
                            "encrypted_content": getattr(item, "encrypted_content", None),
                        }
                    )

            # Handle message items
            elif item.type == "message":
                content = item.content[0].text if item.content else ""
                if content and accumulate_context:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }
                    )

            # Handle function calls
            elif item.type == "function_call":
                function_calls.append(item)
                if accumulate_context:
                    input_items.append(
                        {
                            "type": "function_call",
                            "id": item.id,
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

        # Execute function calls
        for func_call in function_calls:
            tool_name = func_call.name
            call_id = func_call.call_id

            try:
                tool_args = json.loads(func_call.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            yield tool_call_event(tool_name, tool_args, call_id)

            # Execute tool
            try:
                result = self._execute_tool(tool_name, state, tool_args)
            except FatalToolError as e:
                logger.error(f"Fatal tool error, aborting: {e}")
                yield error_event(str(e), recoverable=False)
                return (None, False)

            yield tool_result_event(tool_name, result.to_dict(), call_id, result.success)

            if accumulate_context:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result.to_dict()),
                    }
                )

            # Handle special tools
            if tool_name == "finish_judgments" and result.success:
                warnings = result.data.get("warnings", [])
                yield completed_event(
                    status="success" if not warnings else "needs_review",
                    judgments_count=len(state.judgments),
                    tool_calls=len(state.tool_call_history),
                    warnings=warnings,
                )
                return (None, True)  # Signal completion

            if tool_name == "complete_step" and result.success:
                step_data = result.data
                yield step_completed_event(
                    step_index=state.current_step_index - 1,  # Already advanced
                    category=step_data.get("completed_step", ""),
                    summary=tool_args.get("summary", ""),
                    has_more_steps=step_data.get("status") != "plan_complete",
                )

            # Update step progress for browse_category
            if tool_name == "browse_category" and result.success:
                offset = tool_args.get("offset", 0)
                result_count = result.metadata.get("result_count", 0)
                state.update_step_progress(
                    offset=offset + result_count,
                    products_count=result_count,
                )

        return (function_calls, finished)

    def _handle_no_tool_call(
        self,
        state: AgentState,
        response: Any,
        input_items: list[dict[str, Any]],
    ) -> Generator[AgentEvent, None, None]:
        """Handle case where model returns no function calls."""
        assistant_msg = ""
        for item in response.output:
            if item.type == "message" and item.content:
                assistant_msg = item.content[0].text if item.content else ""
                break

        logger.warning(f"Model returned no function calls: {assistant_msg[:200] if assistant_msg else 'no message'}")

        feedback = "You returned text instead of a tool call. You MUST call a tool every turn. "
        if state.plan_submitted:
            step = state.get_current_step()
            if step:
                feedback += f'Call browse_category(product_class="{step.category}", offset={step.current_offset}).'
            else:
                feedback += "All steps complete - call finish_judgments() to finalize."
        else:
            feedback += "Call list_categories() or search_products() to explore the catalog."

        state.add_guardrail_feedback(feedback)
        yield guardrail_warning_event("no_tool_call", "Agent returned text - reminded to use tools")

    def _execute_tool(self, tool_name: str, state: AgentState, args: dict[str, Any]) -> Any:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            state: Current agent state.
            args: Tool arguments.

        Returns:
            ToolResult from the tool execution.

        Raises:
            FatalToolError: If a fatal infrastructure error occurs.
        """
        from goldendemo.agent.tools.base import ToolResult

        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult.fail(f"Unknown tool: {tool_name}")

        # Clean args - remove null/None values that model sometimes sends
        cleaned_args = {k: v for k, v in args.items() if v is not None and k != "null"}

        try:
            return tool.execute(state, **cleaned_args)
        except Exception as e:
            # Classify error as fatal or recoverable
            if is_fatal_error(e):
                logger.error(f"Fatal error in tool {tool_name}: {e}")
                raise FatalToolError(f"Tool {tool_name} failed with infrastructure error: {e}") from e

            # Recoverable error - return as tool result so agent can adjust
            logger.warning(f"Recoverable error in tool {tool_name}: {e}")
            return ToolResult.fail(f"Tool error (recoverable): {e}")
