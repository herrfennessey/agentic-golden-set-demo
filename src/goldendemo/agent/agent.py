"""Golden set generation agent.

Architecture: Two-Phase Execution Model
========================================

This agent generates search relevance golden sets using a two-phase approach:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    PHASE 1: DISCOVERY                           │
    │                                                                 │
    │  Goal: Understand the catalog and create an exploration plan   │
    │                                                                 │
    │  Tools: list_categories, search_products, submit_plan          │
    │                                                                 │
    │  Flow:                                                          │
    │    1. Agent calls list_categories() to see available classes   │
    │    2. Agent calls search_products(query) to find matches       │
    │    3. Agent analyzes results and identifies relevant categories│
    │    4. Agent calls submit_plan() with ordered category list     │
    │                                                                 │
    │  Context: Accumulates continuously (no reset)                   │
    │  Exits when: submit_plan() is called successfully               │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PHASE 2: EXECUTION                           │
    │                                                                 │
    │  Goal: Browse each planned category and collect judgments      │
    │                                                                 │
    │  Tools: browse_category, complete_step, finish_judgments       │
    │                                                                 │
    │  Flow (per category):                                           │
    │    1. Agent calls browse_category(product_class)               │
    │       - Fetches ALL products in category                       │
    │       - JudgmentSubagent judges each product in parallel       │
    │       - Judgments are saved to state automatically             │
    │    2. Agent calls complete_step() to advance to next category  │
    │    3. Repeat until all categories complete                     │
    │    4. Agent calls finish_judgments() to finalize               │
    │                                                                 │
    │  Context: Resets at each step boundary (keeps context small)   │
    │  Exits when: finish_judgments() passes all guardrails          │
    └─────────────────────────────────────────────────────────────────┘

Key Components:
- GoldenSetAgent: Main orchestrator (this file)
- ResponseRunner: Handles OpenAI API calls and tool dispatch (runtime.py)
- JudgmentSubagent: Evaluates product relevance in parallel (judge.py)
- AgentState: Tracks iteration, judgments, plan progress (state.py)
- Guardrails: Enforce exploration/distribution requirements (guardrails/)

Entry Point:
- run_streaming(query, query_id) -> Generator[AgentEvent, None, None]
- run(query, query_id) -> AgentResult (blocking wrapper)
"""

import logging
from collections.abc import Generator
from typing import Any

from openai import OpenAI
from openai.types.responses import FunctionToolParam

from goldendemo.agent.constants import (
    TOOL_BROWSE_CATEGORY,
    TOOL_COMPLETE_STEP,
    TOOL_FINISH_JUDGMENTS,
    TOOL_LIST_CATEGORIES,
    TOOL_SEARCH_PRODUCTS,
    TOOL_SUBMIT_PLAN,
)
from goldendemo.agent.events import (
    AgentEvent,
    completed_event,
    error_event,
    execution_phase_started_event,
    iteration_start_event,
    plan_step_completed_event,
    started_event,
)
from goldendemo.agent.guardrails import (
    CategoryBrowsingGuardrail,
    IterationBudgetGuardrail,
    MinimumExplorationGuardrail,
    ScoreDistributionGuardrail,
)
from goldendemo.agent.judge import JudgmentSubagent
from goldendemo.agent.prompts import format_system_prompt
from goldendemo.agent.runtime import ResponseRunner
from goldendemo.agent.state import AgentState
from goldendemo.agent.tools import (
    BaseTool,
    BrowseCategoryTool,
    CompleteStepTool,
    FinishJudgmentsTool,
    ListCategoriesTool,
    SearchProductsTool,
    SubmitPlanTool,
)
from goldendemo.clients.weaviate_client import WeaviateClient
from goldendemo.config import settings
from goldendemo.data.models import AgentResult

logger = logging.getLogger(__name__)


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

        # Initialize judgment subagent (shares OpenAI client)
        self.judgment_subagent = JudgmentSubagent(
            openai_client=self.openai_client,
            model=settings.judge_model,
            reasoning_effort=settings.judge_reasoning_effort,
        )

        # Initialize guardrails FIRST (shared by agent and FinishJudgmentsTool)
        self.guardrails = self._init_guardrails()

        # Initialize tools (pass shared guardrails instance)
        self.tools = self._init_tools()

        # Build native OpenAI FunctionToolParam objects for the Responses API
        self.responses_tools: list[FunctionToolParam] = [
            FunctionToolParam(
                type="function",
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=False,  # Allow flexible parameter handling for dynamic schemas
            )
            for tool in self.tools.values()
        ]

        # Initialize runtime layer
        self.runner = ResponseRunner(
            openai_client=self.openai_client,
            model=self.model,
            tools=self.tools,
            responses_tools=self.responses_tools,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
        )

    def _init_tools(self) -> dict[str, BaseTool]:
        """Initialize agent tools.

        Note: self.guardrails must be initialized before calling this method.
        """
        return {
            TOOL_SEARCH_PRODUCTS: SearchProductsTool(self.weaviate_client),
            TOOL_LIST_CATEGORIES: ListCategoriesTool(self.weaviate_client),
            TOOL_BROWSE_CATEGORY: BrowseCategoryTool(self.weaviate_client, judgment_subagent=self.judgment_subagent),
            TOOL_SUBMIT_PLAN: SubmitPlanTool(self.weaviate_client),
            TOOL_COMPLETE_STEP: CompleteStepTool(self.weaviate_client),
            TOOL_FINISH_JUDGMENTS: FinishJudgmentsTool(
                self.weaviate_client,
                guardrails=self.guardrails,  # Use shared instance
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
            yield from self._run_discovery_phase(state)

            if not state.plan_submitted:
                # Discovery phase ended without plan - likely max iterations
                return

            yield execution_phase_started_event(plan_steps=len(state.plan))

            # ==================== PHASE 2: EXECUTION ====================
            yield from self._run_execution_phase(state)

        except Exception as e:
            logger.exception("Agent execution failed")
            yield error_event(str(e), recoverable=False)

    def _run_discovery_phase(self, state: AgentState) -> Generator[AgentEvent, None, None]:
        """Run the discovery phase until plan is submitted.

        Args:
            state: Agent state.

        Yields:
            AgentEvent objects.
        """
        input_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "system",
                "content": format_system_prompt(state),
            },
            {
                "type": "message",
                "role": "user",
                "content": f'Generate a golden set for the search query: "{state.query}"',
            },
        ]

        while not state.plan_submitted and state.iteration < self.max_iterations:
            state.iteration += 1
            yield iteration_start_event(state.iteration, self.max_iterations)

            # Call model
            response = yield from self.runner.call_model(state, input_items, self.guardrails)
            if response is None:
                return  # Error occurred

            # Process response
            result = yield from self.runner.process_response(state, response, input_items, accumulate_context=True)

            if result.error:
                return  # Fatal error occurred

            # Check if plan was just submitted
            if state.plan_submitted:
                break

            # Handle no function calls
            if not result.function_calls:
                yield from self.runner.handle_no_tool_call(state, response, input_items)
                continue

            state.clear_guardrail_feedback()

        # Check if we hit max iterations without plan
        if not state.plan_submitted:
            yield error_event(
                f"Maximum iterations ({self.max_iterations}) reached without plan submission",
                recoverable=False,
            )

    def _run_execution_phase(self, state: AgentState) -> Generator[AgentEvent, None, None]:
        """Run the execution phase until completion.

        Args:
            state: Agent state.

        Yields:
            AgentEvent objects.
        """
        current_step_at_start = state.current_step_index
        input_items = self._build_execution_context(state)

        while state.iteration < self.max_iterations:
            state.iteration += 1
            yield iteration_start_event(state.iteration, self.max_iterations)

            # Check if we moved to a new step - if so, reset context
            if state.current_step_index != current_step_at_start:
                current_step_at_start = state.current_step_index
                input_items = self._build_execution_context(state)

            # Call model
            response = yield from self.runner.call_model(state, input_items, self.guardrails)
            if response is None:
                return  # Error occurred

            # Process response
            result = yield from self.runner.process_response(state, response, input_items, accumulate_context=True)

            if result.error:
                return  # Fatal error occurred

            # Handle special tool results
            finished = yield from self._handle_tool_results(state, result)
            if finished:
                return  # Successfully completed

            # Handle no function calls
            if not result.function_calls:
                yield from self.runner.handle_no_tool_call(state, response, input_items)
                continue

            state.clear_guardrail_feedback()

        # Max iterations reached
        yield error_event(
            f"Maximum iterations ({self.max_iterations}) reached without successful submission",
            recoverable=False,
        )

    @staticmethod
    def _handle_tool_results(state: AgentState, result: Any) -> Generator[AgentEvent, None, bool]:
        """Handle special tool results that affect flow control.

        Args:
            state: Agent state.
            result: ProcessedResult from runtime.

        Yields:
            AgentEvent objects.

        Returns:
            True if execution is complete, False to continue.
        """
        for tool_result in result.tool_results:
            tool_name = tool_result.tool_name

            # Check if finish_judgments completed successfully
            if tool_name == TOOL_FINISH_JUDGMENTS and tool_result.success:
                warnings = []
                if hasattr(tool_result.result, "data") and tool_result.result.data:
                    warnings = tool_result.result.data.get("warnings", [])
                yield completed_event(
                    status="success" if not warnings else "needs_review",
                    judgments_count=len(state.judgments),
                    tool_calls=len(state.tool_call_history),
                    warnings=warnings,
                    token_usage=state.token_usage.to_dict(),
                )
                return True

            # Handle step completion
            if tool_name == TOOL_COMPLETE_STEP and tool_result.success:
                step_idx = state.current_step_index - 1  # Already advanced
                if 0 <= step_idx < len(state.plan):
                    completed_step = state.plan[step_idx]
                    yield plan_step_completed_event(
                        step_index=step_idx,
                        category=completed_step.category,
                        summary=completed_step.summary or "",
                        has_more_steps=state.current_step_index < len(state.plan),
                    )

        return False

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

    @staticmethod
    def _get_execution_user_message(state: AgentState) -> str:
        """Generate the user message for execution phase iterations."""
        step = state.get_current_step()
        if not step:
            return "All plan steps are complete. Call finish_judgments() to finalize your golden set."

        if step.products_processed == 0:
            return f'Start browsing category "{step.category}". Call browse_category(product_class="{step.category}").'
        else:
            # Category already browsed - should call complete_step
            return (
                f'Category "{step.category}" has been browsed ({step.products_processed} products processed). '
                f'Call complete_step(summary="...") to mark this category done and move to the next.'
            )
