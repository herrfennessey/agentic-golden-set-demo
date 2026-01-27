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
    │    2. Agent calls search_products() with query variations      │
    │       - Returns SLIM data (id, name, class, short description) │
    │       - NO judgment - just for understanding the catalog       │
    │    3. Agent analyzes results and identifies relevant categories│
    │    4. Agent calls submit_plan() with search AND category steps │
    │                                                                 │
    │  Context: Accumulates continuously (no reset)                   │
    │  Exits when: submit_plan() is called successfully               │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PHASE 2: EXECUTION                           │
    │                                                                 │
    │  Goal: Execute plan steps and collect judgments                │
    │                                                                 │
    │  Tools: browse_category, finish_judgments                      │
    │                                                                 │
    │  Search steps (auto-execute, no LLM call):                     │
    │    - Executed directly by _execute_search_step()               │
    │    - Fetches products with FULL data for accurate judgment     │
    │    - JudgmentSubagent judges each product in parallel          │
    │    - Auto-completes and advances to next step                  │
    │                                                                 │
    │  Category steps (LLM calls browse_category):                   │
    │    - Agent calls browse_category(product_class)                │
    │    - Fetches ALL products in category with full data           │
    │    - JudgmentSubagent judges each product in parallel          │
    │    - Auto-completes after processing                           │
    │                                                                 │
    │  Completion:                                                    │
    │    - Agent calls finish_judgments() when all steps done        │
    │                                                                 │
    │  Context: Resets at each step boundary (keeps context small)   │
    │  Exits when: finish_judgments() passes all guardrails          │
    └─────────────────────────────────────────────────────────────────┘

Key Design: Discovery vs Execution Data
---------------------------------------
Discovery searches return slim data to keep context manageable while exploring.
Execution fetches full product data (descriptions, attributes, hierarchy) needed
for accurate relevance judgment. This separation prevents context window overflow.

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
    search_step_event,
    started_event,
    validation_phase_event,
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
from goldendemo.agent.state import AgentState, StepType
from goldendemo.agent.tools import (
    BaseTool,
    BrowseCategoryTool,
    CompleteStepTool,
    FinishJudgmentsTool,
    ListCategoriesTool,
    SearchProductsTool,
    SubmitPlanTool,
)
from goldendemo.agent.validator import ValidationSubagent
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

        # Initialize validation subagent (shares OpenAI client)
        self.validation_subagent = ValidationSubagent(
            openai_client=self.openai_client,
            model=settings.validate_model,
            reasoning_effort=settings.validate_reasoning_effort,
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

        Note: self.guardrails and self.validation_subagent must be initialized before calling this method.
        """
        return {
            TOOL_SEARCH_PRODUCTS: SearchProductsTool(self.weaviate_client),
            TOOL_LIST_CATEGORIES: ListCategoriesTool(self.weaviate_client),
            TOOL_BROWSE_CATEGORY: BrowseCategoryTool(self.weaviate_client, judgment_subagent=self.judgment_subagent),
            TOOL_SUBMIT_PLAN: SubmitPlanTool(self.weaviate_client),
            TOOL_COMPLETE_STEP: CompleteStepTool(self.weaviate_client),
            TOOL_FINISH_JUDGMENTS: FinishJudgmentsTool(
                self.weaviate_client,
                guardrails=self.guardrails,
                validator=self.validation_subagent,
            ),
        }

    def _init_guardrails(self) -> dict[str, Any]:
        """Initialize guardrails."""
        return {
            "iteration_budget": IterationBudgetGuardrail(
                max_iterations=self.max_iterations,
            ),
            "minimum_exploration": MinimumExplorationGuardrail(),
            "score_distribution": ScoreDistributionGuardrail(
                min_exact=settings.min_exact_judgments,
                min_total=settings.min_total_judgments,
            ),
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

        Search steps execute automatically without LLM calls.
        Category steps use LLM to call browse_category + complete_step.

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

            # Check if current step is a search step - execute directly without LLM
            current_step = state.get_current_step()
            if current_step and current_step.step_type == StepType.SEARCH:
                yield from self._execute_search_step(state, current_step)
                # Rebuild context after search step auto-completes
                current_step_at_start = state.current_step_index
                input_items = self._build_execution_context(state)
                # Check if plan is now complete after search steps
                if state.is_plan_complete():
                    # All steps done, need one more LLM call for finish_judgments
                    pass
                else:
                    continue  # Move to next step (will increment iteration)

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

            # Handle finish_judgments - emit validation event even on failure
            if tool_name == TOOL_FINISH_JUDGMENTS:
                validation_result = None
                if hasattr(tool_result.result, "data") and tool_result.result.data:
                    validation_result = tool_result.result.data.get("validation_result")

                # Emit validation phase event if validation ran (regardless of success/failure)
                if validation_result and validation_result.get("total_reviewed", 0) > 0:
                    yield validation_phase_event(
                        total_reviewed=validation_result.get("total_reviewed", 0),
                        total_removed=validation_result.get("total_removed", 0),
                        exact_removed=validation_result.get("exact_removed", 0),
                        partial_removed=validation_result.get("partial_removed", 0),
                        exact_remaining=validation_result.get("exact_remaining", 0),
                        partial_remaining=validation_result.get("partial_remaining", 0),
                        removal_details=validation_result.get("removal_details", []),
                        total_adjusted=validation_result.get("total_adjusted", 0),
                        upgrades=validation_result.get("upgrades", 0),
                        downgrades=validation_result.get("downgrades", 0),
                        adjustment_details=validation_result.get("adjustment_details", []),
                    )

                # Only emit completed event and return if successful
                if not tool_result.success:
                    continue

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

            # Handle browse_category auto-completion
            if tool_name == TOOL_BROWSE_CATEGORY and tool_result.success:
                # Check if the result indicates step was completed
                if hasattr(tool_result.result, "data") and tool_result.result.data:
                    if tool_result.result.data.get("step_completed"):
                        # Step was auto-completed by browse_category
                        step_idx = state.current_step_index - 1  # Already advanced
                        if 0 <= step_idx < len(state.plan):
                            completed_step = state.plan[step_idx]
                            yield plan_step_completed_event(
                                step_index=step_idx,
                                category=completed_step.target,
                                summary=completed_step.summary or "",
                                has_more_steps=state.current_step_index < len(state.plan),
                            )

            # Handle manual step completion (fallback/edge case)
            if tool_name == TOOL_COMPLETE_STEP and tool_result.success:
                step_idx = state.current_step_index - 1  # Already advanced
                if 0 <= step_idx < len(state.plan):
                    completed_step = state.plan[step_idx]
                    yield plan_step_completed_event(
                        step_index=step_idx,
                        category=completed_step.target,
                        summary=completed_step.summary or "",
                        has_more_steps=state.current_step_index < len(state.plan),
                    )

        return False

    def _execute_search_step(self, state: AgentState, step: Any) -> Generator[AgentEvent, None, None]:
        """Execute a search step automatically (no LLM call needed).

        Search steps execute their query, judge results, and auto-complete.
        Uses the same judgment pattern as browse_category for consistency.
        Emits tool_call and tool_result events for UI visibility.

        Args:
            state: Agent state.
            step: The search PlanStep to execute.

        Yields:
            AgentEvent objects.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from goldendemo.agent.events import tool_call_event, tool_result_event
        from goldendemo.agent.utils import parse_product_features

        step_index = state.current_step_index
        search_query = step.target

        logger.info(f"[Search Step {step_index + 1}] Executing search: '{search_query}'")

        # Emit tool_call event for UI visibility (like browse_category does via runtime)
        yield tool_call_event(
            tool_name="search_step",
            arguments={"query": search_query, "step_index": step_index},
        )

        try:
            # Fetch products with full data
            results = self.weaviate_client.hybrid_search(
                query=search_query,
                limit=200,
                alpha=0.5,
            )

            total_products = len(results)
            logger.info(f"[Search Step {step_index + 1}] Retrieved {total_products} products")

            # Update state
            state.add_seen_products(results)
            state.record_search(search_query, total_products, 0.5)
            state.record_tool_call("search_step")
            state.update_step_progress(products_count=total_products)

            if total_products == 0:
                state.complete_current_step(f"Search '{search_query}': No products found")
                yield search_step_event(
                    step_index=step_index,
                    query=search_query,
                    products_found=0,
                    judgments_added=0,
                    exact_count=0,
                    partial_count=0,
                )
                return

            # Convert to full product data format (same as browse_category)
            all_products_data = [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.product_class,
                    "category_hierarchy": p.category_hierarchy,
                    "description": p.product_description or "",
                    "attributes": parse_product_features(p.product_features),
                }
                for p in results
            ]

            # Split into chunks and judge in parallel (same pattern as browse_category)
            chunk_size = settings.judge_chunk_size
            chunks = [all_products_data[i : i + chunk_size] for i in range(0, len(all_products_data), chunk_size)]

            logger.info(f"[Search Step {step_index + 1}] Judging {total_products} products in {len(chunks)} chunks")

            total_judgments_added = 0
            batch_exact = 0
            batch_partial = 0

            # Judge all chunks in parallel (same pattern as browse_category)
            with ThreadPoolExecutor(max_workers=settings.judge_max_workers) as executor:
                futures = [
                    executor.submit(self._judge_search_chunk, state, state.query, chunk, idx + 1, len(chunks))
                    for idx, chunk in enumerate(chunks)
                ]

                for future in as_completed(futures):
                    try:
                        valid_judgments, hallucinated_count = future.result()
                        for j in valid_judgments:
                            if j.get("relevance") == 2:
                                batch_exact += 1
                            elif j.get("relevance") == 1:
                                batch_partial += 1
                        judgments_added = state.add_judgments_from_dicts(valid_judgments)
                        total_judgments_added += judgments_added
                        if hallucinated_count > 0:
                            logger.warning(
                                f"[Search Step {step_index + 1}] Hallucinated {hallucinated_count} IDs in chunk"
                            )
                    except Exception as e:
                        logger.error(f"[Search Step {step_index + 1}] Chunk judgment failed: {e}", exc_info=True)

            # Update step with judgments count
            state.update_step_progress(products_count=0, judgments_count=total_judgments_added)

            # Auto-complete the step
            summary = (
                f"Searched '{search_query}': {total_products} products, "
                f"{total_judgments_added} judged (Exact: {batch_exact}, Partial: {batch_partial})"
            )
            state.complete_current_step(summary)

            # Emit tool_result event for UI visibility
            yield tool_result_event(
                tool_name="search_step",
                result={
                    "data": {
                        "judgments_added": total_judgments_added,
                        "exact_count": batch_exact,
                        "partial_count": batch_partial,
                        "step_completed": True,
                    },
                    "metadata": {
                        "query": search_query,
                        "products_found": total_products,
                    },
                },
                success=True,
            )

            # Emit step completed event
            yield search_step_event(
                step_index=step_index,
                query=search_query,
                products_found=total_products,
                judgments_added=total_judgments_added,
                exact_count=batch_exact,
                partial_count=batch_partial,
            )

            logger.info(
                f"[Search Step {step_index + 1}] Completed: {total_products} products, "
                f"{total_judgments_added} judgments"
            )

        except Exception as e:
            logger.error(f"[Search Step {step_index + 1}] Failed: {e}", exc_info=True)
            state.complete_current_step(f"Search failed: {e}")

            # Emit tool_result event for failure
            yield tool_result_event(
                tool_name="search_step",
                result={"error": str(e)},
                success=False,
            )

    def _judge_search_chunk(
        self,
        state: AgentState,
        query: str,
        products_data: list[dict],
        chunk_num: int,
        total_chunks: int,
    ) -> tuple[list[dict], int]:
        """Judge a single chunk of search products (same pattern as browse_category).

        Args:
            state: Agent state (for token tracking).
            query: Original search query to judge against.
            products_data: Products to judge.
            chunk_num: Which chunk this is (for logging).
            total_chunks: Total number of chunks.

        Returns:
            Tuple of (valid_judgments, hallucinated_count).
        """
        # Log chunk details for debugging
        product_ids_sample = [p.get("product_id", "?") for p in products_data[:3]]
        logger.info(
            f"[Search Chunk {chunk_num}/{total_chunks}] Judging {len(products_data)} products "
            f"(first 3 IDs: {product_ids_sample})"
        )

        # Call judgment subagent (same pattern as browse_category._judge_chunk)
        assert self.judgment_subagent is not None
        judgments = self.judgment_subagent.judge_products(query, products_data, state)

        # Validate judgments
        valid_product_ids = {p["product_id"] for p in products_data}
        valid_judgments = []
        hallucinated_ids = []

        for j in judgments:
            if j["product_id"] in valid_product_ids:
                valid_judgments.append(j)
            else:
                hallucinated_ids.append(j["product_id"])

        if hallucinated_ids:
            logger.error(
                f"[Search Chunk {chunk_num}/{total_chunks}] Hallucinated {len(hallucinated_ids)}/{len(judgments)} IDs: "
                f"{hallucinated_ids[:5]}..."
            )

        logger.info(f"[Search Chunk {chunk_num}/{total_chunks}] Added {len(valid_judgments)} valid judgments")

        return valid_judgments, len(hallucinated_ids)

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

        # Handle both step types
        if step.step_type == StepType.SEARCH:
            # Search steps auto-execute, so this message shouldn't normally be reached
            return f'Search step "{step.target}" is pending. It will execute automatically.'

        # Category step - browse_category auto-completes now
        return f'Browse category "{step.target}". Call browse_category(product_class="{step.target}"). Step will auto-complete.'
