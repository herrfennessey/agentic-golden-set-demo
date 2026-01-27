"""Agent state management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from goldendemo.data.models import AgentJudgment, CategoryInfo, ProductSummary, RelevanceScore


class StepType(str, Enum):
    """Type of plan step."""

    SEARCH = "search"
    CATEGORY = "category"


@dataclass
class TokenUsage:
    """Token usage tracking for cost monitoring."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, usage_data: dict) -> None:
        """Add usage from an OpenAI API response.

        Args:
            usage_data: The usage object from response.usage
        """
        self.input_tokens += usage_data.get("input_tokens", 0)
        self.output_tokens += usage_data.get("output_tokens", 0)
        self.total_tokens += usage_data.get("total_tokens", 0)

        # Add detailed breakdowns if available
        input_details = usage_data.get("input_tokens_details", {})
        self.cached_tokens += input_details.get("cached_tokens", 0)

        output_details = usage_data.get("output_tokens_details", {})
        self.reasoning_tokens += output_details.get("reasoning_tokens", 0)

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ExplorationMetrics:
    """Metrics tracking agent's exploration of the catalog."""

    unique_products_seen: int = 0
    search_queries_executed: int = 0
    categories_explored: int = 0
    product_details_retrieved: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "unique_products_seen": self.unique_products_seen,
            "search_queries_executed": self.search_queries_executed,
            "categories_explored": self.categories_explored,
            "product_details_retrieved": self.product_details_retrieved,
        }


@dataclass
class SearchRecord:
    """Record of a search query executed by the agent."""

    query: str
    result_count: int
    alpha: float = 0.5


@dataclass
class PlanStep:
    """A step in the agent's execution plan."""

    step_type: StepType
    target: str  # Query for SEARCH, category name for CATEGORY
    reason: str
    status: str = "pending"  # pending, in_progress, complete
    summary: str | None = None
    products_processed: int = 0
    judgments_added: int = 0

    @property
    def category(self) -> str:
        """Backward compatibility: return target for category steps."""
        return self.target


@dataclass
class AgentState:
    """Current state of the agent during golden set generation.

    Tracks exploration progress, candidate products, and judgments.
    """

    # Query context
    query: str
    query_id: str | None = None

    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 20

    # Exploration tracking
    exploration_metrics: ExplorationMetrics = field(default_factory=ExplorationMetrics)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    available_classes: list[CategoryInfo] = field(default_factory=list)
    browsed_categories: list[str] = field(default_factory=list)
    search_history: list[SearchRecord] = field(default_factory=list)

    # Products seen and candidates
    seen_products: dict[str, ProductSummary] = field(default_factory=dict)
    candidate_products: dict[str, Any] = field(default_factory=dict)  # Full Product objects

    # Judgments
    judgments: list[AgentJudgment] = field(default_factory=list)
    final_reasoning: str = ""

    # Feedback and history
    guardrail_feedback: list[str] = field(default_factory=list)
    tool_call_history: list[str] = field(default_factory=list)
    last_tool_result: Any = None

    # Planning fields
    plan: list[PlanStep] = field(default_factory=list)
    plan_submitted: bool = False
    current_step_index: int = 0

    def add_seen_products(self, products: list[ProductSummary]) -> int:
        """Add products to seen set, returns count of new products."""
        new_count = 0
        for product in products:
            if product.product_id not in self.seen_products:
                self.seen_products[product.product_id] = product
                new_count += 1
        self.exploration_metrics.unique_products_seen = len(self.seen_products)
        return new_count

    def record_search(self, query: str, result_count: int, alpha: float = 0.5) -> None:
        """Record a search query execution."""
        self.search_history.append(SearchRecord(query=query, result_count=result_count, alpha=alpha))
        # Count unique query strings (re-searching same query with different params doesn't count)
        unique_queries = {s.query.lower().strip() for s in self.search_history}
        self.exploration_metrics.search_queries_executed = len(unique_queries)

    def record_category_browse(self, category: str) -> None:
        """Record a category browse."""
        if category not in self.browsed_categories:
            self.browsed_categories.append(category)
            self.exploration_metrics.categories_explored = len(self.browsed_categories)

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.tool_call_history.append(tool_name)

    def add_judgment(self, judgment: AgentJudgment) -> bool:
        """Add a judgment for a new product. Skips if product already judged.

        Args:
            judgment: The judgment to add.

        Returns:
            True if judgment was added, False if product was already judged.
        """
        # Check if product already has a judgment - skip if so
        existing = next((j for j in self.judgments if j.product_id == judgment.product_id), None)

        if existing:
            # Product already judged - skip (don't re-judge during exploration)
            return False

        # New product, add it
        self.judgments.append(judgment)
        return True

    def update_judgment(self, product_id: str, new_relevance: RelevanceScore, new_reasoning: str | None = None) -> bool:
        """Update an existing judgment's relevance (used by validation phase).

        Args:
            product_id: Product to update.
            new_relevance: New relevance score (0=Irrelevant, 1=Partial, 2=Exact).
            new_reasoning: Optional new reasoning.

        Returns:
            True if judgment was updated, False if product not found.
        """
        for j in self.judgments:
            if j.product_id == product_id:
                j.relevance = new_relevance
                if new_reasoning:
                    j.reasoning = new_reasoning
                return True
        return False

    def remove_judgments(self, product_ids: list[str]) -> int:
        """Remove judgments by product ID.

        Args:
            product_ids: List of product IDs to remove.

        Returns:
            Number of judgments actually removed.
        """
        original_count = len(self.judgments)
        ids_to_remove = set(product_ids)
        self.judgments = [j for j in self.judgments if j.product_id not in ids_to_remove]
        return original_count - len(self.judgments)

    def add_judgments_from_dicts(self, judgments_data: list[dict]) -> int:
        """Add judgments from dict format (from subagent).

        Args:
            judgments_data: List of judgment dicts with product_id, relevance, reasoning.

        Returns:
            Number of unique products judged (deduplicates by product_id).
        """
        # Track which product IDs we're adding
        product_ids_in_batch = set()

        for j in judgments_data:
            product_id = str(j["product_id"])
            product_ids_in_batch.add(product_id)

            judgment = AgentJudgment(
                product_id=product_id,
                relevance=j["relevance"],
                reasoning=j["reasoning"],
                confidence=j.get("confidence", 1.0),
            )
            self.add_judgment(judgment)

        # Return count of unique products judged in this batch
        return len(product_ids_in_batch)

    def judgment_counts_by_level(self) -> dict[int, int]:
        """Get count of judgments by relevance level."""
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for j in self.judgments:
            counts[j.relevance] = counts.get(j.relevance, 0) + 1
        return counts

    def add_guardrail_feedback(self, feedback: str) -> None:
        """Add guardrail feedback for the agent to see."""
        self.guardrail_feedback.append(feedback)

    def clear_guardrail_feedback(self) -> None:
        """Clear guardrail feedback for new iteration."""
        self.guardrail_feedback = []

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of current state for prompt injection."""
        return {
            "query": self.query,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "products_seen": self.exploration_metrics.unique_products_seen,
            "searches_executed": self.exploration_metrics.search_queries_executed,
            "categories_explored": self.exploration_metrics.categories_explored,
            "judgments_count": len(self.judgments),
            "guardrail_feedback": self.guardrail_feedback,
        }

    # Plan management methods

    def set_plan(self, steps: list[dict]) -> None:
        """Set the execution plan from submitted steps.

        Steps are reordered so search steps execute first, then categories.
        """
        parsed = []
        for s in steps:
            step_type = StepType(s.get("type", "category"))
            # Use 'query' for search steps, 'category' for category steps
            if step_type == StepType.SEARCH:
                target = s.get("query", "")
            else:
                target = s.get("category", "")
            parsed.append(PlanStep(step_type=step_type, target=target, reason=s["reason"]))

        # Reorder: search steps first, then categories
        search_steps = [s for s in parsed if s.step_type == StepType.SEARCH]
        category_steps = [s for s in parsed if s.step_type == StepType.CATEGORY]
        self.plan = search_steps + category_steps

        self.plan_submitted = True
        self.current_step_index = 0
        if self.plan:
            self.plan[0].status = "in_progress"

    def get_current_step(self) -> PlanStep | None:
        """Get the current plan step being executed."""
        if not self.plan or self.current_step_index >= len(self.plan):
            return None
        return self.plan[self.current_step_index]

    def complete_current_step(self, summary: str) -> bool:
        """Mark the current step as complete and advance to next.

        Returns True if there are more steps, False if plan is complete.
        """
        if not self.plan or self.current_step_index >= len(self.plan):
            return False

        # Complete current step
        self.plan[self.current_step_index].status = "complete"
        self.plan[self.current_step_index].summary = summary

        # Move to next step
        self.current_step_index += 1

        # Mark next step as in_progress if exists
        if self.current_step_index < len(self.plan):
            self.plan[self.current_step_index].status = "in_progress"
            return True
        return False

    def is_plan_complete(self) -> bool:
        """Check if all plan steps are complete."""
        return self.plan_submitted and all(s.status == "complete" for s in self.plan)

    def update_step_progress(self, products_count: int, judgments_count: int = 0) -> None:
        """Update the current step's progress counters.

        Args:
            products_count: Number of products processed to add.
            judgments_count: Number of judgments added to add.
        """
        step = self.get_current_step()
        if step:
            step.products_processed += products_count
            step.judgments_added += judgments_count

    def format_plan_summary(self) -> str:
        """Format the plan for prompt injection."""
        if not self.plan:
            return "No plan submitted yet."

        lines = []
        for i, step in enumerate(self.plan):
            if step.status == "complete":
                marker = "✅"
            elif step.status == "in_progress":
                marker = "🔄"
            else:
                marker = "⏳"

            # Format step description based on type
            if step.step_type == StepType.SEARCH:
                action = f'Search "{step.target}"'
            else:
                action = f'Browse "{step.target}"'

            line = f"{marker} Step {i + 1}: {action} - {step.reason}"
            if step.status == "complete" and step.summary:
                line += f"\n   Summary: {step.summary}"
            elif step.status == "in_progress":
                line += f"\n   Progress: {step.products_processed} products, {step.judgments_added} judgments"
            lines.append(line)

        return "\n".join(lines)
