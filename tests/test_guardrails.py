"""Tests for agent guardrails."""

import pytest

from goldendemo.agent.guardrails import (
    GuardrailAction,
    IterationBudgetGuardrail,
    MinimumExplorationGuardrail,
    ScoreDistributionGuardrail,
)
from goldendemo.agent.state import AgentState, ExplorationMetrics
from goldendemo.data.models import AgentJudgment


@pytest.fixture
def empty_state() -> AgentState:
    """Create an empty agent state."""
    return AgentState(query="test query")


@pytest.fixture
def explored_state() -> AgentState:
    """Create a state with adequate exploration."""
    state = AgentState(query="test query")
    state.exploration_metrics = ExplorationMetrics(
        unique_products_seen=50,
        search_queries_executed=3,
        categories_explored=2,
        product_details_retrieved=10,
    )
    state.iteration = 5
    return state


class TestIterationBudgetGuardrail:
    """Tests for IterationBudgetGuardrail."""

    def test_passes_within_budget(self, empty_state: AgentState):
        """Test that check passes when within budget."""
        guardrail = IterationBudgetGuardrail(max_iterations=20, warn_at_iteration=15)
        empty_state.iteration = 5

        result = guardrail.check(empty_state)

        assert result.passed
        assert result.warning is None

    def test_warns_near_limit(self, empty_state: AgentState):
        """Test that check warns when approaching limit."""
        guardrail = IterationBudgetGuardrail(max_iterations=20, warn_at_iteration=15)
        empty_state.iteration = 16

        result = guardrail.check(empty_state)

        assert result.passed
        assert result.warning is not None
        assert "16/20" in result.warning

    def test_fails_at_max(self, empty_state: AgentState):
        """Test that check fails at max iterations."""
        guardrail = IterationBudgetGuardrail(max_iterations=20)
        empty_state.iteration = 20

        result = guardrail.check(empty_state)

        assert not result.passed
        assert "Maximum iterations" in str(result.message)

    def test_blocks_early_submission(self, empty_state: AgentState):
        """Test that submission is blocked before minimum iterations."""
        guardrail = IterationBudgetGuardrail(min_iterations_before_submit=3)
        empty_state.iteration = 2

        result = guardrail.check(empty_state, is_submission=True)

        assert not result.passed
        assert result.action == GuardrailAction.BLOCK
        assert "Cannot submit before exploring" in str(result.message)

    def test_allows_submission_after_minimum(self, empty_state: AgentState):
        """Test that submission is allowed after minimum iterations."""
        guardrail = IterationBudgetGuardrail(min_iterations_before_submit=3)
        empty_state.iteration = 5

        result = guardrail.check(empty_state, is_submission=True)

        assert result.passed


class TestMinimumExplorationGuardrail:
    """Tests for MinimumExplorationGuardrail."""

    def test_fails_with_no_exploration(self, empty_state: AgentState):
        """Test that check fails with no exploration."""
        guardrail = MinimumExplorationGuardrail(
            min_products_seen=30,
            min_search_queries=2,
            min_categories_explored=1,
        )

        result = guardrail.check(empty_state)

        assert not result.passed
        assert "products" in str(result.message).lower()

    def test_fails_with_partial_exploration(self, empty_state: AgentState):
        """Test that check fails with insufficient exploration."""
        guardrail = MinimumExplorationGuardrail(
            min_products_seen=30,
            min_search_queries=2,
            min_categories_explored=1,
        )
        empty_state.exploration_metrics = ExplorationMetrics(
            unique_products_seen=20,  # Not enough
            search_queries_executed=3,
            categories_explored=2,
        )

        result = guardrail.check(empty_state)

        assert not result.passed
        assert "20/30" in str(result.message)  # Shows current/required

    def test_passes_with_adequate_exploration(self, explored_state: AgentState):
        """Test that check passes with adequate exploration."""
        guardrail = MinimumExplorationGuardrail(
            min_products_seen=30,
            min_search_queries=2,
            min_categories_explored=1,
        )

        result = guardrail.check(explored_state)

        assert result.passed

    def test_all_requirements_checked(self, empty_state: AgentState):
        """Test that all requirements are reported when missing."""
        guardrail = MinimumExplorationGuardrail()

        result = guardrail.check(empty_state)

        assert not result.passed
        # Should mention all three issues
        message = str(result.message)
        assert "products" in message.lower()
        assert "search" in message.lower()
        assert "categor" in message.lower()


class TestScoreDistributionGuardrail:
    """Tests for ScoreDistributionGuardrail."""

    def test_fails_with_too_few_judgments(self, empty_state: AgentState):
        """Test that check fails with too few judgments."""
        guardrail = ScoreDistributionGuardrail(min_total=5)
        empty_state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test reasoning here"),
            AgentJudgment(product_id="p2", relevance=1, reasoning="Test reasoning here"),
        ]

        result = guardrail.check(empty_state)

        assert not result.passed
        assert "Too few judgments" in str(result.message) or "Need at least" in str(result.message)

    def test_fails_with_only_partial_scores(self, empty_state: AgentState):
        """Test that check fails when only Partial scores are present (no Exact)."""
        guardrail = ScoreDistributionGuardrail(min_exact=5, min_total=10)
        empty_state.judgments = [
            AgentJudgment(product_id=f"p{i}", relevance=1, reasoning="Test reasoning here") for i in range(10)
        ]

        result = guardrail.check(empty_state)

        assert not result.passed
        assert "Exact" in str(result.message)

    def test_passes_with_exact_and_partial(self, empty_state: AgentState):
        """Test that check passes with Exact and Partial judgments."""
        guardrail = ScoreDistributionGuardrail(min_exact=2, min_total=5)
        empty_state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test reasoning here"),
            AgentJudgment(product_id="p2", relevance=2, reasoning="Test reasoning here"),
            AgentJudgment(product_id="p3", relevance=1, reasoning="Test reasoning here"),
            AgentJudgment(product_id="p4", relevance=1, reasoning="Test reasoning here"),
            AgentJudgment(product_id="p5", relevance=1, reasoning="Test reasoning here"),
        ]

        result = guardrail.check(empty_state)

        assert result.passed

    def test_passes_with_minimum_thresholds(self, empty_state: AgentState):
        """Test that check passes when minimum thresholds are met."""
        guardrail = ScoreDistributionGuardrail(min_exact=2, min_total=5)
        empty_state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Exact match"),
            AgentJudgment(product_id="p2", relevance=2, reasoning="Exact match"),
            AgentJudgment(product_id="p3", relevance=1, reasoning="Partial match"),
            AgentJudgment(product_id="p4", relevance=1, reasoning="Partial match"),
            AgentJudgment(product_id="p5", relevance=1, reasoning="Partial match"),
        ]

        result = guardrail.check(empty_state)

        assert result.passed

    def test_passes_with_only_exact_scores(self, empty_state: AgentState):
        """Test that check passes with only Exact scores (Partial not required)."""
        guardrail = ScoreDistributionGuardrail(min_exact=5, min_total=10)
        # All Exact scores - should pass since Partial is not required
        empty_state.judgments = [
            AgentJudgment(product_id=f"p{i}", relevance=2, reasoning="Test reasoning here") for i in range(10)
        ]

        result = guardrail.check(empty_state)

        assert result.passed
