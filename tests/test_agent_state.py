"""Tests for agent state management."""

from goldendemo.agent.state import AgentState, ExplorationMetrics, SearchRecord
from goldendemo.data.models import AgentJudgment, ProductSummary


class TestExplorationMetrics:
    """Tests for ExplorationMetrics."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ExplorationMetrics()

        assert metrics.unique_products_seen == 0
        assert metrics.search_queries_executed == 0
        assert metrics.categories_explored == 0
        assert metrics.product_details_retrieved == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ExplorationMetrics(
            unique_products_seen=50,
            search_queries_executed=3,
            categories_explored=2,
            product_details_retrieved=10,
        )

        result = metrics.to_dict()

        assert result["unique_products_seen"] == 50
        assert result["search_queries_executed"] == 3
        assert result["categories_explored"] == 2
        assert result["product_details_retrieved"] == 10


class TestAgentState:
    """Tests for AgentState."""

    def test_initialization(self):
        """Test state initialization."""
        state = AgentState(query="blue sofa", query_id="q123")

        assert state.query == "blue sofa"
        assert state.query_id == "q123"
        assert state.iteration == 0
        assert len(state.judgments) == 0
        assert len(state.seen_products) == 0

    def test_add_seen_products(self):
        """Test adding products to seen set."""
        state = AgentState(query="test")
        products = [
            ProductSummary(
                product_id="p1",
                product_name="Product 1",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
            ),
            ProductSummary(
                product_id="p2",
                product_name="Product 2",
                product_class="Tables",
                category_hierarchy="Furniture/Tables",
            ),
        ]

        new_count = state.add_seen_products(products)

        assert new_count == 2
        assert len(state.seen_products) == 2
        assert state.exploration_metrics.unique_products_seen == 2

    def test_add_seen_products_deduplicates(self):
        """Test that duplicate products are not counted twice."""
        state = AgentState(query="test")
        product = ProductSummary(
            product_id="p1",
            product_name="Product 1",
            product_class="Sofas",
            category_hierarchy="Furniture/Sofas",
        )

        state.add_seen_products([product])
        new_count = state.add_seen_products([product])  # Add same product again

        assert new_count == 0
        assert len(state.seen_products) == 1
        assert state.exploration_metrics.unique_products_seen == 1

    def test_record_search(self):
        """Test recording search queries."""
        state = AgentState(query="test")

        state.record_search("blue sofa", 25, alpha=0.5)
        state.record_search("velvet couch", 15, alpha=0.7)

        assert len(state.search_history) == 2
        assert state.exploration_metrics.search_queries_executed == 2
        assert state.search_history[0].query == "blue sofa"
        assert state.search_history[0].result_count == 25
        assert state.search_history[1].alpha == 0.7

    def test_record_category_browse(self):
        """Test recording category browsing."""
        state = AgentState(query="test")

        state.record_category_browse("Sofas")
        state.record_category_browse("Tables")
        state.record_category_browse("Sofas")  # Duplicate

        assert len(state.browsed_categories) == 2
        assert state.exploration_metrics.categories_explored == 2
        assert "Sofas" in state.browsed_categories
        assert "Tables" in state.browsed_categories

    def test_record_tool_call(self):
        """Test recording tool calls."""
        state = AgentState(query="test")

        state.record_tool_call("search_products")
        state.record_tool_call("browse_category")

        assert len(state.tool_call_history) == 2
        assert state.tool_call_history[0] == "search_products"

    def test_add_judgment(self):
        """Test adding judgments."""
        state = AgentState(query="test")
        judgment = AgentJudgment(
            product_id="p1",
            relevance=2,
            reasoning="Test reasoning",
        )

        state.add_judgment(judgment)

        assert len(state.judgments) == 1
        assert state.judgments[0].product_id == "p1"

    def test_add_judgment_replaces_existing(self):
        """Test that adding judgment for same product replaces existing."""
        state = AgentState(query="test")
        judgment1 = AgentJudgment(
            product_id="p1",
            relevance=2,
            reasoning="First reasoning",
        )
        judgment2 = AgentJudgment(
            product_id="p1",
            relevance=1,
            reasoning="Updated reasoning",
        )

        state.add_judgment(judgment1)
        state.add_judgment(judgment2)

        assert len(state.judgments) == 1
        assert state.judgments[0].relevance == 1
        assert state.judgments[0].reasoning == "Updated reasoning"

    def test_clear_guardrail_feedback(self):
        """Test clearing guardrail feedback."""
        state = AgentState(query="test")
        state.guardrail_feedback = ["Error 1", "Error 2"]

        state.clear_guardrail_feedback()

        assert len(state.guardrail_feedback) == 0

    def test_get_state_summary(self):
        """Test getting state summary."""
        state = AgentState(query="blue sofa", max_iterations=20)
        state.iteration = 5
        state.exploration_metrics = ExplorationMetrics(
            unique_products_seen=30,
            search_queries_executed=2,
            categories_explored=1,
        )
        state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test"),
        ]

        summary = state.get_state_summary()

        assert summary["query"] == "blue sofa"
        assert summary["iteration"] == 5
        assert summary["max_iterations"] == 20
        assert summary["products_seen"] == 30
        assert summary["searches_executed"] == 2
        assert summary["categories_explored"] == 1
        assert summary["judgments_count"] == 1


class TestSearchRecord:
    """Tests for SearchRecord."""

    def test_creation(self):
        """Test search record creation."""
        record = SearchRecord(query="test query", result_count=25, alpha=0.6)

        assert record.query == "test query"
        assert record.result_count == 25
        assert record.alpha == 0.6

    def test_default_alpha(self):
        """Test default alpha value."""
        record = SearchRecord(query="test", result_count=10)

        assert record.alpha == 0.5
