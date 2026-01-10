"""Tests for agent tools."""

from unittest.mock import MagicMock

import pytest

from goldendemo.agent.state import AgentState
from goldendemo.agent.tools import (
    BrowseCategoryTool,
    GetProductDetailsTool,
    ListCategoriesTool,
    SearchProductsTool,
    SubmitJudgmentsTool,
)
from goldendemo.data.models import CategoryInfo, Product, ProductSummary


@pytest.fixture
def mock_weaviate_client() -> MagicMock:
    """Create a mock Weaviate client."""
    return MagicMock()


@pytest.fixture
def agent_state() -> AgentState:
    """Create an agent state for testing."""
    return AgentState(query="blue sofa", query_id="q123")


class TestSearchProductsTool:
    """Tests for SearchProductsTool."""

    def test_tool_properties(self, mock_weaviate_client: MagicMock):
        """Test tool name and description."""
        tool = SearchProductsTool(mock_weaviate_client)

        assert tool.name == "search_products"
        assert "search" in tool.description.lower()
        assert "query" in tool.parameters["properties"]

    def test_execute_success(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test successful search execution."""
        mock_weaviate_client.hybrid_search.return_value = [
            ProductSummary(
                product_id="p1",
                product_name="Blue Velvet Sofa",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
                product_description="A beautiful blue sofa",
                product_features="Color:Blue|Material:Velvet",
            ),
        ]
        tool = SearchProductsTool(mock_weaviate_client)

        result = tool.execute(agent_state, query="blue sofa", limit=10)

        assert result.success
        assert len(result.data) == 1
        assert result.data[0]["product_id"] == "p1"
        assert result.metadata["result_count"] == 1
        mock_weaviate_client.hybrid_search.assert_called_once_with(
            query="blue sofa",
            limit=10,
            alpha=0.5,
        )

    def test_execute_updates_state(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that execution updates agent state."""
        mock_weaviate_client.hybrid_search.return_value = [
            ProductSummary(
                product_id="p1",
                product_name="Product 1",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
            ),
        ]
        tool = SearchProductsTool(mock_weaviate_client)

        tool.execute(agent_state, query="test")

        assert agent_state.exploration_metrics.unique_products_seen == 1
        assert agent_state.exploration_metrics.search_queries_executed == 1
        assert "search_products" in agent_state.tool_call_history

    def test_execute_fails_without_query(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that execution fails without query."""
        tool = SearchProductsTool(mock_weaviate_client)

        result = tool.execute(agent_state)

        assert not result.success
        assert "required" in result.error.lower()

    def test_execute_limits_results(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that limit is capped at 500."""
        mock_weaviate_client.hybrid_search.return_value = []
        tool = SearchProductsTool(mock_weaviate_client)

        tool.execute(agent_state, query="test", limit=600)

        mock_weaviate_client.hybrid_search.assert_called_with(
            query="test",
            limit=500,  # Capped at max
            alpha=0.5,
        )


class TestListCategoriesTool:
    """Tests for ListCategoriesTool."""

    def test_tool_properties(self, mock_weaviate_client: MagicMock):
        """Test tool name and description."""
        tool = ListCategoriesTool(mock_weaviate_client)

        assert tool.name == "list_categories"
        assert "categor" in tool.description.lower()

    def test_execute_success(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test successful category listing."""
        mock_weaviate_client.get_all_classes.return_value = [
            CategoryInfo(
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
                count=500,
            ),
            CategoryInfo(
                product_class="Tables",
                category_hierarchy="Furniture/Tables",
                count=300,
            ),
        ]
        tool = ListCategoriesTool(mock_weaviate_client)

        result = tool.execute(agent_state)

        assert result.success
        assert len(result.data) == 2
        assert result.metadata["total_categories"] == 2


class TestBrowseCategoryTool:
    """Tests for BrowseCategoryTool."""

    def test_tool_properties(self, mock_weaviate_client: MagicMock):
        """Test tool name and description."""
        tool = BrowseCategoryTool(mock_weaviate_client)

        assert tool.name == "browse_category"
        assert "product_class" in tool.parameters["properties"]

    def test_execute_success(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test successful category browsing."""
        mock_weaviate_client.get_by_class.return_value = [
            ProductSummary(
                product_id="p1",
                product_name="Sofa 1",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
            ),
        ]
        tool = BrowseCategoryTool(mock_weaviate_client)

        result = tool.execute(agent_state, product_class="Sofas")

        assert result.success
        assert result.metadata["product_class"] == "Sofas"
        assert agent_state.exploration_metrics.categories_explored == 1

    def test_execute_fails_without_class(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that execution fails without product_class."""
        tool = BrowseCategoryTool(mock_weaviate_client)

        result = tool.execute(agent_state)

        assert not result.success


class TestGetProductDetailsTool:
    """Tests for GetProductDetailsTool."""

    def test_tool_properties(self, mock_weaviate_client: MagicMock):
        """Test tool name and description."""
        tool = GetProductDetailsTool(mock_weaviate_client)

        assert tool.name == "get_product_details"
        assert "product_ids" in tool.parameters["properties"]

    def test_execute_success(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test successful product details retrieval."""
        mock_weaviate_client.get_by_ids.return_value = [
            Product(
                product_id="p1",
                product_name="Blue Sofa",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
                product_description="A nice blue sofa",
            ),
        ]
        tool = GetProductDetailsTool(mock_weaviate_client)

        result = tool.execute(agent_state, product_ids=["p1"])

        assert result.success
        assert result.data[0]["product_description"] == "A nice blue sofa"
        assert agent_state.exploration_metrics.product_details_retrieved == 1

    def test_execute_caches_products(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that products are cached in state."""
        product = Product(
            product_id="p1",
            product_name="Blue Sofa",
            product_class="Sofas",
            category_hierarchy="Furniture/Sofas",
        )
        mock_weaviate_client.get_by_ids.return_value = [product]
        tool = GetProductDetailsTool(mock_weaviate_client)

        tool.execute(agent_state, product_ids=["p1"])

        assert "p1" in agent_state.candidate_products


class TestSubmitJudgmentsTool:
    """Tests for SubmitJudgmentsTool."""

    def test_tool_properties(self, mock_weaviate_client: MagicMock):
        """Test tool name and description."""
        tool = SubmitJudgmentsTool(mock_weaviate_client)

        assert tool.name == "submit_judgments"
        assert "judgments" in tool.parameters["properties"]

    def test_execute_fails_without_judgments(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that execution fails without judgments."""
        tool = SubmitJudgmentsTool(mock_weaviate_client)

        result = tool.execute(agent_state)

        assert not result.success
        assert "required" in result.error.lower()

    def test_execute_accumulates_judgments(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that judgments accumulate across multiple calls."""
        tool = SubmitJudgmentsTool(mock_weaviate_client)

        # First submission
        result1 = tool.execute(
            agent_state,
            judgments=[{"product_id": "p1", "relevance": 2, "reasoning": "Test"}],
        )
        assert result1.success
        assert result1.data["total"] == 1

        # Second submission
        result2 = tool.execute(
            agent_state,
            judgments=[{"product_id": "p2", "relevance": 1, "reasoning": "Test"}],
        )
        assert result2.success
        assert result2.data["total"] == 2

        # Verify state has both judgments
        assert len(agent_state.judgments) == 2

    def test_execute_deduplicates_by_product_id(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that submitting same product_id updates the judgment."""
        tool = SubmitJudgmentsTool(mock_weaviate_client)

        # First submission
        tool.execute(
            agent_state,
            judgments=[{"product_id": "p1", "relevance": 2, "reasoning": "First"}],
        )

        # Submit same product with different relevance
        result = tool.execute(
            agent_state,
            judgments=[{"product_id": "p1", "relevance": 1, "reasoning": "Updated"}],
        )

        assert result.success
        assert result.data["total"] == 1  # Still only 1 judgment
        assert agent_state.judgments[0].relevance == 1  # Updated to new value

    def test_to_openai_schema(self, mock_weaviate_client: MagicMock):
        """Test OpenAI schema generation."""
        tool = SubmitJudgmentsTool(mock_weaviate_client)

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["name"] == "submit_judgments"
        assert "parameters" in schema
