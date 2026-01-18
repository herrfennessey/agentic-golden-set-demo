"""Tests for agent tools."""

from unittest.mock import MagicMock

import pytest

from goldendemo.agent.state import AgentState
from goldendemo.agent.tools import (
    BrowseCategoryTool,
    ListCategoriesTool,
    SearchProductsTool,
)
from goldendemo.data.models import CategoryInfo, ProductSummary


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
        mock_subagent = MagicMock()
        tool = BrowseCategoryTool(mock_weaviate_client, judgment_subagent=mock_subagent)

        assert tool.name == "browse_category"
        assert "product_class" in tool.parameters["properties"]

    def test_execute_success(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test successful category browsing with judgments."""
        mock_weaviate_client.get_by_class.return_value = [
            ProductSummary(
                product_id="p1",
                product_name="Sofa 1",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
            ),
        ]

        # Mock the judgment subagent
        mock_subagent = MagicMock()
        mock_subagent.judge_products.return_value = [{"product_id": "p1", "relevance": 2, "reasoning": "Test"}]

        tool = BrowseCategoryTool(mock_weaviate_client, judgment_subagent=mock_subagent)

        result = tool.execute(agent_state, product_class="Sofas")

        assert result.success
        assert result.metadata["product_class"] == "Sofas"
        assert agent_state.exploration_metrics.categories_explored == 1
        # Check that judgments were added
        assert len(agent_state.judgments) == 1
        assert agent_state.judgments[0].product_id == "p1"

    def test_execute_fails_without_class(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that execution fails without product_class."""
        mock_subagent = MagicMock()
        tool = BrowseCategoryTool(mock_weaviate_client, judgment_subagent=mock_subagent)

        result = tool.execute(agent_state)

        assert not result.success

    def test_execute_fails_without_subagent(
        self,
        mock_weaviate_client: MagicMock,
        agent_state: AgentState,
    ):
        """Test that execution fails when judgment_subagent is not provided."""
        mock_weaviate_client.get_by_class.return_value = [
            ProductSummary(
                product_id="p1",
                product_name="Sofa 1",
                product_class="Sofas",
                category_hierarchy="Furniture/Sofas",
            ),
        ]

        tool = BrowseCategoryTool(mock_weaviate_client, judgment_subagent=None)

        result = tool.execute(agent_state, product_class="Sofas")

        assert not result.success
        assert "requires judgment_subagent" in result.error
