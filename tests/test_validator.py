"""Tests for validation subagent and integration with finish tool."""

from unittest.mock import MagicMock, patch

from goldendemo.agent.state import AgentState
from goldendemo.agent.validator import ValidationSubagent, _format_judgments_for_validation
from goldendemo.data.models import AgentJudgment


class TestFormatJudgmentsForValidation:
    """Tests for _format_judgments_for_validation helper."""

    def test_formats_judgment_with_product(self):
        """Test formatting a judgment with product data."""
        judgments_with_products = [
            {
                "judgment": {
                    "product_id": "p1",
                    "relevance": 2,
                    "reasoning": "Exact match for slow cooker",
                },
                "product": {
                    "product_name": "7 Qt Slow Cooker",
                    "product_class": "Slow Cookers",
                    "product_description": "A 7 quart programmable slow cooker.",
                    "product_features": "capacity: 7 quart",
                },
            }
        ]

        result = _format_judgments_for_validation(judgments_with_products)

        assert "PRODUCT_ID: p1" in result
        assert "Relevance: Exact (2)" in result
        assert "7 Qt Slow Cooker" in result
        assert "Slow Cookers" in result
        assert "7 quart programmable" in result

    def test_formats_partial_relevance(self):
        """Test formatting partial relevance judgment."""
        judgments_with_products = [
            {
                "judgment": {
                    "product_id": "p2",
                    "relevance": 1,
                    "reasoning": "Wrong size",
                },
                "product": {
                    "product_name": "6 Qt Slow Cooker",
                    "product_class": "Slow Cookers",
                },
            }
        ]

        result = _format_judgments_for_validation(judgments_with_products)

        assert "Relevance: Partial (1)" in result

    def test_truncates_long_description(self):
        """Test that long descriptions are truncated."""
        long_desc = "x" * 500
        judgments_with_products = [
            {
                "judgment": {"product_id": "p1", "relevance": 2, "reasoning": "Test"},
                "product": {
                    "product_name": "Test Product",
                    "product_class": "Test",
                    "product_description": long_desc,
                },
            }
        ]

        result = _format_judgments_for_validation(judgments_with_products)

        # Should be truncated at 300 chars + "..."
        assert "..." in result
        assert long_desc not in result  # Full description not present


class TestAgentStateRemoveJudgments:
    """Tests for the remove_judgments method on AgentState."""

    def test_remove_judgments_by_id(self):
        """Test removing judgments by product ID."""
        state = AgentState(query="test")
        state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test 1"),
            AgentJudgment(product_id="p2", relevance=1, reasoning="Test 2"),
            AgentJudgment(product_id="p3", relevance=2, reasoning="Test 3"),
        ]

        removed = state.remove_judgments(["p1", "p3"])

        assert removed == 2
        assert len(state.judgments) == 1
        assert state.judgments[0].product_id == "p2"

    def test_remove_judgments_nonexistent_id(self):
        """Test removing nonexistent IDs doesn't error."""
        state = AgentState(query="test")
        state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test"),
        ]

        removed = state.remove_judgments(["p999", "p888"])

        assert removed == 0
        assert len(state.judgments) == 1

    def test_remove_judgments_empty_list(self):
        """Test removing empty list does nothing."""
        state = AgentState(query="test")
        state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test"),
        ]

        removed = state.remove_judgments([])

        assert removed == 0
        assert len(state.judgments) == 1

    def test_remove_judgments_all(self):
        """Test removing all judgments."""
        state = AgentState(query="test")
        state.judgments = [
            AgentJudgment(product_id="p1", relevance=2, reasoning="Test 1"),
            AgentJudgment(product_id="p2", relevance=1, reasoning="Test 2"),
        ]

        removed = state.remove_judgments(["p1", "p2"])

        assert removed == 2
        assert len(state.judgments) == 0


class TestValidationSubagent:
    """Tests for ValidationSubagent."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        with patch("goldendemo.agent.validator.OpenAI"):
            validator = ValidationSubagent()

            assert validator.model == "gpt-5-nano"
            assert validator.reasoning_effort == "medium"

    def test_initialization_custom(self):
        """Test custom initialization values."""
        mock_client = MagicMock()
        validator = ValidationSubagent(
            openai_client=mock_client,
            model="gpt-5",
            reasoning_effort="high",
        )

        assert validator.openai_client == mock_client
        assert validator.model == "gpt-5"
        assert validator.reasoning_effort == "high"

    def test_validate_empty_judgments(self):
        """Test validating empty list returns empty result."""
        mock_client = MagicMock()
        validator = ValidationSubagent(openai_client=mock_client)

        result = validator.validate_judgments("test query", [])

        assert result == {"keep": [], "adjust": [], "remove": []}
        mock_client.responses.create.assert_not_called()

    def test_validate_judgments_success(self):
        """Test successful validation with keep and remove."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.usage = None

        # Create mock function call output
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.name = "submit_validation"
        mock_function_call.arguments = {
            "keep": ["p1", "p2"],
            "remove": [{"product_id": "p3", "reason": "Wrong size"}],
        }
        mock_response.output = [mock_function_call]

        mock_client.responses.create.return_value = mock_response

        validator = ValidationSubagent(openai_client=mock_client)

        judgments_with_products = [
            {
                "judgment": {"product_id": "p1", "relevance": 2, "reasoning": "Test"},
                "product": {"product_name": "Product 1", "product_class": "Test"},
            },
            {
                "judgment": {"product_id": "p2", "relevance": 1, "reasoning": "Test"},
                "product": {"product_name": "Product 2", "product_class": "Test"},
            },
            {
                "judgment": {"product_id": "p3", "relevance": 2, "reasoning": "Test"},
                "product": {"product_name": "Product 3", "product_class": "Test"},
            },
        ]

        result = validator.validate_judgments("test query", judgments_with_products)

        assert result["keep"] == ["p1", "p2"]
        assert len(result["remove"]) == 1
        assert result["remove"][0]["product_id"] == "p3"
        assert result["remove"][0]["reason"] == "Wrong size"

    def test_validate_judgments_tracks_tokens(self):
        """Test that token usage is tracked when state is provided."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.name = "submit_validation"
        mock_function_call.arguments = {"keep": ["p1"], "remove": []}
        mock_response.output = [mock_function_call]

        mock_client.responses.create.return_value = mock_response

        validator = ValidationSubagent(openai_client=mock_client)
        state = AgentState(query="test")

        judgments_with_products = [
            {
                "judgment": {"product_id": "p1", "relevance": 2, "reasoning": "Test"},
                "product": {"product_name": "Product 1", "product_class": "Test"},
            },
        ]

        validator.validate_judgments("test query", judgments_with_products, state)

        assert state.token_usage.input_tokens == 100
        assert state.token_usage.output_tokens == 50

    def test_validate_judgments_retry_on_failure(self):
        """Test that validation retries on tool call failure."""
        mock_client = MagicMock()

        # First call fails (no tool call)
        mock_response_fail = MagicMock()
        mock_response_fail.status = "completed"
        mock_response_fail.usage = None
        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = []
        mock_response_fail.output = [mock_message]

        # Second call succeeds
        mock_response_success = MagicMock()
        mock_response_success.status = "completed"
        mock_response_success.usage = None
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.name = "submit_validation"
        mock_function_call.arguments = {"keep": ["p1"], "remove": []}
        mock_response_success.output = [mock_function_call]

        mock_client.responses.create.side_effect = [mock_response_fail, mock_response_success]

        validator = ValidationSubagent(openai_client=mock_client)

        judgments_with_products = [
            {
                "judgment": {"product_id": "p1", "relevance": 2, "reasoning": "Test"},
                "product": {"product_name": "Product 1", "product_class": "Test"},
            },
        ]

        result = validator.validate_judgments("test query", judgments_with_products)

        assert result["keep"] == ["p1"]
        assert mock_client.responses.create.call_count == 2


class TestFinishToolValidationIntegration:
    """Tests for validation integration in FinishJudgmentsTool."""

    @staticmethod
    def _mock_passing_guardrails():
        """Create mock guardrails that always pass."""
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.warning = None
        mock_result.message = ""

        mock_guardrail = MagicMock()
        mock_guardrail.check.return_value = mock_result

        return {
            "iteration_budget": mock_guardrail,
            "minimum_exploration": mock_guardrail,
            "score_distribution": mock_guardrail,
            "category_browsing": mock_guardrail,
        }

    def test_finish_without_validator(self):
        """Test finish works without validator (validation skipped)."""
        from goldendemo.agent.tools.finish import FinishJudgmentsTool

        mock_weaviate = MagicMock()
        mock_guardrails = self._mock_passing_guardrails()
        tool = FinishJudgmentsTool(mock_weaviate, guardrails=mock_guardrails, validator=None)

        state = AgentState(query="test query")
        state.judgments = [AgentJudgment(product_id=f"p{i}", relevance=2, reasoning="Test") for i in range(10)]

        with patch.object(tool, "_save_golden_set"):
            result = tool.execute(state, overall_reasoning="Test reasoning")

        assert result.success
        validation_result = result.data.get("validation_result", {})
        assert validation_result.get("total_removed", 0) == 0

    def test_finish_with_validator_removes_products(self):
        """Test that validator removes flagged products."""
        from goldendemo.agent.tools.finish import FinishJudgmentsTool

        mock_weaviate = MagicMock()

        # Mock get_by_ids to return products
        mock_products = []
        for i in range(10):
            mock_product = MagicMock()
            mock_product.product_id = f"p{i}"
            mock_product.product_name = f"Product {i}"
            mock_product.product_class = "Test"
            mock_product.product_description = "Test desc"
            mock_product.product_features = ""
            mock_products.append(mock_product)
        mock_weaviate.get_by_ids.return_value = mock_products

        # Mock validator that removes 2 products
        mock_validator = MagicMock()
        mock_validator.validate_judgments.return_value = {
            "keep": [f"p{i}" for i in range(8)],
            "remove": [
                {"product_id": "p8", "reason": "Wrong size"},
                {"product_id": "p9", "reason": "Hallucinated"},
            ],
        }

        mock_guardrails = self._mock_passing_guardrails()
        tool = FinishJudgmentsTool(mock_weaviate, guardrails=mock_guardrails, validator=mock_validator)

        state = AgentState(query="test query")
        state.judgments = [AgentJudgment(product_id=f"p{i}", relevance=2, reasoning="Test") for i in range(10)]

        with patch.object(tool, "_save_golden_set"):
            result = tool.execute(state, overall_reasoning="Test reasoning")

        assert result.success
        validation_result = result.data.get("validation_result", {})
        assert validation_result.get("total_removed") == 2
        assert validation_result.get("exact_removed") == 2  # All were Exact (relevance=2)
        assert validation_result.get("partial_removed") == 0
        assert len(state.judgments) == 8

    def test_finish_validation_error_continues(self):
        """Test that validation errors don't block save (with warning)."""
        from goldendemo.agent.tools.finish import FinishJudgmentsTool

        mock_weaviate = MagicMock()
        mock_weaviate.get_by_ids.side_effect = Exception("Weaviate error")

        mock_validator = MagicMock()

        mock_guardrails = self._mock_passing_guardrails()
        tool = FinishJudgmentsTool(mock_weaviate, guardrails=mock_guardrails, validator=mock_validator)

        state = AgentState(query="test query")
        state.judgments = [AgentJudgment(product_id=f"p{i}", relevance=2, reasoning="Test") for i in range(10)]

        with patch.object(tool, "_save_golden_set"):
            result = tool.execute(state, overall_reasoning="Test reasoning")

        # Should succeed with warning about validation skip
        assert result.success
        assert any("Validation skipped" in w for w in result.data.get("warnings", []))
