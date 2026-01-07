"""Tests for data models."""

import pytest

from goldendemo.data.models import (
    AgentJudgment,
    GoldenSetConfig,
    Product,
    ProductRelevance,
    RelevanceLabel,
)


class TestRelevanceLabel:
    """Tests for RelevanceLabel enum."""

    def test_score_mapping(self):
        """Test label to score mapping."""
        assert RelevanceLabel.EXACT.score == 2
        assert RelevanceLabel.PARTIAL.score == 1
        assert RelevanceLabel.IRRELEVANT.score == 0

    def test_from_score(self):
        """Test score to label mapping."""
        assert RelevanceLabel.from_score(2) == RelevanceLabel.EXACT
        assert RelevanceLabel.from_score(1) == RelevanceLabel.PARTIAL
        assert RelevanceLabel.from_score(0) == RelevanceLabel.IRRELEVANT

    def test_from_score_invalid(self):
        """Test invalid score raises KeyError."""
        with pytest.raises(KeyError):
            RelevanceLabel.from_score(3)


class TestProduct:
    """Tests for Product model."""

    def test_categories_parsing(self):
        """Test category hierarchy parsing."""
        product = Product(
            product_id="123",
            product_name="Test Product",
            product_class="Sofas",
            category_hierarchy="Furniture / Living Room / Sofas",
        )
        assert product.categories == ["Furniture", "Living Room", "Sofas"]

    def test_categories_empty(self):
        """Test empty category hierarchy."""
        product = Product(
            product_id="123",
            product_name="Test Product",
            product_class="Sofas",
            category_hierarchy="",
        )
        assert product.categories == []

    def test_features_parsing(self):
        """Test product features parsing."""
        product = Product(
            product_id="123",
            product_name="Test Product",
            product_class="Sofas",
            category_hierarchy="Furniture",
            product_features="Color:Blue|Material:Velvet|Size:Large",
        )
        features = product.features_dict
        assert features["Color"] == "Blue"
        assert features["Material"] == "Velvet"
        assert features["Size"] == "Large"

    def test_features_empty(self):
        """Test empty product features."""
        product = Product(
            product_id="123",
            product_name="Test Product",
            product_class="Sofas",
            category_hierarchy="Furniture",
        )
        assert product.features_dict == {}

    def test_to_summary(self):
        """Test conversion to summary."""
        product = Product(
            product_id="123",
            product_name="Test Product",
            product_class="Sofas",
            category_hierarchy="Furniture / Living Room / Sofas",
            product_description="A nice sofa for your living room.",
        )
        summary = product.to_summary()
        assert summary.product_id == "123"
        assert summary.product_name == "Test Product"
        assert summary.product_class == "Sofas"
        assert summary.description_snippet == "A nice sofa for your living room."


class TestProductRelevance:
    """Tests for ProductRelevance model."""

    def test_valid_relevance(self):
        """Test valid relevance scores."""
        for score in [0, 1, 2]:
            pr = ProductRelevance(product_id="123", relevance=score)
            assert pr.relevance == score

    def test_invalid_relevance(self):
        """Test invalid relevance score raises error."""
        with pytest.raises(ValueError):
            ProductRelevance(product_id="123", relevance=3)

    def test_label_property(self):
        """Test label property returns correct label."""
        pr = ProductRelevance(product_id="123", relevance=2)
        assert pr.label == RelevanceLabel.EXACT


class TestAgentJudgment:
    """Tests for AgentJudgment model."""

    def test_with_confidence_and_reasoning(self):
        """Test judgment with all fields."""
        judgment = AgentJudgment(
            product_id="123",
            relevance=2,
            confidence=0.95,
            reasoning="Product matches exactly",
        )
        assert judgment.product_id == "123"
        assert judgment.relevance == 2
        assert judgment.confidence == 0.95
        assert judgment.reasoning == "Product matches exactly"
        assert judgment.label == RelevanceLabel.EXACT

    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            AgentJudgment(product_id="123", relevance=1, confidence=1.5)

        with pytest.raises(ValueError):
            AgentJudgment(product_id="123", relevance=1, confidence=-0.1)


class TestGoldenSetConfig:
    """Tests for GoldenSetConfig model."""

    def test_get_relevance_map(self):
        """Test relevance map generation."""
        config = GoldenSetConfig(
            query_id="Q1",
            query="test query",
            products=[
                ProductRelevance(product_id="P1", relevance=2),
                ProductRelevance(product_id="P2", relevance=1),
                ProductRelevance(product_id="P3", relevance=0),
            ],
        )
        rmap = config.get_relevance_map()
        assert rmap == {"P1": 2, "P2": 1, "P3": 0}

    def test_get_label_map(self):
        """Test label map generation."""
        config = GoldenSetConfig(
            query_id="Q1",
            query="test query",
            products=[
                ProductRelevance(product_id="P1", relevance=2),
                ProductRelevance(product_id="P2", relevance=1),
            ],
        )
        lmap = config.get_label_map()
        assert lmap["P1"] == RelevanceLabel.EXACT
        assert lmap["P2"] == RelevanceLabel.PARTIAL
