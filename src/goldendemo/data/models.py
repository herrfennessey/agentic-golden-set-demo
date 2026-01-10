"""
Data models for the WANDS dataset and golden set generation.

WANDS uses a 3-level relevance scale:
- Exact (2): Product is exactly what user wants
- Partial (1): Product is somewhat relevant
- Irrelevant (0): Product does not match user intent
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RelevanceLabel(str, Enum):
    """WANDS relevance labels."""

    EXACT = "Exact"
    PARTIAL = "Partial"
    IRRELEVANT = "Irrelevant"

    @property
    def score(self) -> int:
        """Convert label to numeric score (0-2)."""
        return {"Exact": 2, "Partial": 1, "Irrelevant": 0}[self.value]

    @classmethod
    def from_score(cls, score: int) -> "RelevanceLabel":
        """Convert numeric score to label."""
        mapping = {2: cls.EXACT, 1: cls.PARTIAL, 0: cls.IRRELEVANT}
        return mapping[score]


# Type alias for relevance scores
RelevanceScore = Literal[0, 1, 2]


class Product(BaseModel):
    """A product from the WANDS dataset."""

    product_id: str
    product_name: str
    product_class: str
    category_hierarchy: str
    product_description: str = ""
    product_features: str = ""
    rating_count: int = 0
    average_rating: float = 0.0
    review_count: int = 0

    @property
    def categories(self) -> list[str]:
        """Parse category hierarchy into list."""
        if not self.category_hierarchy:
            return []
        return [c.strip() for c in self.category_hierarchy.split("/")]

    @property
    def features_dict(self) -> dict[str, str]:
        """Parse product features into dictionary."""
        features: dict[str, str] = {}
        if not self.product_features:
            return features
        for pair in self.product_features.split("|"):
            if ":" in pair:
                key, value = pair.split(":", 1)
                features[key.strip()] = value.strip()
        return features

    def to_summary(self) -> "ProductSummary":
        """Convert to summary for agent tools."""
        return ProductSummary(
            product_id=self.product_id,
            product_name=self.product_name,
            product_class=self.product_class,
            category_hierarchy=self.category_hierarchy,
            product_description=self.product_description,
            product_features=self.product_features,
        )


class ProductSummary(BaseModel):
    """Product data for agent tool responses."""

    product_id: str
    product_name: str
    product_class: str
    category_hierarchy: str
    product_description: str = ""
    product_features: str = ""


class Query(BaseModel):
    """A search query from the WANDS dataset."""

    query_id: str
    query: str
    query_class: str


class WANDSLabel(BaseModel):
    """A ground truth relevance label from WANDS."""

    id: str
    query_id: str
    product_id: str
    label: RelevanceLabel


class ProductRelevance(BaseModel):
    """A product with its relevance score (for golden sets)."""

    product_id: str
    relevance: RelevanceScore

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"Relevance must be 0, 1, or 2, got {v}")
        return v

    @property
    def label(self) -> RelevanceLabel:
        """Get the label for this relevance score."""
        return RelevanceLabel.from_score(self.relevance)


class AgentJudgment(BaseModel):
    """A relevance judgment made by the agent."""

    product_id: str
    relevance: RelevanceScore
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: str = ""

    @property
    def label(self) -> RelevanceLabel:
        """Get the label for this relevance score."""
        return RelevanceLabel.from_score(self.relevance)


class GoldenSetConfig(BaseModel):
    """A complete golden set for a query."""

    query_id: str
    query: str
    query_class: str | None = None
    products: list[ProductRelevance]
    reasoning: str = ""
    created_by: str = "agent"
    created_at: datetime | None = None

    # Human correction tracking
    excluded_products: list[str] = Field(default_factory=list)
    manual_additions: list[ProductRelevance] = Field(default_factory=list)

    def get_relevance_map(self) -> dict[str, int]:
        """Get dict mapping product_id -> relevance score."""
        return {p.product_id: p.relevance for p in self.products}

    def get_label_map(self) -> dict[str, RelevanceLabel]:
        """Get dict mapping product_id -> relevance label."""
        return {p.product_id: p.label for p in self.products}


class AgentResult(BaseModel):
    """Result from the golden set agent."""

    query: str
    status: Literal["success", "needs_review", "failed"]
    products: list[AgentJudgment] | None = None
    reasoning: str | None = None
    review_questions: list[str] | None = None
    tool_call_count: int = 0
    error: str | None = None


class CategoryInfo(BaseModel):
    """Information about a product class/category."""

    product_class: str
    category_hierarchy: str
    count: int
