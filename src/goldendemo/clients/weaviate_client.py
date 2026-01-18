"""
Weaviate client for WANDS product search.

Provides hybrid search (vector + BM25 keyword) over the WANDS product catalog.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.collections import Collection

from goldendemo.config import settings
from goldendemo.data.models import CategoryInfo, Product, ProductSummary

logger = logging.getLogger(__name__)

COLLECTION_NAME = "WandsProduct"


def get_collection_config() -> dict:
    """
    Get the Weaviate collection configuration for WANDS products.

    Uses text2vec-openai for embeddings with hybrid search support.
    """
    return {
        "name": COLLECTION_NAME,
        "description": "WANDS (Wayfair) product catalog for search relevance evaluation",
        "vectorizer_config": Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small",
            vectorize_collection_name=False,
        ),
        "properties": [
            Property(
                name="product_id",
                data_type=DataType.TEXT,
                description="WANDS product identifier",
                skip_vectorization=True,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="product_name",
                data_type=DataType.TEXT,
                description="Product title/name",
                tokenization=Tokenization.WORD,
            ),
            Property(
                name="product_class",
                data_type=DataType.TEXT_ARRAY,
                description="Product category classes (can belong to multiple)",
                skip_vectorization=True,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="category_hierarchy",
                data_type=DataType.TEXT,
                description="Full category path (slash-delimited)",
                tokenization=Tokenization.WORD,
            ),
            Property(
                name="product_description",
                data_type=DataType.TEXT,
                description="Detailed product description",
                tokenization=Tokenization.WORD,
            ),
            Property(
                name="product_features",
                data_type=DataType.TEXT,
                description="Product attributes (pipe-delimited key:value pairs)",
                tokenization=Tokenization.WORD,
            ),
            Property(
                name="average_rating",
                data_type=DataType.NUMBER,
                description="Average product rating",
                skip_vectorization=True,
            ),
            Property(
                name="rating_count",
                data_type=DataType.INT,
                description="Number of ratings",
                skip_vectorization=True,
            ),
            Property(
                name="review_count",
                data_type=DataType.INT,
                description="Number of reviews",
                skip_vectorization=True,
            ),
        ],
    }


class WeaviateClient:
    """
    Client for WANDS product search via Weaviate.

    Supports:
    - Hybrid search (vector + BM25 keyword matching)
    - Filtering by product class
    - Batch operations for data loading
    """

    def __init__(self, url: str | None = None, openai_api_key: str | None = None):
        """
        Initialize the Weaviate client.

        Args:
            url: Weaviate server URL. Defaults to settings.weaviate_url.
            openai_api_key: OpenAI API key for embeddings. Defaults to settings.openai_api_key.
        """
        self.url = url or settings.weaviate_url
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self._client: weaviate.WeaviateClient | None = None

    @contextmanager
    def connect(self) -> Iterator["WeaviateClient"]:
        """Context manager for connecting to Weaviate."""
        try:
            parsed = urlparse(self.url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080

            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
                headers={"X-OpenAI-Api-Key": self.openai_api_key},
            )
            yield self
        finally:
            if self._client:
                self._client.close()
                self._client = None

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Get the Weaviate client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Not connected. Use 'with client.connect():' context manager.")
        return self._client

    @property
    def collection(self) -> Collection:
        """Get the WANDS product collection."""
        return self.client.collections.get(COLLECTION_NAME)

    def is_ready(self) -> bool:
        """Check if Weaviate is ready."""
        try:
            return bool(self.client.is_ready())
        except Exception:
            logger.debug("Weaviate readiness check failed", exc_info=True)
            return False

    def collection_exists(self) -> bool:
        """Check if the WANDS collection exists."""
        return bool(self.client.collections.exists(COLLECTION_NAME))

    def create_collection(self, delete_existing: bool = False) -> None:
        """
        Create the WANDS product collection.

        Args:
            delete_existing: If True, delete existing collection first.
        """
        if delete_existing and self.collection_exists():
            logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
            self.client.collections.delete(COLLECTION_NAME)

        if not self.collection_exists():
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            config = get_collection_config()
            self.client.collections.create(**config)
            logger.info("Collection created successfully")

    def get_collection_count(self) -> int:
        """Get the number of objects in the collection."""
        if not self.collection_exists():
            return 0
        result = self.collection.aggregate.over_all(total_count=True)
        return result.total_count or 0

    def get_all_product_ids(self) -> set[str]:
        """Get all product IDs currently in the collection."""
        if not self.collection_exists():
            return set()

        product_ids: set[str] = set()
        for obj in self.collection.iterator(return_properties=["product_id"]):
            pid = obj.properties.get("product_id")
            if pid:
                product_ids.add(str(pid))

        return product_ids

    def delete_products(self, product_ids: list[str]) -> int:
        """
        Delete products by their IDs.

        Note: Returns the number of delete operations attempted, not the actual
        number of objects deleted (Weaviate's delete_many doesn't return counts).
        """
        deleted = 0
        for pid in product_ids:
            self.collection.data.delete_many(where=Filter.by_property("product_id").equal(pid))
            deleted += 1
            if deleted % 100 == 0:
                logger.info(f"Deleted {deleted} products...")

        return deleted

    def insert_products(self, products: list[Product], batch_size: int = 100) -> int:
        """
        Insert products into Weaviate.

        Args:
            products: List of Product objects to insert.
            batch_size: Number of products per batch.

        Returns:
            Number of products inserted.
        """
        inserted = 0
        with self.collection.batch.fixed_size(batch_size=batch_size) as batch:
            for product in products:
                # Split pipe-delimited product_class into array
                product_classes = product.product_class.split("|") if product.product_class else []

                batch.add_object(
                    properties={
                        "product_id": product.product_id,
                        "product_name": product.product_name,
                        "product_class": product_classes,  # Now an array
                        "category_hierarchy": product.category_hierarchy,
                        "product_description": product.product_description,
                        "product_features": product.product_features,
                        "average_rating": product.average_rating,
                        "rating_count": product.rating_count,
                        "review_count": product.review_count,
                    }
                )
                inserted += 1

                if inserted % 1000 == 0:
                    logger.info(f"Inserted {inserted} products...")

        return inserted

    def hybrid_search(
        self,
        query: str,
        limit: int = 50,
        alpha: float = 0.5,
    ) -> list[ProductSummary]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            alpha: Balance between vector (1.0) and keyword (0.0). Default 0.5.

        Returns:
            List of ProductSummary objects.
        """
        results = self.collection.query.hybrid(
            query=query,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True),
        )

        return [ProductSummary.model_validate(obj.properties) for obj in results.objects]

    def get_by_class(self, product_class: str, limit: int = 100) -> list[ProductSummary]:
        """
        Get all products in a specific product class.

        Args:
            product_class: The product class to filter by.
            limit: Maximum number of results.

        Returns:
            List of ProductSummary objects.
        """
        results = self.collection.query.fetch_objects(
            filters=Filter.by_property("product_class").contains_any([product_class]),
            limit=limit,
        )

        return [ProductSummary.model_validate(obj.properties) for obj in results.objects]

    def get_by_ids(self, product_ids: list[str], limit: int = 50) -> list[Product]:
        """
        Get full product details for specific IDs.

        Args:
            product_ids: List of product IDs to retrieve.
            limit: Maximum number of IDs to process (default 50).

        Returns:
            List of Product objects (order not guaranteed to match input).
        """
        if not product_ids:
            return []

        limited_ids = product_ids[:limit]

        # Build OR filter for batch fetch instead of N individual queries
        combined_filter = Filter.by_property("product_id").equal(limited_ids[0])
        for pid in limited_ids[1:]:
            combined_filter = combined_filter | Filter.by_property("product_id").equal(pid)

        results = self.collection.query.fetch_objects(
            filters=combined_filter,
            limit=len(limited_ids),
        )

        return [Product.model_validate(obj.properties) for obj in results.objects]

    def get_all_classes(self) -> list[CategoryInfo]:
        """
        Get all unique product classes with counts.

        Since product_class is now an array, we need to flatten and count each class.
        A product with ["Accent Chairs", "Office Chairs"] counts for both classes.

        Returns:
            List of CategoryInfo objects sorted by product_class.
        """
        # Iterate through all products and count each class
        class_counts: dict[str, int] = {}
        class_to_hierarchy: dict[str, str] = {}

        for obj in self.collection.iterator(return_properties=["product_class", "category_hierarchy"]):
            # product_class is now an array
            product_classes = obj.properties.get("product_class", [])
            if not isinstance(product_classes, list):
                product_classes = [product_classes] if product_classes else []

            category_hierarchy = str(obj.properties.get("category_hierarchy", ""))

            # Count each class
            for product_class in product_classes:
                if product_class:
                    product_class = str(product_class).strip()
                    class_counts[product_class] = class_counts.get(product_class, 0) + 1
                    # Store hierarchy for first occurrence
                    if product_class not in class_to_hierarchy:
                        class_to_hierarchy[product_class] = category_hierarchy

        # Build CategoryInfo objects
        classes = []
        for product_class, count in class_counts.items():
            classes.append(
                CategoryInfo(
                    product_class=product_class,
                    category_hierarchy=class_to_hierarchy.get(product_class, ""),
                    count=count,
                )
            )

        return sorted(classes, key=lambda x: x.product_class)
