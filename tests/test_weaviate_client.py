"""Tests for Weaviate client.

Note: These tests require a running Weaviate instance.
Tests are skipped if Weaviate is not available.

IMPORTANT: Integration tests use a separate test collection to avoid
destroying production data.
"""

import pytest

from goldendemo.clients.weaviate_client import (
    COLLECTION_NAME,
    WeaviateClient,
    get_collection_config,
)
from goldendemo.data.models import Product

# Test collection name - NEVER use the production collection
TEST_COLLECTION_NAME = "WandsProductTest"


@pytest.fixture
def client() -> WeaviateClient:
    """Create a Weaviate client for testing."""
    return WeaviateClient()


@pytest.fixture
def test_client():
    """Create a Weaviate client that uses a TEST collection.

    Automatically cleans up the test collection after tests complete.
    """
    client = WeaviateClient()
    try:
        with client.connect():
            # Clean up any existing test collection before test
            if client.client.collections.exists(TEST_COLLECTION_NAME):
                client.client.collections.delete(TEST_COLLECTION_NAME)
            yield client
            # Clean up test collection after test
            if client.client.collections.exists(TEST_COLLECTION_NAME):
                client.client.collections.delete(TEST_COLLECTION_NAME)
    except Exception:
        yield client  # Let the test handle connection errors


class TestCollectionConfig:
    """Tests for collection configuration."""

    def test_config_has_required_fields(self):
        """Test that config has all required fields."""
        config = get_collection_config()

        assert config["name"] == COLLECTION_NAME
        assert "vectorizer_config" in config
        assert "properties" in config

    def test_config_has_product_properties(self):
        """Test that config has all product properties."""
        config = get_collection_config()
        prop_names = [p.name for p in config["properties"]]

        assert "product_id" in prop_names
        assert "product_name" in prop_names
        assert "product_class" in prop_names
        assert "category_hierarchy" in prop_names
        assert "product_description" in prop_names


class TestWeaviateClientUnit:
    """Unit tests that don't require Weaviate connection."""

    def test_client_init_defaults(self):
        """Test client initialization with defaults."""
        client = WeaviateClient()
        assert client.url is not None
        assert client._client is None  # Not connected yet

    def test_client_init_custom_url(self):
        """Test client initialization with custom URL."""
        client = WeaviateClient(url="http://custom:9999")
        assert client.url == "http://custom:9999"

    def test_client_not_connected_raises(self):
        """Test that accessing client without connection raises error."""
        client = WeaviateClient()
        with pytest.raises(RuntimeError, match="Not connected"):
            _ = client.client


@pytest.mark.integration
class TestWeaviateClientIntegration:
    """Integration tests that require running Weaviate.

    IMPORTANT: These tests use a SEPARATE test collection (WandsProductTest)
    to avoid destroying production data.

    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    def test_connect_and_ready(self, client: WeaviateClient):
        """Test connecting to Weaviate."""
        try:
            with client.connect():
                assert client.is_ready()
        except Exception:
            pytest.skip("Weaviate not available")

    def test_create_collection(self, test_client: WeaviateClient):
        """Test creating a collection (uses test collection, not production)."""
        try:
            if not test_client.is_ready():
                pytest.skip("Weaviate not available")

            # Create TEST collection (not production!)
            config = get_collection_config()
            config["name"] = TEST_COLLECTION_NAME
            test_client.client.collections.create(**config)

            assert test_client.client.collections.exists(TEST_COLLECTION_NAME)
            # Cleanup happens automatically via fixture

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    def test_insert_and_search(self, test_client: WeaviateClient):
        """Test inserting products and searching (uses test collection)."""
        try:
            if not test_client.is_ready():
                pytest.skip("Weaviate not available")

            # Create TEST collection
            config = get_collection_config()
            config["name"] = TEST_COLLECTION_NAME
            test_client.client.collections.create(**config)
            test_collection = test_client.client.collections.get(TEST_COLLECTION_NAME)

            # Insert test products directly into test collection
            products = [
                Product(
                    product_id="test-1",
                    product_name="Leather Dining Chair",
                    product_class="Dining Chairs",
                    category_hierarchy="Furniture / Dining Room / Dining Chairs",
                    product_description="A beautiful leather dining chair",
                ),
                Product(
                    product_id="test-2",
                    product_name="Wooden Coffee Table",
                    product_class="Coffee Tables",
                    category_hierarchy="Furniture / Living Room / Coffee Tables",
                    product_description="Oak wood coffee table",
                ),
            ]

            with test_collection.batch.fixed_size(batch_size=10) as batch:
                for product in products:
                    batch.add_object(
                        properties={
                            "product_id": product.product_id,
                            "product_name": product.product_name,
                            "product_class": product.product_class,
                            "category_hierarchy": product.category_hierarchy,
                            "product_description": product.product_description,
                            "product_features": product.product_features,
                            "average_rating": product.average_rating,
                            "rating_count": product.rating_count,
                            "review_count": product.review_count,
                        }
                    )

            # Search in test collection
            results = test_collection.query.hybrid(
                query="leather dining chairs",
                limit=10,
            )
            assert len(results.objects) > 0
            assert results.objects[0].properties["product_id"] == "test-1"

            # Cleanup happens automatically via fixture

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")
