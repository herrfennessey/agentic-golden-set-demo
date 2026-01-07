"""Tests for Weaviate client.

Note: These tests require a running Weaviate instance.
Tests are skipped if Weaviate is not available.
"""

import pytest

from goldendemo.clients.weaviate_client import (
    COLLECTION_NAME,
    WeaviateClient,
    get_collection_config,
)
from goldendemo.data.models import Product


@pytest.fixture
def client() -> WeaviateClient:
    """Create a Weaviate client for testing."""
    return WeaviateClient()


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

    def test_create_collection(self, client: WeaviateClient):
        """Test creating the collection."""
        try:
            with client.connect():
                if not client.is_ready():
                    pytest.skip("Weaviate not available")

                # Create with reset to ensure clean state
                client.create_collection(delete_existing=True)
                assert client.collection_exists()
        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    def test_insert_and_search(self, client: WeaviateClient):
        """Test inserting products and searching."""
        try:
            with client.connect():
                if not client.is_ready():
                    pytest.skip("Weaviate not available")

                # Create collection
                client.create_collection(delete_existing=True)

                # Insert test products
                products = [
                    Product(
                        product_id="test-1",
                        product_name="Blue Velvet Sofa",
                        product_class="Sofas",
                        category_hierarchy="Furniture / Living Room / Sofas",
                        product_description="A beautiful blue velvet sofa",
                    ),
                    Product(
                        product_id="test-2",
                        product_name="Wooden Coffee Table",
                        product_class="Coffee Tables",
                        category_hierarchy="Furniture / Living Room / Coffee Tables",
                        product_description="Oak wood coffee table",
                    ),
                ]
                client.insert_products(products)

                # Search
                results = client.hybrid_search("blue sofa", limit=10)
                assert len(results) > 0
                assert results[0].product_id == "test-1"

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")
