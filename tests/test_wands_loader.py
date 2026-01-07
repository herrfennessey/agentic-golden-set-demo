"""Tests for WANDS data loader."""

from pathlib import Path
from textwrap import dedent

import pytest

from goldendemo.data.models import RelevanceLabel
from goldendemo.data.wands_loader import WANDSLoader


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample WANDS data."""
    data_dir = tmp_path / "wands"
    data_dir.mkdir()

    # Create sample product.csv
    products_csv = dedent("""
        product_id,product_name,product_class,category hierarchy,product_description,product_features,rating_count,average_rating,review_count
        1,Blue Velvet Sofa,Sofas,Furniture / Living Room / Sofas,A beautiful blue velvet sofa,Color:Blue|Material:Velvet,100,4.5,50
        2,Leather Sectional,Sectionals,Furniture / Living Room / Sectionals,Modern leather sectional,Color:Brown|Material:Leather,200,4.2,80
        3,Wooden Coffee Table,Coffee Tables,Furniture / Living Room / Coffee Tables,Oak wood coffee table,Material:Wood|Style:Modern,150,4.8,75
    """).strip()
    (data_dir / "product.csv").write_text(products_csv)

    # Create sample query.csv
    queries_csv = dedent("""
        query_id,query,query_class
        Q1,blue sofa,Sofas
        Q2,coffee table,Coffee Tables
    """).strip()
    (data_dir / "query.csv").write_text(queries_csv)

    # Create sample label.csv
    labels_csv = dedent("""
        id,query_id,product_id,label
        1,Q1,1,Exact
        2,Q1,2,Partial
        3,Q1,3,Irrelevant
        4,Q2,3,Exact
        5,Q2,1,Irrelevant
    """).strip()
    (data_dir / "label.csv").write_text(labels_csv)

    return data_dir


@pytest.fixture
def loader(sample_data_dir: Path) -> WANDSLoader:
    """Create a loader with sample data."""
    return WANDSLoader(data_dir=sample_data_dir)


class TestWANDSLoader:
    """Tests for WANDSLoader."""

    def test_get_product(self, loader: WANDSLoader):
        """Test getting a single product."""
        product = loader.get_product("1")
        assert product is not None
        assert product.product_id == "1"
        assert product.product_name == "Blue Velvet Sofa"
        assert product.product_class == "Sofas"

    def test_get_product_not_found(self, loader: WANDSLoader):
        """Test getting a non-existent product."""
        product = loader.get_product("999")
        assert product is None

    def test_get_products(self, loader: WANDSLoader):
        """Test getting multiple products."""
        products = loader.get_products(["1", "3"])
        assert len(products) == 2
        assert products[0].product_id == "1"
        assert products[1].product_id == "3"

    def test_get_all_products(self, loader: WANDSLoader):
        """Test getting all products."""
        products = loader.get_all_products()
        assert len(products) == 3

    def test_get_query(self, loader: WANDSLoader):
        """Test getting a single query."""
        query = loader.get_query("Q1")
        assert query is not None
        assert query.query_id == "Q1"
        assert query.query == "blue sofa"
        assert query.query_class == "Sofas"

    def test_get_all_queries(self, loader: WANDSLoader):
        """Test getting all queries."""
        queries = loader.get_all_queries()
        assert len(queries) == 2

    def test_get_labels_for_query(self, loader: WANDSLoader):
        """Test getting labels for a query."""
        labels = loader.get_labels_for_query("Q1")
        assert len(labels) == 3

        # Check label types
        label_map = {lbl.product_id: lbl.label for lbl in labels}
        assert label_map["1"] == RelevanceLabel.EXACT
        assert label_map["2"] == RelevanceLabel.PARTIAL
        assert label_map["3"] == RelevanceLabel.IRRELEVANT

    def test_get_ground_truth_map(self, loader: WANDSLoader):
        """Test getting ground truth map."""
        gt = loader.get_ground_truth_map("Q1")
        assert gt["1"] == RelevanceLabel.EXACT
        assert gt["2"] == RelevanceLabel.PARTIAL
        assert gt["3"] == RelevanceLabel.IRRELEVANT

    def test_get_ground_truth_scores(self, loader: WANDSLoader):
        """Test getting ground truth scores."""
        scores = loader.get_ground_truth_scores("Q1")
        assert scores == {"1": 2, "2": 1, "3": 0}

    def test_get_products_by_class(self, loader: WANDSLoader):
        """Test getting products by class."""
        products = loader.get_products_by_class("Sofas")
        assert len(products) == 1
        assert products[0].product_name == "Blue Velvet Sofa"

    def test_get_all_classes(self, loader: WANDSLoader):
        """Test getting all product classes."""
        classes = loader.get_all_classes()
        assert len(classes) == 3

        class_names = [c.product_class for c in classes]
        assert "Sofas" in class_names
        assert "Sectionals" in class_names
        assert "Coffee Tables" in class_names

    def test_get_queries_by_class(self, loader: WANDSLoader):
        """Test getting queries by class."""
        queries = loader.get_queries_by_class("Sofas")
        assert len(queries) == 1
        assert queries[0].query == "blue sofa"

    def test_get_all_query_classes(self, loader: WANDSLoader):
        """Test getting all query classes."""
        classes = loader.get_all_query_classes()
        assert "Sofas" in classes
        assert "Coffee Tables" in classes

    def test_get_stats(self, loader: WANDSLoader):
        """Test getting dataset statistics."""
        stats = loader.get_stats()
        assert stats["products"] == 3
        assert stats["queries"] == 2
        assert stats["total_labels"] == 5
        assert stats["label_distribution"]["Exact"] == 2
        assert stats["label_distribution"]["Partial"] == 1
        assert stats["label_distribution"]["Irrelevant"] == 2

    def test_missing_file_raises_error(self, tmp_path: Path):
        """Test that missing files raise appropriate errors."""
        loader = WANDSLoader(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.get_all_products()
