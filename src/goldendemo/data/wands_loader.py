"""
WANDS dataset loader.

Loads and indexes the WANDS dataset (products, queries, labels) for efficient access.
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from goldendemo.config import settings
from goldendemo.data.models import (
    CategoryInfo,
    Product,
    Query,
    RelevanceLabel,
    WANDSLabel,
)

logger = logging.getLogger(__name__)


class WANDSLoader:
    """
    Loader for the WANDS dataset.

    Provides efficient access to products, queries, and relevance labels.
    Data is loaded lazily and cached for performance.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the WANDS loader.

        Args:
            data_dir: Path to WANDS data directory. Defaults to settings.wands_dir.
        """
        self.data_dir = data_dir or settings.wands_dir

        # Lazy-loaded data
        self._products_df: pd.DataFrame | None = None
        self._queries_df: pd.DataFrame | None = None
        self._labels_df: pd.DataFrame | None = None

        # Indexes (built on first access)
        self._products_by_id: dict[str, Product] | None = None
        self._queries_by_id: dict[str, Query] | None = None
        self._labels_by_query: dict[str, list[WANDSLabel]] | None = None
        self._products_by_class: dict[str, list[str]] | None = None

    def _load_products_df(self) -> pd.DataFrame:
        """Load products CSV into DataFrame."""
        if self._products_df is None:
            path = self.data_dir / "product.csv"
            if not path.exists():
                raise FileNotFoundError(f"Product file not found: {path}. Run 'make download-wands' first.")
            logger.info(f"Loading products from {path}")
            self._products_df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            logger.info(f"Loaded {len(self._products_df)} products")
        return self._products_df

    def _load_queries_df(self) -> pd.DataFrame:
        """Load queries CSV into DataFrame."""
        if self._queries_df is None:
            path = self.data_dir / "query.csv"
            if not path.exists():
                raise FileNotFoundError(f"Query file not found: {path}. Run 'make download-wands' first.")
            logger.info(f"Loading queries from {path}")
            self._queries_df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            logger.info(f"Loaded {len(self._queries_df)} queries")
        return self._queries_df

    def _load_labels_df(self) -> pd.DataFrame:
        """Load labels CSV into DataFrame."""
        if self._labels_df is None:
            path = self.data_dir / "label.csv"
            if not path.exists():
                raise FileNotFoundError(f"Label file not found: {path}. Run 'make download-wands' first.")
            logger.info(f"Loading labels from {path}")
            self._labels_df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            logger.info(f"Loaded {len(self._labels_df)} labels")
        return self._labels_df

    def _build_product_index(self) -> dict[str, Product]:
        """Build product index by ID."""
        if self._products_by_id is None:
            df = self._load_products_df()
            self._products_by_id = {}
            for row in df.to_dict("records"):
                product = Product(
                    product_id=str(row["product_id"]),
                    product_name=str(row["product_name"]),
                    product_class=str(row["product_class"]),
                    category_hierarchy=str(row["category hierarchy"]),
                    product_description=str(row.get("product_description", "")),
                    product_features=str(row.get("product_features", "")),
                    rating_count=int(float(row.get("rating_count", 0) or 0)),
                    average_rating=float(row.get("average_rating", 0) or 0),
                    review_count=int(float(row.get("review_count", 0) or 0)),
                )
                self._products_by_id[product.product_id] = product
        return self._products_by_id

    def _build_query_index(self) -> dict[str, Query]:
        """Build query index by ID."""
        if self._queries_by_id is None:
            df = self._load_queries_df()
            self._queries_by_id = {}
            for row in df.to_dict("records"):
                query = Query(
                    query_id=str(row["query_id"]),
                    query=str(row["query"]),
                    query_class=str(row["query_class"]),
                )
                self._queries_by_id[query.query_id] = query
        return self._queries_by_id

    def _build_labels_index(self) -> dict[str, list[WANDSLabel]]:
        """Build labels index by query ID."""
        if self._labels_by_query is None:
            df = self._load_labels_df()
            self._labels_by_query = {}
            for row in df.to_dict("records"):
                label = WANDSLabel(
                    id=str(row["id"]),
                    query_id=str(row["query_id"]),
                    product_id=str(row["product_id"]),
                    label=RelevanceLabel(str(row["label"])),
                )
                if label.query_id not in self._labels_by_query:
                    self._labels_by_query[label.query_id] = []
                self._labels_by_query[label.query_id].append(label)
        return self._labels_by_query

    def _build_products_by_class(self) -> dict[str, list[str]]:
        """Build index of product IDs by product class."""
        if self._products_by_class is None:
            products = self._build_product_index()
            self._products_by_class = {}
            for pid, product in products.items():
                if product.product_class not in self._products_by_class:
                    self._products_by_class[product.product_class] = []
                self._products_by_class[product.product_class].append(pid)
        return self._products_by_class

    # Public API

    def get_product(self, product_id: str) -> Product | None:
        """Get a product by ID."""
        return self._build_product_index().get(product_id)

    def get_products(self, product_ids: list[str]) -> list[Product]:
        """Get multiple products by ID."""
        index = self._build_product_index()
        return [index[pid] for pid in product_ids if pid in index]

    def get_all_products(self) -> list[Product]:
        """Get all products."""
        return list(self._build_product_index().values())

    def get_query(self, query_id: str) -> Query | None:
        """Get a query by ID."""
        return self._build_query_index().get(query_id)

    def get_all_queries(self) -> list[Query]:
        """Get all queries."""
        return list(self._build_query_index().values())

    def get_labels_for_query(self, query_id: str) -> list[WANDSLabel]:
        """Get all relevance labels for a query."""
        return self._build_labels_index().get(query_id, [])

    def get_ground_truth_map(self, query_id: str) -> dict[str, RelevanceLabel]:
        """Get ground truth relevance map for a query (product_id -> label)."""
        labels = self.get_labels_for_query(query_id)
        return {label.product_id: label.label for label in labels}

    def get_ground_truth_scores(self, query_id: str) -> dict[str, int]:
        """Get ground truth scores for a query (product_id -> 0/1/2)."""
        labels = self.get_labels_for_query(query_id)
        return {label.product_id: label.label.score for label in labels}

    def get_products_by_class(self, product_class: str) -> list[Product]:
        """Get all products in a product class."""
        index = self._build_products_by_class()
        product_ids = index.get(product_class, [])
        return self.get_products(product_ids)

    def get_all_classes(self) -> list[CategoryInfo]:
        """Get all unique product classes with counts."""
        index = self._build_products_by_class()
        products_index = self._build_product_index()

        classes = []
        for product_class, pids in sorted(index.items()):
            # Get category hierarchy from first product
            sample = products_index.get(pids[0]) if pids else None
            hierarchy = sample.category_hierarchy if sample else ""

            classes.append(
                CategoryInfo(
                    product_class=product_class,
                    category_hierarchy=hierarchy,
                    count=len(pids),
                )
            )
        return classes

    def get_queries_by_class(self, query_class: str) -> list[Query]:
        """Get all queries in a query class."""
        queries = self._build_query_index()
        return [q for q in queries.values() if q.query_class == query_class]

    def get_all_query_classes(self) -> list[str]:
        """Get all unique query classes."""
        queries = self._build_query_index()
        return sorted(set(q.query_class for q in queries.values()))

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        products = self._build_product_index()
        queries = self._build_query_index()
        labels = self._build_labels_index()

        total_labels = sum(len(lbls) for lbls in labels.values())

        # Count labels by type
        label_counts = {"Exact": 0, "Partial": 0, "Irrelevant": 0}
        for lbls in labels.values():
            for label in lbls:
                label_counts[label.label.value] += 1

        return {
            "products": len(products),
            "queries": len(queries),
            "total_labels": total_labels,
            "label_distribution": label_counts,
            "product_classes": len(self._build_products_by_class()),
            "query_classes": len(self.get_all_query_classes()),
        }


# Global instance for convenience
@lru_cache
def get_wands_loader() -> WANDSLoader:
    """Get the global WANDS loader instance."""
    return WANDSLoader()
