"""Tests for agent utilities."""

import pytest

from goldendemo.agent.utils import find_matching_category, normalize_for_matching
from goldendemo.data.models import CategoryInfo


class TestNormalizeForMatching:
    """Tests for normalize_for_matching function."""

    def test_basic_ascii(self):
        """Test that basic ASCII text is lowercased and stripped."""
        assert normalize_for_matching("Hello World") == "hello world"
        assert normalize_for_matching("  SOFAS  ") == "sofas"

    def test_accented_characters(self):
        """Test that accented characters are converted to ASCII."""
        assert normalize_for_matching("Décor") == "decor"
        assert normalize_for_matching("café") == "cafe"
        assert normalize_for_matching("naïve") == "naive"
        assert normalize_for_matching("résumé") == "resume"

    def test_null_bytes(self):
        """Test that null bytes are removed."""
        assert normalize_for_matching("Hello\x00World") == "helloworld"
        assert normalize_for_matching("\x00test\x00") == "test"

    def test_malformed_unicode_from_llm(self):
        """Test handling of malformed Unicode patterns from LLM output.

        LLMs sometimes output \\x00e9 (null + 'e9') instead of é (\\xe9).
        We detect this pattern and convert it to the proper character.
        """
        # The malformed pattern: null byte followed by hex chars
        malformed = "Kids Wall D\x00e9cor"
        # After fixing malformed unicode: "Kids Wall Décor"
        # After ASCII normalization: "kids wall decor"
        result = normalize_for_matching(malformed)
        assert result == "kids wall decor"
        assert "\x00" not in result
        assert "e9" not in result  # The hex chars should be converted

    def test_various_unicode_forms(self):
        """Test that different Unicode normalization forms are handled."""
        # NFC vs NFD forms of é
        nfc = "caf\u00e9"  # é as single codepoint
        nfd = "cafe\u0301"  # e + combining acute accent
        assert normalize_for_matching(nfc) == normalize_for_matching(nfd)

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_for_matching("") == ""
        assert normalize_for_matching("   ") == ""


class TestFindMatchingCategory:
    """Tests for find_matching_category function."""

    @pytest.fixture
    def categories(self):
        """Create test categories."""
        return [
            CategoryInfo(product_class="Sofas", category_hierarchy="Furniture / Seating", count=100),
            CategoryInfo(product_class="Kids Wall Décor", category_hierarchy="Home Decor / Kids", count=50),
            CategoryInfo(product_class="Coffee Tables", category_hierarchy="Furniture / Tables", count=75),
            CategoryInfo(product_class="Café Furniture", category_hierarchy="Outdoor / Dining", count=25),
        ]

    def test_exact_match(self, categories):
        """Test exact string match."""
        result = find_matching_category("Sofas", categories)
        assert result == "Sofas"

    def test_exact_match_with_accent(self, categories):
        """Test exact match with accented characters."""
        result = find_matching_category("Kids Wall Décor", categories)
        assert result == "Kids Wall Décor"

    def test_ascii_fallback_for_accent(self, categories):
        """Test ASCII fallback when accents don't match exactly."""
        # Query without accent should match category with accent
        result = find_matching_category("Kids Wall Decor", categories)
        assert result == "Kids Wall Décor"

    def test_case_insensitive_ascii_match(self, categories):
        """Test case-insensitive matching through ASCII normalization."""
        result = find_matching_category("SOFAS", categories)
        assert result == "Sofas"

        result = find_matching_category("coffee tables", categories)
        assert result == "Coffee Tables"

    def test_malformed_unicode_match(self, categories):
        """Test that malformed Unicode from LLM matches the correct category."""
        # LLM outputs null byte + hex chars instead of proper Unicode
        # \x00e9 should be interpreted as \xe9 (é)
        malformed = "Kids Wall D\x00e9cor"
        result = find_matching_category(malformed, categories)
        # Now this should match because we fix the malformed pattern
        assert result == "Kids Wall Décor"

    def test_no_match_returns_none(self, categories):
        """Test that non-matching category returns None."""
        result = find_matching_category("Nonexistent Category", categories)
        assert result is None

    def test_empty_categories(self):
        """Test with empty category list."""
        result = find_matching_category("Sofas", [])
        assert result is None

    def test_whitespace_handling(self, categories):
        """Test that whitespace is handled correctly."""
        result = find_matching_category("  Sofas  ", categories)
        assert result == "Sofas"

    def test_returns_original_category_name(self, categories):
        """Test that the original category name is returned, not normalized version."""
        # Even if we search with lowercase, we should get back the original
        result = find_matching_category("café furniture", categories)
        assert result == "Café Furniture"
