"""Shared utilities for agent tools."""

import re
import unicodedata

__all__ = [
    "find_matching_category",
    "normalize_for_matching",
    "parse_product_features",
]


def parse_product_features(features: str) -> dict[str, list[str]]:
    """Parse pipe-delimited features, grouping empty keys with previous key.

    Handles multi-value attributes like:
        dswoodtone : red wood| : barn brown| : cottonwood
    Which becomes:
        {"dswoodtone": ["red wood", "barn brown", "cottonwood"]}
    """
    result: dict[str, list[str]] = {}
    current_key: str | None = None

    if not features:
        return result

    for pair in features.split("|"):
        if ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key, value = key.strip(), value.strip()

        if key:  # New key
            current_key = key
            if key not in result:
                result[key] = []
            if value:
                result[key].append(value)
        elif current_key and value:  # Empty key = append to previous
            result[current_key].append(value)

    return result


def _fix_malformed_unicode(text: str) -> str:
    """Fix malformed Unicode patterns from LLM output.

    LLMs sometimes output \\x00XX (null + two hex chars) instead of \\xXX.
    For example: 'D\\x00e9cor' instead of 'Décor' (\\xe9 = é).

    This function detects and fixes these patterns.
    """

    # Pattern: null byte followed by exactly two hex digits
    # We capture the two hex digits to convert them to the actual character
    def replace_malformed(match: re.Match[str]) -> str:
        hex_chars: str = match.group(1)
        try:
            # Convert hex string to character (e.g., "e9" -> é)
            return chr(int(hex_chars, 16))
        except (ValueError, OverflowError):
            # If conversion fails, just remove the null byte
            return hex_chars

    # Match null byte followed by two hex digits
    return re.sub(r"\x00([0-9a-fA-F]{2})", replace_malformed, text)


def normalize_for_matching(text: str) -> str:
    """Normalize text to ASCII for fuzzy matching.

    Handles common LLM Unicode issues:
    - Malformed Unicode (\\x00e9 → é)
    - Null bytes in strings
    - Accented characters (é → e)
    - Various Unicode normalization forms

    Args:
        text: Input text, possibly with Unicode issues.

    Returns:
        Lowercase ASCII-only string for comparison.
    """
    # First, fix malformed Unicode patterns like \x00e9 -> é
    text = _fix_malformed_unicode(text)

    # Remove any remaining null bytes and control characters
    text = "".join(c for c in text if ord(c) >= 32 and c != "\x00")

    # Normalize to NFKD form (decomposes characters, e.g., é → e + combining accent)
    text = unicodedata.normalize("NFKD", text)

    # Strip to ASCII (removes combining marks and non-ASCII chars)
    text = text.encode("ascii", "ignore").decode("ascii")

    return text.lower().strip()


def find_matching_category(query: str, categories: list) -> str | None:
    """Find a category matching the query, with fuzzy Unicode handling.

    Tries exact match first, then falls back to ASCII-normalized comparison.

    Args:
        query: Category name from LLM (may have Unicode issues).
        categories: List of CategoryInfo objects with product_class attribute.

    Returns:
        The exact product_class string from categories if found, None otherwise.
    """
    query_stripped = query.strip()

    # Try exact match first
    for cat in categories:
        if cat.product_class == query_stripped:
            return str(cat.product_class)

    # Try normalized match
    query_normalized = normalize_for_matching(query)
    for cat in categories:
        if normalize_for_matching(cat.product_class) == query_normalized:
            return str(cat.product_class)

    return None
