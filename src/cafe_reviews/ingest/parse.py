"""Safe parsers for the raw Zomato fields.

`reviews_list` is stored as a stringified Python list. It is parsed with
`ast.literal_eval`, which only accepts Python literals. `eval()` is never used.
"""

from __future__ import annotations

import ast
import hashlib
import re
import unicodedata
from typing import Any


class ReviewsParseError(ValueError):
    """Raised when a reviews_list cell is not a well-formed literal list."""


def parse_literal_list(cell: Any) -> list[Any]:
    """Parse a stringified Python list. Missing cells become an empty list.

    Raises ReviewsParseError if the cell is not a string holding a list literal.
    """
    if cell is None or (isinstance(cell, float) and cell != cell):  # NaN
        return []
    if not isinstance(cell, str):
        raise ReviewsParseError(f"expected str, got {type(cell).__name__}")
    try:
        value = ast.literal_eval(cell)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ReviewsParseError(f"not a Python literal: {type(exc).__name__}") from exc
    if not isinstance(value, list):
        raise ReviewsParseError(f"literal is a {type(value).__name__}, not a list")
    return value


def as_rating_text_pair(item: Any) -> tuple[str | None, str]:
    """Check that one reviews_list element is a (rating, text) pair of strings.

    The rating may be None. Raises ReviewsParseError otherwise.
    """
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise ReviewsParseError("element is not a 2-item tuple")
    rating, text = item
    if rating is not None and not isinstance(rating, str):
        raise ReviewsParseError("rating is not a string")
    if not isinstance(text, str):
        raise ReviewsParseError("text is not a string")
    return rating, text


def parse_review_rating(raw: str | None, pattern: str) -> float | None:
    """Extract the numeric rating from a string like 'Rated 4.0'.

    Returns None if the value is missing or does not match `pattern`
    (a regex with one capture group, taken from config).
    """
    if raw is None:
        return None
    m = re.match(pattern, raw)
    if m is None:
        return None
    return float(m.group(1))


def fold_text(s: Any) -> str:
    """Lowercase and strip accents so 'Café' and 'cafe' compare equal."""
    if not isinstance(s, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def review_hash(rating: str | None, text: str) -> int:
    """Stable 64-bit hash of an exact (rating, text) pair, for exact-duplicate counts."""
    payload = f"{rating}\x1f{text}".encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
