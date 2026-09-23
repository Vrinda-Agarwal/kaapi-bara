import math

import pytest

from cafe_reviews.ingest.parse import (
    ReviewsParseError,
    as_rating_text_pair,
    fold_text,
    parse_literal_list,
    parse_review_rating,
    review_hash,
)

PATTERN = r"^\s*Rated\s+(\d+(?:\.\d+)?)\s*$"


def test_parse_literal_list_tuples():
    assert parse_literal_list("[('Rated 4.0', 'good')]") == [("Rated 4.0", "good")]


@pytest.mark.parametrize("cell", [None, math.nan, "[]"])
def test_parse_literal_list_empty(cell):
    assert parse_literal_list(cell) == []


@pytest.mark.parametrize(
    "cell", ["[('Rated 4.0', 'oops'", "__import__('os').getcwd()", "('a', 'b')", "{'a': 1}", 5]
)
def test_parse_literal_list_rejects(cell):
    with pytest.raises(ReviewsParseError):
        parse_literal_list(cell)


def test_as_rating_text_pair():
    assert as_rating_text_pair(("Rated 5.0", "x")) == ("Rated 5.0", "x")
    assert as_rating_text_pair((None, "x")) == (None, "x")
    for bad in [("a",), ("a", "b", "c"), (1.0, "x"), ("a", None), "ab"]:
        with pytest.raises(ReviewsParseError):
            as_rating_text_pair(bad)


@pytest.mark.parametrize(
    "raw, expected",
    [("Rated 4.0", 4.0), ("Rated 3.5", 3.5), ("  Rated 5  ", 5.0), ("Rated", None),
     ("4.0", None), (None, None), ("Rated 4.0 stars", None)],
)
def test_parse_review_rating(raw, expected):
    assert parse_review_rating(raw, PATTERN) == expected


def test_fold_text():
    assert fold_text("Café KAAPI") == "cafe kaapi"
    assert fold_text(None) == ""


def test_review_hash_stable_and_distinct():
    assert review_hash("Rated 4.0", "a") == review_hash("Rated 4.0", "a")
    assert review_hash("Rated 4.0", "a") != review_hash("Rated 5.0", "a")
    assert review_hash(None, "a") != review_hash("None", "a\x1f")
