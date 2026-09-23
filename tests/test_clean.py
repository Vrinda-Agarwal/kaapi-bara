import pandas as pd
import pytest

from cafe_reviews.clean.reviews import clean_reviews, normalize_text, repair_mojibake, review_id
from cafe_reviews.validate.schemas import reviews_schema


def _double_mojibake(s: str) -> str:
    return s.encode("utf-8").decode("latin-1").encode("utf-8").decode("latin-1")


def test_repair_mojibake_reverses_double_encoding():
    fixed, changed = repair_mojibake(_double_mojibake("café ☕ nice"))
    assert fixed == "café ☕ nice"
    assert changed


def test_repair_leaves_clean_text():
    assert repair_mojibake("plain text") == ("plain text", False)


def test_normalize_strips_rated_prefix_and_whitespace():
    assert normalize_text("RATED\n   good   food ") == "good food"


def test_clean_reviews_dedup_and_conflicts():
    raw = pd.DataFrame({
        "rating": [5.0, 5.0, 4.0, 3.0, 1.0, 2.0],
        "review_text": ["Great place", "great  place", "Okay food", "okay food", "", "bad"],
    })
    clean, log = clean_reviews(raw)
    assert log["n_dropped_empty_text"] == 1
    assert log["n_dropped_exact_duplicates"] == 1  # "Great place" twice at 5.0
    assert log["n_dropped_conflicting_rating"] == 2  # "okay food" at 4.0 and 3.0
    assert sorted(clean["review_text"]) == ["Great place", "bad"]
    reviews_schema.validate(clean)


def test_review_id_stable_and_case_insensitive():
    assert review_id(5.0, "Great place") == review_id(5.0, "great  place")
    assert review_id(5.0, "Great place") != review_id(4.0, "Great place")


def test_schema_rejects_bad_rating():
    df = pd.DataFrame({"review_id": ["0123456789abcdef"], "rating": [6.0], "review_text": ["x"], "n_words": [1]})
    with pytest.raises(Exception):
        reviews_schema.validate(df)
