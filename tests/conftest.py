"""Synthetic fixtures. No real review text or real restaurant data."""

from __future__ import annotations

import pandas as pd
import pytest

COLUMNS = [
    "url", "address", "name", "online_order", "book_table", "rate", "votes", "phone",
    "location", "rest_type", "dish_liked", "cuisines", "approx_cost(for two people)",
    "reviews_list", "menu_item", "listed_in(type)", "listed_in(city)",
]


def _reviews(*pairs):
    return repr(list(pairs))


@pytest.fixture
def raw_df() -> pd.DataFrame:
    shared = _reviews(("Rated 5.0", "RATED\n  alpha beta"), ("Rated 3.0", "RATED\n  gamma"))
    rows = [
        # Same cafe listed under two categories, identical reviews.
        dict(url="https://x.test/a?ctx=1", address="1 Road", name="Bean There Cafe",
             rest_type="Cafe", cuisines="Cafe, Beverages", **{"listed_in(type)": "Cafes"},
             reviews_list=shared, rate="4.1/5"),
        dict(url="https://x.test/a?ctx=2", address="1 Road", name="Bean There Cafe",
             rest_type="Cafe", cuisines="Cafe, Beverages", **{"listed_in(type)": "Delivery"},
             reviews_list=shared, rate="4.1/5"),
        # Second branch of the same name (possible chain), accented name, half star.
        dict(url="https://x.test/b", address="2 Street", name="Bean There Café",
             rest_type="Café", cuisines="Coffee", **{"listed_in(type)": "Dine-out"},
             reviews_list=_reviews(("Rated 4.5", "delta 12/03/2018"), (None, "epsilon")),
             rate="NEW"),
        # Not a cafe. Mojibake name, byte escapes in text.
        dict(url="https://x.test/c", address="3 Lane", name="Dosa CafÃ© Corner",
             rest_type="Quick Bites", cuisines="South Indian", **{"listed_in(type)": "Delivery"},
             reviews_list=_reviews(("Rated 1.0", "zeta \\xf0\\x9f\\x98\\x8b")), rate="3.0 /5"),
        # Empty and malformed reviews_list.
        dict(url="https://x.test/d", address="4 Main", name="Plain Diner",
             rest_type="Casual Dining", cuisines="North Indian", **{"listed_in(type)": "Buffet"},
             reviews_list="[]", rate="-"),
        dict(url="https://x.test/e", address="5 Cross", name="Broken Row",
             rest_type="Casual Dining", cuisines="Chinese", **{"listed_in(type)": "Buffet"},
             reviews_list="[('Rated 4.0', 'unterminated", rate=None),
    ]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df["listed_in(city)"] = "Test Area"
    return df


@pytest.fixture
def raw_csv(tmp_path, raw_df):
    path = tmp_path / "zomato.csv"
    raw_df.to_csv(path, index=False)
    return path
