"""pandera schemas for pipeline outputs."""

from __future__ import annotations

import pandera.pandas as pa

VALID_RATINGS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

reviews_schema = pa.DataFrameSchema(
    {
        "review_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{16}$"), unique=True),
        "rating": pa.Column(float, pa.Check.isin(VALID_RATINGS)),
        "review_text": pa.Column(str, pa.Check.str_length(min_value=1)),
        "n_words": pa.Column(int, pa.Check.ge(1)),
    },
    strict=False,
    coerce=True,
)


def aspect_flags_schema(aspects: list[str]) -> pa.DataFrameSchema:
    cols = {"review_id": pa.Column(str, unique=True)}
    for a in aspects:
        for d in ("pos", "neg"):
            cols[f"{d}_{a}"] = pa.Column(int, pa.Check.isin([0, 1]))
    return pa.DataFrameSchema(cols, strict=False, coerce=True)
