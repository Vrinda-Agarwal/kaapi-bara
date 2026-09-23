"""`make aspects`: run the lexicon extractor on all cleaned reviews -> DuckDB."""

from __future__ import annotations

import duckdb

from cafe_reviews.aspects.lexicon import LexiconExtractor
from cafe_reviews.config import load_config
from cafe_reviews.ingest.build import DB_PATH, load_reviews
from cafe_reviews.validate.schemas import aspect_flags_schema


def main() -> None:
    cfg = load_config("aspects")
    ext = LexiconExtractor.from_config(cfg)
    reviews = load_reviews()
    flags = ext.extract_frame(reviews)
    flags = aspect_flags_schema(list(cfg["aspects"])).validate(flags)
    with duckdb.connect(str(DB_PATH)) as con:
        con.register("flags_df", flags)
        con.execute("CREATE OR REPLACE TABLE aspects_lexicon AS SELECT * FROM flags_df")
    prev = flags.drop(columns="review_id").mean().round(3)
    print(f"{ext.version}: {len(flags)} reviews")
    print(prev.to_string())


if __name__ == "__main__":
    main()
