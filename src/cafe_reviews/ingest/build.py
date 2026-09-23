"""`make data`: raw file -> cleaned, validated reviews in DuckDB.

Reads the primary Kaggle file if present, otherwise the fallback review file
(see configs/data_source.yaml). Only the fallback path is implemented end to
end; the primary file lacks a verified schema in this environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

from cafe_reviews.clean.reviews import clean_reviews
from cafe_reviews.config import REPO_ROOT, load_config
from cafe_reviews.ingest.profile_alt import load_alt
from cafe_reviews.validate.schemas import reviews_schema

INTERIM = REPO_ROOT / "data" / "interim"
PROCESSED = REPO_ROOT / "data" / "processed"
DB_PATH = PROCESSED / "reviews.duckdb"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config("data_source")
    if (REPO_ROOT / cfg["primary_source"]["path"]).exists():
        raise SystemExit(
            "Primary zomato.csv found. Its pipeline is not built yet; run `make inspect` "
            "and adapt this stage to the verified schema."
        )
    src = cfg["fallback_source"]
    path = REPO_ROOT / src["path"]
    digest = sha256(path)
    if digest != src["sha256"]:
        raise SystemExit(f"Checksum mismatch for {path}: {digest}")

    raw = load_alt(cfg)
    clean, log = clean_reviews(raw)
    clean = reviews_schema.validate(clean)
    log["source"] = src["origin_url"]
    log["sha256"] = digest

    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (INTERIM / "clean_log.json").write_text(json.dumps(log, indent=2) + "\n")

    DB_PATH.unlink(missing_ok=True)
    with duckdb.connect(str(DB_PATH)) as con:
        con.register("clean_df", clean)
        con.execute("CREATE TABLE reviews AS SELECT * FROM clean_df")
    print(json.dumps(log, indent=2))


def load_reviews():
    """Cleaned reviews table as a DataFrame."""
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute("SELECT * FROM reviews ORDER BY review_id").df()


if __name__ == "__main__":
    main()
