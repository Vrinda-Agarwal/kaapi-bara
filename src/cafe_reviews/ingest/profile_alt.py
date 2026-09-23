"""Count-only profile of the fallback review file (rating + review_text).

Prints counts, never review text.
"""

from __future__ import annotations

import re
import zipfile

import pandas as pd

from cafe_reviews.config import REPO_ROOT, load_config

COFFEE_RE = re.compile(r"\b(coffee|cafe|café|latte|cappuccino|espresso|kaapi)\b", re.IGNORECASE)


def load_alt(cfg: dict) -> pd.DataFrame:
    src = cfg["fallback_source"]
    with zipfile.ZipFile(REPO_ROOT / src["path"]) as z, z.open(src["member"]) as f:
        return pd.read_csv(f)


def main() -> None:
    cfg = load_config("data_source")
    df = load_alt(cfg)
    t = df["review_text"].fillna("")
    words = t.str.split().str.len()
    print("rows", len(df), "cols", list(df.columns))
    print("exact duplicate rows", int(df.duplicated().sum()), "duplicate texts", int(t.duplicated().sum()))
    print("text starts with RATED", int(t.str.strip().str.upper().str.startswith("RATED").sum()))
    print("contains U+00C3 (mojibake marker)", int(t.str.contains("Ã", regex=False).sum()))
    print("non-ascii", int((~t.map(str.isascii)).sum()))
    print("word count quantiles", words.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(), "empty", int((words == 0).sum()))
    print("mentions a coffee/cafe term", int(t.map(lambda s: bool(COFFEE_RE.search(s))).sum()))


if __name__ == "__main__":
    main()
