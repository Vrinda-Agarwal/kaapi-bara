"""Gold set sampling and label loading.

`sample` writes the labelling sheet (with text) to data/interim/ (gitignored).
Committed labels live in annotation/gold_labels.csv: review_id + flags only.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from cafe_reviews.config import REPO_ROOT, load_config
from cafe_reviews.ingest.build import load_reviews

SHEET = REPO_ROOT / "data" / "interim" / "gold_sheet.jsonl"
LABELS = REPO_ROOT / "annotation" / "gold_labels.csv"

# Short codes used while labelling -> aspect names.
CODES = {
    "cof": "coffee", "food": "food", "serv": "service_staff", "wait": "wait_time",
    "val": "value", "amb": "ambience", "seat": "seating_space", "noise": "noise_crowding",
    "wifi": "wifi_work", "clean": "cleanliness", "loc": "location_access",
}


def sample_ids() -> pd.DataFrame:
    g = load_config("gold")
    reviews = load_reviews()
    return reviews.sample(n=g["n"], random_state=g["seed"]).reset_index(drop=True)


def write_sheet() -> None:
    s = sample_ids()
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8") as f:
        for i, r in s.iterrows():
            f.write(json.dumps({"i": i, "review_id": r.review_id, "text": r.review_text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(s)} rows to {SHEET}")


def parse_codes(code: str) -> dict[str, int]:
    """'food+ serv- val+-' -> flags. '+-' means mixed (both flags)."""
    aspects = list(CODES.values())
    flags = {f"{d}_{a}": 0 for a in aspects for d in ("pos", "neg")}
    for tok in code.split():
        name = tok.rstrip("+-")
        signs = tok[len(name):]
        if name not in CODES or not signs:
            raise ValueError(f"bad code token: {tok!r}")
        a = CODES[name]
        if "+" in signs:
            flags[f"pos_{a}"] = 1
        if "-" in signs:
            flags[f"neg_{a}"] = 1
    return flags


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(LABELS, dtype={"review_id": str})
    return df




CODES_FILE = REPO_ROOT / "data" / "interim" / "gold_codes.tsv"


def build_labels() -> pd.DataFrame:
    """gold_codes.tsv (index -> codes) + sample order -> annotation/gold_labels.csv."""
    s = sample_ids()
    codes: dict[int, str] = {}
    for line in CODES_FILE.read_text(encoding="utf-8").splitlines():
        idx, _, code = line.partition("\t")
        codes[int(idx)] = code.strip()
    missing = set(range(len(s))) - set(codes)
    if missing:
        raise ValueError(f"no labels for sample rows {sorted(missing)[:10]}")
    rows = [{"review_id": s.loc[i, "review_id"], **parse_codes(codes[i])} for i in range(len(s))]
    out = pd.DataFrame(rows)
    out.to_csv(LABELS, index=False)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["sheet"]:
        write_sheet()
    elif sys.argv[1:] == ["labels"]:
        df = build_labels()
        print(f"wrote {len(df)} labels to {LABELS}")
        print(df.drop(columns="review_id").sum().to_string())
