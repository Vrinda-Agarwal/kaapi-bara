"""Clean and deduplicate the fallback review file.

Steps (each counted and written to data/interim/clean_log.json):
1. Repair mojibake by reversing repeated UTF-8-read-as-Latin-1 decoding.
2. Strip a leading "RATED" marker and collapse whitespace.
3. Drop reviews with empty text.
4. Drop exact duplicates of (rating, normalized text).
5. Drop texts that appear with more than one rating (ambiguous label).

Defaults picked here (not owner decisions yet): the duplicate key in step 4 and
the conflict rule in step 5.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

_MOJIBAKE_MARKER = re.compile("[ÃÂâ]")
# C1 control characters and leftover mojibake runs after repair.
_C1 = re.compile("[\u0080-\u009f]")
_RESIDUAL = re.compile("(?:[ÃÂ][\u0080-¿ -ÿ]?)+")
_RATED_PREFIX = re.compile(r"^\s*RATED\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")


def repair_mojibake(s: str, max_rounds: int = 8) -> tuple[str, bool]:
    """Undo repeated UTF-8 -> Latin-1 misdecoding. Returns (text, changed)."""
    original = s
    for _ in range(max_rounds):
        if not _MOJIBAKE_MARKER.search(s):
            break
        try:
            fixed = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if fixed == s:
            break
        s = fixed
    s = _C1.sub("", s)
    s = _RESIDUAL.sub(" ", s)
    return s, s != original


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _RATED_PREFIX.sub("", s)
    return _WS.sub(" ", s).strip()


def dedup_key(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def review_id(rating: float, text: str) -> str:
    """Stable ID from rating and normalized text. Safe to commit (no text)."""
    return hashlib.sha1(f"{rating:.1f}\x1f{dedup_key(text)}".encode()).hexdigest()[:16]


def clean_reviews(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"n_raw": int(len(raw))}
    df = raw[["rating", "review_text"]].copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)

    repaired = df["review_text"].map(repair_mojibake)
    df["review_text"] = repaired.map(lambda x: x[0]).map(normalize_text)
    log["n_mojibake_repaired"] = int(repaired.map(lambda x: x[1]).sum())

    empty = df["review_text"].str.len() == 0
    log["n_dropped_empty_text"] = int(empty.sum())
    df = df[~empty]

    missing_rating = df["rating"].isna()
    log["n_dropped_missing_rating"] = int(missing_rating.sum())
    df = df[~missing_rating]

    df["_key"] = df["review_text"].map(dedup_key)
    before = len(df)
    df = df.drop_duplicates(subset=["rating", "_key"], keep="first")
    log["n_dropped_exact_duplicates"] = int(before - len(df))

    n_ratings = df.groupby("_key")["rating"].transform("nunique")
    conflict = n_ratings > 1
    log["n_dropped_conflicting_rating"] = int(conflict.sum())
    df = df[~conflict]

    df["review_id"] = [review_id(r, t) for r, t in zip(df["rating"], df["review_text"], strict=True)]
    df["n_words"] = df["review_text"].str.split().str.len().astype(int)
    df = df.drop(columns="_key").sort_values("review_id").reset_index(drop=True)
    log["n_clean"] = int(len(df))
    return df[["review_id", "rating", "review_text", "n_words"]], log
