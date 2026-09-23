"""Phase 0 inspection of the raw Zomato file.

Checks every unverified assumption listed in CLAUDE.md and reports counts for
candidate cafe rules and target thresholds. The raw file is only read, never
modified. The report contains counts and restaurant names only, never review
text.

Usage:
    python -m cafe_reviews.ingest.inspect_raw [--raw PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cafe_reviews.config import REPO_ROOT, load_config
from cafe_reviews.ingest.parse import (
    ReviewsParseError,
    as_rating_text_pair,
    fold_text,
    parse_literal_list,
    parse_review_rating,
    review_hash,
)

# Heuristic markers of UTF-8 text decoded as Latin-1/cp1252 ("CafÃ©", "â€™").
# These are detection heuristics, not a proof of mojibake.
MOJIBAKE_RE = re.compile(r"\u00c3[\x80-\xbf]|\u00e2\u20ac|\u00c2[\xa0-\xbf]|\ufffd")
# Literal backslash byte escapes left in text, e.g. "\\xf0\\x9f".
BYTE_ESCAPE_RE = re.compile(r"\\x[0-9a-fA-F]{2}")
# Weak signals that a review text contains a date. Counted, not trusted.
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
ID_LIKE_COLUMN_RE = re.compile(r"date|time|user|reviewer|author|\bid\b|_id", re.IGNORECASE)
REVIEWS_PER_CAFE_BINS = (1, 5, 10, 20, 50)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def strip_query(url: Any) -> str:
    return url.split("?", 1)[0] if isinstance(url, str) else ""


def restaurant_key(df: pd.DataFrame, cols: dict[str, str]) -> pd.Series:
    """Provisional restaurant key: folded name + folded address.

    This is an inspection default, not the final dedup rule (that is the
    owner's decision in Phase 1).
    """
    name = df[cols["name"]].map(fold_text).str.strip()
    addr = df[cols["address"]].map(fold_text).str.strip()
    return name + " @ " + addr


@dataclass
class ReviewScan:
    """Per-review arrays built while streaming reviews_list. No text is kept."""

    row_idx: list[int] = field(default_factory=list)
    rating: list[float] = field(default_factory=list)  # NaN when unparsed
    hash: list[int] = field(default_factory=list)
    row_status: Counter = field(default_factory=Counter)
    parse_error_kinds: Counter = field(default_factory=Counter)
    element_error_rows: int = 0
    element_types: Counter = field(default_factory=Counter)
    element_lengths: Counter = field(default_factory=Counter)
    raw_rating_values: Counter = field(default_factory=Counter)
    text_starts_with_rated: int = 0
    text_empty: int = 0
    text_mojibake: int = 0
    text_byte_escapes: int = 0
    text_non_ascii: int = 0
    text_date_like: int = 0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_idx": np.asarray(self.row_idx, dtype=np.int64),
                "rating": np.asarray(self.rating, dtype=float),
                "hash": np.asarray(self.hash, dtype=np.uint64),
            }
        )


def scan_reviews(cells: pd.Series, rating_regex: str) -> ReviewScan:
    scan = ReviewScan()
    for row_idx, cell in zip(range(len(cells)), cells, strict=True):
        try:
            items = parse_literal_list(cell)
        except ReviewsParseError as exc:
            scan.row_status["parse_failed"] += 1
            scan.parse_error_kinds[str(exc).split(":")[0]] += 1
            continue
        if not items:
            scan.row_status["empty_list"] += 1
            continue
        scan.row_status["parsed_nonempty"] += 1
        row_had_bad_element = False
        for item in items:
            scan.element_types[type(item).__name__] += 1
            if isinstance(item, (tuple, list)):
                scan.element_lengths[len(item)] += 1
            try:
                raw_rating, text = as_rating_text_pair(item)
            except ReviewsParseError:
                row_had_bad_element = True
                continue
            value = parse_review_rating(raw_rating, rating_regex)
            if value is None:
                scan.raw_rating_values[repr(raw_rating)] += 1
            scan.row_idx.append(row_idx)
            scan.rating.append(np.nan if value is None else value)
            scan.hash.append(review_hash(raw_rating, text))
            stripped = text.strip()
            if not stripped:
                scan.text_empty += 1
            if stripped.upper().startswith("RATED"):
                scan.text_starts_with_rated += 1
            if MOJIBAKE_RE.search(text):
                scan.text_mojibake += 1
            if BYTE_ESCAPE_RE.search(text):
                scan.text_byte_escapes += 1
            if not text.isascii():
                scan.text_non_ascii += 1
            if DATE_RE.search(text):
                scan.text_date_like += 1
        scan.element_error_rows += int(row_had_bad_element)
    return scan


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], cols: dict[str, str]) -> pd.Series:
    """Rows matching any condition of a candidate cafe rule."""
    mask = pd.Series(False, index=df.index)
    for cond in rule["any_of"]:
        folded = df[cols[cond["column"]]].map(fold_text)
        for term in cond["contains"]:
            mask |= folded.str.contains(fold_text(term), regex=False)
    return mask


def describe_reviews(reviews: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    """Counts for a set of reviews already restricted and exact-deduped."""
    rated = reviews["rating"].dropna()
    out: dict[str, Any] = {
        "n_reviews": int(len(reviews)),
        "n_rated": int(len(rated)),
        "positive_share_by_threshold": {},
    }
    for t in thresholds:
        n_pos = int((rated >= t).sum())
        out["positive_share_by_threshold"][str(t)] = {
            "n_positive": n_pos,
            "share": round(n_pos / len(rated), 4) if len(rated) else None,
        }
    return out


def dedup_reviews_within_key(reviews: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Attach restaurant key and drop exact (rating, text) repeats within a key."""
    r = reviews.assign(key=keys.to_numpy()[reviews["row_idx"].to_numpy()])
    return r.drop_duplicates(subset=["key", "hash"])


def inspect(df: pd.DataFrame, header: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    cols: dict[str, str] = cfg["columns"]
    report: dict[str, Any] = {}

    # 1. Shape and columns.
    expected_cols = list(cols.values())
    report["shape"] = {
        "n_rows": int(len(df)),
        "n_cols": len(header),
        "expected_n_rows_approx": cfg["expected_n_rows_approx"],
        "expected_n_cols": cfg["expected_n_cols"],
        "columns": header,
        "expected_columns_missing": [c for c in expected_cols if c not in header],
    }
    missing = report["shape"]["expected_columns_missing"]
    if missing:
        # Later checks depend on these columns. Stop rather than guess.
        report["stopped_early"] = f"expected columns missing: {missing}"
        return report

    # 2. reviews_list structure and 3. rating format.
    scan = scan_reviews(df[cols["reviews_list"]], cfg["review_rating_regex"])
    reviews = scan.frame()
    rated = reviews["rating"].dropna()
    report["reviews_list"] = {
        "row_status": dict(sorted(scan.row_status.items())),
        "parse_error_kinds": dict(sorted(scan.parse_error_kinds.items())),
        "rows_with_non_pair_elements": scan.element_error_rows,
        "element_types": dict(sorted(scan.element_types.items())),
        "element_lengths": {str(k): v for k, v in sorted(scan.element_lengths.items())},
        "n_reviews_total_raw": int(len(reviews)),
    }
    value_counts = rated.value_counts().sort_index()
    report["review_ratings"] = {
        "n_matching_pattern": int(len(rated)),
        "n_not_matching_pattern": int(reviews["rating"].isna().sum()),
        "top_non_matching_values": dict(scan.raw_rating_values.most_common(10)),
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        "n_non_integer": int((rated % 1 != 0).sum()),
    }

    # 4. Dates and reviewer IDs.
    report["dates_and_ids"] = {
        "id_or_date_like_columns": [c for c in header if ID_LIKE_COLUMN_RE.search(c)],
        "element_lengths": report["reviews_list"]["element_lengths"],
        "reviews_with_date_like_text": scan.text_date_like,
        "note": "Date-like text is a weak regex signal (e.g. a date mentioned in the review), "
        "not a review timestamp.",
    }

    # 5. Duplication across rows.
    keys = restaurant_key(df, cols)
    rows_per_key = keys.value_counts()
    name_folded = df[cols["name"]].map(fold_text).str.strip()
    addresses_per_name = (
        pd.DataFrame({"name": name_folded, "key": keys}).groupby("name")["key"].nunique()
    )
    key_to_listed = df.groupby(keys)[cols["listed_type"]].nunique()
    reviews_k = reviews.assign(key=keys.to_numpy()[reviews["row_idx"].to_numpy()])
    within_key_dedup = dedup_reviews_within_key(reviews, keys)
    # Do all rows of a restaurant carry the same set of reviews?
    row_sets = reviews_k.groupby("row_idx")["hash"].agg(lambda h: hash(frozenset(h.tolist())))
    row_sets = row_sets.reindex(range(len(df)), fill_value=hash(frozenset()))
    sets_per_key = pd.Series(row_sets.to_numpy(), index=keys.to_numpy()).groupby(level=0).nunique()
    multi_row_keys = rows_per_key[rows_per_key > 1].index
    report["duplication"] = {
        "restaurant_key_definition": "fold_text(name) + ' @ ' + fold_text(address) (provisional)",
        "n_unique_url": int(df[cols["url"]].nunique()),
        "n_unique_url_without_query": int(df[cols["url"]].map(strip_query).nunique()),
        "n_unique_name_address": int(keys.nunique()),
        "n_unique_name": int(name_folded.nunique()),
        "rows_per_restaurant": {
            "max": int(rows_per_key.max()),
            "median": float(rows_per_key.median()),
            "n_restaurants_with_gt1_row": int((rows_per_key > 1).sum()),
            "n_rows_in_those_restaurants": int(rows_per_key[rows_per_key > 1].sum()),
        },
        "n_restaurants_listed_under_gt1_listing_type": int((key_to_listed > 1).sum()),
        "names_with_gt1_address": {
            "n_names": int((addresses_per_name > 1).sum()),
            "top10": {k: int(v) for k, v in addresses_per_name.nlargest(10).items()},
        },
        "reviews": {
            "n_raw": int(len(reviews_k)),
            "n_unique_within_restaurant": int(len(within_key_dedup)),
            "n_unique_global": int(reviews_k["hash"].nunique()),
        },
        "multi_row_restaurants_with_identical_review_sets": int(
            (sets_per_key.reindex(multi_row_keys) == 1).sum()
        ),
        "n_multi_row_restaurants": int(len(multi_row_keys)),
    }

    # 6. Encoding.
    names = df[cols["name"]].fillna("")
    name_moji = names[names.map(lambda s: bool(MOJIBAKE_RE.search(s)))]
    report["encoding"] = {
        "names_with_mojibake_markers": int(name_moji.nunique()),
        "example_names_with_markers": sorted(name_moji.unique().tolist())[:10],
        "names_non_ascii": int(names[~names.map(str.isascii)].nunique()),
        "reviews_with_mojibake_markers": scan.text_mojibake,
        "reviews_with_literal_byte_escapes": scan.text_byte_escapes,
        "reviews_non_ascii": scan.text_non_ascii,
        "reviews_text_starting_with_RATED": scan.text_starts_with_rated,
        "reviews_empty_text": scan.text_empty,
        "note": "Mojibake markers are a regex heuristic and can include false positives.",
    }

    # Aggregate rate column: format only. Leakage rule: never a feature.
    rate = df[cols["aggregate_rate"]].astype("string").str.strip()
    non_numeric = rate[~rate.fillna("").str.match(r"^\d(\.\d)?\s*/\s*5$")]
    report["aggregate_rate_format"] = {
        "n_missing": int(rate.isna().sum()),
        "top_values_not_like_x_over_5": {
            str(k): int(v) for k, v in non_numeric.value_counts(dropna=False).head(10).items()
        },
    }

    # 7. Candidate cafe rules.
    thresholds = [float(t) for t in cfg["candidate_target_thresholds"]]
    rules = cfg["candidate_cafe_rules"]
    rule_keys: dict[str, set[str]] = {}
    rule_rows: dict[str, Any] = {}
    for rule_name, rule in rules.items():
        mask = rule_mask(df, rule, cols)
        rows_idx = np.flatnonzero(mask.to_numpy())
        kset = set(keys.iloc[rows_idx].tolist())
        rule_keys[rule_name] = kset
        r = within_key_dedup[within_key_dedup["key"].isin(kset)]
        per_cafe = r.groupby("key").size().reindex(sorted(kset), fill_value=0)
        rule_rows[rule_name] = {
            "description": rule["description"],
            "n_rows": int(mask.sum()),
            "n_cafes": len(kset),
            "cafes_with_at_least_n_unique_reviews": {
                str(n): int((per_cafe >= n).sum()) for n in REVIEWS_PER_CAFE_BINS
            },
            "median_unique_reviews_per_cafe": float(per_cafe.median()) if len(per_cafe) else None,
            **describe_reviews(r, thresholds),
        }
    names_sorted = list(rules)
    report["candidate_cafe_rules"] = rule_rows
    report["candidate_rule_overlap_n_cafes"] = {
        a: {b: len(rule_keys[a] & rule_keys[b]) for b in names_sorted} for a in names_sorted
    }
    report["all_restaurants_reference"] = describe_reviews(within_key_dedup, thresholds)
    return report


# ---------------------------------------------------------------- rendering


def _table(rows: list[list[Any]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(report: dict[str, Any], meta: dict[str, Any]) -> str:
    out: list[str] = ["# Raw data inspection report", ""]
    out += [
        f"- File: `{meta['raw_file']}`",
        f"- sha256: `{meta['sha256']}`",
        f"- Size: {meta['size_bytes']} bytes",
        f"- Generated by: `python -m cafe_reviews.ingest.inspect_raw` (pandas {meta['pandas']})",
        "- Contains counts and restaurant names only. No review text.",
        "",
    ]
    s = report["shape"]
    out += [
        "## 1. Shape",
        "",
        f"Rows: {s['n_rows']} (assumed about {s['expected_n_rows_approx']}). "
        f"Columns: {s['n_cols']} (assumed {s['expected_n_cols']}).",
        "",
        "Columns: " + ", ".join(f"`{c}`" for c in s["columns"]),
        "",
        f"Expected columns missing: {s['expected_columns_missing'] or 'none'}",
        "",
    ]
    if "stopped_early" in report:
        out += [f"**Stopped early:** {report['stopped_early']}", ""]
        return "\n".join(out)

    rl = report["reviews_list"]
    out += [
        "## 2. reviews_list structure",
        "",
        f"Row parse status: {rl['row_status']}",
        "",
        f"Parse error kinds: {rl['parse_error_kinds'] or 'none'}",
        "",
        f"Element types: {rl['element_types']}. Element lengths: {rl['element_lengths']}.",
        "",
        f"Rows with any element that is not a (str, str) pair: {rl['rows_with_non_pair_elements']}",
        "",
        f"Total reviews across all rows (before any dedup): {rl['n_reviews_total_raw']}",
        "",
    ]
    rr = report["review_ratings"]
    out += [
        "## 3. Review rating format",
        "",
        f"Matching the configured pattern: {rr['n_matching_pattern']}. "
        f"Not matching: {rr['n_not_matching_pattern']}.",
        "",
        f"Most common non-matching raw values: {rr['top_non_matching_values'] or 'none'}",
        "",
        f"Non-integer (e.g. half-star) ratings: {rr['n_non_integer']}",
        "",
        _table([[k, v] for k, v in rr["value_counts"].items()], ["rating", "count"]),
        "",
    ]
    d = report["dates_and_ids"]
    out += [
        "## 4. Review dates and reviewer IDs",
        "",
        f"Columns with date/time/user/id-like names: {d['id_or_date_like_columns'] or 'none'}",
        "",
        f"reviews_list element lengths: {d['element_lengths']} "
        "(2 means only rating and text)",
        "",
        f"Reviews whose text contains a date-like pattern: {d['reviews_with_date_like_text']}. "
        + d["note"],
        "",
    ]
    du = report["duplication"]
    rp = du["rows_per_restaurant"]
    rv = du["reviews"]
    out += [
        "## 5. Duplication across rows",
        "",
        f"Restaurant key used here: {du['restaurant_key_definition']}",
        "",
        _table(
            [
                ["unique url", du["n_unique_url"]],
                ["unique url without query string", du["n_unique_url_without_query"]],
                ["unique name + address", du["n_unique_name_address"]],
                ["unique name", du["n_unique_name"]],
                ["restaurants with more than 1 row", rp["n_restaurants_with_gt1_row"]],
                ["rows belonging to those restaurants", rp["n_rows_in_those_restaurants"]],
                ["max rows for one restaurant", rp["max"]],
                ["median rows per restaurant", rp["median"]],
                [
                    "restaurants listed under more than 1 listing type",
                    du["n_restaurants_listed_under_gt1_listing_type"],
                ],
                ["names with more than 1 address (possible chains)", du["names_with_gt1_address"]["n_names"]],
                [
                    "multi-row restaurants whose rows all carry the same review set",
                    f"{du['multi_row_restaurants_with_identical_review_sets']} of {du['n_multi_row_restaurants']}",
                ],
                ["reviews, raw", rv["n_raw"]],
                ["reviews, unique (rating, text) within restaurant", rv["n_unique_within_restaurant"]],
                ["reviews, unique (rating, text) globally", rv["n_unique_global"]],
            ],
            ["measure", "count"],
        ),
        "",
        "Names with the most distinct addresses: "
        + ", ".join(f"{k} ({v})" for k, v in du["names_with_gt1_address"]["top10"].items()),
        "",
    ]
    e = report["encoding"]
    out += [
        "## 6. Encoding",
        "",
        _table(
            [
                ["distinct names with mojibake markers", e["names_with_mojibake_markers"]],
                ["distinct names with non-ASCII characters", e["names_non_ascii"]],
                ["reviews with mojibake markers", e["reviews_with_mojibake_markers"]],
                ["reviews with literal \\xNN byte escapes", e["reviews_with_literal_byte_escapes"]],
                ["reviews with non-ASCII characters", e["reviews_non_ascii"]],
                ["reviews whose text starts with 'RATED'", e["reviews_text_starting_with_RATED"]],
                ["reviews with empty text", e["reviews_empty_text"]],
            ],
            ["measure", "count"],
        ),
        "",
        "Example names with markers: " + (", ".join(e["example_names_with_markers"]) or "none"),
        "",
        e["note"],
        "",
        "Aggregate `rate` values not shaped like `x.x/5` (format check only; never a feature): "
        f"{report['aggregate_rate_format']['top_values_not_like_x_over_5']}",
        "",
    ]
    rules = report["candidate_cafe_rules"]
    thresholds = list(report["all_restaurants_reference"]["positive_share_by_threshold"])
    out += [
        "## 7. Candidate cafe rules (decision pending)",
        "",
        "Review counts below are after dropping exact (rating, text) repeats within the "
        "provisional restaurant key. That dedup is for counting only.",
        "",
    ]
    bins = [str(n) for n in REVIEWS_PER_CAFE_BINS]
    out.append(
        _table(
            [
                [name, r["description"], r["n_rows"], r["n_cafes"]]
                + [r["cafes_with_at_least_n_unique_reviews"][b] for b in bins]
                + [r["median_unique_reviews_per_cafe"], r["n_reviews"], r["n_rated"]]
                for name, r in rules.items()
            ],
            ["rule", "description", "rows", "cafes"]
            + [f"cafes >= {b} reviews" for b in bins]
            + ["median reviews/cafe", "reviews", "rated reviews"],
        )
    )
    ov = report["candidate_rule_overlap_n_cafes"]
    names = list(ov)
    out += [
        "",
        "Overlap (number of cafes in both rules):",
        "",
        _table([[a] + [ov[a][b] for b in names] for a in names], ["rule"] + names),
        "",
        "## 8. Candidate target thresholds (decision pending)",
        "",
        "Share of rated reviews at or above each threshold.",
        "",
    ]
    ref = report["all_restaurants_reference"]
    rows = [
        [name]
        + [
            f"{r['positive_share_by_threshold'][t]['n_positive']} ({_pct(r['positive_share_by_threshold'][t]['share'])})"
            for t in thresholds
        ]
        for name, r in [*rules.items(), ("all restaurants (reference)", ref)]
    ]
    out += [_table(rows, ["rule"] + [f">= {t}" for t in thresholds]), ""]
    return "\n".join(out)


# --------------------------------------------------------------------- CLI


def read_raw(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [c for c in cfg["columns"].values() if c in header]
    df = pd.read_csv(path, usecols=wanted, dtype=str, keep_default_na=True)
    return df, header


def main(argv: list[str] | None = None) -> int:
    cfg = load_config("cafe_definition")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / cfg["raw_file"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "inspection")
    args = parser.parse_args(argv)

    if not args.raw.exists():
        print(f"Raw file not found: {args.raw}")
        print("Download zomato.csv manually from Kaggle into data/raw/ (see DATA.md).")
        return 1

    meta = {
        "raw_file": str(args.raw.relative_to(REPO_ROOT) if args.raw.is_relative_to(REPO_ROOT) else args.raw),
        "sha256": sha256_file(args.raw),
        "size_bytes": args.raw.stat().st_size,
        "pandas": pd.__version__,
    }
    print(f"sha256 {meta['sha256']}  {meta['raw_file']}")
    df, header = read_raw(args.raw, cfg)
    report = inspect(df, header, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inspection_summary.json").write_text(
        json.dumps({"meta": meta, **report}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "inspection_report.md").write_text(render_markdown(report, meta), encoding="utf-8")
    print(f"Wrote {args.out / 'inspection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
