"""Phase 2 EDA: rating distribution and review length (counts only).

Segment, price, and neighbourhood views from the plan need cafe attributes,
which the fallback data does not have.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafe_reviews.config import REPO_ROOT
from cafe_reviews.ingest.build import INTERIM, load_reviews

RESULTS = REPO_ROOT / "reports" / "results"
FIGURES = REPO_ROOT / "reports" / "figures"


def summary() -> dict:
    df = load_reviews()
    clean_log = json.loads((INTERIM / "clean_log.json").read_text())
    counts = df["rating"].value_counts().sort_index()
    by_rating_words = df.groupby("rating")["n_words"].median()
    return {
        "clean_log": clean_log,
        "rating_counts": {f"{k:.1f}": int(v) for k, v in counts.items()},
        "share_5_star": round(float((df.rating == 5).mean()), 4),
        "share_ge_4_5": round(float((df.rating >= 4.5).mean()), 4),
        "share_ge_4": round(float((df.rating >= 4).mean()), 4),
        "words_quantiles": {str(q): float(v) for q, v in df.n_words.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()},
        "median_words_by_rating": {f"{k:.1f}": float(v) for k, v in by_rating_words.items()},
    }


def main() -> None:
    s = summary()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eda_summary.json").write_text(json.dumps(s, indent=2) + "\n")

    ratings = list(s["rating_counts"])
    vals = np.array(list(s["rating_counts"].values()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(ratings, vals, color=["#C44E52" if r != "5.0" else "#4C72B0" for r in ratings])
    axes[0].set_xlabel("Review rating")
    axes[0].set_ylabel("Reviews (after cleaning)")
    axes[0].set_title("Rating distribution (5.0 = target class)")
    med = s["median_words_by_rating"]
    axes[1].plot(list(med), list(med.values()), marker="o", color="#4C72B0")
    axes[1].set_xlabel("Review rating")
    axes[1].set_ylabel("Median words per review")
    axes[1].set_title("Review length by rating")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_ratings.png", dpi=150)
    plt.close(fig)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
