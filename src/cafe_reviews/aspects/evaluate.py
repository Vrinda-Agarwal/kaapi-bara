"""Per-aspect F1 of each extractor against the gold set.

Extractors:
- lexicon-v1: rule-based (configs/aspects.yaml), evaluated directly.
- tfidf-lr-cv: one TF-IDF + logistic regression per flag, trained on the gold
  set itself and scored with 5-fold cross-validated predictions. Only fitted for
  flags with at least `min_pos` positive gold examples.

Outputs reports/results/extractor_f1.csv (counts and scores only).
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from cafe_reviews.aspects.gold import load_gold
from cafe_reviews.config import REPO_ROOT, load_config
from cafe_reviews.ingest.build import DB_PATH, load_reviews

RESULTS = REPO_ROOT / "reports" / "results"
SEED = 20260923
MIN_POS = 15


def flag_columns(aspects: list[str]) -> list[str]:
    return [f"{d}_{a}" for a in aspects for d in ("pos", "neg")]


def score(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n_gold_pos": int(y.sum()),
        "n_pred_pos": int(p.sum()),
        "precision": round(precision_score(y, p, zero_division=0), 3) if p.sum() else None,
        "recall": round(recall_score(y, p, zero_division=0), 3) if y.sum() else None,
        "f1": round(f1_score(y, p, zero_division=0), 3) if (y.sum() or p.sum()) else None,
    }


def cv_predictions(texts: pd.Series, y: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(model, texts, y, cv=cv)


def main() -> None:
    aspects = list(load_config("aspects")["aspects"])
    cols = flag_columns(aspects)
    gold = load_gold()
    reviews = load_reviews().set_index("review_id")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        lex = con.execute("SELECT * FROM aspects_lexicon").df().set_index("review_id")
    gold = gold.set_index("review_id")
    lex = lex.loc[gold.index]
    texts = reviews.loc[gold.index, "review_text"]

    rows = []
    for c in cols:
        y = gold[c].to_numpy()
        rows.append({"extractor": "lexicon-v1", "flag": c, **score(y, lex[c].to_numpy())})
        if y.sum() >= MIN_POS:
            p = cv_predictions(texts, y)
            rows.append({"extractor": "tfidf-lr-cv", "flag": c, **score(y, p)})
        else:
            rows.append({"extractor": "tfidf-lr-cv", "flag": c, "n_gold_pos": int(y.sum()),
                         "n_pred_pos": None, "precision": None, "recall": None, "f1": None})
    out = pd.DataFrame(rows)

    # Micro-averaged F1 over all flags, per extractor (lexicon: all flags).
    y_all = gold[cols].to_numpy().ravel()
    micro = {"lexicon-v1": round(f1_score(y_all, lex[cols].to_numpy().ravel()), 3)}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "extractor_f1.csv", index=False)
    meta = {"n_gold": int(len(gold)), "seed": SEED, "min_pos_for_supervised": MIN_POS,
            "labeler": load_config("gold")["labeler"], "micro_f1": micro}
    (RESULTS / "extractor_f1_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out.pivot(index="flag", columns="extractor", values="f1").loc[cols].to_string())
    print(meta)


if __name__ == "__main__":
    main()
