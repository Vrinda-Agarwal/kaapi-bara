"""`make train`: predict whether a review is 5-star.

Models
- B0: majority class.
- M1: TF-IDF word (1-2) + char (3-5) n-grams, logistic regression.
- M2: lexicon aspect flags + log word count, logistic regression.
- M3: histogram gradient-boosted trees on the M2 features, explained with SHAP.
B1 (cafe attributes) and M4 (transformer) are not run: the fallback data has no
cafe attributes and model downloads are blocked in the build environment.

Evaluation: stratified 80/20 split, test set used once. 5-fold CV on the training
part gives CV metrics and the decision threshold (max F1 on out-of-fold
predictions). Grouped CV by cafe is impossible: the data has no cafe identifier.
"""

from __future__ import annotations

import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.sparse import hstack
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cafe_reviews.config import REPO_ROOT, load_config
from cafe_reviews.ingest.build import DB_PATH, load_reviews

RESULTS = REPO_ROOT / "reports" / "results"
FIGURES = REPO_ROOT / "reports" / "figures"


def load_modeling_frame() -> tuple[pd.DataFrame, list[str]]:
    cfg = load_config("modeling")
    reviews = load_reviews()
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        flags = con.execute("SELECT * FROM aspects_lexicon").df()
    df = reviews.merge(flags, on="review_id", validate="one_to_one")
    df["y"] = (df["rating"] >= cfg["target_threshold"]).astype(int)
    df["log_words"] = np.log1p(df["n_words"])
    flag_cols = [c for c in flags.columns if c != "review_id"]
    return df, flag_cols


class TextModel:
    def __init__(self, seed: int):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=50000)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True,
                                    max_features=100000)
        self.clf = LogisticRegression(max_iter=3000, C=4.0, random_state=seed)

    def fit(self, texts, y):
        X = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts):
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


def make_model(name: str, seed: int):
    if name == "B0_majority":
        return DummyClassifier(strategy="prior")
    if name == "M1_tfidf_lr":
        return TextModel(seed)
    if name == "M2_aspects_lr":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "M3_aspects_gbt":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              random_state=seed)
    raise ValueError(name)


def fit_predict(name, seed, train, test, feats):
    m = make_model(name, seed)
    if name == "M1_tfidf_lr":
        m.fit(train["review_text"], train["y"])
        return m, m.predict_proba(test["review_text"])
    m.fit(train[feats], train["y"])
    return m, m.predict_proba(test[feats])[:, 1]


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, p >= t, average="macro") for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metrics(y, p, thr) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "macro_f1": round(f1_score(y, p >= thr, average="macro"), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "threshold": round(thr, 2),
    }


def main() -> None:
    cfg = load_config("modeling")
    seed = cfg["seed"]
    df, flag_cols = load_modeling_frame()
    feats = flag_cols + ["log_words"]
    train, test = train_test_split(df, test_size=cfg["test_size"], stratify=df["y"], random_state=seed)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    names = ["B0_majority", "M1_tfidf_lr", "M2_aspects_lr", "M3_aspects_gbt"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows, test_probs = [], {}
    for name in names:
        oof = np.zeros(len(train))
        for tr, va in skf.split(train, train["y"]):
            _, oof[va] = fit_predict(name, seed, train.iloc[tr], train.iloc[va], feats)
        thr = best_threshold(train["y"], oof) if name != "B0_majority" else 0.5
        cv = metrics(train["y"], oof, thr)
        model, p_test = fit_predict(name, seed, train, test, feats)
        test_probs[name] = p_test
        te = metrics(test["y"], p_test, thr)
        rows.append({"model": name, **{f"cv_{k}": v for k, v in cv.items() if k != "threshold"},
                     "threshold": thr, **{f"test_{k}": v for k, v in te.items() if k != "threshold"}})
        if name == "M3_aspects_gbt":
            gbt = model
    res = pd.DataFrame(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "prediction_results.csv", index=False)
    meta = {"target": f"rating >= {cfg['target_threshold']}", "seed": seed, "test_size": cfg["test_size"],
            "n_train": int(len(train)), "n_test": int(len(test)),
            "positive_rate_train": round(float(train["y"].mean()), 4),
            "positive_rate_test": round(float(test["y"].mean()), 4),
            "not_run": {"B1": "no cafe attributes in fallback data",
                        "M4": "model downloads blocked; skipped per owner"},
            "grouping": "none possible (no cafe identifier); exact duplicates removed before split"}
    (RESULTS / "prediction_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Reliability plot (test set).
    rel = []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    for name in names[1:]:
        frac, mean_p = calibration_curve(test["y"], test_probs[name], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=3, label=name)
        rel += [{"model": name, "mean_predicted": round(float(a), 4), "observed_rate": round(float(b), 4)}
                for a, b in zip(mean_p, frac, strict=True)]
    ax.set_xlabel("Mean predicted probability of 5 stars")
    ax.set_ylabel("Observed share of 5-star reviews")
    ax.set_title("Reliability on held-out reviews")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rel).to_csv(RESULTS / "reliability.csv", index=False)

    # SHAP for M3 on the test set.
    explainer = shap.TreeExplainer(gbt)
    sv = explainer.shap_values(test[feats])
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0),
                        "mean_shap_when_1": [float(sv[test[f].to_numpy() == 1, i].mean())
                                             if f != "log_words" and (test[f] == 1).any() else np.nan
                                             for i, f in enumerate(feats)]})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.round(5).to_csv(RESULTS / "shap_m3.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("M3 features most associated with predictions")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_m3.png", dpi=150)
    plt.close(fig)

    print(res.to_string(index=False))
    print(json.dumps(meta, indent=2))
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
