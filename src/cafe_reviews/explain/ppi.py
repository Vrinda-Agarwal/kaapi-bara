"""Phase 6: prediction-powered inference (PPI) for aspect prevalence, and a
comparison of conclusions across label sources.

PPI for a mean (implemented by hand; the `ppi_py` package is not a project
dependency and was not verified in this environment). TODO(citation): PPI
(Angelopoulos et al.).

  theta_ppi = mean_{unlabeled}(f) - mean_{labeled}(f - y)
  var       = var_{unlabeled}(f) / N + var_{labeled}(f - y) / n

Labeled set: the 400 gold reviews (y = gold flag, f = lexicon flag).
Unlabeled set: every other cleaned review (f = lexicon flag).
Reviews are treated as independent. The fallback data has no cafe identifier, so
clustered PPI is neither needed nor possible here.

Intervals compared per flag:
- naive: lexicon prevalence over all reviews (binomial CI), ignores extractor error;
- gold-only: gold prevalence over 400 reviews (normal CI);
- PPI: combines both.

Extractor comparison: on the 400 gold reviews, fit the same pooled LPM once
with gold flags and once with lexicon flags, and compare coefficient signs.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from cafe_reviews.aspects.gold import load_gold
from cafe_reviews.models.train import FIGURES, RESULTS, load_modeling_frame

Z = 1.959964


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
    return m, m - Z * se, m + Z * se


def ppi_mean(f_unl: np.ndarray, f_lab: np.ndarray, y_lab: np.ndarray) -> tuple[float, float, float]:
    rect = f_lab - y_lab
    theta = float(f_unl.mean() - rect.mean())
    se = float(np.sqrt(f_unl.var(ddof=1) / len(f_unl) + rect.var(ddof=1) / len(rect)))
    return theta, theta - Z * se, theta + Z * se


def main() -> None:
    df, flag_cols = load_modeling_frame()
    gold = load_gold().set_index("review_id")
    is_lab = df["review_id"].isin(gold.index)
    lab = df[is_lab].set_index("review_id").loc[gold.index]
    unl = df[~is_lab]

    rows = []
    for c in flag_cols:
        n_m, n_lo, n_hi = mean_ci(df[c].to_numpy(float))
        g_m, g_lo, g_hi = mean_ci(gold[c].to_numpy(float))
        p_m, p_lo, p_hi = ppi_mean(unl[c].to_numpy(float), lab[c].to_numpy(float), gold[c].to_numpy(float))
        for method, m, lo, hi in [("naive_lexicon", n_m, n_lo, n_hi), ("gold_only", g_m, g_lo, g_hi),
                                  ("ppi", p_m, p_lo, p_hi)]:
            rows.append({"flag": c, "method": method, "estimate": round(m, 4),
                         "ci_low": round(max(lo, 0.0) if not np.isnan(lo) else lo, 4),
                         "ci_high": round(hi, 4)})
    prev = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    prev.to_csv(RESULTS / "prevalence_ppi.csv", index=False)

    # Figure: naive vs gold-only vs PPI.
    order = [c for c in flag_cols]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    offsets = {"naive_lexicon": -0.25, "gold_only": 0.0, "ppi": 0.25}
    colors = {"naive_lexicon": "#8C8C8C", "gold_only": "#DD8452", "ppi": "#4C72B0"}
    for method, off in offsets.items():
        sub = prev[prev.method == method].set_index("flag").loc[order]
        y = np.arange(len(order)) + off
        ax.errorbar(sub.estimate * 100, y, xerr=[(sub.estimate - sub.ci_low) * 100, (sub.ci_high - sub.estimate) * 100],
                    fmt="o", ms=3, color=colors[method], lw=1, label=method.replace("_", " "))
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of reviews with the flag (%), 95% CI")
    ax.set_title("Aspect prevalence: naive lexicon vs gold-only vs PPI")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "ppi_prevalence.png", dpi=150)
    plt.close(fig)

    # Extractor comparison on the gold reviews.
    keep = [c for c in flag_cols if gold[c].sum() >= 10 and lab[c].sum() >= 10]
    y = lab["y"].astype(float).to_numpy()
    lw = lab["log_words"].to_numpy()

    def fit(flags: pd.DataFrame):
        X = sm.add_constant(np.column_stack([flags[keep].to_numpy(float), lw]))
        r = sm.OLS(y, X).fit(cov_type="HC1")
        return pd.DataFrame({"coef_pp": r.params[1:-1] * 100, "p": r.pvalues[1:-1]}, index=keep)

    g, lx = fit(gold), fit(lab)
    comp = pd.DataFrame({"flag": keep, "gold_coef_pp": g.coef_pp.round(2).to_numpy(), "gold_p": g.p.round(4).to_numpy(),
                         "lexicon_coef_pp": lx.coef_pp.round(2).to_numpy(), "lexicon_p": lx.p.round(4).to_numpy()})
    comp["same_sign"] = np.sign(comp.gold_coef_pp) == np.sign(comp.lexicon_coef_pp)
    comp.to_csv(RESULTS / "extractor_conclusions.csv", index=False)
    summary = {"n_labeled": int(len(lab)), "n_unlabeled": int(len(unl)),
               "flags_compared": len(keep), "same_sign": int(comp.same_sign.sum()),
               "ppi_note": "independent reviews assumed; no cafe clustering possible"}
    (RESULTS / "ppi_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(prev.pivot(index="flag", columns="method", values="estimate").loc[order].to_string())
    print(comp.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
