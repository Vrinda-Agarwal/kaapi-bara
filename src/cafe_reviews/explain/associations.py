"""`make explain`: how aspect praise/complaints are associated with 5-star reviews.

Design limits (fallback data): there is no cafe identifier, so the planned
within-cafe fixed-effects model, cafe-clustered SEs, between-cafe model,
mixed-effects check, and cafe-weighted check cannot be run. This module fits
pooled review-level models instead:

- Main: linear probability model (OLS), y = 1[5-star] on all pos_/neg_ flags +
  log word count, HC1 robust SEs. Coefficients read as percentage-point
  differences in the chance of a 5-star review, holding other flags fixed.
- Robustness: logistic regression with the same terms (sign and significance).
- Benjamini-Hochberg over all flag coefficients.
- H1: bootstrap (reviews resampled) share of draws in which neg_service_staff
  is the most negative neg_ coefficient, plus pairwise Wald tests.
- H2: asymmetry = beta_pos + beta_neg per aspect with CI (negative = the
  complaint penalty outweighs the praise premium).

All results are associations, not effects.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from cafe_reviews.config import REPO_ROOT, load_config
from cafe_reviews.models.train import FIGURES, RESULTS, load_modeling_frame

N_BOOT = 1000
MIN_PREVALENCE_COUNT = 30  # flags rarer than this are dropped from the models
WIDE_CI_PP = 15  # H2: a 95% CI wider than this (pp) is called inconclusive


def fit_ols(df: pd.DataFrame, cols: list[str]):
    X = sm.add_constant(df[cols].astype(float))
    return sm.OLS(df["y"].astype(float), X).fit(cov_type="HC1")


def main() -> None:
    seed = load_config("modeling")["seed"]
    aspects = list(load_config("aspects")["aspects"])
    df, flag_cols = load_modeling_frame()
    counts = df[flag_cols].sum()
    kept = [c for c in flag_cols if counts[c] >= MIN_PREVALENCE_COUNT]
    dropped = {c: int(counts[c]) for c in flag_cols if c not in kept}
    cols = kept + ["log_words"]

    ols = fit_ols(df, cols)
    logit = sm.Logit(df["y"].astype(float), sm.add_constant(df[cols].astype(float))).fit(disp=0, cov_type="HC1")
    ci = ols.conf_int()
    coef = pd.DataFrame({
        "term": kept,
        "n_reviews_with_flag": [int(counts[c]) for c in kept],
        "lpm_coef_pp": (ols.params[kept] * 100).round(2).to_numpy(),
        "lpm_ci_low_pp": (ci.loc[kept, 0] * 100).round(2).to_numpy(),
        "lpm_ci_high_pp": (ci.loc[kept, 1] * 100).round(2).to_numpy(),
        "lpm_p": ols.pvalues[kept].to_numpy(),
        "logit_coef": logit.params[kept].round(3).to_numpy(),
        "logit_p": logit.pvalues[kept].to_numpy(),
    })
    coef["lpm_p_bh"] = multipletests(coef["lpm_p"], method="fdr_bh")[1]
    coef["logit_p_bh"] = multipletests(coef["logit_p"], method="fdr_bh")[1]
    coef["signs_agree"] = np.sign(coef["lpm_coef_pp"]) == np.sign(coef["logit_coef"])

    # H1: is neg_service_staff the largest complaint penalty?
    neg_terms = [c for c in kept if c.startswith("neg_")]
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(df), len(df))
        b = fit_ols(df.iloc[idx], cols).params[neg_terms]
        wins += int(b.idxmin() == "neg_service_staff")
    pairwise = []
    for t in neg_terms:
        if t == "neg_service_staff":
            continue
        test = ols.t_test(f"neg_service_staff - {t} = 0")
        pairwise.append({"other": t, "diff_pp": round(float(test.effect[0]) * 100, 2),
                         "p": float(test.pvalue)})
    pw = pd.DataFrame(pairwise)
    pw["p_bh"] = multipletests(pw["p"], method="fdr_bh")[1]
    ranked = coef[coef.term.str.startswith("neg_")].sort_values("lpm_coef_pp")
    h1 = {
        "most_negative_neg_term": ranked.iloc[0]["term"],
        "neg_service_staff_coef_pp": float(coef.set_index("term").loc["neg_service_staff", "lpm_coef_pp"]),
        "bootstrap_share_service_most_negative": round(wins / N_BOOT, 3),
        "pairwise_vs_service": pw.round(4).to_dict("records"),
    }
    n_sig_worse = int(((pw["diff_pp"] < 0) & (pw["p_bh"] < 0.05)).sum())
    if h1["most_negative_neg_term"] == "neg_service_staff" and n_sig_worse == len(pw):
        h1["verdict"] = "supported"
    elif h1["most_negative_neg_term"] == "neg_service_staff":
        h1["verdict"] = "partly supported (largest point estimate, not separable from every other complaint)"
    else:
        h1["verdict"] = "not supported"

    # H2: asymmetry per aspect where both flags are in the model.
    asym = []
    for a in aspects:
        p, n = f"pos_{a}", f"neg_{a}"
        if p in kept and n in kept:
            t = ols.t_test(f"{p} + {n} = 0")
            lo, hi = t.conf_int()[0]
            asym.append({"aspect": a, "pos_pp": round(ols.params[p] * 100, 2),
                         "neg_pp": round(ols.params[n] * 100, 2),
                         "asymmetry_pp": round(float(t.effect[0]) * 100, 2),
                         "ci_low_pp": round(lo * 100, 2), "ci_high_pp": round(hi * 100, 2),
                         "p": float(t.pvalue)})
        else:
            asym.append({"aspect": a, "pos_pp": None, "neg_pp": None, "asymmetry_pp": None,
                         "ci_low_pp": None, "ci_high_pp": None, "p": None,
                         "note": "a flag is too rare to estimate"})
    asym = pd.DataFrame(asym)

    def h2_status(a: str, expect: str) -> str:
        r = asym.set_index("aspect").loc[a]
        if r["asymmetry_pp"] is None or pd.isna(r["asymmetry_pp"]):
            return "not testable (too few reviews with this flag)"
        if expect == "penalty":
            return "consistent" if r["ci_high_pp"] < 0 else "not consistent"
        # Symmetric or delighter: the CI does not exclude zero from above (penalty not dominant).
        if r["ci_high_pp"] < 0:
            return "not consistent"
        if r["ci_high_pp"] - r["ci_low_pp"] > WIDE_CI_PP:
            return "inconclusive (CI too wide to tell symmetric from penalty-dominant)"
        return "consistent"

    h2 = {"penalty_dominant_expected": {a: h2_status(a, "penalty") for a in ["cleanliness", "wait_time", "wifi_work"]},
          "symmetric_or_delighter_expected": {a: h2_status(a, "sym") for a in ["coffee", "ambience"]}}
    statuses = list(h2["penalty_dominant_expected"].values()) + list(h2["symmetric_or_delighter_expected"].values())
    n_cons = sum(s == "consistent" for s in statuses)
    n_not = sum(s == "not consistent" for s in statuses)
    if n_cons == len(statuses):
        h2["verdict"] = "supported"
    elif n_cons and not n_not:
        h2["verdict"] = "partly supported (no aspect contradicts it; some are inconclusive or not testable)"
    elif n_cons:
        h2["verdict"] = "mixed"
    else:
        h2["verdict"] = "not supported"

    verdicts = {
        "H1": h1,
        "H2": h2,
        "H3": {"verdict": "not testable", "reason": "no price tier or cafe segment in fallback data"},
        "H4": {"verdict": "not testable", "reason": "no cafe-level price or segment in fallback data"},
        "H5": {"verdict": "not testable", "reason": "no neighbourhood in fallback data"},
        "design": {"model": "pooled review-level LPM with HC1 SEs; logit robustness",
                   "n_reviews": int(len(df)), "bootstrap_draws": N_BOOT, "seed": seed,
                   "dropped_rare_flags": dropped,
                   "not_run": ["within-cafe fixed effects", "cafe-clustered SEs", "between-cafe model",
                               "mixed-effects check", "cafe-weighted check"],
                   "reason_not_run": "no cafe identifier in fallback data",
                   "measurement": "aspect flags from lexicon-v1 (micro-F1 0.71 vs single-annotator gold); "
                                  "misclassification biases coefficients, usually toward zero"},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    coef.round(6).to_csv(RESULTS / "associations_coefficients.csv", index=False)
    asym.round(6).to_csv(RESULTS / "associations_asymmetry.csv", index=False)
    (RESULTS / "hypotheses.json").write_text(json.dumps(verdicts, indent=2, default=str) + "\n")

    # Coefficient plot.
    plot = coef.sort_values("lpm_coef_pp")
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#C44E52" if t.startswith("neg_") else "#4C72B0" for t in plot.term]
    ax.errorbar(plot.lpm_coef_pp, plot.term, xerr=[plot.lpm_coef_pp - plot.lpm_ci_low_pp,
                plot.lpm_ci_high_pp - plot.lpm_coef_pp], fmt="none", ecolor="grey", lw=1)
    ax.scatter(plot.lpm_coef_pp, plot.term, c=colors, zorder=3, s=20)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Difference in chance of a 5-star review (percentage points, 95% CI)")
    ax.set_title("Aspect mentions associated with 5-star reviews (pooled LPM)")
    fig.tight_layout()
    fig.savefig(FIGURES / "associations.png", dpi=150)
    plt.close(fig)

    print(coef.round(4).to_string(index=False))
    print(asym.to_string(index=False))
    print(json.dumps({k: v.get("verdict") for k, v in verdicts.items() if "verdict" in v}, indent=2))
    print(json.dumps(h1, indent=2, default=str))
    print(json.dumps(h2, indent=2))


if __name__ == "__main__":
    main()
