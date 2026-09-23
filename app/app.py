"""Streamlit dashboard: reads only aggregate outputs in reports/ (no review text).

Run: uv run streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"


def read_json(name: str) -> dict:
    return json.loads((RES / name).read_text())


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name)


st.set_page_config(page_title="Filter Kaapi to Flat Whites", layout="wide")
st.title("From Filter Kaapi to Flat Whites")
st.caption("What separates 5-star restaurant reviews in Bengaluru (Zomato, circa 2019). "
           "All results are associations, not causal effects.")

missing = [n for n in ["eda_summary.json", "prediction_results.csv", "hypotheses.json"] if not (RES / n).exists()]
if missing:
    st.error(f"Missing result files: {missing}. Run `make data aspects train explain report` first.")
    st.stop()

eda = read_json("eda_summary.json")
pred = read_csv("prediction_results.csv")
pred_meta = read_json("prediction_meta.json")
hyp = read_json("hypotheses.json")
coef = read_csv("associations_coefficients.csv")
asym = read_csv("associations_asymmetry.csv")
f1 = read_csv("extractor_f1.csv")
f1_meta = read_json("extractor_f1_meta.json")
prev = read_csv("prevalence_ppi.csv")
concl = read_csv("extractor_conclusions.csv")
shap_df = read_csv("shap_m3.csv")

st.warning(
    "Data caveat: the planned Kaggle file (with restaurant names, prices, and locations) could not be "
    "obtained in the build environment. These results use a public review-level extract with only a "
    "rating and review text. Within-cafe analysis and H3 to H5 could not be run. Aspect labels in the "
    "gold set come from a single AI annotator, not people. See the Limitations tab."
)

tabs = st.tabs(["Overview", "Data", "Aspects and extractors", "Prediction", "Associations", "Limitations"])

with tabs[0]:
    log = eda["clean_log"]
    best = pred[pred.model != "B0_majority"].sort_values("test_roc_auc", ascending=False).iloc[0]
    c = st.columns(4)
    c[0].metric("Reviews analysed", f"{log['n_clean']:,}", help=f"from {log['n_raw']:,} raw rows")
    c[1].metric("Share rated 5.0", f"{eda['share_5_star']:.1%}")
    c[2].metric(f"Best test ROC-AUC ({best.model})", f"{best.test_roc_auc:.3f}")
    c[3].metric("Lexicon micro-F1 vs gold", f"{f1_meta['micro_f1']['lexicon-v1']:.2f}")
    st.subheader("Hypothesis verdicts")
    rows = []
    labels = {
        "H1": "Service complaints carry the largest penalty",
        "H2": "Cleanliness/wait/Wi-Fi hurt more than they help; coffee/ambience symmetric",
        "H3": "Aspect weights differ by price tier",
        "H4": "Price level not associated with rating once segment is controlled",
        "H5": "Neighbourhood differences vanish after controls",
    }
    for h, text in labels.items():
        rows.append({"Hypothesis": h, "Statement": text, "Verdict": hyp[h]["verdict"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    h1 = hyp["H1"]
    st.markdown(
        f"**H1.** The largest complaint penalty is `{h1['most_negative_neg_term']}`, not service. "
        f"A service complaint goes with a {h1['neg_service_staff_coef_pp']:.1f} pp lower chance of a 5-star review; "
        f"service was the most negative complaint in {h1['bootstrap_share_service_most_negative']:.0%} of "
        f"{hyp['design']['bootstrap_draws']} bootstrap draws."
    )

with tabs[1]:
    st.subheader("Cleaning")
    steps = pd.DataFrame([
        ("Raw rows", log["n_raw"]),
        ("Mojibake repaired (kept)", log["n_mojibake_repaired"]),
        ("Dropped: empty text", -log["n_dropped_empty_text"]),
        ("Dropped: exact duplicate (rating, text)", -log["n_dropped_exact_duplicates"]),
        ("Dropped: same text, different ratings", -log["n_dropped_conflicting_rating"]),
        ("Clean reviews", log["n_clean"]),
    ], columns=["Step", "Rows"])
    st.dataframe(steps, hide_index=True)
    st.caption(f"Source: {log['source']} (sha256 {log['sha256'][:12]}...)")
    st.subheader("Ratings")
    st.image(str(FIG / "eda_ratings.png"))

with tabs[2]:
    st.subheader("Extractor accuracy on the gold set")
    st.caption(f"Gold set: {f1_meta['n_gold']} randomly sampled reviews. Labeler: {f1_meta['labeler']}. "
               "tfidf-lr-cv is trained on the gold set with 5-fold cross-validation; blank = too few positives.")
    piv = f1.pivot(index="flag", columns="extractor", values="f1")
    piv["gold positives"] = f1.drop_duplicates("flag").set_index("flag")["n_gold_pos"]
    st.dataframe(piv.loc[f1.flag.drop_duplicates()], width="stretch")
    st.subheader("How common is each aspect? Naive vs prediction-powered estimates")
    flag = st.selectbox("Flag", prev.flag.drop_duplicates().tolist(), index=3)
    sub = prev[prev.flag == flag]
    fig = go.Figure()
    for _, r in sub.iterrows():
        fig.add_trace(go.Scatter(x=[r.estimate * 100], y=[r.method], mode="markers",
                                 error_x=dict(type="data", symmetric=False,
                                              array=[(r.ci_high - r.estimate) * 100],
                                              arrayminus=[(r.estimate - r.ci_low) * 100]),
                                 name=r.method))
    fig.update_layout(xaxis_title="Share of reviews (%), 95% CI", showlegend=False, height=260,
                      margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch")
    st.image(str(FIG / "ppi_prevalence.png"), width=700)

with tabs[3]:
    st.subheader("Predicting a 5-star review")
    st.caption(f"Target: {pred_meta['target']}. Train {pred_meta['n_train']:,} / test {pred_meta['n_test']:,} "
               f"reviews, stratified split, seed {pred_meta['seed']}. Test set used once. "
               f"Not run: B1 ({pred_meta['not_run']['B1']}), M4 ({pred_meta['not_run']['M4']}).")
    show = pred[["model", "test_roc_auc", "test_pr_auc", "test_macro_f1", "test_brier",
                 "cv_roc_auc", "cv_pr_auc", "cv_macro_f1", "cv_brier", "threshold"]]
    st.dataframe(show, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "reliability.png"))
    c2.image(str(FIG / "shap_m3.png"))
    st.dataframe(shap_df.head(12), hide_index=True)

with tabs[4]:
    st.subheader("Aspect mentions associated with a 5-star review")
    st.caption("Pooled linear probability model, HC1 robust SEs, Benjamini-Hochberg adjusted p-values. "
               "Coefficients are percentage-point differences, holding the other flags fixed.")
    c = coef.sort_values("lpm_coef_pp")
    fig = go.Figure(go.Scatter(
        x=c.lpm_coef_pp, y=c.term, mode="markers",
        marker=dict(color=["#C44E52" if t.startswith("neg_") else "#4C72B0" for t in c.term]),
        error_x=dict(type="data", symmetric=False, array=c.lpm_ci_high_pp - c.lpm_coef_pp,
                     arrayminus=c.lpm_coef_pp - c.lpm_ci_low_pp),
        customdata=c[["n_reviews_with_flag", "lpm_p_bh"]].to_numpy(),
        hovertemplate="%{y}: %{x:.1f} pp<br>reviews with flag: %{customdata[0]}<br>BH p: %{customdata[1]:.3g}",
    ))
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=650, xaxis_title="Percentage points (95% CI)", margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch")
    st.subheader("H2: praise vs complaint asymmetry")
    st.dataframe(asym, hide_index=True, width="stretch")
    st.json(hyp["H2"])
    st.subheader("Do conclusions depend on the extractor? (400 gold reviews)")
    st.dataframe(concl, hide_index=True, width="stretch")

with tabs[5]:
    d = hyp["design"]
    st.markdown(f"""
- **Data substitute.** The Kaggle file named in the plan could not be reached (network policy blocked Kaggle;
  the only GitHub copy found stores it in Git LFS, which this environment could not fetch). The fallback file has
  {log['n_raw']:,} rows but only `rating` and `review_text`; {log['n_dropped_exact_duplicates']:,} rows were exact
  duplicates. There are no restaurant names, so the analysis covers restaurants in general, not only cafes.
- **No within-cafe design.** Not run: {', '.join(d['not_run'])}. Pooled estimates mix within- and
  between-restaurant differences.
- **Measurement.** {d['measurement']}. PPI shows the lexicon under-counts complaints.
- **Gold labels.** One annotator (the project assistant), no agreement statistic. Treat F1 as agreement with
  that annotator.
- **Rare aspects.** Dropped from the models (fewer than 30 reviews): {d['dropped_rare_flags']}.
  Coffee and Wi-Fi estimates are weak.
- **Era.** Reviews date from around 2019 and describe that period.
""")
