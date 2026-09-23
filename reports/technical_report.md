# Technical report: what goes with a 5-star review

All numbers below come from files in `reports/results/`, produced by the `make` targets named
next to each section. All findings are associations. None of them show that an aspect causes a
rating.

## 1. Data (`make data`, `make report`)

**Planned source.** Kaggle `himanshupoddar/zomato-bangalore-restaurants`, file `zomato.csv`
(restaurant rows with a `reviews_list` column). It could not be obtained in the build environment:
Kaggle was blocked by the network policy, and the one public GitHub copy found stores the file in
Git LFS, which the environment could not download.

**Source used.** A public review-level extract,
`klimanyusuf/Zomato-Rating/Dataset_zomato_reviews.zip` (sha256 in `DATA.md`). It has two columns,
`rating` and `review_text`, and appears to derive from the same Zomato Bangalore scrape (half-star
ratings, the same review style). That link is not verified. It has **no restaurant name, price,
location, or date**.

Cleaning (`reports/results/eda_summary.json` -> `clean_log`):

| Step | Rows |
|---|---|
| Raw rows | 27,762 |
| Mojibake repaired (kept) | 2,646 |
| Dropped: empty text | 23 |
| Dropped: exact duplicate of (rating, normalized text) | 17,212 |
| Dropped: same text with different ratings | 78 |
| **Clean reviews** | **10,449** |

62% of raw rows were exact duplicates. The duplicate rule (same rating and same lower-cased,
whitespace-collapsed text) is a default chosen during the build, not an owner decision.

Ratings after cleaning: 27.1% are 5.0, 30.5% are 4.5 or higher, 61.7% are 4.0 or higher. Median
review length is 44 words.

Because there is no restaurant identifier, the analysis covers **Bengaluru restaurant reviews in
general**, not cafes. Only 204 reviews (2.0%) carry a lexicon-detected positive coffee judgement.

## 2. Aspect extraction (`make aspects`)

Taxonomy and definitions: `annotation/codebook.md` (draft v1). Each review gets `pos_<aspect>` and
`neg_<aspect>` flags.

**Gold set.** Simple random sample of 400 clean reviews (seed 20260923). Labelled by a single
annotator, the project assistant, blind to the extractor output. There is no second annotator and no
agreement statistic, so the F1 scores measure agreement with one annotator, not accuracy against
human truth. Labels: `annotation/gold_labels.csv` (IDs and flags only).

**Extractors.**
- `lexicon-v1`: aspect terms and cues from `configs/aspects.yaml`, clause-level sentiment with a
  3-token negation window.
- `tfidf-lr-cv`: TF-IDF + logistic regression per flag, trained on the gold set, scored with
  5-fold cross-validated predictions (flags with at least 15 gold positives).
- Zero-shot NLI and LLM extractors were not run (model downloads blocked; skipped per owner).

Lexicon results (`extractor_f1.csv`), micro-F1 over all flags **0.71**:

| Flag | Gold positives | Precision | Recall | F1 |
|---|---|---|---|---|
| pos_food | 285 | 0.94 | 0.82 | 0.87 |
| neg_food | 142 | 0.80 | 0.47 | 0.59 |
| pos_service_staff | 114 | 0.82 | 0.76 | 0.79 |
| neg_service_staff | 61 | 0.66 | 0.38 | 0.48 |
| pos_wait_time | 36 | 0.72 | 0.78 | 0.75 |
| neg_wait_time | 34 | 0.63 | 0.56 | 0.59 |
| pos_value | 72 | 0.81 | 0.65 | 0.72 |
| neg_value | 43 | 0.83 | 0.47 | 0.60 |
| pos_ambience | 116 | 0.90 | 0.69 | 0.78 |
| pos_cleanliness | 11 | 0.75 | 0.82 | 0.78 |

The lexicon is precise but misses many complaints (recall 0.38 to 0.56 for the main `neg_` flags).
The supervised extractor beat it only on `pos_food` (0.90) and `neg_food` (0.71). The lexicon was
not tuned on the gold set.

## 3. Prediction (`make train`)

Target: rating = 5.0. Stratified 80/20 split (8,359 / 2,090), seed 20260923. Thresholds chosen by
max macro-F1 on 5-fold out-of-fold predictions; the test set was scored once
(`prediction_results.csv`).

| Model | Test ROC-AUC | Test PR-AUC | Test macro-F1 | Test Brier |
|---|---|---|---|---|
| B0 majority | 0.500 | 0.271 | 0.422 | 0.198 |
| M1 TF-IDF word+char, LR | **0.839** | **0.632** | **0.737** | **0.143** |
| M2 aspect flags, LR | 0.717 | 0.430 | 0.627 | 0.176 |
| M3 aspect flags, GBT | 0.728 | 0.456 | 0.632 | 0.174 |

Not run: B1 (no cafe attributes), M4 (transformer; downloads blocked). CV could not be grouped by
cafe (no identifier). Reliability curves: `figures/reliability.png`. SHAP for M3 (`shap_m3.csv`):
the features with the largest mean |SHAP| are `neg_food` (0.63), `pos_food` (0.45), review length
(0.33), and `neg_service_staff` (0.31). Among reviews that have the flag, `neg_service_staff` has
the most negative average SHAP value (-2.27 log-odds), followed by `neg_cleanliness` (-1.81) and
`neg_food` (-1.48).

The aspect flags carry much less signal than the full text (ROC-AUC 0.72 to 0.73 vs 0.84). Part of
that gap is extractor error (section 2).

## 4. Associations and hypotheses (`make explain`)

**Design actually used.** Pooled review-level linear probability model: 1[5-star] on all flags with
at least 30 reviews plus log word count, HC1 robust SEs, Benjamini-Hochberg across flags. Logistic
regression as a robustness check (signs agree for 20 of 21 flags; the one disagreement,
`pos_value`, is near zero in both). Dropped as too rare: `neg_wifi_work` (13 reviews).

**Not run** (no restaurant identifier): within-cafe fixed effects, cafe-clustered SEs, between-cafe
model, mixed-effects check, cafe-weighted check. The pooled estimates mix differences within and
between restaurants.

Main coefficients (`associations_coefficients.csv`), percentage points (pp), 95% CI, all BH p < 0.05:

| Flag | Reviews | pp | 95% CI |
|---|---|---|---|
| neg_food | 2,146 | -17.4 | -19.0 to -15.9 |
| neg_seating_space | 305 | -11.6 | -15.3 to -7.9 |
| neg_service_staff | 729 | -10.7 | -12.6 to -8.8 |
| neg_cleanliness | 187 | -8.2 | -11.3 to -5.1 |
| neg_value | 550 | -7.7 | -10.1 to -5.3 |
| neg_wait_time | 693 | -4.4 | -7.1 to -1.8 |
| neg_ambience | 329 | -4.4 | -8.0 to -0.8 |
| pos_service_staff | 3,032 | +8.7 | +6.6 to +10.9 |
| pos_food | 6,787 | +14.1 | +12.2 to +16.0 |

Praise of wait time, value, ambience, seating, cleanliness, or location is not associated with a
5-star rating once food and service are in the model (all CIs include zero).

**H1: service complaints carry the largest penalty.** **Not supported.** The largest complaint
penalty is `neg_food` (-17.4 pp); `neg_service_staff` is -10.7 pp. Service was the most negative
complaint in 0% of 1,000 bootstrap draws, and the food-service gap is significant (6.7 pp, BH
p < 0.001). Service complaints are more strongly associated with fewer 5-star ratings than
complaints about wait time, ambience, noise, or location (BH p < 0.01 each). They are not
distinguishable from complaints about value, cleanliness, seating, or coffee. On the 400 gold reviews with
hand labels the ordering holds (food -25.2 pp, service -15.7 pp; `extractor_conclusions.csv`).

**H2: some aspects hurt more than they help; coffee and ambience are symmetric.** **Partly
supported** (`associations_asymmetry.csv`, asymmetry = pos + neg coefficient):
- cleanliness: -9.0 pp (CI -14.7 to -3.3), penalty-dominant, as predicted;
- wait time: -5.0 pp (CI -9.0 to -1.0), penalty-dominant, as predicted;
- Wi-Fi/work: not testable (too few complaints);
- ambience: -3.3 pp (CI -7.5 to +1.0), consistent with symmetry;
- coffee: -9.3 pp (CI -21.3 to +2.8), inconclusive (only 30 coffee complaints).

The pattern is broader than H2 predicted: value, seating, and food are also penalty-dominant.

**H3, H4, H5.** **Not testable.** They need price tier, cafe segment, and neighbourhood, which the
fallback data does not have.

## 5. Extension (`make explain`)

**Prediction-powered prevalence** (`prevalence_ppi.csv`, `figures/ppi_prevalence.png`).
PPI for a mean, implemented by hand (`ppi_py` was not added). Labeled set: 400 gold reviews;
unlabeled: 10,049. Reviews are treated as independent. TODO(citation): PPI method.

| Flag | Naive lexicon | Gold only (n=400) | PPI |
|---|---|---|---|
| neg_food | 20.5% (19.8 to 21.3) | 35.5% (30.8 to 40.2) | 35.3% (30.7 to 39.9) |
| neg_service_staff | 7.0% (6.5 to 7.5) | 15.3% (11.7 to 18.8) | 13.4% (10.0 to 16.9) |
| neg_value | 5.3% (4.8 to 5.7) | 10.8% (7.7 to 13.8) | 10.0% (7.4 to 12.5) |
| pos_coffee | 2.0% (1.7 to 2.2) | 2.3% (0.8 to 3.7) | 3.2% (1.9 to 4.6) |

The naive intervals are narrow and in the wrong place for complaints: the lexicon misses about
half of them. PPI moves the estimate to where the gold labels put it, with intervals slightly
narrower than gold-only for most flags.

**Extractor comparison** (`extractor_conclusions.csv`). On the gold reviews, 10 of 12 comparable
coefficients have the same sign with gold and lexicon flags. Lexicon-based complaint coefficients
are smaller in magnitude (e.g. `neg_wait_time` -3.5 vs -14.9 pp), consistent with attenuation from
missed complaints. The full-sample coefficients in section 4 are therefore likely too small for
complaints.

## 6. Limitations

1. **Wrong unit of analysis for the original question.** No restaurant identifier: no within-cafe
   comparison, no cafe-level controls, no grouped CV, no cafe subset.
2. **Unverified provenance** of the fallback file; license not recorded.
3. **Single non-human annotator** for the gold set, no agreement measure.
4. **Extractor error**: complaint recall about 0.4 to 0.6; coefficients attenuated.
5. **Rare cafe aspects**: coffee and Wi-Fi are too rare for firm conclusions.
6. **Near-duplicates** (same review with small edits) were not removed; only exact duplicates.
7. **Era**: reviews date from around 2019.

## Reproduce

```
uv sync
make data aspects train explain report   # needs data/raw/alt/Dataset_zomato_reviews.zip
make test
make app
```

Seeds: 20260923 (gold sample, splits, CV, bootstrap, models). A full rebuild from raw reproduced
every file in `reports/results/` byte for byte.
