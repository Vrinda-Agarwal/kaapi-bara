# From Filter Kaapi to Flat Whites

Which parts of a restaurant visit are associated with a review being 5-star? A study of Zomato
reviews from Bengaluru, scraped around 2019. All findings are associations, not causal effects, and
they describe that dataset and that era.

## Read this first

The plan was built around the Kaggle file `zomato.csv`, which links each review to a restaurant,
its price, and its neighbourhood. That file could not be obtained in the build environment. The
results here use a public review-level extract with only `rating` and `review_text`
(see `DATA.md`). As a result:

- the analysis covers restaurant reviews in general, not only cafes;
- the within-cafe comparison at the heart of the research question could not be run;
- hypotheses H3, H4, and H5 could not be tested.

The gold labels used to score the aspect extractors come from a single annotator, the project
assistant, not from people.

## Results in brief

Full details, with the source file for every number: `reports/technical_report.md`.

- **Data:** 27,762 raw rows, 62% exact duplicates, leaving 10,449 unique reviews; 27.1% are
  rated 5.0.
- **Prediction** (held-out 2,090 reviews): TF-IDF text model ROC-AUC 0.839, PR-AUC 0.632,
  macro-F1 0.737. Aspect flags alone: ROC-AUC 0.72 to 0.73. Majority baseline: 0.500.
- **Associations** (pooled model, percentage points in the chance of a 5-star review):
  food complaints -17.4, seating complaints -11.6, service complaints -10.7, cleanliness
  complaints -8.2, value complaints -7.7; food praise +14.1, service praise +8.7. Praise of other
  aspects is not associated with a 5-star rating once food and service are in the model.

| Hypothesis | Verdict |
|---|---|
| H1 Service complaints carry the largest penalty | Not supported: food complaints are larger |
| H2 Cleanliness/wait/Wi-Fi hurt more than they help; coffee/ambience symmetric | Partly supported |
| H3 Aspect weights differ by price tier | Not testable with this data |
| H4 Price not associated once segment is controlled | Not testable with this data |
| H5 Neighbourhood differences vanish after controls | Not testable with this data |

- **Extractor error:** the lexicon extractor finds about half of the complaints (recall 0.38 to
  0.56). Prediction-powered inference moves the estimated share of reviews with a food complaint
  from 20.5% (naive) to 35.3%.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Put the data file in place (not committed; see `DATA.md`):
`data/raw/alt/Dataset_zomato_reviews.zip`.

## Reproduce

```
make data       # clean, dedupe, validate -> data/processed/reviews.duckdb
make aspects    # lexicon flags, extractor F1 against annotation/gold_labels.csv
make train      # prediction models, test metrics, reliability, SHAP
make explain    # association models, hypothesis verdicts, PPI
make report     # EDA figure and summary
make test       # unit tests on synthetic data
make app        # Streamlit dashboard (reads reports/ only)
```

Seeds are fixed (20260923). A full rebuild from raw reproduces every file in `reports/results/`.

## Layout

```
configs/            data source, aspect lexicon, gold sample, modeling settings
annotation/         codebook draft, gold labels (IDs and flags only)
src/cafe_reviews/   ingest, clean, validate, aspects, models, explain, viz
reports/            technical_report.md, results/ (aggregate CSV/JSON), figures/
app/                Streamlit dashboard
tests/              pytest, synthetic fixtures only
```

`make inspect` runs the Phase 0 inspection for the original Kaggle file if it is ever placed at
`data/raw/zomato.csv`.
