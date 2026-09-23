PY := uv run python

.PHONY: inspect data aspects gold gold-labels train explain report app test

## Phase 0: checksum + inspection report for data/raw/zomato.csv (primary source only)
inspect:
	$(PY) -m cafe_reviews.ingest.inspect_raw

## Phase 1: raw -> cleaned, validated reviews in data/processed/reviews.duckdb
data:
	$(PY) -m cafe_reviews.ingest.build

## Phase 3: lexicon aspect flags, gold labels, extractor F1
aspects:
	$(PY) -m cafe_reviews.aspects.run
	$(PY) -m cafe_reviews.aspects.evaluate

## Rebuild annotation/gold_labels.csv from the labelling codes (data/interim, gitignored)
gold-labels:
	$(PY) -m cafe_reviews.aspects.gold labels

## Rebuild the (gitignored) labelling sheet with review text for the gold sample
gold:
	$(PY) -m cafe_reviews.aspects.gold sheet

## Phase 4: prediction models, test metrics, reliability, SHAP
train:
	$(PY) -m cafe_reviews.models.train

## Phases 5-6: association models, hypothesis verdicts, PPI
explain:
	$(PY) -m cafe_reviews.explain.associations
	$(PY) -m cafe_reviews.explain.ppi

## Phase 2 figures + summary used by the report and dashboard
report:
	$(PY) -m cafe_reviews.viz.eda

## Phase 7: dashboard (reads reports/ only)
app:
	uv run streamlit run app/app.py

test:
	uv run pytest
