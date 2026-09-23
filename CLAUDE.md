# From Filter Kaapi to Flat Whites

A research project on what separates 5-star café reviews in Bengaluru. It's part of an MS CS/AI application portfolio, so rigor and honesty about limitations matter more than impressive-looking numbers.

## Research question

**Primary:** Among reviews of the *same* café, which aspects of the experience, and in which direction (praise vs. complaint), are most strongly associated with a review being highly positive?

**Secondary:** (a) Do those associations differ by café segment and price tier? (b) Across cafés, how much of the variation in average rating goes with price, neighborhood, and segment?

All findings are associational. In code comments, figure titles, and reports, write "associated with", never "causes", "drives", or "influences".

### Hypotheses
- H1: Service complaints carry the largest within-café penalty of any aspect.
- H2: Some aspects (cleanliness, wait time, Wi-Fi) hurt a lot when negative but barely help when positive; coffee and ambience behave more symmetrically or as "delighters".
- H3: Ambience, seating, and Wi-Fi matter more at premium cafés; coffee and value matter more at budget ones.
- H4: Price level isn't associated with average rating once segment is controlled; value-for-money sentiment is.
- H5: Neighborhood differences mostly disappear once café segment and price are controlled.

## Data

- Source: the Kaggle dataset "Zomato Bangalore Restaurants" (`himanshupoddar/zomato-bangalore-restaurants`), file `zomato.csv`. I download it manually into `data/raw/`.
- Do NOT download it yourself, scrape any website, or call the Google Maps/Places APIs.
- Status (2026-09-23): the Kaggle file could not be obtained. With owner approval the project uses a substitute review-level file (rating + text only); see `DATA.md`. Never scrape review sites.
- The dataset was originally scraped from Zomato around 2019. All findings describe that era.
- `data/` is gitignored. Never commit raw or processed review text. Record the sha256 of the raw file in `DATA.md`.
- Parse any stringified Python lists safely. Never use `eval()`.

### Unverified assumptions (check on load, report back, never assume)
- Roughly 51.7k rows and 17 columns.
- `reviews_list` is a stringified list of (rating, text) tuples.
- Individual review ratings look like "Rated 4.0". Half-stars may exist.
- There are no review dates or reviewer IDs.
- The same restaurant appears in many rows (multiple listing categories, chain branches).
- Restaurant names or review text may have encoding problems (mojibake).

## Leakage rules

- Never use these as features: the café's aggregate `rate`, `votes`, `dish_liked`.
- Mask café names in review text before fitting any text model.
- Split by café, and by chain brand where identifiable, using grouped CV.
- The final held-out set of cafés is used once, and only when I say so.

## Aspect taxonomy (11)

`coffee`, `food`, `service_staff`, `wait_time`, `value`, `ambience`, `seating_space`, `noise_crowding`, `wifi_work`, `cleanliness`, `location_access`

- Definitions live in `annotation/codebook.md`. You can draft it; I finalize it.
- Review-level features: `pos_<aspect>` and `neg_<aspect>` flags. Both 0 means not mentioned; both 1 means mixed.

## Stack

- Python 3.12, managed with uv (`pyproject.toml` + `uv.lock`). Decided in Phase 0.
- Core libraries: pandas, DuckDB, pandera, scikit-learn, statsmodels, SHAP, matplotlib/plotly, pytest. PyYAML for reading `configs/`.
- Hugging Face transformers only from Phase 3 onward; Streamlit only in Phase 7.
- Don't add other dependencies without asking. No Airflow, Spark, Docker, or Kubernetes.

## Repo structure

```
README.md  DATA.md  CLAUDE.md  pyproject.toml  Makefile
configs/        cafe_definition.yaml, aspects.yaml, prompts/
data/           raw/ interim/ processed/   (gitignored)
annotation/     codebook.md, gold_labels.csv (review IDs + labels only, no text)
src/cafe_reviews/
  ingest/ clean/ validate/ aspects/ features/ models/ explain/ viz/
notebooks/      01_eda, 02_aspect_validation, 03_prediction, 04_drivers
reports/        figures/, technical_report.md
app/            Streamlit dashboard
tests/
```

## Engineering rules

- **Code placement:** logic lives in `src/cafe_reviews/`. Notebooks only call functions and plot.
- **Pipeline stages:** each stage reads the previous stage's output, writes its own output, and gives the same result when rerun. Raw data is never modified.
- **Config:** café definition, thresholds, aspect lexicon, and LLM prompts go in `configs/`, never hardcoded.
- **Reproducibility:** fix random seeds and record them alongside outputs.
- **Tests:** pytest for parsing, deduplication, and validation, using small synthetic fixtures rather than real review text.
- **Makefile targets:** `inspect`, `data`, `aspects`, `train`, `explain`, `report`, `test`.
- **Git:** small, focused commits with clear messages. `main` is the only branch. Push to `main`; don't create other branches or open PRs unless asked.
- **No AI attribution:** commits are authored and committed as Vrinda Agarwal <vrindarocks777@gmail.com>. No `Co-Authored-By`, session links, "Generated with" lines, or any other AI or tool branding in commit messages, PR titles or bodies, code, comments, or docs.
- **Secrets:** API keys come only from environment variables. Never print, log, or commit them.
- **Prose style (README, reports):** plain and direct. No em dashes, no corporate phrasing.

## Honesty rules (important)

- **Library APIs:** if you're unsure a function or argument exists, check the installed version (signature, `help()`, or source) before using it. If you can't verify it, say so.
- **Numbers:** every number in a summary, README, or report must come from an actual run in this repo, with the script or output file it came from. Never fill in numbers from memory or estimates.
- **Contradicting data:** when the data contradicts an assumption in this file, stop and tell me. Don't work around it silently.
- **Guesses and defaults:** label anything that is a guess, a default you picked, or unverified.
- **Citations:** don't cite papers or sources in docs unless I've given them to you. Leave `TODO(citation)` instead.

## Working style

- **One phase at a time.** End each phase with a short summary: what you did, what you checked and how, anything surprising, and open decisions. Then stop and wait for my go-ahead.
- **Decisions are mine.** Stop and ask rather than choosing on:
  - the café inclusion rule
  - the target threshold
  - near-duplicate rules
  - any step that drops a meaningful share of rows (tell me how many)
- **Paid calls:** before any LLM or API call that costs money, estimate the cost and ask.
- **Options over polish:** when a choice is subjective, show me two or three options with tradeoffs. I usually refine things myself.

## Phases

| Phase | Work | Done when |
|---|---|---|
| 0. Setup + inspection | Scaffold the repo, record the checksum, write the inspection script | The inspection report answers every unverified assumption above, and candidate café rules and target thresholds are proposed with counts |
| 1. Pipeline | Parse, clean, dedupe, validate (pandera), load DuckDB tables `cafes` and `reviews` | `make data` rebuilds everything from raw, and tests pass |
| 2. EDA + codebook | Rating distribution; segment, price, and neighborhood views; codebook draft | EDA notebook exists and codebook v1 is ready for my review |
| 3. Gold set + extractors | Gold set is a simple random sample of about 400 to 500 reviews (random, not stratified; only IDs and labels committed). Three extractors: lexicon + sentence sentiment, zero-shot NLI, LLM with JSON output (versioned prompt, cached outputs) | A per-aspect F1 table exists for all three extractors |
| 4. Prediction | B0 majority class; B1 café attributes + LR; M1 TF-IDF (word + char n-grams) + LR; M2 aspect flags + LR; M3 gradient-boosted trees + SHAP; M4 transformer only if I approve. Grouped CV. Metrics: ROC-AUC, PR-AUC, macro-F1, Brier score with reliability plot | Results table on held-out cafés |
| 5. Explanation | Within-café fixed-effects regression with SEs clustered by café; between-café model; mixed-effects robustness check; Benjamini-Hochberg correction; café-weighted sensitivity check | Every hypothesis has a verdict with confidence intervals |
| 6. Extension | Compare conclusions across extractors; prediction-powered inference. Verify what `ppi_py` supports, especially for clustered data, before implementing | Figure comparing naive vs. PPI intervals |
| 7. Write-up | README, technical report, Streamlit dashboard | Someone else can clone the repo and reproduce the results |
