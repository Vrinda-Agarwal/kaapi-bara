# Data

## Planned source (not obtained)

- Kaggle dataset `himanshupoddar/zomato-bangalore-restaurants`, file `zomato.csv`.
- Not available in the build environment: the network policy blocked Kaggle, and the one public
  GitHub copy found (`prithvi-jpg/Scrweb---datascience`, a Git LFS pointer with size 574,072,999
  bytes and oid `4b0c91c3...9d43d`) could not be downloaded because the environment could not fetch
  Git LFS objects.
- `make inspect` still works on this file if it is placed at `data/raw/zomato.csv`.

## Source used

The owner approved using a substitute on 2026-09-23.

| Field | Value |
|---|---|
| File | `data/raw/alt/Dataset_zomato_reviews.zip` (member `Zomato_reviews.csv`) |
| Origin | https://raw.githubusercontent.com/klimanyusuf/Zomato-Rating/master/Dataset_zomato_reviews.zip |
| sha256 | `706f9faa1c6e7ec1929307797db1faeef1554315803460553de64a77d076d324` |
| Retrieved | 2026-09-23 |
| Columns | `rating` (1.0 to 5.0 in half steps), `review_text` |
| Rows | 27,762 raw, 10,449 after cleaning |

`make data` checks the sha256 before using the file.

Unverified:
- that these reviews come from the same Zomato Bangalore scrape as the Kaggle file (the half-star
  ratings and review style match, but there is no shared ID);
- the license and terms of use. TODO: find and record them.

What the file lacks: restaurant name or ID, price, location, cuisine, and review date.

## What is committed

- Nothing under `data/` (gitignored). No raw or processed review text is ever committed.
- `annotation/gold_labels.csv`: review IDs (a hash of rating and normalized text) and flags only.
- `reports/results/`: counts, metrics, and coefficients only.
