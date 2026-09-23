# Data

## Source

- Kaggle dataset: `himanshupoddar/zomato-bangalore-restaurants`
- File: `zomato.csv`
- Originally scraped from Zomato around 2019. Every finding in this repo
  describes that snapshot, not the current state of any café.
- License and terms of use: TODO, check the Kaggle dataset page and record them
  here. Not yet verified.

## Getting the file

Download it manually from Kaggle and save it as `data/raw/zomato.csv`.
The pipeline never downloads or scrapes anything.

## Checksum

Record the sha256 of the exact file used. `make inspect` prints it and also writes
it to `reports/inspection/inspection_report.md`.

| File | sha256 | Recorded on |
|---|---|---|
| `data/raw/zomato.csv` | TODO: not recorded yet (file not available when Phase 0 was scaffolded) | |

## What is committed

- Nothing under `data/` (gitignored). No raw or processed review text is ever
  committed.
- `annotation/gold_labels.csv` holds review IDs and labels only, no text.
- Inspection reports contain counts only, no review text.
