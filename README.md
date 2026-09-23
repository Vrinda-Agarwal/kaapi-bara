# From Filter Kaapi to Flat Whites

A study of which parts of a café visit are associated with a review being highly
positive, using Zomato reviews of Bengaluru cafés scraped around 2019.

All findings are associational. They describe that dataset and that era, not
cafés today.

## Status

Phase 0 (setup and inspection). No results yet.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Download `zomato.csv` from the Kaggle dataset
`himanshupoddar/zomato-bangalore-restaurants` and put it in `data/raw/`.
See `DATA.md`.

## Commands

```
make inspect   # checksum and inspection report for the raw file
make test      # unit tests on synthetic fixtures
```

Other targets (`data`, `aspects`, `train`, `explain`, `report`) are placeholders
until their phase is built.
