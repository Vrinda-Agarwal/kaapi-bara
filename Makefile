PY := uv run python

.PHONY: inspect data aspects train explain report test

## Phase 0: checksum + inspection report for data/raw/zomato.csv
inspect:
	$(PY) -m cafe_reviews.ingest.inspect_raw

## Not built yet. Each target is filled in by its phase.
data:
	@echo "make data: not implemented yet (Phase 1)"; exit 1
aspects:
	@echo "make aspects: not implemented yet (Phase 3)"; exit 1
train:
	@echo "make train: not implemented yet (Phase 4)"; exit 1
explain:
	@echo "make explain: not implemented yet (Phase 5)"; exit 1
report:
	@echo "make report: not implemented yet (Phase 7)"; exit 1

test:
	uv run pytest
