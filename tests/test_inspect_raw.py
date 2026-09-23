import json

from cafe_reviews.config import load_config
from cafe_reviews.ingest import inspect_raw
from cafe_reviews.ingest.inspect_raw import inspect, read_raw, render_markdown


def _run(raw_csv):
    cfg = load_config("cafe_definition")
    df, header = read_raw(raw_csv, cfg)
    return inspect(df, header, cfg)


def test_shape_and_structure(raw_csv):
    r = _run(raw_csv)
    assert r["shape"]["n_rows"] == 6
    assert r["shape"]["n_cols"] == 17
    assert r["shape"]["expected_columns_missing"] == []
    assert r["reviews_list"]["row_status"] == {
        "empty_list": 1, "parse_failed": 1, "parsed_nonempty": 4,
    }
    assert r["reviews_list"]["element_lengths"] == {"2": 7}
    assert r["reviews_list"]["n_reviews_total_raw"] == 7


def test_ratings(raw_csv):
    rr = _run(raw_csv)["review_ratings"]
    assert rr["n_not_matching_pattern"] == 1  # the None rating
    assert rr["n_non_integer"] == 1  # the 4.5
    assert rr["value_counts"] == {"1.0": 1, "3.0": 2, "4.5": 1, "5.0": 2}


def test_duplication(raw_csv):
    du = _run(raw_csv)["duplication"]
    assert du["n_unique_url"] == 6
    assert du["n_unique_url_without_query"] == 5
    assert du["n_unique_name_address"] == 5
    # "Bean There Cafe" and "Bean There Café" fold to the same name.
    assert du["names_with_gt1_address"]["n_names"] == 1
    assert du["reviews"] == {"n_raw": 7, "n_unique_within_restaurant": 5, "n_unique_global": 5}
    assert du["multi_row_restaurants_with_identical_review_sets"] == 1
    assert du["n_restaurants_listed_under_gt1_listing_type"] == 1


def test_encoding_and_dates(raw_csv):
    r = _run(raw_csv)
    assert r["encoding"]["names_with_mojibake_markers"] == 1
    assert r["encoding"]["reviews_with_literal_byte_escapes"] == 1
    assert r["encoding"]["reviews_text_starting_with_RATED"] == 4
    assert r["dates_and_ids"]["reviews_with_date_like_text"] == 1
    assert r["dates_and_ids"]["id_or_date_like_columns"] == []


def test_candidate_rules_and_thresholds(raw_csv):
    r = _run(raw_csv)
    rules = r["candidate_cafe_rules"]
    # Two Bean There branches match on rest_type (accent folded).
    assert rules["R1_rest_type_cafe"]["n_cafes"] == 2
    assert rules["R1_rest_type_cafe"]["n_rows"] == 3
    # Only the listing-category row(s) with "Cafes".
    assert rules["R2_listed_cafes"]["n_cafes"] == 1
    # The mojibake name "Dosa CafÃ© Corner" folds to "cafa©", so the name rule misses it.
    assert rules["R4_name_keyword"]["n_cafes"] == 2
    t = rules["R1_rest_type_cafe"]["positive_share_by_threshold"]
    # Unique rated reviews in R1: 5.0, 3.0, 4.5 (the None rating is unrated).
    assert rules["R1_rest_type_cafe"]["n_rated"] == 3
    assert t["5.0"]["n_positive"] == 1
    assert t["4.5"]["n_positive"] == 2
    assert t["4.0"]["n_positive"] == 2
    assert r["candidate_rule_overlap_n_cafes"]["R1_rest_type_cafe"]["R2_listed_cafes"] == 1


def test_stops_when_columns_missing(raw_df, tmp_path):
    path = tmp_path / "z.csv"
    raw_df.drop(columns=["reviews_list"]).to_csv(path, index=False)
    r = _run(path)
    assert "stopped_early" in r
    assert "reviews_list" in r["shape"]["expected_columns_missing"]


def test_main_writes_report_without_review_text(raw_csv, tmp_path):
    out = tmp_path / "out"
    assert inspect_raw.main(["--raw", str(raw_csv), "--out", str(out)]) == 0
    md = (out / "inspection_report.md").read_text(encoding="utf-8")
    js = json.loads((out / "inspection_summary.json").read_text(encoding="utf-8"))
    assert len(js["meta"]["sha256"]) == 64
    for fragment in ["alpha beta", "gamma", "delta", "epsilon", "zeta"]:
        assert fragment not in md
        assert fragment not in json.dumps(js)
    # Rerun gives identical output.
    first = md
    inspect_raw.main(["--raw", str(raw_csv), "--out", str(out)])
    assert (out / "inspection_report.md").read_text(encoding="utf-8") == first


def test_render_stopped_early():
    md = render_markdown(
        {"shape": {"n_rows": 0, "n_cols": 0, "expected_n_rows_approx": 1, "expected_n_cols": 1,
                   "columns": [], "expected_columns_missing": ["x"]}, "stopped_early": "x"},
        {"raw_file": "f", "sha256": "0", "size_bytes": 0, "pandas": "0"},
    )
    assert "Stopped early" in md


def test_main_missing_file(tmp_path):
    assert inspect_raw.main(["--raw", str(tmp_path / "nope.csv"), "--out", str(tmp_path)]) == 1
