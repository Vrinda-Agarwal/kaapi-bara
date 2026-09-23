import pytest

from cafe_reviews.aspects.gold import parse_codes
from cafe_reviews.aspects.lexicon import LexiconExtractor
from cafe_reviews.config import load_config


@pytest.fixture(scope="module")
def ext():
    return LexiconExtractor.from_config(load_config("aspects"))


def on(flags):
    return {k for k, v in flags.items() if v}


def test_mixed_review_splits_on_but(ext):
    assert on(ext.extract("The coffee was great but the staff were rude.")) == {"pos_coffee", "neg_service_staff"}


def test_negation_flips_polarity(ext):
    assert on(ext.extract("The food was not good.")) == {"neg_food"}


def test_aspect_cue_sets_direction(ext):
    assert "neg_value" in on(ext.extract("Totally overpriced."))


def test_neutral_mention_adds_nothing(ext):
    assert on(ext.extract("We ordered pasta.")) == set()


def test_parse_codes():
    f = parse_codes("food+- serv- cof+")
    assert on(f) == {"pos_food", "neg_food", "neg_service_staff", "pos_coffee"}
    assert on(parse_codes("")) == set()
    with pytest.raises(ValueError):
        parse_codes("food")
