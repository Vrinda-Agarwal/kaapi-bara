"""Lexicon + clause-level sentiment aspect extractor.

For each clause (sentences split further on "but"/"however"):
- an aspect-specific cue (e.g. "overpriced") sets that aspect's direction;
- otherwise, an aspect term takes the clause's net sentiment, counted from the
  general positive/negative word lists with a simple negation window.
Clauses with net sentiment 0 add nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\.{2,}|\n+|;|\bbut\b|\bhowever\b|\bthough\b|\balthough\b", re.I)
_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)?|n't")


def _phrase_regex(phrases: list[str]) -> re.Pattern | None:
    if not phrases:
        return None
    alts = sorted((re.escape(p.lower()) for p in phrases), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(?:" + "|".join(alts) + r")(?![a-z])")


@dataclass
class LexiconExtractor:
    aspects: dict
    positive: set[str]
    negative: set[str]
    negators: set[str]
    window: int
    version: str

    @classmethod
    def from_config(cls, cfg: dict) -> LexiconExtractor:
        s = cfg["sentiment"]
        compiled = {
            a: {
                "terms": _phrase_regex(spec["terms"]),
                "pos": _phrase_regex(spec.get("pos_cues", [])),
                "neg": _phrase_regex(spec.get("neg_cues", [])),
            }
            for a, spec in cfg["aspects"].items()
        }
        return cls(
            aspects=compiled,
            positive={w.lower() for w in s["positive"]},
            negative={w.lower() for w in s["negative"]},
            negators={w.lower() for w in s["negators"]},
            window=int(s["negation_window"]),
            version=cfg["extractor_version"],
        )

    def clause_score(self, clause: str) -> int:
        tokens = _TOKEN.findall(clause.replace("n't", " n't"))
        score = 0
        for i, tok in enumerate(tokens):
            pol = 1 if tok in self.positive else -1 if tok in self.negative else 0
            if pol == 0:
                continue
            if any(t in self.negators for t in tokens[max(0, i - self.window) : i]):
                pol = -pol
            score += pol
        return score

    def extract(self, text: str) -> dict[str, int]:
        flags = {f"{d}_{a}": 0 for a in self.aspects for d in ("pos", "neg")}
        for clause in _CLAUSE_SPLIT.split(text.lower()):
            if not clause or not clause.strip():
                continue
            score = None
            for a, rx in self.aspects.items():
                if rx["neg"] is not None and rx["neg"].search(clause):
                    flags[f"neg_{a}"] = 1
                    continue
                if rx["pos"] is not None and rx["pos"].search(clause):
                    flags[f"pos_{a}"] = 1
                    continue
                if rx["terms"] is not None and rx["terms"].search(clause):
                    if score is None:
                        score = self.clause_score(clause)
                    if score > 0:
                        flags[f"pos_{a}"] = 1
                    elif score < 0:
                        flags[f"neg_{a}"] = 1
        return flags

    def extract_frame(self, reviews: pd.DataFrame) -> pd.DataFrame:
        rows = [self.extract(t) for t in reviews["review_text"]]
        out = pd.DataFrame(rows, index=reviews.index).astype(int)
        out.insert(0, "review_id", reviews["review_id"].to_numpy())
        return out
