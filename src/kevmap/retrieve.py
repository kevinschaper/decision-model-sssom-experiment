"""Lexical candidate retrieval over Mondo labels + exact synonyms.

BM25 over normalised tokens for recall, rapidfuzz token-set similarity to rerank the top of that list.
Deliberately simple: the point of the experiment is the decision model, and retrieval recall@k is
reported separately so its misses are never charged to the model.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

_TOKEN = re.compile(r"[a-z0-9]+")


def norm_tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


@dataclass
class Retriever:
    ids: list[str]  # one entry per (term, name) pair
    names: list[str]
    bm25: BM25Okapi

    @classmethod
    def from_terms(cls, terms: pd.DataFrame) -> "Retriever":
        ids, names = [], []
        for r in terms.itertuples():
            for n in [r.label, *r.synonyms]:
                ids.append(r.id)
                names.append(n)
        return cls(ids, names, BM25Okapi([norm_tokens(n) for n in names]))

    def search(self, query: str, k: int = 10, pool: int = 60) -> list[tuple[str, float]]:
        """Top-k Mondo ids for a source label, best name per id, fused score in [0, 1]."""
        scores = self.bm25.get_scores(norm_tokens(query))
        top = np.argpartition(-scores, min(pool, len(scores) - 1))[:pool]
        best: dict[str, float] = {}
        smax = float(scores[top].max()) or 1.0
        for i in top:
            fz = fuzz.token_set_ratio(query, self.names[i]) / 100.0
            s = 0.5 * (scores[i] / smax) + 0.5 * fz
            if s > best.get(self.ids[i], -1):
                best[self.ids[i]] = s
        return sorted(best.items(), key=lambda x: -x[1])[:k]


def retrieve_all(ret: "Retriever", queries: dict[str, str], k: int = 250, pool: int = 400) -> pd.DataFrame:
    """Top-k for every (source id -> label); one row per hit. Cached by items.py so variants share it."""
    rows = []
    for sid, q in queries.items():
        for rank, (mid, score) in enumerate(ret.search(q, k=k, pool=pool)):
            rows.append({"sctid": sid, "rank": rank, "mondo_id": mid, "score": round(score, 4)})
    return pd.DataFrame(rows)
