"""Tiered evaluation items. Each item is one Kev request: a source SNOMED disorder plus k Mondo candidates.

Tiers
  easy            gold in the lexical top-k; candidates are whatever retrieval returned
  hard            gold in the lexical top-k; the weakest lexical candidates are replaced by Mondo
                  neighbours of the gold (parent, children, siblings) — the near-misses a small
                  model should confuse and a bigger one shouldn't
  none-narrower   SNOMED disorder with no Mondo mapping whose is_a parent IS mapped; the parent's
                  Mondo term is forced into the candidates. Correct answer: no exact match
                  (and the parent's Mondo term is broader than the source)
  retrieval-miss  gold exists but retrieval didn't surface it; correct answer for the match question
                  is "none" — measures abstention when the truth is absent
"""

import json
import math
import random

import pandas as pd

from .paths import RETRIEVAL, items_path
from .retrieve import Retriever, retrieve_all

NONE = "none"


def _text(v) -> str | None:
    """Parquet hands back NaN for a missing string; keep it out of the JSON."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def _neighbours(mid: str, parents: dict[str, list[str]], children: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(mondo id, kind) for the gold's parents, children and siblings; kind is the predicate truth for that
    candidate (source is narrower than a parent, broader than a child, related to a sibling)."""
    ps = parents.get(mid, [])
    sibs = [s for p in ps for s in children.get(p, []) if s != mid]
    seen, out = {mid}, []
    for x, kind in [*((p, "mondo-parent") for p in ps), *((c, "mondo-child") for c in children.get(mid, [])),
                    *((s, "mondo-sibling") for s in sibs)]:
        if x not in seen:
            seen.add(x)
            out.append((x, kind))
    return out


def _retrieval(snomed: pd.DataFrame, mondo: pd.DataFrame, sctids: list[str]) -> pd.DataFrame:
    """Top-250 lexical hits for every mapped/unmapped source we might sample, cached on disk."""
    if RETRIEVAL.exists():
        cached = pd.read_parquet(RETRIEVAL)
        if set(sctids) <= set(cached.sctid):
            return cached
    ret = Retriever.from_terms(mondo)
    labels = dict(zip(snomed.sctid, snomed.label))
    df = retrieve_all(ret, {s: labels[s] for s in sctids}, k=250)
    RETRIEVAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RETRIEVAL)
    return df


def build_items(
    snomed: pd.DataFrame,
    mondo: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    k: int = 8,
    n_mapped: int = 1200,
    n_none: int = 300,
    n_removed: int = 0,
    n_hard_inject: int = 4,
    seed: int = 7,
    shuffle: bool = False,
    labels_only: bool = False,
    match_only: bool = False,
    ambiguous_share: float = 0.0,
    name: str = "items",
    exclude: set[str] | None = None,
) -> list[dict]:
    """Variant knobs: `shuffle` randomises option order (else retrieval rank); `labels_only` drops synonyms and
    definitions from the state (needed to fit k=250 in the 8k context); `match_only` asks no relation questions;
    `ambiguous_share` is the fraction of mapped items drawn from sources whose lexical top-1 is not the gold or
    whose top-1/top-2 margin is < 0.05, the cases where a model can add something over string matching."""
    rng = random.Random(seed)
    mterm = mondo.set_index("id")
    parents = {r.id: list(r.parents) for r in mondo.itertuples()}
    children: dict[str, list[str]] = {}
    for c, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(c)
    sterm = snomed.set_index("sctid")
    exact = gold[gold.predicate == "skos:exactMatch"]
    counts = exact.sctid.value_counts()
    exact = exact[exact.sctid.map(counts) == 1]
    exact = exact[exact.sctid.isin(sterm.index)]
    gold_of = dict(zip(exact.sctid, exact.mondo_id))
    mapped_ids = set(gold_of)
    exclude = exclude or set()  # sources reserved for evaluation (training-set builds pass every eval set's sources)
    none_pool = [
        r.sctid for r in snomed.itertuples()
        if r.sctid not in mapped_ids and r.sctid not in exclude and any(p in mapped_ids for p in r.parents)
    ]
    none_pick = rng.sample(none_pool, min(n_none, len(none_pool)))

    hits = _retrieval(snomed, mondo, sorted(mapped_ids) + none_pick)
    top = {s: g for s, g in hits.sort_values(["sctid", "rank"]).groupby("sctid")}

    def lex(sid: str) -> list[tuple[str, float]]:
        g = top.get(sid)
        return list(zip(g.mondo_id, g.score)) if g is not None else []

    def ambiguous(sid: str) -> bool:
        h = lex(sid)
        if not h or h[0][0] != gold_of[sid]:
            return True
        return len(h) > 1 and h[0][1] - h[1][1] < 0.05

    def source_block(sid: str) -> dict:
        r = sterm.loc[sid]
        return {
            "sctid": sid,
            "label": r.label,
            "synonyms": list(r.synonyms)[:6],
            "definition": _text(r.definition),
            "parents": [sterm.loc[p].label for p in r.parents if p in sterm.index][:4],
        }

    all_mapped = sorted(mapped_ids - exclude)
    amb = [s for s in all_mapped if ambiguous(s)]
    n_amb = min(int(n_mapped * ambiguous_share), len(amb))
    picked = rng.sample(amb, n_amb)
    rest = [s for s in all_mapped if s not in set(picked)]
    picked += rng.sample(rest, min(n_mapped - n_amb, len(rest)))
    rng.shuffle(picked)

    items: list[dict] = []
    for i, sid in enumerate(picked):
        src = source_block(sid)
        gid = gold_of[sid]
        cands = [{"id": m, "score": s, "source": "lexical"} for m, s in lex(sid)[:k]]
        ids = [c["id"] for c in cands]
        if gid not in ids:
            tier = "retrieval-miss"
        elif i % 2 == 0:
            tier = "easy"
        else:
            tier = "hard"
            # one parent and one child first (predicate truths narrower/broader), then siblings (related)
            nb = [(n, kind) for n, kind in _neighbours(gid, parents, children) if n not in ids]
            rng.shuffle(nb)
            inject = [x for x in nb if x[1] == "mondo-parent"][:1] + [x for x in nb if x[1] == "mondo-child"][:1]
            inject += [x for x in nb if x[1] == "mondo-sibling"][: n_hard_inject - len(inject)]
            keep = [c for c in cands if c["id"] == gid] + [c for c in cands if c["id"] != gid]
            keep = keep[: max(1, k - len(inject))]
            cands = keep + [{"id": n, "score": None, "source": kind} for n, kind in inject]
            cands = sorted(cands, key=lambda c: -(c["score"] or 0.0))
        if shuffle:
            rng.shuffle(cands)
        items.append(_item(f"{tier}:{sid}", tier, src, cands, gid, "skos:exactMatch", mterm,
                           ambiguous=ambiguous(sid), labels_only=labels_only, match_only=match_only))

    # gold-removed: an ordinary mapped item with the gold deleted from the list; the only right answer is none
    removed_pool = [s for s in all_mapped if s not in set(picked) and lex(s) and lex(s)[0][0] == gold_of[s]]
    for sid in rng.sample(removed_pool, min(n_removed, len(removed_pool))):
        src = source_block(sid)
        cands = [{"id": m, "score": s, "source": "lexical"} for m, s in lex(sid) if m != gold_of[sid]][:k]
        if shuffle:
            rng.shuffle(cands)
        items.append(_item(f"gold-removed:{sid}", "gold-removed", src, cands, None, None, mterm,
                           labels_only=labels_only, match_only=match_only, removed_gold=gold_of[sid]))

    for sid in none_pick:
        src = source_block(sid)
        parent_gold = next(gold_of[p] for p in sterm.loc[sid].parents if p in mapped_ids)
        cands = [{"id": m, "score": s, "source": "lexical"} for m, s in lex(sid)[:k]]
        if parent_gold not in [c["id"] for c in cands]:
            cands = cands[: k - 1] + [{"id": parent_gold, "score": None, "source": "mapped-parent"}]
        if shuffle:
            rng.shuffle(cands)
        items.append(_item(f"none-narrower:{sid}", "none-narrower", src, cands, None, None, mterm,
                           broader=parent_gold, labels_only=labels_only, match_only=match_only))

    rng.shuffle(items)
    out = items_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


def option_name(i: int) -> str:
    """A, B, ... Z, AA, AB, ...: option names for up to 255 candidates."""
    name = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def _item(item_id, tier, src, cands, gold_id, gold_predicate, mterm, broader=None, ambiguous=None,
          labels_only=False, match_only=False, removed_gold=None) -> dict:
    out_cands = []
    for n, c in enumerate(cands):
        t = mterm.loc[c["id"]]
        out_cands.append(
            {
                "option": option_name(n),
                "id": c["id"],
                "label": t.label,
                "synonyms": [] if labels_only else list(t.synonyms)[:4],
                "definition": None if labels_only else _text(t.definition),
                "retrieval_score": c["score"],
                "source": c["source"],
            }
        )
    lexical = [c for c in cands if c["score"] is not None]
    return {
        "item_id": item_id,
        "tier": tier,
        "source": src,
        "candidates": out_cands,
        "gold_id": gold_id,
        "gold_predicate": gold_predicate,
        "broader_id": broader,
        "removed_gold": removed_gold,
        "ambiguous": ambiguous,
        "lexical_top1": max(lexical, key=lambda c: c["score"])["id"] if lexical else None,
        "match_only": match_only,
    }


def load_items(name: str = "items") -> list[dict]:
    return [json.loads(line) for line in items_path(name).open()]
