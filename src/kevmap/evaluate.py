"""Score a run against the gold, per tier, and write metrics.json + a cross-model comparison table.

Metrics on the match question (candidate-or-none):
  acc          argmax == correct option (the gold candidate, or none when the gold is absent)
  brier        multi-class Brier score of the reported probabilities
  ece          expected calibration error of the top probability, 10 equal-width bins
  coverage@5   share of items you could auto-accept, taking them in order of top probability,
               before the running error rate exceeds 5%   (the System-One selling point)
On the per-candidate relation questions:
  rel_acc      argmax relation == truth for the candidates whose relation we know
               (gold -> same; the mapped parent's Mondo term in none-narrower -> narrower)
  fp_same      share of *non-gold* candidates the model calls "same" (over-mapping rate)
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .items import NONE, load_items
from .paths import RESULTS


def result_dir(model: str, items: str = "items"):
    return RESULTS / model / items


def _correct_option(item: dict) -> str:
    if item["gold_id"] is None:
        return NONE
    for c in item["candidates"]:
        if c["id"] == item["gold_id"]:
            return c["option"]
    return NONE  # retrieval-miss


RELATION_OF_KIND = {"mapped-parent": "narrower", "mondo-parent": "narrower", "mondo-child": "broader",
                    "mondo-sibling": "related"}


def _relation_truth(item: dict, cand: dict) -> str | None:
    """What the source's relation to this candidate is, where we know it: the gold is `same`; an injected Mondo
    parent / child / sibling of the gold (or the mapped parent in none-narrower) follows the hierarchy."""
    if cand["id"] == item["gold_id"]:
        return "same"
    if cand["id"] == item.get("broader_id"):
        return "narrower"
    return RELATION_OF_KIND.get(cand["source"])


def _lexical_top1(item: dict) -> str | None:
    """The retriever's own answer: the candidate with the highest retrieval score (injected neighbours have none)."""
    lex = [c for c in item["candidates"] if c.get("retrieval_score") is not None]
    return max(lex, key=lambda c: c["retrieval_score"])["id"] if lex else None


def _ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def _coverage(conf: np.ndarray, hit: np.ndarray, budget: float = 0.05) -> float:
    order = np.argsort(-conf)
    errs = np.cumsum(~hit[order])
    n = np.arange(1, len(order) + 1)
    ok = np.where(errs / n <= budget)[0]
    return float((ok[-1] + 1) / len(order)) if len(ok) else 0.0


def score_run(model: str, items_name: str = "items") -> dict:
    items = {it["item_id"]: it for it in load_items(items_name)}
    out_dir = result_dir(model, items_name)
    recs = [json.loads(line) for line in (out_dir / "responses.jsonl").open()]
    rows = []
    rel_rows = []
    for r in recs:
        it = items[r["item_id"]]
        truth = _correct_option(it)
        m = r["answers"]["match"]
        probs = m["probabilities"]
        keys = list(probs)
        p = np.array([probs[k] for k in keys])
        y = np.array([1.0 if k == truth else 0.0 for k in keys])
        rows.append(
            {
                "item_id": it["item_id"],
                "tier": it["tier"],
                "truth": truth,
                "pred": m["choice"],
                "hit": m["choice"] == truth,
                "lex_hit": bool(it["gold_id"]) and _lexical_top1(it) == it["gold_id"],  # a retriever never abstains
                "ambiguous": it.get("ambiguous"),
                "p_top": float(p.max()),
                "p_truth": float(probs.get(truth, 0.0)),
                "brier": float(((p - y) ** 2).sum()),
                "confidence": m.get("confidence", (p.max() - 1 / len(p)) / (1 - 1 / len(p))),  # Hopper omits it
                "latency_ms": r.get("latency_ms"),
            }
        )
        for c in it["candidates"]:
            a = r["answers"].get(f"rel_{c['option']}")
            if not a:
                continue
            rel_rows.append(
                {
                    "tier": it["tier"],
                    "cand": c["id"],
                    "is_gold": c["id"] == it["gold_id"],
                    "truth": _relation_truth(it, c),
                    "pred": a["choice"],
                    "p_same": a["probabilities"].get("same", 0.0),
                }
            )
    df = pd.DataFrame(rows)
    rel = pd.DataFrame(rel_rows)

    def block(d: pd.DataFrame, rd: pd.DataFrame) -> dict:
        conf, hit = d.p_top.to_numpy(), d.hit.to_numpy()
        out = {
            "n": int(len(d)),
            "acc": float(hit.mean()),
            "lexical_top1_acc": float(d.lex_hit.mean()),
            "brier": float(d.brier.mean()),
            "ece": _ece(conf, hit),
            "coverage@5": _coverage(conf, hit),
            "coverage@10": _coverage(conf, hit, 0.10),
            "auto_accept_precision": float(hit[conf >= 0.9].mean()) if (conf >= 0.9).any() else None,
            "n_auto_accept": int((conf >= 0.9).sum()),
        }
        # abstention as its own skill: a model can lift aggregate accuracy just by saying `none` more often
        tn, pn = (d.truth == NONE).to_numpy(), (d.pred == NONE).to_numpy()
        out["none_rate"] = float(pn.mean())
        out["none_precision"] = float((tn & pn).sum() / pn.sum()) if pn.any() else None
        out["none_recall"] = float((tn & pn).sum() / tn.sum()) if tn.any() else None
        out["acc_gold_present"] = float(hit[~tn].mean()) if (~tn).any() else None
        out["acc_gold_absent"] = float(hit[tn].mean()) if tn.any() else None
        k = rd[rd.truth.notna()]
        # predicate accuracy on the classes that matter for a mapping: same / narrower / broader / no-map,
        # where no-map collapses related and unrelated (neither yields a row; siblings are the no-map truth)
        collapse = {"related": "no-map", "unrelated": "no-map"}
        out["rel_acc"] = float((k.pred.replace(collapse) == k.truth.replace(collapse)).mean()) if len(k) else None
        out["rel_acc_by_truth"] = (
            {t: {"acc": float((g.pred.replace(collapse) == collapse.get(t, t)).mean()), "n": int(len(g))}
             for t, g in k.groupby("truth")} if len(k) else {}
        )
        out["rel_confusion"] = (
            {t: g.pred.value_counts().to_dict() for t, g in k.groupby("truth")} if len(k) else {}
        )
        ng = rd[~rd.is_gold]
        out["fp_same"] = float((ng.pred == "same").mean()) if len(ng) else None
        return out

    metrics = {"model": model, "items": items_name, "overall": block(df, rel), "tiers": {}}
    for tier, d in df.groupby("tier"):
        metrics["tiers"][tier] = block(d, rel[rel.tier == tier])
    if df.ambiguous.notna().any():
        for flag, label in [(True, "lex-ambiguous"), (False, "lex-clear")]:
            d = df[(df.ambiguous == flag)]
            if len(d):
                metrics["tiers"][label] = block(d, rel.iloc[0:0])
    metrics["median_latency_ms"] = float(df.latency_ms.median()) if df.latency_ms.notna().any() else None
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    df.to_csv(out_dir / "per_item.tsv", sep="\t", index=False)
    return metrics


def comparison_table(models: list[str], items_name: str = "items") -> str:
    ms = [json.loads((result_dir(m, items_name) / "metrics.json").read_text()) for m in models]
    tiers = sorted({t for m in ms for t in m["tiers"]})
    lines = ["| metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]

    def fmt(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    for key in ["n", "acc", "acc_gold_present", "acc_gold_absent", "none_rate", "none_precision", "none_recall",
                "lexical_top1_acc", "brier", "ece", "coverage@5", "coverage@10", "auto_accept_precision",
                "n_auto_accept", "rel_acc", "fp_same"]:
        lines.append(f"| overall {key} | " + " | ".join(fmt(m["overall"][key]) for m in ms) + " |")
    for t in tiers:
        for key in ["acc", "lexical_top1_acc", "coverage@5", "rel_acc"]:
            lines.append(f"| {t} {key} | " + " | ".join(fmt(m["tiers"].get(t, {}).get(key)) for m in ms) + " |")
    lines.append("| median latency ms | " + " | ".join(fmt(m["median_latency_ms"]) for m in ms) + " |")
    table = "\n".join(lines)
    (RESULTS / f"comparison-{items_name}.md").write_text(table + "\n")
    return table


def retrieval_report(items_name: str = "items") -> dict:
    """Recall@k of the lexical retriever on mapped items, so its misses are visible and not charged to the model."""
    by = defaultdict(lambda: [0, 0])
    for it in load_items(items_name):
        if it["gold_id"] is None:
            continue
        by["mapped"][1] += 1
        by["mapped"][0] += it["tier"] != "retrieval-miss"
    return {t: {"recall@k": n / d, "n": d} for t, (n, d) in by.items()}


def paired_bootstrap(model_a: str, model_b: str, items_name: str = "items", n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy difference B - A with a 95% paired bootstrap interval over items, overall and per tier."""
    a = pd.read_csv(result_dir(model_a, items_name) / "per_item.tsv", sep="\t").set_index("item_id")
    b = pd.read_csv(result_dir(model_b, items_name) / "per_item.tsv", sep="\t").set_index("item_id")
    j = a[["tier", "hit"]].join(b[["hit"]], rsuffix="_b", how="inner")
    rng = np.random.default_rng(seed)

    def ci(d: pd.DataFrame) -> dict:
        diff = (d.hit_b.astype(float) - d.hit.astype(float)).to_numpy()
        boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
        return {"n": int(len(diff)), "delta": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                                             float(np.percentile(boots, 97.5))]}

    out = {"a": model_a, "b": model_b, "items": items_name, "overall": ci(j)}
    out["tiers"] = {t: ci(d) for t, d in j.groupby("tier")}
    return out
