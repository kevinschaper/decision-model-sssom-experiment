import argparse
import json
import sys

from . import evaluate, gold, items, mondo, snomed, sssom_out
from .paths import RESULTS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kevmap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="term tables, gold, retrieval, tiered items")
    b.add_argument("--k", type=int, default=8)
    b.add_argument("--n-mapped", type=int, default=1200)
    b.add_argument("--n-none", type=int, default=300)
    b.add_argument("--n-removed", type=int, default=0, help="mapped items served with the gold deleted from the list")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--reuse-terms", action="store_true", help="skip rebuilding the Mondo/SNOMED term tables")
    b.add_argument("--shuffle", action="store_true", help="randomise option order (default: retrieval rank)")
    b.add_argument("--labels-only", action="store_true", help="no synonyms/definitions in the state (large k)")
    b.add_argument("--match-only", action="store_true", help="no per-candidate relation questions")
    b.add_argument("--ambiguous-share", type=float, default=0.0, help="share of items with ambiguous lexical top-1")
    b.add_argument("--name", default="items", help="item set name -> data/<name>.jsonl, results/<model>/<name>/")
    r = sub.add_parser("run", help="send items to a running kev.serve")
    r.add_argument("--model", required=True)
    r.add_argument("--port", type=int, default=8009)
    r.add_argument("--limit", type=int)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--tier", action="append")
    r.add_argument("--items", default="items")
    r.add_argument("--base-url", help="hosted endpoint, e.g. https://api.typesafe.ai (bearer key from KEVMAP_API_KEY)")
    r.add_argument("--request-model", default="kev-latest", help='"model" field sent in each request (Jev: jev-latest)')
    e = sub.add_parser("eval", help="score runs and write the comparison table")
    e.add_argument("--model", action="append", help="default: every results/<model>/<items> with responses")
    e.add_argument("--items", default="items")
    s = sub.add_parser("sssom", help="write results/<model>/mappings.sssom.tsv")
    s.add_argument("--model", required=True)
    s.add_argument("--min-confidence", type=float, default=0.0)
    s.add_argument("--items", default="items")
    td = sub.add_parser("train-data", help="labelled records for kev.train, disjoint from every eval set")
    td.add_argument("--out", default="data/train.jsonl")
    td.add_argument("--n-mapped", type=int, default=3000)
    td.add_argument("--n-none", type=int, default=600)
    td.add_argument("--n-removed", type=int, default=400)
    td.add_argument("--seed", type=int, default=11)
    td.add_argument("--eval-sets", default="items,items-v2,wide64,wide250,k8lab")
    c = sub.add_parser("compare", help="paired bootstrap of accuracy: B minus A")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--items", default="items")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        if a.reuse_terms:
            m, sn = mondo.load_mondo_terms(), snomed.load_snomed_disorders()
        else:
            m = mondo.build_mondo_terms()
            print(f"mondo terms: {len(m)}", file=sys.stderr)
            sn = snomed.build_snomed_disorders()
            print(f"snomed disorders: {len(sn)}", file=sys.stderr)
        g = gold.build_gold(set(m.id))
        print(f"gold rows: {len(g)} ({(g.predicate == 'skos:exactMatch').sum()} exact)", file=sys.stderr)
        its = items.build_items(
            sn, m, g, k=a.k, n_mapped=a.n_mapped, n_none=a.n_none, n_removed=a.n_removed, seed=a.seed,
            shuffle=a.shuffle, labels_only=a.labels_only, match_only=a.match_only,
            ambiguous_share=a.ambiguous_share, name=a.name,
        )
        from collections import Counter

        print("items:", dict(Counter(i["tier"] for i in its)), file=sys.stderr)
        print("ambiguous:", sum(1 for i in its if i.get("ambiguous")), file=sys.stderr)
        print("retrieval:", evaluate.retrieval_report(a.name), file=sys.stderr)
    elif a.cmd == "run":
        from .run import run

        run(a.model, a.port, a.limit, a.workers, a.tier, a.items, a.base_url, a.request_model)
    elif a.cmd == "eval":
        models = a.model or sorted(p.parent.parent.name for p in RESULTS.glob(f"*/{a.items}/responses.jsonl"))
        for m in models:
            met = evaluate.score_run(m, a.items)
            print(m, json.dumps(met["overall"]), file=sys.stderr)
        print(evaluate.comparison_table(models, a.items))
        print("retrieval:", evaluate.retrieval_report(a.items))
    elif a.cmd == "train-data":
        from . import train_data

        m, sn = mondo.load_mondo_terms(), snomed.load_snomed_disorders()
        g = gold.load_gold()
        excl = train_data.eval_sources(a.eval_sets.split(","))
        its = items.build_items(
            sn, m, g, k=8, n_mapped=a.n_mapped, n_none=a.n_none, n_removed=a.n_removed, seed=a.seed,
            shuffle=True, ambiguous_share=0.5, name="train-items", exclude=excl,
        )
        assert not ({i["source"]["sctid"] for i in its} & excl), "training sources overlap an eval set"
        n = train_data.write_training(its, a.out)
        from collections import Counter

        print(f"excluded {len(excl)} eval sources; wrote {n} records -> {a.out}", file=sys.stderr)
        print("tiers:", dict(Counter(i["tier"] for i in its)), file=sys.stderr)
    elif a.cmd == "compare":
        res = evaluate.paired_bootstrap(a.a, a.b, a.items)
        fmt = lambda r: f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] n={r['n']}"  # noqa: E731
        print(f"{a.b} - {a.a} on {a.items}: overall {fmt(res['overall'])}")
        for t, r in res["tiers"].items():
            print(f"  {t:15} {fmt(r)}")
    elif a.cmd == "sssom":
        n = sssom_out.write_sssom(a.model, a.min_confidence, a.items)
        print(f"wrote {n} mappings", file=sys.stderr)
