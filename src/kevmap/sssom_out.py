"""Write the model's decisions as an SSSOM TSV (YAML header + tab table)."""

import json
from datetime import date

import yaml

from .items import NONE, load_items
from .paths import RESULTS
from .questions import RELATION_TO_PREDICATE


def result_dir(model: str, items: str = "items"):
    return RESULTS / model / items

COLUMNS = [
    "subject_id", "subject_label", "predicate_id", "object_id", "object_label",
    "mapping_justification", "confidence", "mapping_tool", "comment",
]


def _tool_version(server: dict) -> str:
    m = (server.get("models") or [{}])[0] if isinstance(server, dict) else {}
    return f"{m.get('run', '?')} on {m.get('base', '?')} ({m.get('backend', '?')}, {m.get('dtype', '?')})"


def write_sssom(model: str, min_confidence: float = 0.0, items_name: str = "items") -> int:
    items = {it["item_id"]: it for it in load_items(items_name)}
    out_dir = result_dir(model, items_name)
    recs = [json.loads(line) for line in (out_dir / "responses.jsonl").open()]
    server = json.loads((out_dir / "server.json").read_text())
    rows = []
    for r in recs:
        it = items[r["item_id"]]
        m = r["answers"]["match"]
        if m["choice"] == NONE:
            continue
        cand = next(c for c in it["candidates"] if c["option"] == m["choice"])
        p_match = m["probabilities"][m["choice"]]
        if p_match < min_confidence:
            continue
        rel_ans = r["answers"].get(f"rel_{cand['option']}", {})
        rel = rel_ans.get("choice", "same")
        p_same = rel_ans.get("probabilities", {}).get("same", 0.0)
        pred = RELATION_TO_PREDICATE.get(rel)
        if pred is None:
            continue
        rows.append(
            {
                "subject_id": f"SCTID:{it['source']['sctid']}",
                # SNOMED CT labels are licensed content: withheld, as Mondo's own SCTID rows do
                "subject_label": "",
                "predicate_id": pred,
                "object_id": cand["id"],
                "object_label": cand["label"],
                "mapping_justification": "semapv:CompositeMatching",
                "confidence": f"{p_match:.4f}",
                "mapping_tool": model,
                "comment": f"tier={it['tier']} relation={rel} p_same={p_same:.3f}",
            }
        )
    header = {
        "mapping_set_id": f"https://w3id.org/monarch/kev-mapping-experiment/{model}",
        "mapping_set_title": f"SNOMED CT -> Mondo candidate mappings decided by {model}",
        "mapping_date": date.today().isoformat(),
        "mapping_tool": model,
        "mapping_tool_version": _tool_version(server),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "comment": "subject_label withheld: SNOMED CT labels are licensed content (SNOMED International "
        "Affiliate License); SCTIDs only, as in Mondo's own SCTID mappings.",
        "subject_source": "SNOMED CT US Edition 2026-03-01",
        "object_source": "obo:mondo/releases/2026-05-05",
        "curie_map": {"SCTID": "http://snomed.info/id/", "MONDO": "http://purl.obolibrary.org/obo/MONDO_",
                      "skos": "http://www.w3.org/2004/02/skos/core#", "semapv": "https://w3id.org/semapv/vocab/"},
    }
    out = out_dir / "mappings.sssom.tsv"
    with out.open("w") as f:
        for line in yaml.safe_dump(header, sort_keys=False).splitlines():
            f.write(f"# {line}\n")
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(row[c] for c in COLUMNS) + "\n")
    return len(rows)
