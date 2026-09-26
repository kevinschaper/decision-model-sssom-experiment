"""Training records for kev.train: an eval-style item plus the gold labels, in Kev's labelled-request format
({"state", "questions": {id: {..., "label"}}}). Sources in any eval set are excluded upstream (items.build_items)."""

import json

from .evaluate import _relation_truth
from .items import NONE, load_items
from .paths import DATA
from .questions import build_request


def eval_sources(names: list[str]) -> set[str]:
    out = set()
    for n in names:
        if (DATA / f"{n}.jsonl").exists():
            out |= {it["source"]["sctid"] for it in load_items(n)}
    return out


def to_training_record(item: dict) -> dict:
    req = build_request(item)
    match_label = NONE
    for c in item["candidates"]:
        if c["id"] == item["gold_id"]:
            match_label = c["option"]
    questions = {"match": {**req["questions"]["match"], "label": match_label, "src": f"snomed-mondo/{item['tier']}"}}
    for c in item["candidates"]:
        truth = _relation_truth(item, c)
        q = req["questions"].get(f"rel_{c['option']}")
        if truth and q:  # only candidates whose relation to the source is known get a labelled question
            questions[f"rel_{c['option']}"] = {**q, "label": truth, "src": "snomed-mondo/relation"}
    return {"state": req["state"], "questions": questions}


def write_training(items: list[dict], out) -> int:
    n = 0
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(to_training_record(it)) + "\n")
            n += 1
    return n
