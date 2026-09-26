"""Turn an eval item into a /v1/systemone request, and read the answers back.

One request per source term. The state carries the source concept and every candidate in full; the
questions are one Choice over the candidates (+ none) and, per candidate, one Choice over the
relation between source and candidate. Everything is answered in a single prefill.
"""

from .items import NONE

RELATIONS = {
    "same": "The source and the candidate denote the same disease",
    "narrower": "The source is a subtype or more specific form of the candidate",
    "broader": "The source is a more general category that includes the candidate",
    "related": "Related diseases, but neither the same nor one a subtype of the other",
    "unrelated": "Unrelated",
}
RELATION_TO_PREDICATE = {
    "same": "skos:exactMatch",
    "narrower": "skos:broadMatch",  # source narrower than object => source broadMatch object
    "broader": "skos:narrowMatch",
    "related": "skos:relatedMatch",
}


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def build_request(item: dict, model: str = "kev-latest", relation_for: list[dict] | None = None) -> dict:
    """One /v1/systemone request. `relation_for` restricts the relation questions to those candidates (the
    second-stage call for match-only items); by default every candidate gets one unless the item is match-only."""
    src = item["source"]
    labels_only = all(not c["synonyms"] and not c["definition"] for c in item["candidates"])
    state = {
        "task": "Match a clinical terminology concept to a disease ontology term.",
        "source_concept": _strip(
            {
                "terminology": "SNOMED CT",
                "name": src["label"],
                "synonyms": src["synonyms"],
                "definition": src["definition"],
                "parent_concepts": src["parents"],
            }
        ),
    }
    if not labels_only:  # labels-only sets (large k) carry each candidate once, in the choice criteria
        state["candidate_terms"] = [
            _strip(
                {
                    "option": c["option"],
                    "ontology": "Mondo",
                    "name": c["label"],
                    "synonyms": c["synonyms"],
                    "definition": c["definition"],
                }
            )
            for c in item["candidates"]
        ]
    else:
        state["candidate_terms"] = "Mondo disease terms, listed as the options of the match question"
    match_criteria = {c["option"]: c["label"] for c in item["candidates"]}
    match_criteria[NONE] = "No candidate denotes the same disease as the source concept"
    questions = {
        "match": {
            "type": "choice",
            "instructions": "Which candidate term denotes the same disease as the source concept? "
            "Answer none if no candidate is the same disease (a candidate that is merely a parent, "
            "child or sibling of the source is not the same disease).",
            "criteria": match_criteria,
        }
    }
    rel_cands = relation_for if relation_for is not None else ([] if item.get("match_only") else item["candidates"])
    for c in rel_cands:
        questions[f"rel_{c['option']}"] = {
            "type": "choice",
            "instructions": f"How does the source concept relate to candidate {c['option']} ({c['label']})?",
            "criteria": RELATIONS,
        }
    return {"state": state, "model": model, "questions": questions}
