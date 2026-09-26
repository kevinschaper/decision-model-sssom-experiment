"""Mondo term table from the cached semsql sqlite: label, exact synonyms, definition, parents; obsoletes dropped."""

import sqlite3

import pandas as pd

from .paths import MONDO_DB, MONDO_TERMS


def build_mondo_terms() -> pd.DataFrame:
    c = sqlite3.connect(MONDO_DB)
    deprecated = {r[0] for r in c.execute("select id from deprecated_node where id like 'MONDO:%'")}
    labels = dict(c.execute("select subject, value from rdfs_label_statement where subject like 'MONDO:%'"))
    syn: dict[str, list[str]] = {}
    for s, v in c.execute("select subject, value from has_exact_synonym_statement where subject like 'MONDO:%'"):
        syn.setdefault(s, []).append(v)
    defs = dict(
        c.execute("select subject, value from statements where predicate='IAO:0000115' and subject like 'MONDO:%'")
    )
    parents: dict[str, list[str]] = {}
    for s, o in c.execute(
        "select subject, object from edge where predicate='rdfs:subClassOf' "
        "and subject like 'MONDO:%' and object like 'MONDO:%'"
    ):
        parents.setdefault(s, []).append(o)
    rows = [
        {
            "id": i,
            "label": lab,
            "synonyms": sorted(set(syn.get(i, [])) - {lab}),
            "definition": defs.get(i),
            "parents": sorted(parents.get(i, [])),
        }
        for i, lab in labels.items()
        if i not in deprecated
    ]
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    MONDO_TERMS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MONDO_TERMS)
    return df


def load_mondo_terms() -> pd.DataFrame:
    return pd.read_parquet(MONDO_TERMS)
