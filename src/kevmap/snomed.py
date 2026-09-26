"""SNOMED CT disorder table straight from RF2: preferred term, FSN, synonyms, text definition, is_a parents.

Restricted to descendants of 64572001 |Disease (disorder)|, which is where Mondo's SCTID mappings live.
"""

from collections import deque

import duckdb
import pandas as pd

from .paths import SNOMED_RF2, SNOMED_TERMS, SNOMED_VERSION

DISEASE_ROOT = "64572001"
IS_A = "116680003"
FSN, SYNONYM, DEFINITION = "900000000000003001", "900000000000013009", "900000000000550004"
US_LANG_REFSET, PREFERRED = "900000000000509007", "900000000000548007"


def _rf2(name: str) -> str:
    return f"read_csv('{SNOMED_RF2 / name}', delim='\\t', header=true, quote='', all_varchar=true)"


def build_snomed_disorders() -> pd.DataFrame:
    term = "Terminology/"
    con = duckdb.connect()
    rel = con.execute(
        f"select sourceId, destinationId from {_rf2(term + f'sct2_Relationship_Snapshot_{SNOMED_VERSION}.txt')} "
        f"where active='1' and typeId='{IS_A}'"
    ).fetchall()
    children: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    for s, d in rel:
        children.setdefault(d, []).append(s)
        parents.setdefault(s, []).append(d)
    # closure under is_a from the disease root
    disorders, q = {DISEASE_ROOT}, deque([DISEASE_ROOT])
    while q:
        for ch in children.get(q.popleft(), []):
            if ch not in disorders:
                disorders.add(ch)
                q.append(ch)
    disorders.discard(DISEASE_ROOT)

    desc = con.execute(
        f"select d.conceptId, d.typeId, d.term, coalesce(l.acceptabilityId, '') as acc "
        f"from {_rf2(term + f'sct2_Description_Snapshot-en_{SNOMED_VERSION}.txt')} d "
        f"left join (select referencedComponentId, acceptabilityId "
        f"  from {_rf2(f'Refset/Language/der2_cRefset_LanguageSnapshot-en_{SNOMED_VERSION}.txt')} "
        f"  where active='1' and refsetId='{US_LANG_REFSET}') l on l.referencedComponentId = d.id "
        f"where d.active='1'"
    ).fetchall()
    defs = dict(
        con.execute(
            f"select conceptId, term from {_rf2(term + f'sct2_TextDefinition_Snapshot-en_{SNOMED_VERSION}.txt')} "
            f"where active='1' and typeId='{DEFINITION}'"
        ).fetchall()
    )
    fsn: dict[str, str] = {}
    pref: dict[str, str] = {}
    syns: dict[str, set[str]] = {}
    for cid, tid, t, acc in desc:
        if cid not in disorders:
            continue
        if tid == FSN:
            fsn[cid] = t
        elif tid == SYNONYM:
            if acc == PREFERRED and cid not in pref:
                pref[cid] = t
            else:
                syns.setdefault(cid, set()).add(t)
    rows = []
    for cid in sorted(disorders):
        f = fsn.get(cid)
        if not f:
            continue
        label = pref.get(cid) or f.rsplit(" (", 1)[0]
        tag = f[f.rfind("(") + 1 : -1] if f.endswith(")") else ""
        rows.append(
            {
                "sctid": cid,
                "label": label,
                "fsn": f,
                "semantic_tag": tag,
                "synonyms": sorted(syns.get(cid, set()) - {label}),
                "definition": defs.get(cid),
                "parents": sorted(p for p in parents.get(cid, []) if p in disorders),
            }
        )
    df = pd.DataFrame(rows)
    SNOMED_TERMS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SNOMED_TERMS)
    return df


def load_snomed_disorders() -> pd.DataFrame:
    return pd.read_parquet(SNOMED_TERMS)
