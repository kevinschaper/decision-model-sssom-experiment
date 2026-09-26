"""Gold standard: Mondo's own SCTID mappings (skos:exactMatch, plus the handful of broadMatch rows)."""

import pandas as pd

from .paths import GOLD, MONDO_SSSOM


def build_gold(mondo_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(MONDO_SSSOM, sep="\t", comment="#", dtype=str)
    df = df[df.object_id.str.startswith("SCTID:")].copy()
    df["sctid"] = df.object_id.str.removeprefix("SCTID:")
    df = df.rename(columns={"subject_id": "mondo_id", "subject_label": "mondo_label", "predicate_id": "predicate"})
    # Mondo's row says MONDO <pred> SCTID; we map SCTID -> MONDO, so broadMatch flips to narrowMatch.
    df["predicate"] = df.predicate.map({"skos:exactMatch": "skos:exactMatch", "skos:broadMatch": "skos:narrowMatch"})
    df = df[df.mondo_id.isin(mondo_ids)][["sctid", "mondo_id", "mondo_label", "predicate"]]
    df = df.drop_duplicates().reset_index(drop=True)
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GOLD)
    return df


def load_gold() -> pd.DataFrame:
    return pd.read_parquet(GOLD)
