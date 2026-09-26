"""Locations of inputs (all pre-existing on this machine) and of derived artefacts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Inputs. Mondo's SSSOM (the gold standard) is the copy medic-ingest already downloads;
# Mondo itself is the semsql build OAK caches; SNOMED is the RF2 snapshot in the bdc project.
MONDO_SSSOM = ROOT.parent / "medic-ingest" / "data" / "mondo.sssom.tsv"
MONDO_DB = Path.home() / ".data" / "oaklib" / "mondo.db"
SNOMED_RF2 = (
    Path.home()
    / "Monarch/bdc/snomed/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260301T120000Z/Snapshot"
)
SNOMED_VERSION = "US1000124_20260301"

# Derived
MONDO_TERMS = DATA / "mondo_terms.parquet"
SNOMED_TERMS = DATA / "snomed_disorders.parquet"
GOLD = DATA / "gold.parquet"
RETRIEVAL = DATA / "retrieval_top250.parquet"


def items_path(name: str = "items"):
    return DATA / f"{name}.jsonl"


ITEMS = items_path()
