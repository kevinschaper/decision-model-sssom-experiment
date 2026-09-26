import numpy as np

from kevmap.evaluate import _correct_option, _coverage, _ece
from kevmap.items import NONE, option_name
from kevmap.questions import RELATION_TO_PREDICATE, build_request


def test_coverage_stops_at_error_budget():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    hit = np.array([True, True, False, True, True])
    # first two are right (0 errors), third wrong -> 1/3 > 5%; never recovers within 5 items
    assert _coverage(conf, hit, budget=0.05) == 2 / 5
    assert _coverage(conf, np.ones(5, bool)) == 1.0
    assert _coverage(conf, np.zeros(5, bool)) == 0.0


def test_ece_is_zero_when_perfectly_calibrated():
    conf = np.array([0.8] * 10)
    hit = np.array([True] * 8 + [False] * 2)
    assert abs(_ece(conf, hit)) < 1e-9


def _item(gold_id=None, cands=("MONDO:1", "MONDO:2"), tier="easy", broader=None):
    return {
        "item_id": "x",
        "tier": tier,
        "source": {"sctid": "1", "label": "foo", "synonyms": [], "definition": None, "parents": ["bar"]},
        "candidates": [
            {"option": o, "id": c, "label": f"disease {o}", "synonyms": [], "definition": None,
             "retrieval_score": 1.0, "source": "lexical"}
            for o, c in zip("AB", cands)
        ],
        "gold_id": gold_id,
        "gold_predicate": "skos:exactMatch" if gold_id else None,
        "broader_id": broader,
    }


def test_correct_option_maps_gold_to_letter_and_absent_to_none():
    assert _correct_option(_item("MONDO:2")) == "B"
    assert _correct_option(_item(None)) == NONE
    assert _correct_option(_item("MONDO:9")) == NONE  # retrieval miss


def test_request_shape_and_gold_is_not_marked():
    req = build_request(_item("MONDO:2"))
    assert set(req["questions"]) == {"match", "rel_A", "rel_B"}
    assert list(req["questions"]["match"]["criteria"]) == ["A", "B", NONE]
    assert "gold" not in str(req) and "MONDO:" not in str(req["state"])
    assert set(req["questions"]["rel_A"]["criteria"]) == set(RELATION_TO_PREDICATE) | {"unrelated"}


def test_option_names_cover_255():
    names = [option_name(i) for i in range(255)]
    assert names[:3] == ["A", "B", "C"] and names[25:28] == ["Z", "AA", "AB"] and len(set(names)) == 255
