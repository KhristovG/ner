from __future__ import annotations

from bank_ner.evaluation import exact_span_metrics, presence_metrics
from bank_ner.schema import Entity


def test_exact_span_metrics_penalize_wrong_boundary() -> None:
    gold = {"a": [Entity(9, 15, "PERSON", "Антону")]}
    predicted = {"a": [Entity(9, 14, "PERSON", "Антон")]}
    metric = exact_span_metrics(gold, predicted)["PERSON"]
    assert metric["tp"] == 0
    assert metric["fp"] == 1
    assert metric["fn"] == 1


def test_presence_metrics_support_product_binary_decision() -> None:
    gold = {"a": [Entity(0, 4, "PERSON", "Анна")], "b": []}
    predicted = {
        "a": [Entity(0, 4, "PERSON", "Анна")],
        "b": [Entity(0, 3, "ORG", "МТС")],
    }
    metric = presence_metrics(gold, predicted)
    assert metric["tp"] == 1
    assert metric["fp"] == 1
