"""Transparent, exact-span metrics for a precision-first NER decision."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from bank_ner.schema import LABELS, Entity


@dataclass(frozen=True, slots=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall, "f1": self.f1}


def _entity_key(entity: Entity) -> tuple[int, int, str]:
    return (entity.start, entity.end, entity.label)


def exact_span_metrics(
    gold: Mapping[str, Iterable[Entity]], predicted: Mapping[str, Iterable[Entity]]
) -> dict[str, dict[str, int | float]]:
    """Compute micro exact-span metrics per label and overall.

    The mapping key is an example id. Entity text is intentionally not part of the
    key: schema validation already guarantees it matches the source text.
    """

    per_label: dict[str, Counts] = {label: Counts() for label in LABELS}
    total = Counts()
    for example_id in gold.keys() | predicted.keys():
        gold_by_label: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
        predicted_by_label: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
        for entity in gold.get(example_id, []):
            gold_by_label[entity.label].add(_entity_key(entity))
        for entity in predicted.get(example_id, []):
            predicted_by_label[entity.label].add(_entity_key(entity))
        for label in LABELS:
            correct = len(gold_by_label[label] & predicted_by_label[label])
            false_positive = len(predicted_by_label[label] - gold_by_label[label])
            false_negative = len(gold_by_label[label] - predicted_by_label[label])
            old = per_label[label]
            per_label[label] = Counts(
                old.tp + correct,
                old.fp + false_positive,
                old.fn + false_negative,
            )
            total = Counts(
                total.tp + correct,
                total.fp + false_positive,
                total.fn + false_negative,
            )
    report = {label: per_label[label].to_dict() for label in sorted(LABELS)}
    return {**report, "overall": total.to_dict()}


def presence_metrics(
    gold: Mapping[str, Iterable[Entity]],
    predicted: Mapping[str, Iterable[Entity]],
    labels: frozenset[str] = LABELS,
) -> dict[str, int | float]:
    """Query-level presence metrics for downstream routing decisions."""

    counts = Counts()
    for example_id in gold.keys() | predicted.keys():
        actual = any(entity.label in labels for entity in gold.get(example_id, []))
        forecast = any(entity.label in labels for entity in predicted.get(example_id, []))
        counts = Counts(
            counts.tp + int(actual and forecast),
            counts.fp + int(not actual and forecast),
            counts.fn + int(actual and not forecast),
        )
    return counts.to_dict()
