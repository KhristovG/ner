"""Deterministic precision guards applied after model inference."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bank_ner.schema import Entity


def normalize_surface(value: str) -> str:
    """Normalize only for matching a reviewed exception list, never for spans."""

    value = value.casefold().replace("ё", "е")
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def load_blocked_persons(path: str | Path) -> frozenset[str]:
    """Load a small, reviewable list of surfaces forbidden as PERSON."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    allowed_lines = (line for line in lines if line.strip() and not line.startswith("#"))
    return frozenset(normalize_surface(line) for line in allowed_lines)


def filter_entities(
    entities: Iterable[Entity],
    *,
    blocked_persons: frozenset[str],
    thresholds: Mapping[str, float] | None = None,
    scores: Mapping[tuple[int, int, str], float] | None = None,
) -> list[Entity]:
    """Remove known false positives and scores below label-specific thresholds."""

    kept: list[Entity] = []
    for entity in entities:
        if entity.label == "PERSON" and normalize_surface(entity.text) in blocked_persons:
            continue
        if thresholds and scores:
            score = scores.get((entity.start, entity.end, entity.label))
            if score is not None and score < thresholds.get(entity.label, 0.0):
                continue
        kept.append(entity)
    return kept


def coerce_scored_entities(
    raw_entities: Iterable[Mapping[str, Any]],
    source_text: str,
) -> tuple[list[Entity], dict[tuple[int, int, str], float]]:
    """Convert a model adapter's span dictionaries to checked entities.

    Keep this adapter boundary narrow: inspect a new model version's actual response
    once and adapt it here, instead of spreading response-shape assumptions around
    the service code.
    """

    entities: list[Entity] = []
    scores: dict[tuple[int, int, str], float] = {}
    for raw in raw_entities:
        entity = Entity(
            start=int(raw["start"]),
            end=int(raw["end"]),
            label=str(raw["label"]).upper(),
            text=str(raw.get("text", source_text[int(raw["start"]) : int(raw["end"])])),
        )
        entity.validate(source_text)
        entities.append(entity)
        if "score" in raw:
            scores[(entity.start, entity.end, entity.label)] = float(raw["score"])
    return entities, scores
