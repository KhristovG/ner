"""Canonical data contract shared by annotation, training and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = frozenset({"PERSON", "ORG", "CONTACT_ROLE"})


@dataclass(frozen=True, slots=True, order=True)
class Entity:
    """A character-level entity span using Python's [start:end) convention."""

    start: int
    end: int
    label: str
    text: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Entity:
        required = {"start", "end", "label", "text"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Entity lacks required fields: {sorted(missing)}")
        return cls(
            start=int(value["start"]),
            end=int(value["end"]),
            label=str(value["label"]),
            text=str(value["text"]),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {"start": self.start, "end": self.end, "label": self.label, "text": self.text}

    def validate(self, source_text: str, allowed_labels: frozenset[str] = LABELS) -> None:
        if self.label not in allowed_labels:
            raise ValueError(f"Unknown label {self.label!r}; allowed: {sorted(allowed_labels)}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid span [{self.start}:{self.end})")
        if self.end > len(source_text):
            message = f"Span [{self.start}:{self.end}) is outside text of length {len(source_text)}"
            raise ValueError(message)
        actual = source_text[self.start : self.end]
        if actual != self.text:
            raise ValueError(f"Entity text mismatch: expected {actual!r}, got {self.text!r}")


@dataclass(frozen=True, slots=True)
class Example:
    """One JSONL record used for gold data and model training."""

    id: str
    text: str
    entities: tuple[Entity, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Example:
        required = {"id", "text", "entities"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Example lacks required fields: {sorted(missing)}")
        entities = value["entities"]
        if not isinstance(entities, list):
            raise ValueError("'entities' must be a JSON array")
        example = cls(
            id=str(value["id"]),
            text=str(value["text"]),
            entities=tuple(Entity.from_dict(entity) for entity in entities),
        )
        example.validate()
        return example

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Example id must not be empty")
        if not self.text:
            raise ValueError(f"Example {self.id!r} has empty text")
        seen: set[tuple[int, int, str]] = set()
        occupied: list[tuple[int, int]] = []
        for entity in self.entities:
            entity.validate(self.text)
            key = (entity.start, entity.end, entity.label)
            if key in seen:
                raise ValueError(f"Duplicate entity {key} in {self.id!r}")
            seen.add(key)
            for start, end in occupied:
                if max(start, entity.start) < min(end, entity.end):
                    raise ValueError(f"Overlapping entities are not allowed in {self.id!r}")
            occupied.append((entity.start, entity.end))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "entities": [e.to_dict() for e in self.entities]}


def read_jsonl(path: str | Path) -> list[Example]:
    """Load and validate a JSONL dataset, reporting the precise bad line."""

    examples: list[Example] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            examples.append(Example.from_dict(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"{path}: no examples found")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError(f"{path}: example ids must be unique")
    return examples


def write_jsonl(examples: Iterable[Example], path: str | Path) -> None:
    """Write validated examples as UTF-8 JSONL for a controlled data export."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in examples:
        example.validate()
        rows.append(json.dumps(example.to_dict(), ensure_ascii=False))
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
