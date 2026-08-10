from __future__ import annotations

from bank_ner.postprocess import filter_entities, normalize_surface
from bank_ner.schema import Entity


def test_normalize_surface_ignores_case_punctuation_and_yo() -> None:
    assert normalize_surface("  Ёлкин,  Дом ") == "елкин дом"


def test_blocked_person_is_removed() -> None:
    result = filter_entities(
        [Entity(0, 6, "PERSON", "Пушкин"), Entity(10, 13, "ORG", "МТС")],
        blocked_persons=frozenset({"пушкин"}),
    )
    assert result == [Entity(10, 13, "ORG", "МТС")]
