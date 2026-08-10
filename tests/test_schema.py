from __future__ import annotations

import pytest

from bank_ner.schema import Entity, Example


def test_valid_example_accepts_python_end_exclusive_span() -> None:
    example = Example.from_dict(
        {
            "id": "one",
            "text": "переведи Антону",
            "entities": [{"start": 9, "end": 15, "label": "PERSON", "text": "Антону"}],
        }
    )
    assert example.entities == (Entity(9, 15, "PERSON", "Антону"),)


def test_invalid_surface_is_rejected() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        Example.from_dict(
            {
                "id": "one",
                "text": "переведи Антону",
                "entities": [{"start": 9, "end": 15, "label": "PERSON", "text": "Антон"}],
            }
        )
