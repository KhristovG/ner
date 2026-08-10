from __future__ import annotations

from bank_ner.llm_contract import validate_llm_annotation


def test_llm_contract_sends_non_high_confidence_to_review() -> None:
    accepted, review = validate_llm_annotation(
        {
            "entities": [
                {
                    "start": 9,
                    "end": 15,
                    "label": "PERSON",
                    "text": "Антону",
                    "confidence": "high",
                },
                {
                    "start": 18,
                    "end": 31,
                    "label": "ORG",
                    "text": "Мосэнергосбыт",
                    "confidence": "medium",
                },
            ]
        },
        "Переведи Антону в Мосэнергосбыт",
    )
    assert [entity.label for entity in accepted] == ["PERSON"]
    assert [entity.label for entity in review] == ["ORG"]
