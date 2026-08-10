"""Strict contract for local LLM pre-annotation of character spans."""

from __future__ import annotations

from typing import Any

from bank_ner.schema import LABELS, Entity

ANNOTATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "bank_ner_annotation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end", "label", "text", "confidence"],
                    "properties": {
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 1},
                        "label": {"type": "string", "enum": sorted(LABELS)},
                        "text": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                },
            }
        },
    },
}


def system_prompt(blocked_persons: list[str] | None = None) -> str:
    """Return the annotation instruction to pair with a provider's JSON schema."""

    blocked = ", ".join(blocked_persons or []) or "нет дополнительного списка"
    return f"""Ты размечаешь русские запросы банковского приложения.
Верни только JSON, соответствующий предоставленной JSON Schema.

Разрешённые сущности:
- PERSON: имя, фамилия или ФИО реального частного получателя.
- ORG: получатель платежа, поставщик услуги или сервис.
- CONTACT_ROLE: роль личного контакта (маме, папе, сестре, тренеру и формы).

Критичные правила precision:
1) Размечай только точные символы, которые есть во входном тексте. Не исправляй
   опечатки, не нормализуй форму, не придумывай сущность.
2) Если не уверен, верни пустой массив entities или confidence=low. Не угадывай.
3) Не размечай знаменитостей, исторических лиц, блюда и рецепты как PERSON,
   даже если они стоят после слов «перевод» или «переведи».
4) Все найденные сущности должны быть непересекающимися.
5) start включителен, end исключителен, отсчёт с нуля по Python-строке.

Известные запрещённые PERSON-поверхности: {blocked}."""


def validate_llm_annotation(
    response: dict[str, Any], source_text: str
) -> tuple[list[Entity], list[Entity]]:
    """Validate an LLM response and split high-confidence from review-needed spans."""

    if set(response) != {"entities"} or not isinstance(response["entities"], list):
        raise ValueError("LLM response must contain exactly the 'entities' array")
    accepted: list[Entity] = []
    needs_review: list[Entity] = []
    for raw in response["entities"]:
        expected = {"start", "end", "label", "text", "confidence"}
        if set(raw) != expected:
            raise ValueError(f"Unexpected LLM entity fields: {sorted(raw)}")
        entity = Entity.from_dict(raw)
        entity.validate(source_text)
        if raw["confidence"] not in {"high", "medium", "low"}:
            raise ValueError(f"Unknown confidence: {raw['confidence']!r}")
        (accepted if raw["confidence"] == "high" else needs_review).append(entity)
    return accepted, needs_review
