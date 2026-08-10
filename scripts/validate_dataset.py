"""Validate a JSONL NER dataset before training or scoring."""

from __future__ import annotations

import argparse
from collections import Counter

from bank_ner.schema import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate banking NER JSONL annotations.")
    parser.add_argument("path", help="Path to a JSONL file")
    args = parser.parse_args()

    examples = read_jsonl(args.path)
    label_counts = Counter(entity.label for example in examples for entity in example.entities)
    negatives = sum(not example.entities for example in examples)
    print(f"OK: {len(examples)} examples, {negatives} negative examples")
    print("Entities:", dict(sorted(label_counts.items())))


if __name__ == "__main__":
    main()
