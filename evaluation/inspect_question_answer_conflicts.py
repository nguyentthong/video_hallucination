#!/usr/bin/env python3
"""
Find repeated questions with conflicting answers inside benchmark JSON files.

For each JSON file, the script builds tuples of:
    (question, level, answer)

where level is "target" or "sub-question". It then reports any question text
that appears more than once in the same file with different answers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


QuestionTuple = Tuple[str, str, str]


def _normalise_answer(answer: Any) -> str:
    return str(answer).strip().lower()


def extract_question_tuples(data: Dict[str, Any]) -> List[QuestionTuple]:
    records: List[QuestionTuple] = []

    for question, answer in zip(data.get("questions", []), data.get("answers", [])):
        records.append((str(question), "target", str(answer)))

    sub_questions = data.get("sub-questions", [])
    sub_answers = data.get("sub-answers", [])
    for questions, answers in zip(sub_questions, sub_answers):
        for question, answer in zip(questions, answers):
            records.append((str(question), "sub-question", str(answer)))

    return records


def find_conflicts(records: Iterable[QuestionTuple]) -> Dict[str, List[QuestionTuple]]:
    by_question: Dict[str, List[QuestionTuple]] = defaultdict(list)
    for question, level, answer in records:
        by_question[question.strip()].append((question, level, answer))

    conflicts: Dict[str, List[QuestionTuple]] = {}
    for question, occurrences in by_question.items():
        answers = {_normalise_answer(answer) for _, _, answer in occurrences}
        if len(answers) > 1:
            conflicts[question] = occurrences
    return conflicts


def iter_json_files(benchmark_dir: Path) -> Iterable[Path]:
    yield from sorted(benchmark_dir.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print benchmark JSON files that repeat a question with different answers."
    )
    parser.add_argument(
        "--benchmark_dir",
        default="benchmark",
        type=Path,
        help="Directory containing benchmark JSON files. Default: benchmark",
    )
    args = parser.parse_args()

    total_files = 0
    files_with_conflicts = 0
    total_conflicting_questions = 0

    for json_path in iter_json_files(args.benchmark_dir):
        total_files += 1
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        records = extract_question_tuples(data)
        conflicts = find_conflicts(records)
        if not conflicts:
            continue

        files_with_conflicts += 1
        total_conflicting_questions += len(conflicts)
        print(f"\n{json_path}")
        for question, occurrences in conflicts.items():
            print(f"  Question: {question}")
            for _, level, answer in occurrences:
                print(f"    - level={level:<12} answer={answer}")

    print("\nSummary")
    print(f"  JSON files scanned          : {total_files}")
    print(f"  Files with conflicts        : {files_with_conflicts}")
    print(f"  Conflicting question texts  : {total_conflicting_questions}")


if __name__ == "__main__":
    main()
