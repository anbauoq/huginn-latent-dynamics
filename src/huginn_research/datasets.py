"""JSONL loading, task auto-detection, and example normalization."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from huginn_research import answers

Task = Literal["numeric", "multiple-choice"]

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that solves problems carefully and precisely."

ANSWER_TAG_INSTRUCTION = (
    "Think through the problem, then give your final answer wrapped exactly "
    "like this: <answer>...</answer>."
)


@dataclass
class Example:
    id: str
    question: str
    expected_answer: str
    task: Task
    options: list[str] | None = None


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def detect_task(item: dict[str, Any]) -> Task:
    """Detect whether a JSONL item is numeric or multiple-choice."""
    if item.get("options"):
        return "multiple-choice"
    return "numeric"


def normalize_example(item: dict[str, Any], task: Task | Literal["auto"], index: int) -> Example:
    """Normalize a raw JSONL item into an Example using the correct task adapter."""
    resolved_task: Task = detect_task(item) if task == "auto" else task
    item = dict(item)
    item.setdefault("id", str(index))

    if resolved_task == "multiple-choice":
        normalized = answers.mc_process_item(item)
        return Example(
            id=str(normalized["id"]),
            question=normalized["question"],
            expected_answer=normalized["answer"],
            task="multiple-choice",
            options=normalized["options"],
        )

    normalized = answers.numeric_process_item(item)
    return Example(
        id=str(normalized["id"]),
        question=normalized["question"],
        expected_answer=normalized["answer"],
        task="numeric",
        options=None,
    )


def load_examples(path: str, task: Task | Literal["auto"], limit: int | None = None) -> list[Example]:
    """Read a JSONL file and normalize every item, optionally truncated to `limit`."""
    examples = []
    for index, item in enumerate(read_jsonl(path)):
        examples.append(normalize_example(item, task, index))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_prompt(question: str) -> str:
    """Build the user-turn text asking the model to answer inside <answer> tags."""
    return f"{question}\n\n{ANSWER_TAG_INSTRUCTION}"


def extract_answer(task: Task, generated_text: str, options: list[str] | None) -> str:
    if task == "multiple-choice":
        return answers.mc_extract_answer(generated_text, options)
    return answers.numeric_extract_answer(generated_text)
