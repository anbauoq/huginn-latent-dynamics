"""Task-specific answer normalization and extraction (numeric and multiple-choice)."""

import re
from typing import Any, Optional


def normalize_numeric_token(s: str) -> str:
    """Normalize a numeric string for comparison.

    Strips whitespace/currency/percent signs and commas, and treats a
    '.'-as-thousands-separator pattern like '1.300' as 1300 while leaving
    genuine decimals like '1.3' untouched.
    """
    s = s.strip().replace(" ", "")
    s = s.replace("$", "").replace("%", "").replace(",", "")

    m = re.fullmatch(r"(-?\d+)\.(\d{3})", s)
    if m:
        return m.group(1) + m.group(2)

    return s


def numeric_process_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a numeric-task JSONL item to {id, question, answer}."""
    raw_answer = str(item["answer"]).strip()
    return {
        "id": item.get("id"),
        "question": item["question"],
        "answer": normalize_numeric_token(raw_answer),
    }


def numeric_extract_answer(text: str) -> str:
    """Extract the final numeric answer from model output.

    Returns the numeric candidate that appears LAST in the text, across
    <ans>/<answer> blocks, \\boxed{...}, and anchored phrases like
    "final answer is 42".
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()
    candidates: list[tuple[int, str]] = []

    def _extract_last_number(blob: str) -> Optional[str]:
        if not blob:
            return None
        numbers = re.findall(r"[-+]?\$?\d+(?:[.,]\d+)?%?", blob)
        if not numbers:
            return None
        return normalize_numeric_token(numbers[-1])

    for m in re.finditer(r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>", t, flags=re.IGNORECASE | re.DOTALL):
        block = (m.group(2) or "").strip()
        clean = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()
        num = _extract_last_number(clean)
        if num is not None:
            candidates.append((m.start(), num))

    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL):
        inside = (m.group(1) or "").strip()
        while True:
            new_inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()
            if new_inside == inside:
                break
            inside = new_inside
        num = _extract_last_number(inside)
        if num is not None:
            candidates.append((m.start(), num))

    for m in re.finditer(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)\s*(?:is|:|=)?\s*([-+]?\$?\d+(?:[.,]\d+)?%?)",
        t,
        flags=re.IGNORECASE,
    ):
        candidates.append((m.start(1), normalize_numeric_token(m.group(1))))

    if not candidates:
        return "no_final_answer"

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def mc_process_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a multiple-choice JSONL item to {id, question, answer, options}."""
    stem = item["question"].split("\n# Answer option:")[0].strip()
    opts = item["options"]
    q_formatted = stem + "\n\nOptions:\n" + "\n".join(opts)

    return {
        "id": item["id"],
        "question": q_formatted,
        "answer": item["answer"].strip().upper(),
        "options": opts,
    }


def mc_extract_answer(text: str, options: Optional[list[str]] = None) -> str:
    """Extract the final MCQ letter (A-E) from model output.

    Rule: always return the answer candidate that appears LAST in the text.
    Candidates can be a letter from <ans>/<answer>, \\boxed{...}, or anchored
    phrases, or a number inside \\boxed{...} mapped to a letter via `options`.
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    letter_candidates: list[tuple[int, str]] = []
    num_candidates: list[tuple[int, float]] = []

    def _pick_last_letter(blob: str) -> Optional[str]:
        if not blob:
            return None
        s = blob.strip()

        m = re.fullmatch(r"[A-E]", s, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

        toks = re.findall(r"\b([A-E])\b", s, flags=re.IGNORECASE)
        if toks:
            return toks[-1].upper()

        br = re.findall(r"[\(\[\{<\*\"']\s*([A-E])\s*[\)\]\}>\"\*']", s, flags=re.IGNORECASE)
        if br:
            return br[-1].upper()

        return None

    def _unwrap_latex_wrappers(s: str) -> str:
        out = (s or "").strip()
        while True:
            new_out = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", out).strip()
            if new_out == out:
                break
            out = new_out
        return out

    def _extract_num(s: str) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
        return m.group(0) if m else None

    def _map_num_to_letter(target: float) -> Optional[str]:
        if not options:
            return None
        for opt in options:
            mo = re.match(r"\s*([A-E])\s*\)\s*(.*)$", (opt or "").strip(), flags=re.IGNORECASE)
            if not mo:
                continue
            letter = mo.group(1).upper()
            rhs_num = _extract_num(mo.group(2).strip())
            if not rhs_num:
                continue
            try:
                val = float(rhs_num)
            except ValueError:
                continue
            if abs(val - target) <= 1e-6:
                return letter
        return None

    for m in re.finditer(r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>", t, flags=re.IGNORECASE | re.DOTALL):
        block = (m.group(2) or "").strip()
        block = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()
        letter = _pick_last_letter(block)
        if letter:
            letter_candidates.append((m.start(), letter))

    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL):
        inside = _unwrap_latex_wrappers((m.group(1) or "").strip())

        letter = _pick_last_letter(inside)
        if letter:
            letter_candidates.append((m.start(), letter))
            continue

        if options:
            num_str = _extract_num(inside)
            if num_str:
                try:
                    num_candidates.append((m.start(), float(num_str)))
                except ValueError:
                    pass

    for m in re.finditer(
        r"""
        \b(?:the\s*)?
        final\s+
        answer
        \s*(?:is|:|=)\s*
        (?:\*\*|__)?\s*
        [\(\[\{<"']?\s*
        ([A-E])
        \s*[\)\]\}>\"']?
        \b
        """,
        t,
        flags=re.VERBOSE,
    ):
        letter_candidates.append((m.start(1), m.group(1).upper()))

    last_letter_pos = max((p for p, _ in letter_candidates), default=-1)
    last_num_pos = max((p for p, _ in num_candidates), default=-1)

    if last_num_pos > last_letter_pos:
        target = next(v for p, v in num_candidates if p == last_num_pos)
        mapped = _map_num_to_letter(target)
        if mapped:
            return mapped

    if letter_candidates:
        letter_candidates.sort(key=lambda x: x[0])
        return letter_candidates[-1][1]

    return "no_final_answer"


def check_correct(task: str, extracted: str, expected: str) -> bool:
    """Compare an extracted answer against the expected answer for a task type."""
    if extracted == "no_final_answer":
        return False
    if task == "numeric":
        if normalize_numeric_token(extracted) == normalize_numeric_token(expected):
            return True
        try:
            return abs(float(normalize_numeric_token(extracted)) - float(normalize_numeric_token(expected))) < 1e-6
        except ValueError:
            return False
    return extracted.strip().upper() == expected.strip().upper()
