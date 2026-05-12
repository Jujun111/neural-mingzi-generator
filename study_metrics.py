import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


VALID_NAME_RE = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
COMPOUND_PROMPT_ALIASES = {
    "\u6b27\u9633": {"\u6b27\u9633", "\u6b50\u967d"},
    "\u53f8\u9a6c": {"\u53f8\u9a6c", "\u53f8\u99ac"},
}


def canonicalize_compound_prefix(value: str) -> str:
    for canonical, aliases in COMPOUND_PROMPT_ALIASES.items():
        if value in aliases:
            return canonical
    return value


def teacher_forced_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    pad_idx: int,
    device: str,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(x_batch)
            logits = logits.view(-1, logits.size(-1))
            y_flat = y_batch.view(-1)
            loss = criterion(logits, y_flat)

            non_pad_tokens = int((y_flat != pad_idx).sum().item())
            total_loss += float(loss.item())
            total_tokens += non_pad_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {"cross_entropy": avg_loss, "perplexity": math.exp(avg_loss)}


def distinct_n(names: Sequence[str], n: int) -> float:
    ngrams: List[tuple[str, ...]] = []
    for name in names:
        chars = list(name)
        if len(chars) < n:
            continue
        ngrams.extend(tuple(chars[index : index + n]) for index in range(len(chars) - n + 1))

    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def length_distribution(names: Sequence[str]) -> Dict[str, float]:
    counts = Counter(len(name) for name in names)
    total = len(names) or 1
    return {str(length): round(count / total, 6) for length, count in sorted(counts.items())}


def consecutive_duplicate_ratio(names: Sequence[str]) -> float:
    if not names:
        return 0.0
    flagged = sum(
        1 for name in names if any(name[index] == name[index + 1] for index in range(len(name) - 1))
    )
    return flagged / len(names)


def any_duplicate_ratio(names: Sequence[str]) -> float:
    if not names:
        return 0.0
    flagged = sum(1 for name in names if len(set(name)) < len(name))
    return flagged / len(names)


def invalid_output_ratio(names: Sequence[str]) -> float:
    if not names:
        return 0.0
    invalid = sum(1 for name in names if not VALID_NAME_RE.fullmatch(name))
    return invalid / len(names)


def novelty_ratio(names: Sequence[str], training_names: Iterable[str]) -> float:
    if not names:
        return 0.0
    training_set = set(training_names)
    novel = sum(1 for name in names if name not in training_set)
    return novel / len(names)


def train_overlap_ratio(names: Sequence[str], training_names: Iterable[str]) -> float:
    if not names:
        return 0.0
    training_set = set(training_names)
    overlap = sum(1 for name in names if name in training_set)
    return overlap / len(names)


def unique_name_ratio(names: Sequence[str]) -> float:
    if not names:
        return 0.0
    return len(set(names)) / len(names)


def compound_prompt_compatibility(prompted_records: Sequence[Dict[str, str]]) -> float:
    relevant_records = [
        record
        for record in prompted_records
        if canonicalize_compound_prefix(record["prompt"]) in COMPOUND_PROMPT_ALIASES
    ]
    if not relevant_records:
        return 0.0

    compatible = 0
    for record in relevant_records:
        prompt = record["prompt"]
        name = record["name"]
        normalized_prompt = canonicalize_compound_prefix(prompt)
        normalized_prefix = canonicalize_compound_prefix(name[:2]) if len(name) >= 2 else name
        normalized_name = normalized_prefix + name[2:] if len(name) >= 2 else name
        if normalized_name.startswith(normalized_prompt) and len(name) in {len(prompt) + 1, len(prompt) + 2}:
            compatible += 1

    return compatible / len(relevant_records)


def summarize_generation_metrics(
    names: Sequence[str],
    training_names: Sequence[str],
    prompted_records: Sequence[Dict[str, str]] | None = None,
) -> Dict[str, object]:
    metrics = {
        "sample_count": len(names),
        "unique_name_ratio": round(unique_name_ratio(names), 6),
        "distinct_1": round(distinct_n(names, 1), 6),
        "distinct_2": round(distinct_n(names, 2), 6),
        "novelty_ratio": round(novelty_ratio(names, training_names), 6),
        "train_overlap_ratio": round(train_overlap_ratio(names, training_names), 6),
        "consecutive_duplicate_ratio": round(consecutive_duplicate_ratio(names), 6),
        "any_duplicate_ratio": round(any_duplicate_ratio(names), 6),
        "invalid_output_ratio": round(invalid_output_ratio(names), 6),
        "length_distribution": length_distribution(names),
    }

    if prompted_records is not None:
        metrics["compound_surname_compatibility"] = round(
            compound_prompt_compatibility(prompted_records), 6
        )

    return metrics
