"""Held-out evaluation for the agent's structured responses."""
from __future__ import annotations

import json
import random
from collections import defaultdict

from agent import FIELDS


def response_object(text):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    try:
        obj = json.loads(text, parse_constant=reject_constant)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def evaluation_example(row):
    reference = response_object(row["completion"][0]["content"])
    if reference is None:
        raise ValueError("Expected an assistant completion containing a JSON object")
    prompt = next(message["content"] for message in reversed(row["prompt"])
                  if message["role"] == "user")
    task = row.get("task") or ("search" if "action" in reference else "extraction")
    if task not in {"search", "extraction"}:
        raise ValueError(f"Unsupported evaluation task: {task}")
    return task, prompt, reference


def split_for_validation(dataset, fraction, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(dataset):
        task, prompt, _ = evaluation_example(row)
        if task == "search":
            source = prompt.partition("\nSearch calls already used:")[0]
        else:
            header, marker, abstract = prompt.partition("\nAbstract:\n")
            # Synthetic variants share the first sentence but have different paper IDs.
            if marker and header.startswith("Paper ID: synthetic-"):
                source = abstract.partition(". ")[0]
            else:
                source = header if marker else prompt
        groups[task][source].append(index)

    rng = random.Random(seed)
    train_indices, eval_indices = [], []
    for task_groups in groups.values():
        sources = list(task_groups.values())
        rng.shuffle(sources)
        count = min(len(sources) - 1, max(1, round(len(sources) * fraction)))
        eval_indices.extend(index for group in sources[:count] for index in group)
        train_indices.extend(index for group in sources[count:] for index in group)
    if not eval_indices:
        raise ValueError("Cannot hold out a source group. Supply --eval-dataset.")
    return dataset.select(sorted(train_indices)), dataset.select(sorted(eval_indices))


def search_schema_valid(obj):
    if obj.get("action") == "finish":
        return set(obj) == {"action"}
    query = obj.get("query")
    limit = obj.get("limit")
    return (set(obj) == {"action", "provider", "query", "limit"}
            and obj["action"] == "search" and obj.get("provider") in ("openalex", "arxiv")
            and isinstance(query, str) and bool(query.strip())
            and type(limit) is int and limit > 0)


def extraction_field_valid(item):
    return (isinstance(item, dict) and set(item) == {"value", "evidence"}
            and (all(value is None for value in item.values())
                 or all(isinstance(value, str) and value.strip() for value in item.values())))


def score_responses(examples):
    scores = defaultdict(list)
    counts = defaultdict(int)
    for row, prediction in examples:
        task, prompt, reference = evaluation_example(row)
        obj = response_object(prediction)
        counts[task] += 1
        scores["json_valid_rate"].append(obj is not None)
        scores[f"{task}_exact_match"].append(obj == reference)
        obj = obj or {}
        if task == "search":
            scores["search_schema_valid_rate"].append(search_schema_valid(obj))
            scores["search_action_accuracy"].append(obj.get("action") == reference["action"])
            if reference["action"] == "search":
                scores["search_provider_accuracy"].append(
                    obj.get("action") == "search" and obj.get("provider") == reference["provider"])
        else:
            scores["extraction_schema_valid_rate"].append(
                set(obj) == set(FIELDS) and all(extraction_field_valid(obj[key]) for key in FIELDS))
            abstract = " ".join(prompt.partition("\nAbstract:\n")[2].split())
            for key in FIELDS:
                item = obj.get(key)
                scores["extraction_field_exact_match"].append(item == reference[key])
                if isinstance(item, dict) and item.get("evidence") is not None:
                    evidence = item["evidence"]
                    scores["extraction_evidence_grounded_rate"].append(
                        isinstance(evidence, str) and bool(evidence.strip())
                        and " ".join(evidence.split()) in abstract)

    metrics = {"eval_examples": sum(counts.values())}
    metrics.update({f"eval_{task}_examples": count for task, count in counts.items()})
    for name, values in scores.items():
        metrics[f"eval_{name}"] = sum(values) / len(values)
        metrics[f"eval_{name}_count"] = len(values)
    return metrics
