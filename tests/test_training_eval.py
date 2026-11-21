import json
from pathlib import Path

from training_eval import evaluation_example, score_responses, split_for_validation


class Examples(list):
    def select(self, indices):
        return Examples(self[index] for index in indices)


def load_examples():
    path = Path(__file__).resolve().parents[1] / "data/train.jsonl"
    return Examples(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def test_validation_keeps_variants_of_each_source_together():
    dataset = load_examples()
    train, validation = split_for_validation(dataset, 0.2, seed=42)

    def sources(rows):
        result = set()
        for row in rows:
            task, prompt, _ = evaluation_example(row)
            if task == "search":
                source = prompt.splitlines()[0]
            else:
                source = prompt.splitlines()[2].split(". ", 1)[0]
            result.add((task, source))
        return result

    assert len(train) + len(validation) == len(dataset)
    assert sources(train).isdisjoint(sources(validation))
    assert {row["task"] for row in validation} == {"search", "extraction"}


def test_metrics_include_invalid_responses_and_use_task_specific_denominators():
    dataset = load_examples()
    search = next(row for row in dataset if evaluation_example(row)[2].get("action") == "search")
    finish = next(row for row in dataset if evaluation_example(row)[2].get("action") == "finish")
    extraction = next(row for row in dataset if row["task"] == "extraction")
    metrics = score_responses([
        (search, search["completion"][0]["content"]),
        (finish, "invalid JSON"),
        (extraction, extraction["completion"][0]["content"]),
    ])

    assert metrics["eval_json_valid_rate"] == 2 / 3
    assert metrics["eval_search_action_accuracy"] == 0.5
    assert metrics["eval_search_provider_accuracy_count"] == 1
    assert metrics["eval_search_provider_accuracy"] == 1
    assert metrics["eval_extraction_field_exact_match_count"] == 4
    assert metrics["eval_extraction_field_exact_match"] == 1
    assert metrics["eval_extraction_evidence_grounded_rate"] == 1
