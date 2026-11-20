"""Create SYNTHETIC examples of both model skills; not a factual paper dataset.

Each JSONL row is a TRL conversational prompt/completion example. The prompts
match agent.py at inference time. Use real, annotated abstracts for useful models.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prompts import EXTRACT_SYSTEM, PLAN_SYSTEM

TOPICS = [
    ("How does sleep restriction affect attention in shift workers?", "sleep restriction attention shift workers", "openalex"),
    ("What are the effects of mindfulness interventions on anxiety in adolescents?", "mindfulness adolescents anxiety randomized trial", "openalex"),
    ("Do graph neural networks improve molecular property prediction?", "graph neural networks molecular property prediction", "arxiv"),
    ("How effective are transformer models for clinical note classification?", "transformer clinical note classification evaluation", "openalex"),
    ("What are the limitations of retrieval augmented generation in biomedicine?", "retrieval augmented generation biomedical evaluation", "arxiv"),
    ("Does exercise improve glycemic control in type 2 diabetes?", "exercise intervention type 2 diabetes glycemic control", "openalex"),
    ("How does federated learning affect model privacy and accuracy?", "federated learning privacy utility tradeoff", "arxiv"),
    ("How do urban green spaces relate to mental health?", "urban green space mental health cohort", "openalex"),
    ("What is the efficacy of active learning in low resource NLP?", "active learning low resource NLP benchmark", "arxiv"),
    ("Do probiotics reduce antibiotic-associated diarrhea?", "probiotics antibiotic associated diarrhea meta analysis", "openalex"),
    ("Are vision transformers robust to domain shift?", "vision transformer robustness domain shift", "arxiv"),
    ("How do heat waves affect mortality among older adults?", "heat waves older adults mortality epidemiology", "openalex"),
]

STUDIES = [
    {"topic": "sleep restriction and attention", "method": "We conducted a randomized crossover experiment with two sleep schedules.",
     "sample": "Thirty-two night-shift nurses completed both conditions.",
     "finding": "Reaction times were longer after the restricted-sleep condition.",
     "limitation": "The short follow-up limits conclusions about chronic exposure."},
    {"topic": "mindfulness for adolescents", "method": "A cluster randomized controlled trial compared mindfulness classes with standard instruction.",
     "sample": "The analysis included 240 students from eight schools.",
     "finding": "Average anxiety scores decreased more in the intervention arm.",
     "limitation": "Outcomes were self-reported and assessors were not blinded."},
    {"topic": "graph neural networks for chemistry", "method": "We benchmarked message-passing networks against fingerprint baselines.",
     "sample": "Evaluation covered 11 public molecular property datasets.",
     "finding": "The graph models improved mean classification AUROC on seven datasets.",
     "limitation": "Some benchmark datasets contained scaffold overlap across splits."},
    {"topic": "clinical note classification", "method": "We trained a transformer classifier and a linear baseline with identical folds.",
     "sample": "We used 5,600 de-identified clinical notes.",
     "finding": "The transformer achieved higher macro-F1 than the baseline.",
     "limitation": "The single-hospital source may reduce external validity."},
    {"topic": "federated learning", "method": "We simulated federated optimization across distributed client devices.",
     "sample": "The experiments used five image and text benchmarks.",
     "finding": "Non-IID client splits reduced average predictive accuracy.",
     "limitation": "Simulated clients do not fully reproduce real network conditions."},
    {"topic": "urban greenness and mental health", "method": "We conducted a prospective observational cohort analysis.",
     "sample": "The cohort followed 1,820 adult residents for two years.",
     "finding": "Higher neighborhood greenness was associated with lower distress scores.",
     "limitation": "Residual socioeconomic confounding cannot be excluded."},
    {"topic": "exercise and glycemic control", "method": "A systematic review pooled randomized exercise interventions.",
     "sample": "Twelve trials contributed data from 1,090 adults.",
     "finding": "The pooled estimate favored exercise for HbA1c reduction.",
     "limitation": "Treatment intensity varied substantially across included trials."},
    {"topic": "low resource NLP", "method": "We compared uncertainty sampling with random sampling in a controlled benchmark.",
     "sample": "Six text classification corpora were evaluated.",
     "finding": "Active learning used fewer labeled examples at matched accuracy.",
     "limitation": "The evaluation did not include annotation time or cost."},
]


def make_search_example(rng: random.Random, idx: int) -> dict:
    question, query, provider = TOPICS[idx % len(TOPICS)]
    variant = (idx // len(TOPICS) + idx) % 4
    if variant == 0:
        prompt = (f"Research question: {question}\nSearch calls already used: 0/2\n"
                  "Previous search outcomes (not paper text):\n[]\nReturn one search or finish action as JSON.")
        response = {"action": "search", "provider": provider, "query": query, "limit": 8}
    elif variant == 1:
        prompt = (f"Research question: {question}\nSearch calls already used: 1/2\n"
                  "Previous search outcomes (not paper text):\n" +
                  json.dumps([{"step": 1, "action": "search", "provider": provider,
                               "query": query, "returned": 0, "ids": []}]) +
                  "\nReturn one search or finish action as JSON.")
        response = {"action": "search", "provider": "arxiv" if provider == "openalex" else "openalex",
                    "query": query.split(" ")[0] + " " + " ".join(query.split(" ")[-2:]), "limit": 8}
    elif variant == 2:
        prompt = (f"Research question: {question}\nSearch calls already used: 1/2\n"
                  "Previous search outcomes (not paper text):\n" +
                  json.dumps([{"step": 1, "action": "search", "provider": provider,
                               "query": query, "returned": 8, "ids": ["sample-id-1", "sample-id-2"]}]) +
                  "\nReturn one search or finish action as JSON.")
        response = {"action": "finish"}
    else:
        prompt = (f"Research question: {question}\nSearch calls already used: 0/2\n"
                  "Previous search outcomes (not paper text):\n[]\nReturn one search or finish action as JSON.")
        response = {"action": "search", "provider": provider, "query": query + " comparison", "limit": 8}
    return {"task": "search", "prompt": [{"role": "system", "content": PLAN_SYSTEM},
                                       {"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": json.dumps(response, ensure_ascii=False)}]}


def make_extraction_example(rng: random.Random, idx: int) -> dict:
    study = STUDIES[idx % len(STUDIES)]
    first = f"We investigated {study['topic']}."
    sentences = [study["method"], study["sample"], study["finding"], study["limitation"]]
    # Synthetic incomplete abstracts teach nulls rather than unsupported inference.
    if idx % 5 == 1:
        sentences[3] = ""
    if idx % 7 == 2:
        sentences[1] = ""
    if idx % 11 == 3:
        sentences[0] = ""
    abstract = " ".join([first] + [s for s in sentences if s])
    labels = {}
    for key, evidence in zip(("methodology", "dataset_or_sample", "key_findings", "limitations"), sentences):
        labels[key] = {"value": evidence or None, "evidence": evidence or None}
    return {"task": "extraction", "prompt": [{"role": "system", "content": EXTRACT_SYSTEM},
                                            {"role": "user", "content":
                                             f"Paper ID: synthetic-{idx:04d}\nAbstract:\n{abstract}"}],
            "completion": [{"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--n", type=int, default=120, help="Total examples, half search and half extraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n < 4:
        parser.error("--n must be at least 4")
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n // 2):
        rows.append(make_search_example(rng, i))
    for i in range(args.n - len(rows)):
        rows.append(make_extraction_example(rng, i))
    rng.shuffle(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic examples to {path}")


if __name__ == "__main__":
    main()
