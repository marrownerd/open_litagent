"""Bounded model-directed academic search and grounded extraction."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from llm import ChatModel
from prompts import EXTRACT_SYSTEM, PLAN_SYSTEM, SYNTH_SYSTEM
from tools import Paper, rank_papers, search


FIELDS = ("methodology", "dataset_or_sample", "key_findings", "limitations")


def parse_json_object(text: str) -> dict[str, Any]:
    """Accept a JSON object, even if the LLM wrapped it in code fences."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean)
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found")
        obj = json.loads(clean[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model response must be a JSON object")
    return obj


def empty_insights() -> dict[str, dict[str, None]]:
    return {k: {"value": None, "evidence": None} for k in FIELDS}


def check_insights(raw: dict[str, Any], abstract: str) -> dict[str, dict[str, str | None]]:
    """Reject unsupported evidence. An exact quote is necessary, not sufficient,
    for scientific correctness; human verification is still required."""
    checked = empty_insights()
    # Collapsing whitespace handles abstracts from Atom feeds without inventing words.
    normalized = " ".join(abstract.split()).casefold()
    for key in FIELDS:
        item = raw.get(key)
        if not isinstance(item, dict):
            continue
        value, evidence = item.get("value"), item.get("evidence")
        if not isinstance(value, str) or not value.strip():
            continue
        if not isinstance(evidence, str) or not evidence.strip():
            continue
        if " ".join(evidence.split()).casefold() not in normalized:
            continue
        checked[key] = {"value": value.strip()[:550], "evidence": evidence.strip()[:500]}
    return checked


@dataclass
class ReviewedPaper:
    paper: Paper
    insights: dict[str, dict[str, str | None]]

    def to_dict(self) -> dict[str, Any]:
        return {"paper": self.paper.to_dict(), "insights": self.insights}


@dataclass
class Review:
    question: str
    papers: list[ReviewedPaper] = field(default_factory=list)
    summary: str = ""
    search_log: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "papers": [p.to_dict() for p in self.papers],
                "summary": self.summary, "search_log": self.search_log, "errors": self.errors}

    def to_markdown(self) -> str:
        lines = ["# Literature review", "", "**Question:** " + self.question, "", "## Comparison", "",
                 "| # | Paper | Methodology | Dataset / sample | Key findings | Limitations |",
                 "|---|---|---|---|---|---|"]
        for i, item in enumerate(self.papers, 1):
            p = item.paper
            title = _escape_cell(p.title)
            ref = f"[{i}] [{title}]({p.url}) ({p.year or 'n.d.'})"
            fields = [_escape_cell((item.insights[k].get("value") or "Not stated in abstract")) for k in FIELDS]
            lines.append("| " + " | ".join([str(i), ref] + fields) + " |")
        if not self.papers:
            lines.append("| – | No matching papers with abstracts retrieved | – | – | – | – |")
        lines.extend(["", "## Synthesis", "", self.summary or "No evidence available for synthesis.",
                      "", "## Sources", ""])
        for i, item in enumerate(self.papers, 1):
            p = item.paper
            doi = " DOI: " + p.doi if p.doi else ""
            lines.append(f"[{i}] {p.title} ({p.year or 'n.d.'}). {p.url}.{doi}")
        lines.extend(["", "*Extracted from abstracts only. Quotations are checked for occurrence, not scientific entailment.*"])
        return "\n".join(lines) + "\n"


def _escape_cell(text: str) -> str:
    return " ".join(str(text).replace("|", "\\|").split())[:550]


class LiteratureAgent:
    def __init__(self, model: ChatModel,
                 search_fn: Callable[[str, str, int], list[Paper]] = search,
                 *, max_searches: int = 2, per_search: int = 8,
                 max_papers: int = 5, provider: str = "both"):
        if not (1 <= max_searches <= 6 and 1 <= per_search <= 25 and 1 <= max_papers <= 20):
            raise ValueError("Invalid search limits")
        if provider not in {"both", "openalex", "arxiv"}:
            raise ValueError("provider must be both, openalex, or arxiv")
        self.model, self.search_fn = model, search_fn
        self.max_searches, self.per_search, self.max_papers = max_searches, per_search, max_papers
        self.provider = provider

    def _plan(self, question: str, history: list[dict[str, Any]], count: int) -> dict[str, Any]:
        prompt = (f"Research question: {question}\n"
                  f"Search calls already used: {count}/{self.max_searches}\n"
                  "Previous search outcomes (not paper text):\n" + json.dumps(history[-4:]) +
                  "\nReturn one search or finish action as JSON.")
        try:
            action = parse_json_object(self.model.generate([
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": prompt}], max_new_tokens=160))
        except (ValueError, TypeError):
            action = {}
        if action.get("action") == "finish" and count >= 1:
            return {"action": "finish"}
        if action.get("action") != "search" or not isinstance(action.get("query"), str):
            action = {"action": "search", "provider": "openalex", "query": question}
        if self.provider != "both":
            action["provider"] = self.provider
        if action.get("provider") not in {"openalex", "arxiv"}:
            action["provider"] = "openalex"
        action["query"] = action["query"].strip()[:250] or question[:250]
        action["limit"] = self.per_search
        return action

    def _extract(self, paper: Paper) -> dict[str, dict[str, str | None]]:
        if not paper.abstract.strip():
            return empty_insights()
        prompt = "Paper ID: " + paper.id + "\nAbstract:\n" + paper.abstract[:7000]
        try:
            obj = parse_json_object(self.model.generate([
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": prompt}], max_new_tokens=430))
            return check_insights(obj, paper.abstract[:7000])
        except (ValueError, TypeError):
            return empty_insights()

    def _synthesize(self, question: str, papers: list[ReviewedPaper]) -> str:
        if not papers:
            return "No matching papers with abstracts could be retrieved. Try a narrower question or an API key."
        evidence = [{"ref": f"[{i}]", "title": p.paper.title,
                     "year": p.paper.year, "insights": p.insights}
                    for i, p in enumerate(papers, 1)]
        raw = self.model.generate([
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": "Question: " + question +
             "\nEvidence:\n" + json.dumps(evidence, ensure_ascii=False) +
             "\nWrite a concise comparison and the main evidence gaps."}], max_new_tokens=400).strip()
        # Avoid unsupported paper number citations; grounded table is always available.
        if any(int(n) < 1 or int(n) > len(papers) for n in re.findall(r"\[(\d+)\]", raw)):
            return "Evidence is tabulated above. Verify each quoted abstract before drawing conclusions."
        return raw or "Evidence is tabulated above; verify the abstracts before drawing conclusions."

    def run(self, question: str) -> Review:
        if not question.strip():
            raise ValueError("Question must not be empty")
        review = Review(question=question.strip())
        fetched: list[Paper] = []
        seen_actions: set[tuple[str, str]] = set()
        for attempt in range(self.max_searches):
            action = self._plan(review.question, review.search_log, attempt)
            if action["action"] == "finish":
                if fetched:
                    break
                action = {"action": "search",
                          "provider": self.provider if self.provider != "both" else "openalex",
                          "query": review.question, "limit": self.per_search}
            provider, query = action["provider"], action["query"]
            signature = (provider, query.casefold())
            if signature in seen_actions:
                # Force a new provider rather than endlessly repeat identical API calls.
                alt = "arxiv" if provider == "openalex" else "openalex"
                if self.provider == "both" and (alt, query.casefold()) not in seen_actions:
                    provider = alt
                else:
                    break
            seen_actions.add((provider, query.casefold()))
            entry = {"step": attempt + 1, "action": "search", "provider": provider,
                     "query": query, "limit": self.per_search}
            try:
                results = self.search_fn(provider, query, self.per_search)
                entry["returned"] = len(results)
                entry["ids"] = [p.id for p in results]
                fetched.extend(results)
            except Exception as exc:
                # A failed API call should not invalidate results from earlier calls.
                entry["error"] = str(exc)
                review.errors.append(f"{provider}: {exc}")
            review.search_log.append(entry)
        selected = rank_papers(review.question, fetched, self.max_papers)
        for paper in selected:
            try:
                insights = self._extract(paper)
            except Exception as exc:
                review.errors.append(f"extract {paper.id}: {exc}")
                insights = empty_insights()
            review.papers.append(ReviewedPaper(paper=paper, insights=insights))
        try:
            review.summary = self._synthesize(review.question, review.papers)
        except Exception as exc:
            review.errors.append(f"synthesis: {exc}")
            review.summary = "See the evidence table and linked abstracts above."
        return review
