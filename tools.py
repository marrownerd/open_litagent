"""Free academic search tools. Works without credentials on modest workloads.

Use OPENALEX_API_KEY for a larger OpenAlex rate budget. No full-text claims are made:
OpenAlex abstracts are reconstructed from inverted indices and arXiv from Atom XML.
"""
from __future__ import annotations

import math
import os
import re
import time
import threading
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_REQUEST = 0.0


class SearchError(RuntimeError):
    """A provider failed or returned an unusable response."""


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    year: int | None
    authors: list[str]
    doi: str | None
    url: str
    citations: int
    source: str
    venue: str | None = None
    relevance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "OpenLiteratureAgent/1.0 (academic metadata client)",
        "Accept": "application/json, application/atom+xml;q=0.9",
    })
    return session


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: dict[int, str] = {}
    for word, indices in index.items():
        for pos in indices:
            if isinstance(pos, int) and pos >= 0:
                positions[pos] = word
    return " ".join(positions[pos] for pos in sorted(positions))


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip(), flags=re.I).lower()


def openalex_parse(work: dict[str, Any]) -> Paper:
    location = work.get("primary_location") or {}
    journal = location.get("source") or {}
    doi = normalize_doi(work.get("doi"))
    paper_id = work.get("id") or ("doi:" + doi if doi else "")
    if not paper_id:
        raise SearchError("OpenAlex result missing identifier")
    return Paper(
        id=str(paper_id),
        title=(work.get("display_name") or work.get("title") or "Untitled").strip(),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        year=work.get("publication_year"),
        authors=[item.get("author", {}).get("display_name", "") for item in (work.get("authorships") or [])
                 if item.get("author", {}).get("display_name")][:12],
        doi=doi,
        url=("https://doi.org/" + quote(doi, safe="/")) if doi else (location.get("landing_page_url") or str(paper_id)),
        citations=max(0, work.get("cited_by_count") or 0),
        source="openalex",
        venue=journal.get("display_name"),
    )


def search_openalex(query: str, limit: int = 8, *, session: requests.Session | None = None,
                    api_key: str | None = None) -> list[Paper]:
    if not query.strip():
        raise ValueError("Search query must not be empty")
    key = api_key if api_key is not None else os.getenv("OPENALEX_API_KEY")
    params: dict[str, Any] = {"search": query[:250], "per_page": max(1, min(limit, 25)),
                              "filter": "has_abstract:true"}
    if key:
        params["api_key"] = key
    session = session or build_session()
    try:
        response = session.get("https://api.openalex.org/works", params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload.get("results"), list):
            raise ValueError("Invalid OpenAlex results response")
        parsed = []
        for work in payload["results"]:
            if not isinstance(work, dict):
                continue
            try:
                parsed.append(openalex_parse(work))
            except (SearchError, ValueError, TypeError):
                continue
        return parsed
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise SearchError(f"OpenAlex request failed: {exc}") from exc


def search_arxiv(query: str, limit: int = 8, *, session: requests.Session | None = None) -> list[Paper]:
    if not query.strip():
        raise ValueError("Search query must not be empty")
    # Use AND across a few keywords rather than demanding an exact long phrase.
    keywords = [w for w in re.findall(r"[a-zA-Z0-9-]+", query.lower())
                if len(w) > 2 and w not in _STOPWORDS][:4]
    if not keywords:
        raise ValueError("arXiv query needs at least one keyword")
    search_query = " AND ".join("all:" + w for w in keywords)
    session = session or build_session()
    # Respect arXiv's legacy API request-spacing rule within this process.
    global _ARXIV_LAST_REQUEST
    with _ARXIV_LOCK:
        elapsed = time.monotonic() - _ARXIV_LAST_REQUEST if _ARXIV_LAST_REQUEST else 999.0
        if elapsed < 3.1:
            time.sleep(3.1 - elapsed)
        _ARXIV_LAST_REQUEST = time.monotonic()
    try:
        response = session.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": search_query,
                    "start": 0, "max_results": max(1, min(limit, 25)),
                    "sortBy": "relevance", "sortOrder": "descending"},
            timeout=35,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as exc:
        raise SearchError(f"arXiv request failed: {exc}") from exc
    atom = "{http://www.w3.org/2005/Atom}"
    arxiv = "{http://arxiv.org/schemas/atom}"
    papers = []
    for entry in root.findall(atom + "entry"):
        identifier = (entry.findtext(atom + "id") or "").strip()
        if not identifier:
            continue
        title = " ".join((entry.findtext(atom + "title") or "Untitled").split())
        abstract = " ".join((entry.findtext(atom + "summary") or "").split())
        authors = [(a.findtext(atom + "name") or "") for a in entry.findall(atom + "author")]
        published = entry.findtext(atom + "published") or ""
        year = int(published[:4]) if re.match(r"^\d{4}", published) else None
        doi = normalize_doi(entry.findtext(arxiv + "doi"))
        papers.append(Paper(
            id=identifier, title=title, abstract=abstract, year=year,
            authors=authors[:12], doi=doi, url=identifier,
            citations=0, source="arxiv", venue="arXiv",
        ))
    return papers


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Prefer entries with more abstract text and citation metadata."""
    results: dict[str, Paper] = {}
    for paper in papers:
        title_key = re.sub(r"\W+", "", paper.title.lower())
        key = "doi:" + paper.doi.lower() if paper.doi else "title:" + title_key
        if key not in results or (len(paper.abstract), paper.citations) > (
            len(results[key].abstract), results[key].citations
        ):
            results[key] = paper
    return list(results.values())


_STOPWORDS = set("a an and are as at be by can do does for from how in into is of on or the to via vs what which with among about study studies effects impact using use their were was this that between".split())


def _terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in _STOPWORDS}


def rank_papers(question: str, papers: list[Paper], limit: int = 6) -> list[Paper]:
    """Lexical relevance first; weak tie-breakers for cited and newer work."""
    question_terms = _terms(question)
    candidates: list[Paper] = []
    for paper in deduplicate(papers):
        if not paper.abstract.strip():
            continue  # Can't do grounded abstract extraction without an abstract.
        title_overlap = len(question_terms & _terms(paper.title)) / max(1, len(question_terms))
        body_overlap = len(question_terms & _terms(paper.abstract)) / max(1, len(question_terms))
        if question_terms and title_overlap == 0 and body_overlap == 0:
            continue
        citation_bonus = min(1, math.log1p(paper.citations) / 10)
        freshness_bonus = max(0.0, min(1.0, ((paper.year or 2000) - 2000) / 30))
        paper.relevance = round(0.55 * title_overlap + 0.35 * body_overlap
                                + 0.05 * citation_bonus + 0.05 * freshness_bonus, 4)
        candidates.append(paper)
    return sorted(candidates, key=lambda p: (-p.relevance, -(p.year or 0)))[:limit]


def search(provider: str, query: str, limit: int, session: requests.Session | None = None) -> list[Paper]:
    if provider == "openalex":
        return search_openalex(query, limit, session=session)
    if provider == "arxiv":
        return search_arxiv(query, limit, session=session)
    raise ValueError(f"Unsupported provider: {provider}")
