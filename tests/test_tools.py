from tools import (Paper, deduplicate, openalex_parse, rank_papers,
                   reconstruct_abstract, search_arxiv, search_openalex, SearchError)
import pytest


class Response:
    def __init__(self, json_data=None, content=b""):
        self.payload = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def paper(id, title="Graph neural networks for molecular prediction", doi=None, abstract="Graph neural networks improve molecular prediction."):
    return Paper(id=id, title=title, abstract=abstract, year=2023, authors=["A"],
                 doi=doi, url="https://example.org/" + id, citations=20, source="openalex")


def test_reconstruct_abstract():
    assert reconstruct_abstract({"Improves": [1], "Training": [0], "models": [2]}) == "Training Improves models"
    assert reconstruct_abstract(None) == ""


def test_openalex_parsing_and_request():
    data = {"results": [{"id": "https://openalex.org/W1", "title": "Evaluation of Graph Models",
                         "publication_year": 2024, "cited_by_count": 3,
                         "doi": "https://doi.org/10.5555/ABC", "authorships": [],
                         "abstract_inverted_index": {"Networks": [0], "work": [1]}}]}
    sess = Session(Response(json_data=data))
    results = search_openalex("graph models", 3, session=sess, api_key="demo")
    assert len(results) == 1
    assert results[0].doi == "10.5555/abc"
    assert results[0].abstract == "Networks work"
    assert sess.calls[0][1]["params"]["api_key"] == "demo"


def test_arxiv_parsing():
    xml = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
    <entry><id>http://arxiv.org/abs/2401.00001v1</id>
    <title>GNN Work</title><summary>Some abstract</summary>
    <published>2024-01-02T00:00:00Z</published><author><name>Ada</name></author></entry>
    </feed>'''
    results = search_arxiv("graph neural networks", session=Session(Response(content=xml)))
    assert results[0].year == 2024
    assert results[0].authors == ["Ada"]
    assert results[0].abstract == "Some abstract"


def test_ranking_and_deduplication():
    results = [paper("a", doi="10.1/same", abstract=""),
               paper("b", doi="10.1/same"),
               paper("c", title="Unrelated title", abstract="No terms match.")]
    assert [p.id for p in deduplicate(results)] == ["b", "c"]
    assert [p.id for p in rank_papers("graph neural networks molecular prediction", results)] == ["b"]


def test_provider_failure():
    with pytest.raises(ValueError):
        search_openalex(" ")
    with pytest.raises(SearchError):
        search_openalex("graph", session=Session(Response(json_data={"unexpected": []})), api_key="")
