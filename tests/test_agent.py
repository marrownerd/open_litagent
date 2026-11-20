import json
from agent import LiteratureAgent, check_insights, parse_json_object
from tools import Paper


ABSTRACT = ("A randomized trial enrolled 30 adults. We compared therapy to placebo. "
            "The treatment improved sleep duration. Follow-up lasted only one week.")


class FakeModel:
    def __init__(self):
        self.queries = []

    def generate(self, messages, max_new_tokens=350):
        system = messages[0]["content"]
        if "paper-search controller" in system:
            self.queries.append(messages[1]["content"])
            if len(self.queries) == 1:
                return json.dumps({"action": "search", "provider": "openalex",
                                   "query": "sleep improvement randomized trial", "limit": 8})
            return '{"action":"finish"}'
        if "exclusively" in system:
            return json.dumps({
                "methodology": {"value": "Randomized trial", "evidence": "A randomized trial enrolled 30 adults."},
                "dataset_or_sample": {"value": "30 adults", "evidence": "enrolled 30 adults"},
                "key_findings": {"value": "Improved sleep duration", "evidence": "The treatment improved sleep duration."},
                "limitations": {"value": "Long-term follow-up missing", "evidence": "long-term outcomes unreported"},
            })
        return "The study reports improved sleep after treatment [1], with limited follow-up [1]."


def test_evidence_validation_rejects_hallucinations():
    raw = {"methodology": {"value": "Randomized", "evidence": "A randomized trial"},
           "limitations": {"value": "Long-term harms", "evidence": "long-term harms"}}
    verified = check_insights(raw, ABSTRACT)
    assert verified["methodology"]["value"] == "Randomized"
    assert verified["limitations"]["value"] is None
    assert verified["dataset_or_sample"]["value"] is None


def test_parse_json_code_fence():
    assert parse_json_object('```json\n{"action": "finish"}\n```')["action"] == "finish"


def test_agent_end_to_end_mocked():
    searches = []
    def search_fake(provider, query, limit):
        searches.append((provider, query, limit))
        return [Paper(id="doi:test", title="Sleep improvement randomized trial", abstract=ABSTRACT,
                      year=2024, authors=["A"], doi="10.1111/example",
                      url="https://doi.org/10.1111/example", citations=0, source="openalex")]
    agent = LiteratureAgent(FakeModel(), search_fn=search_fake, max_searches=2, max_papers=3)
    review = agent.run("sleep improvement randomized trial")
    assert len(searches) == 1
    assert len(review.papers) == 1
    assert review.papers[0].insights["dataset_or_sample"]["value"] == "30 adults"
    assert review.papers[0].insights["limitations"]["value"] is None
    assert "Sleep improvement" in review.to_markdown()
    assert review.search_log[0]["returned"] == 1
    assert review.to_dict()["papers"][0]["paper"]["id"] == "doi:test"


def test_fallback_if_invalid_tool_call():
    class InvalidModel(FakeModel):
        def generate(self, messages, max_new_tokens=350):
            if "paper-search controller" in messages[0]["content"]:
                return "not valid JSON"
            return super().generate(messages, max_new_tokens)
    def fail(provider, query, limit):
        raise RuntimeError("503")
    review = LiteratureAgent(InvalidModel(), search_fn=fail, max_searches=1).run("sleep trial")
    assert review.papers == []
    assert review.errors
    assert "No matching papers" in review.summary
