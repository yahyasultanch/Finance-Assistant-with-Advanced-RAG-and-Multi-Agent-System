"""Checks for citation verification and the LangGraph flow (stub LLM, no API calls)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agents import memory as mem
from agents.citations import check_claim, make_source, numbers_in, number_supported
from agents.graph import Draft, Judgement, Judgements, Plan, Task, build_brief_graph
from agents.tools import ResearchContext, ROLES

RISK = make_source("filing", "10-K#Risk Factors", "NVDA 10-K Risk Factors", "https://sec.gov/x",
                   "2026-02-25", "Revenue was $46,743 million, up 56% from the prior year.")


def test_numbers_skip_years_and_small_counts():
    assert numbers_in("Revenue rose 56% to $46.7 billion in fiscal 2025 across 3 segments") == [56.0, 46.7]


def test_number_supported_across_unit_scales():
    source = [46_743_000_000.0, 56.0]
    assert number_supported(46.7, source)        # billions
    assert number_supported(46_743, source)      # millions
    assert number_supported(56, source)
    assert not number_supported(52.0, source)


def test_check_claim_statuses():
    sources = {RISK["id"]: RISK}
    ok = check_claim({"text": "Revenue was $46.7 billion, up 56%.", "sources": [RISK["id"]]}, sources)
    bad_number = check_claim({"text": "Revenue was $52.1 billion.", "sources": [RISK["id"]]}, sources)
    bad_id = check_claim({"text": "Revenue grew.", "sources": ["S000000"]}, sources)
    assert (ok["status"], bad_number["status"], bad_id["status"]) == ("ok", "number_mismatch", "bad_citation")


def test_source_ids_are_stable_content_hashes():
    again = make_source("filing", "10-K#Risk Factors", "other title", "https://sec.gov/x",
                        "2026-02-25", RISK["text"])
    other = make_source("filing", "10-K#Risk Factors", "", "https://sec.gov/x", "", "Different text.")
    assert again["id"] == RISK["id"] != other["id"]


class StubLLM:
    """Scripted outputs: the first draft cites a missing id, the second is fixed."""

    def __init__(self):
        self.drafts = 0
        self.plans = 0

    def structured(self, messages, model):
        if model is Plan:
            self.plans += 1
            return Plan(tasks=[Task(researcher="filings", objective="risks"),
                               Task(researcher="news", objective="events")])
        if model is Draft:
            self.drafts += 1
            source = "S000000" if self.drafts == 1 else RISK["id"]
            return Draft(risks=[{"text": "Revenue was $46.7 billion.", "sources": [source]}])
        if model is Judgements:
            return Judgements(judgements=[Judgement(index=0, supported=True)])
        raise AssertionError(model)

    def run_tools(self, messages, tools, max_steps):
        return f"- Revenue was $46,743 million [{RISK['id']}]", [RISK], [
            {"tool": "search_filings", "args": {}, "ms": 1, "sources": 1, "error": None, "preview": ""}]


class StubOrchestrator:
    def __init__(self):
        self.llm = StubLLM()
        self.ctx = ResearchContext(
            ticker="NVDA", retriever=None,
            vector_store=SimpleNamespace(size=1, documents=[{"metadata": {"form_type": "10-K"}}]),
            news_items=[], company_info={"name": "NVIDIA"}, price_metrics={}, metrics=[],
        )

    def load_context(self, ticker, use_cache, days_back):
        return self.ctx

    def context(self, ticker):
        return self.ctx

    def tools(self, ticker):
        return {name: None for _, names in ROLES.values() for name in names}

    def score(self, ticker, summaries):
        return {"sentiment": {},
                "prediction": {"predictions": {"short_term": {"target_price": 104.0},
                                               "long_term": {"target_price": 120.0}}},
                "rating": {"rating": "HOLD", "composite_score": 50, "price_target": 1.0}}

    def finish_report(self, state, track_record=None):
        return {"summaries": state["summaries"], "verification": state["verification"],
                "price_metrics": {"current_price": 100.0}}

    def price_closes(self, ticker):
        return [("2026-01-02", 100.0)]


def test_graph_revises_failed_draft_then_pauses_for_review():
    orc = StubOrchestrator()
    graph = build_brief_graph(orc, InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    paused = graph.invoke({"ticker": "NVDA", "review": True}, config)
    assert paused["__interrupt__"][0].value["coverage"] == 1.0

    done = graph.invoke(Command(resume={"approved": True, "note": "ok"}), config)
    verification = done["result"]["verification"]
    assert (verification["first_pass_coverage"], verification["coverage"], verification["revisions"]) == (0.0, 1.0, 1)
    assert (orc.llm.plans, orc.llm.drafts) == (2, 2)
    # Two tasks per plan ran as parallel researcher branches, twice
    assert sum(e["node"].startswith("researcher") for e in done["trace"]) == 4
    assert done["result"]["summaries"]["risks"]["summary"] == \
        "- Revenue was $46.7 billion. [1:10-K#Risk Factors]"


def test_graph_rejected_review_skips_report():
    graph = build_brief_graph(StubOrchestrator(), InMemorySaver())
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"ticker": "NVDA", "review": True}, config)
    done = graph.invoke(Command(resume={"approved": False}), config)
    assert "result" not in done


def test_ledger_scores_due_calls_and_tracks_pending_ones():
    entries = [
        {"kind": "7d", "made_on": "2026-01-01", "price": 100.0, "target": 110.0, "due": "2026-01-08"},
        {"kind": "12m", "made_on": "2026-01-01", "price": 100.0, "target": 90.0, "due": "2027-01-01"},
    ]
    closes = [("2026-01-05", 105.0), ("2026-01-09", 120.0)]
    due, pending = mem.score_entries(entries, closes)
    # First close on/after the due date is 120: target 110 undershot by 8.33%, direction right
    assert (due["status"], due["actual"], due["error_pct"], due["direction_hit"]) == ("scored", 120.0, -8.33, True)
    assert (pending["status"], pending["actual"]) == ("pending", 120.0)
    summary = mem.summarize([due, pending])
    assert (summary["scored"], summary["pending"], summary["direction_hit_rate"]) == (1, 1, 1.0)


def test_memory_carries_over_between_runs():
    store = InMemoryStore()
    graph = build_brief_graph(StubOrchestrator(), InMemorySaver(), store)

    first = graph.invoke({"ticker": "NVDA"}, {"configurable": {"thread_id": "m1"}})
    assert first["memory"] == {}
    assert [b["rating"] for b in mem.recent_briefs(store, "NVDA")] == ["HOLD"]
    assert sorted(e["kind"] for e in mem.ledger(store, "NVDA")) == ["12m", "7d", "rating_target"]

    second = graph.invoke({"ticker": "NVDA"}, {"configurable": {"thread_id": "m2"}})
    memory_id = second["memory"]["id"]
    assert second["sources"][memory_id]["kind"] == "memory"
    assert "rating HOLD" in second["memory"]["text"]
    assert "memory of earlier briefs loaded" in [t for t in second["trace"] if t["node"] == "planner"][0]["detail"]
