"""Checks for citation verification and the LangGraph flow (stub LLM, no API calls)."""
import sys
from pathlib import Path
from datetime import date, timedelta
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agents import memory as mem
from agents.citations import check_claim, make_source, numbers_in, number_supported, reattribute
from agents.graph import Draft, Judgement, Judgements, Omissions, Plan, Task, build_brief_graph
from agents.tools import ResearchContext, ROLES
from eval.run_eval import event_recall

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
    """Scripted outputs: the first draft has an unsupported figure, the second is fixed."""

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
            # First draft: a figure no retrieved source contains, so citation repair can't save it
            if self.drafts == 1:
                return Draft(risks=[{"text": "Revenue was $99.9 billion.", "sources": ["S000000"]}])
            return Draft(risks=[{"text": "Revenue was $46.7 billion.", "sources": [RISK["id"]]}])
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
    # Parallel researcher branches: 2 planned + the mandatory recent-events task, then 2 on revision
    assert sum(e["node"].startswith("researcher") for e in done["trace"]) == 5
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
    assert sorted(e["kind"] for e in mem.ledger(store, "NVDA")) == ["12m", "7d"]

    # A same-day rerun must not treat its own earlier output as the last brief
    assert graph.invoke({"ticker": "NVDA"}, {"configurable": {"thread_id": "m2"}})["memory"] == {}

    # Date today's record as yesterday's: now it is an earlier brief
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    record = mem.recent_briefs(store, "NVDA")[0]
    store.delete(("memory", "NVDA"), record["date"])
    mem.remember_brief(store, {**record, "date": yesterday})
    second = graph.invoke({"ticker": "NVDA"}, {"configurable": {"thread_id": "m3"}})
    memory_id = second["memory"]["id"]
    assert second["sources"][memory_id]["kind"] == "memory"
    assert "rating HOLD" in second["memory"]["text"]
    assert "memory of earlier briefs loaded" in [t for t in second["trace"] if t["node"] == "planner"][0]["detail"]


def test_nan_price_is_not_remembered_and_old_null_records_still_load():
    store = InMemoryStore()
    nan = float("nan")
    verification = {"claims": [], "coverage": 1.0}
    prediction = {"predictions": {"short_term": {"target_price": nan}, "long_term": {"target_price": nan}}}
    mem.remember_brief(store, mem.brief_record("NVDA", "2026-11-25", nan, {"price_target": nan}, verification))
    mem.record_predictions(store, mem.ledger_entries_for("NVDA", "2026-11-25", nan, prediction))
    # A call stored before prices were checked
    store.put(("ledger", "NVDA"), "2026-11-24:7d", {"ticker": "NVDA", "kind": "7d", "made_on": "2026-11-24",
                                                    "price": None, "target": None, "due": "2026-12-01"})

    assert mem.ledger(store, "NVDA") == []
    assert "price then n/a" in mem.format_memory(mem.recent_briefs(store, "NVDA"))


class OmissionLLM(StubLLM):
    """Draft is fully verified, but leaves out the recent 8-K on the first pass."""

    def __init__(self):
        super().__init__()
        self.checks = 0
        self.analyst_prompts = []

    def structured(self, messages, model):
        if model is Omissions:
            self.checks += 1
            return Omissions(missing=[0] if self.checks == 1 else [])
        if model is Draft:
            self.analyst_prompts.append(messages[-1]["content"])
            self.drafts += 1
            return Draft(risks=[{"text": "Revenue was $46.7 billion.", "sources": [RISK["id"]]}])
        return super().structured(messages, model)


def test_missing_material_event_triggers_a_revision():
    orc = StubOrchestrator()
    orc.llm = OmissionLLM()
    orc.ctx.news_items = [{"source": "SEC 8-K", "date": date.today().isoformat(), "url": "u",
                           "title": "8-K: NVIDIA Announces CEO Transition", "summary": "Jensen Huang resigns."}]
    done = build_brief_graph(orc, InMemorySaver()).invoke({"ticker": "NVDA"}, {"configurable": {"thread_id": "e1"}})

    verification = done["result"]["verification"]
    # Every claim verified, yet the omission alone sent the draft back once
    assert (verification["first_pass_coverage"], verification["revisions"]) == (1.0, 1)
    assert "CEO Transition" in orc.llm.analyst_prompts[1]
    assert verification["missing_events"] == []


def test_event_recall_counts_filings_cited_by_verified_claims():
    folder = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000082"
    events = [
        {"date": "2026-11-12", "headline": "NVIDIA Announces CEO Transition", "url": f"{folder}/nvda-8k.htm"},
        {"date": "2026-11-23", "headline": "Board Reinstates Jensen Huang", "url": f"{folder}9/nvda-8k.htm"},
        {"date": "2026-11-01", "headline": "", "url": f"{folder}7/nvda-8k.htm"},  # no press release: not in the key
    ]
    # Cited via the exhibit (same accession folder, CIK not zero-padded), with a text fragment
    exhibit = folder.replace("/1045810/", "/0001045810/") + "/ex99.htm#:~:text=Jensen"
    result = {"summaries": {"events": {"citations": [{"url": exhibit}]}}}
    covered, key = event_recall(result, events)
    assert [e["date"] for e in covered] == ["2026-11-12"] and len(key) == 2


def test_misattributed_citation_is_repaired_to_the_source_with_the_figures():
    ceo = make_source("event", "8-K#Event", "", "u1", "2026-11-12", "8-K 2026-11-12: Jensen Huang resigns as CEO.")
    results = make_source("event", "8-K#Event", "", "u2", "2026-11-17",
                          "8-K 2026-11-17: Q3 revenue of $115.8 billion, up 103% year over year.")
    sources = {s["id"]: s for s in (ceo, results)}
    claim = {"text": "On November 17, NVIDIA reported revenue of $115.8 billion, up 103%.", "sources": [ceo["id"]]}

    repaired = reattribute(check_claim(claim, sources), sources)
    assert (repaired["status"], repaired["sources"]) == ("ok", [results["id"]])
    # Figures no retrieved source contains: left as a failure
    made_up = {"text": "Revenue was $999.9 billion.", "sources": [ceo["id"]]}
    assert reattribute(check_claim(made_up, sources), sources)["status"] == "number_mismatch"
