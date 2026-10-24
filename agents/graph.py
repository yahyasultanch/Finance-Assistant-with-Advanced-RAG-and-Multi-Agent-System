"""LangGraph orchestration for analyst briefs and follow-up chat.

Brief graph:

    ingest -> planner --Send--> researcher x N (parallel tool loops) -> analyst -> verifier
                 ^                                                                  |
                 +---------- coverage < MIN_COVERAGE, revisions left: replan <-----+
                                                                                    v
                                    score (sentiment, rating, prediction) -> review -> report

Long-term memory (a LangGraph Store, injected into nodes as `store`): the
planner loads earlier briefs for the ticker and registers them as a citable
source, the analyst writes what changed, and the report node scores the
prediction ledger and records this brief and its predictions.

State holds only plain data so the SQLite checkpointer can persist it; the
retriever, models and tools live on the orchestrator (`orc`) and are looked
up by ticker inside the nodes.
"""
import logging
import operator
from datetime import date
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from agents import memory as mem
from agents.citations import check_claim, deep_link, make_source
from agents.tools import ROLES

logger = logging.getLogger(__name__)

# Report section key -> (title, what to cover, max claims)
SECTIONS = {
    "overview": ("Business Overview", "business model, segments, products and customers", 3),
    "drivers": ("Key Growth Drivers", "growth drivers and opportunities, backed by reported numbers", 3),
    "risks": ("Risk Factors", "the most significant company-specific risks", 3),
    "events": ("Recent Material Events", "recent 8-K events, earnings releases and announcements", 5),
    "outlook": ("Financial Outlook", "guidance and forward-looking expectations", 3),
    "changes": ("Changes Since Last Brief", "what changed versus the previous brief in memory "
                "(rating, figures, events), backed by new sources", 3),
}
MIN_COVERAGE = 0.8   # share of claims that must verify before skipping a revision
MAX_REVISIONS = 1
MAX_TASKS = 6

DEFAULT_PLAN = [
    {"researcher": "filings", "objective": "Business model, segments and strategy from the latest 10-K Business section"},
    {"researcher": "filings", "objective": "The most significant risk factors from the 10-K and recent 10-Qs"},
    {"researcher": "financials", "objective": "Latest annual and quarterly growth, margins, cash and debt"},
    {"researcher": "news", "objective": "Recent material events: 8-K earnings releases and announcements"},
    {"researcher": "filings", "objective": "Outlook and guidance from MD&A and the latest earnings press release"},
]


def merge_dicts(a: Dict, b: Dict) -> Dict:
    return {**a, **b}


class BriefState(TypedDict, total=False):
    ticker: str
    use_cache: bool
    days_back: int
    save_format: str
    review: bool
    plan: List[Dict[str, str]]
    memory: Dict[str, str]          # {"id": source id, "text": earlier briefs}
    feedback: str
    revisions: int
    findings: Annotated[List[Dict[str, Any]], operator.add]
    sources: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    trace: Annotated[List[Dict[str, Any]], operator.add]
    draft: Dict[str, List[Dict[str, Any]]]
    verification: Dict[str, Any]
    summaries: Dict[str, Any]
    scores: Dict[str, Any]
    approved: bool
    review_note: str
    result: Dict[str, Any]


class ChatState(TypedDict):
    ticker: str
    messages: Annotated[List[Dict[str, Any]], operator.add]


# Structured LLM outputs

class Task(BaseModel):
    researcher: Literal["filings", "financials", "news"]
    objective: str


class Plan(BaseModel):
    tasks: List[Task] = Field(min_length=1, max_length=MAX_TASKS)


class Claim(BaseModel):
    text: str
    sources: List[str]


class Draft(BaseModel):
    overview: List[Claim] = []
    drivers: List[Claim] = []
    risks: List[Claim] = []
    events: List[Claim] = []
    outlook: List[Claim] = []
    changes: List[Claim] = []


class Judgement(BaseModel):
    index: int
    supported: bool
    reason: str = ""


class Judgements(BaseModel):
    judgements: List[Judgement]


def build_summaries(claims: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Turn verified claims into the per-section summaries the reporter and UI
    use. Sources get one report-wide number in order of first use, so the
    same source has the same [n:label] everywhere.
    """
    numbers: Dict[str, int] = {}
    summaries = {}
    for key in SECTIONS:
        lines, citations, contexts = [], [], []
        for claim in claims:
            if claim["section"] != key or claim["status"] != "verified":
                continue
            refs = []
            for sid in claim["sources"]:
                source = sources[sid]
                numbers.setdefault(sid, len(numbers) + 1)
                ref = f"[{numbers[sid]}:{source['label']}]"
                refs.append(ref)
                citation = {
                    "number": numbers[sid], "reference": ref, "source": source["title"],
                    "section": source["label"], "url": deep_link(source),
                    "date": source["date"], "tier": source["tier"],
                }
                if citation not in citations:
                    citations.append(citation)
                    contexts.append({"text": source["text"], "metadata": {"source_id": sid}})
            lines.append(f"- {claim['text']} {' '.join(refs)}")
        summaries[key] = {
            "summary": "\n".join(lines) or "No verified information available.",
            "citations": citations,
            "contexts": contexts,
        }
    return summaries


def route_after_verify(state: BriefState) -> str:
    return "planner" if state.get("feedback") else "score"


def route_after_review(state: BriefState) -> str:
    return "report" if state.get("approved", True) else END


def build_brief_graph(orc, checkpointer=None, store=None):
    """Compile the brief graph. `orc` is the Orchestrator (LLM, tools, models); `store` holds long-term memory."""

    def ingest(state: BriefState) -> Dict[str, Any]:
        ctx = orc.load_context(state["ticker"], state.get("use_cache", True), state.get("days_back", 365))
        return {
            "revisions": 0,
            "trace": [{"node": "ingest", "detail": f"{ctx.vector_store.size} chunks, "
                                                   f"{len(ctx.news_items)} news items, "
                                                   f"{len(ctx.metrics)} XBRL metrics"}],
        }

    def planner(state: BriefState, *, store: BaseStore) -> Dict[str, Any]:
        ctx = orc.context(state["ticker"])
        feedback = state.get("feedback", "")
        update: Dict[str, Any] = {}

        # Long-term memory: earlier briefs become a citable source for "what changed"
        memory = state.get("memory")
        if memory is None and store is not None:
            briefs = mem.recent_briefs(store, ctx.ticker)
            memory = {}
            if briefs:
                source = make_source(
                    kind="memory", label=f"MEMORY#{briefs[0]['date']}",
                    title=f"Earlier {ctx.ticker} briefs (latest {briefs[0]['date']})", url="",
                    date=briefs[0]["date"], text=mem.format_memory(briefs), tier="memory",
                )
                memory = {"id": source["id"], "text": source["text"]}
                update["sources"] = {source["id"]: source}
            update["memory"] = memory

        roles = "\n".join(f"- {name}: {desc}" for name, (desc, _) in ROLES.items())
        sections = "\n".join(f"- {title}: {goal}" for title, goal, _ in SECTIONS.values())
        if feedback:
            ask = ("A first draft failed verification for these claims:\n"
                   f"{feedback}\n\nPlan 1-3 targeted tasks that find direct evidence or the "
                   "correct figures for them.")
        elif memory:
            ask = (f"Earlier briefs from memory:\n{memory['text']}\n\nPlan 3-{MAX_TASKS} research "
                   "tasks that together cover every report section, including one that checks "
                   "what changed since the latest earlier brief.")
        else:
            ask = (f"Plan 3-{MAX_TASKS} research tasks that together cover every report section "
                   "(there is no earlier brief, so skip 'Changes Since Last Brief').")
        messages = [
            {"role": "system", "content": (
                f"You are the research lead for an equity analyst brief on {ctx.ticker} "
                f"({ctx.company_info.get('name', '')}, {ctx.company_info.get('sector', '')}). "
                "Split the work between researchers. Respond in JSON: "
                '{"tasks": [{"researcher": "filings|financials|news", "objective": "..."}]}')},
            {"role": "user", "content": (
                f"Researchers:\n{roles}\n\nReport sections:\n{sections}\n\n"
                f"Indexed documents (chunks by form#section): {ctx.inventory()}\n\n{ask}")},
        ]
        try:
            tasks = [t.model_dump() for t in orc.llm.structured(messages, Plan).tasks]
        except Exception as e:
            logger.warning(f"Planner failed, using fallback plan: {e}")
            tasks = ([{"researcher": "filings", "objective": f"Find evidence for: {feedback[:800]}"}]
                     if feedback else DEFAULT_PLAN)
        detail = f"{len(tasks)} tasks" + (" (memory of earlier briefs loaded)" if memory else "")
        return {**update, "plan": tasks,
                "trace": [{"node": "planner", "detail": detail, "tasks": tasks}]}

    def dispatch(state: BriefState) -> List[Send]:
        # Fan out: one researcher run per task, executed in parallel
        return [Send("researcher", {"ticker": state["ticker"], "task": task}) for task in state["plan"]]

    def researcher(payload: Dict[str, Any]) -> Dict[str, Any]:
        task, ticker = payload["task"], payload["ticker"]
        role = task["researcher"]
        description, tool_names = ROLES[role]
        all_tools = orc.tools(ticker)
        system = (
            f"You are the {role} researcher on an equity research team covering {ticker}. "
            f"Your focus: {description}\n"
            "Use the tools to gather evidence for your objective: 2-4 targeted calls, several "
            "at once if they are independent. Then report findings as concise bullet points. "
            "Each bullet states specific facts (figures, dates, names) copied exactly from the "
            "tool results and ends with the id(s) of its sources exactly as shown, e.g. "
            "[S1a2b3c]. Only use ids that appear in tool results. If a search returns nothing, "
            "retry with different filters before concluding evidence is missing."
        )
        notes, sources, calls = orc.llm.run_tools(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Objective: {task['objective']}"}],
            {name: all_tools[name] for name in tool_names},
            max_steps=4,
        )
        node = f"researcher:{role}"
        return {
            "findings": [{**task, "notes": notes}],
            "sources": {s["id"]: s for s in sources},
            "trace": [{"node": node, "objective": task["objective"], **call} for call in calls]
                     or [{"node": node, "objective": task["objective"], "detail": "no tool calls"}],
        }

    def analyst(state: BriefState) -> Dict[str, Any]:
        notes = "\n\n".join(
            f"## {f['researcher']}: {f['objective']}\n{f['notes']}" for f in state["findings"]
        )
        sections = "\n".join(
            f"- {key} ({title}, up to {n} claims): {goal}" for key, (title, goal, n) in SECTIONS.items()
        )
        failed = [c for c in state.get("verification", {}).get("claims", []) if c["status"] != "verified"]
        revision = ""
        if failed:
            revision = "\n\nThese claims from the previous draft failed verification; fix them " \
                       "using the new notes or drop them:\n" + "\n".join(
                           f"- {c['text']} ({c['status']}: {c['reason']})" for c in failed)
        memory = state.get("memory") or {}
        if memory:
            remembered = (f"\n\nEarlier briefs (memory) - cite prior-brief facts with "
                          f"[{memory['id']}], and current facts with new sources:\n{memory['text']}")
        else:
            remembered = "\n\nThere is no earlier brief: leave 'changes' empty."
        messages = [
            {"role": "system", "content": (
                f"You are a senior equity analyst writing a brief on {state['ticker']}. Use ONLY "
                "facts in the research notes. Each claim is one specific sentence (figures, dates, "
                "names) citing the source ids from the notes that directly support it. Copy figures "
                "exactly as they appear in the notes. Never cite an id that is not in the notes. "
                "XBRL figures are reported results, never projections. If the notes have no "
                "evidence for a section, return an empty list for it; never write claims about "
                "missing evidence. "
                'Respond in JSON: {"overview": [{"text": "...", "sources": ["S1a2b3c"]}], '
                '"drivers": [...], "risks": [...], "events": [...], "outlook": [...], '
                '"changes": [...]}')},
            {"role": "user", "content": f"Sections:\n{sections}\n\nResearch notes:\n{notes}"
                                        f"{remembered}{revision}"},
        ]
        try:
            draft = orc.llm.structured(messages, Draft).model_dump()
        except Exception as e:
            logger.error(f"Analyst failed: {e}")
            draft = Draft().model_dump()
        total = sum(len(v) for v in draft.values())
        return {"draft": draft, "trace": [{"node": "analyst", "detail": f"{total} claims drafted"}]}

    def verifier(state: BriefState) -> Dict[str, Any]:
        sources = state.get("sources", {})
        claims = [{"section": key, **claim} for key in SECTIONS for claim in state["draft"].get(key, [])]
        checked = [check_claim(claim, sources) for claim in claims]

        # LLM judge for claims that passed the deterministic checks
        to_judge = [c for c in checked if c["status"] == "ok"]
        verdicts: Dict[int, Judgement] = {}
        if to_judge:
            listing = "\n\n".join(
                f"Claim {i}: {c['text']}\n" + "\n".join(
                    f"[{sid}] {sources[sid]['text'][:1500]}" for sid in c["sources"])
                for i, c in enumerate(to_judge)
            )
            messages = [
                {"role": "system", "content": (
                    "You are a strict fact checker. For each claim, decide whether its cited "
                    "sources directly support every specific in it (figures, dates, names, "
                    "direction of change). Paraphrase is fine; specifics the sources do not "
                    "state are not, and neither is misframing them (e.g. calling reported "
                    'results "projected"). Respond in JSON: {"judgements": [{"index": 0, '
                    '"supported": true, "reason": "short reason"}]}')},
                {"role": "user", "content": listing},
            ]
            try:
                verdicts = {j.index: j for j in orc.llm.structured(messages, Judgements).judgements}
            except Exception as e:
                logger.error(f"Judge failed: {e}")
        for i, claim in enumerate(to_judge):
            verdict = verdicts.get(i)
            claim["status"] = "verified" if verdict and verdict.supported else "unsupported"
            claim["reason"] = verdict.reason if verdict else "no verdict from judge"

        verified = sum(c["status"] == "verified" for c in checked)
        coverage = verified / len(checked) if checked else 0.0
        revisions = state.get("revisions", 0)
        revise = coverage < MIN_COVERAGE and revisions < MAX_REVISIONS
        previous = state.get("verification")
        feedback = "\n".join(
            f"- [{c['section']}] {c['text']} -> {c['status']}: {c['reason']}"
            for c in checked if c["status"] != "verified"
        )
        return {
            "verification": {
                "claims": checked,
                "verified": verified,
                "total": len(checked),
                "coverage": coverage,
                "first_pass_coverage": previous["first_pass_coverage"] if previous else coverage,
                "revisions": revisions + int(revise),
            },
            "feedback": feedback if revise else "",
            "revisions": revisions + int(revise),
            "trace": [{"node": "verifier", "detail": f"{verified}/{len(checked)} verified "
                                                     f"({coverage:.0%}){' - revising' if revise else ''}"}],
        }

    def score(state: BriefState) -> Dict[str, Any]:
        summaries = build_summaries(state["verification"]["claims"], state.get("sources", {}))
        return {"summaries": summaries, "scores": orc.score(state["ticker"], summaries)}

    def review(state: BriefState) -> Dict[str, Any]:
        if not state.get("review"):
            return {"approved": True}
        rating = state["scores"]["rating"]
        verification = state["verification"]
        decision = interrupt({
            "ticker": state["ticker"],
            "rating": rating.get("rating"),
            "score": rating.get("composite_score"),
            "price_target": rating.get("price_target"),
            "coverage": verification["coverage"],
            "claims": [{"section": c["section"], "text": c["text"], "status": c["status"]}
                       for c in verification["claims"]],
        })
        return {"approved": bool(decision.get("approved", True)), "review_note": decision.get("note", "")}

    def report(state: BriefState, *, store: BaseStore) -> Dict[str, Any]:
        if store is None:
            return {"result": orc.finish_report(state)}
        ticker = state["ticker"]
        # Score earlier calls first, so the report shows the track record so far
        scored = mem.score_entries(mem.ledger(store, ticker), orc.price_closes(ticker))
        track_record = mem.summarize(scored)
        result = orc.finish_report(state, track_record=track_record)

        # Then remember this brief and log its price calls
        today = date.today().isoformat()
        price = result["price_metrics"].get("current_price") or 0.0
        rating, prediction = state["scores"]["rating"], state["scores"]["prediction"]
        mem.remember_brief(store, mem.brief_record(ticker, today, price, rating, state["verification"]))
        entries = mem.ledger_entries_for(ticker, today, price, rating, prediction)
        mem.record_predictions(store, entries)
        return {
            "result": {**result, "track_record": track_record},
            "trace": [{"node": "report", "detail": f"saved brief to memory, logged {len(entries)} "
                                                   f"predictions; {mem.format_track_record(track_record)}"}],
        }

    graph = StateGraph(BriefState)
    for name, node in [("ingest", ingest), ("planner", planner), ("researcher", researcher),
                       ("analyst", analyst), ("verifier", verifier), ("score", score),
                       ("review", review), ("report", report)]:
        graph.add_node(name, node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "planner")
    graph.add_conditional_edges("planner", dispatch, ["researcher"])
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "verifier")
    graph.add_conditional_edges("verifier", route_after_verify, ["planner", "score"])
    graph.add_edge("score", "review")
    graph.add_conditional_edges("review", route_after_review, ["report", END])
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def build_chat_graph(orc, checkpointer=None, store=None, history: int = 12):
    """
    Follow-up Q&A. Short-term memory is the checkpointer: each thread_id's
    messages persist in SQLite, so a question can refer to earlier answers.
    Long-term memory (earlier briefs, prediction track record) comes from the store.
    """

    def chat(state: ChatState, *, store: BaseStore) -> Dict[str, Any]:
        ticker = state["ticker"]
        ctx = orc.context(ticker)
        history_notes = ""
        if store is not None:
            briefs = mem.recent_briefs(store, ticker)
            if briefs:
                scored = mem.score_entries(mem.ledger(store, ticker), orc.price_closes(ticker))
                history_notes = (f"\n\nEarlier briefs on {ticker} (long-term memory):\n"
                                 f"{mem.format_memory(briefs)}\nPrediction track record: "
                                 f"{mem.format_track_record(mem.summarize(scored))}")
        system = (
            f"You are an equity research assistant for {ticker}. Answer using the tools and "
            "cite every fact with the source id exactly as it appears in tool results, e.g. "
            "[S1a2b3c]. The brief's numbered references like [3:10-K#MD&A] are not valid "
            "citations here: look facts up with the tools. Be concise. "
            f"The latest analyst brief, for context:\n{ctx.report[:6000] or 'none generated yet'}"
            f"{history_notes}"
        )
        # ponytail: last-N message window; summarize older turns if chats run long
        recent = [{"role": m["role"], "content": m["content"]} for m in state["messages"][-history:]]
        answer, sources, calls = orc.llm.run_tools(
            [{"role": "system", "content": system}] + recent, orc.tools(ticker), max_steps=4)
        # Show the sources it cited; if it cited none, the ones it consulted
        unique = {s["id"]: s for s in sources}
        shown = [s for s in unique.values() if s["id"] in answer] or list(unique.values())[:8]
        return {"messages": [{
            "role": "assistant",
            "content": answer,
            "sources": [{"id": s["id"], "label": s["label"], "title": s["title"], "url": deep_link(s)}
                        for s in shown],
            "tool_calls": [{"tool": c["tool"], "args": c["args"]} for c in calls],
        }]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)
