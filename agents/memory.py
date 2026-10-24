"""Long-term memory and the prediction ledger, kept in a LangGraph Store.

The checkpointer remembers one run (or one chat thread); the store remembers
across runs. Two namespaces per ticker:
  ("memory", TICKER): one record per published brief (rating, target,
      verified claims), so the next brief can say what changed
  ("ledger", TICKER): every price call with the price when it was made,
      scored against actual prices once its horizon passes
Keys are dates, so re-running a ticker on the same day replaces that day's
records instead of double-counting them.
"""
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ledger entry kind -> horizon in days
HORIZONS = {"7d": 7, "12m": 365, "rating_target": 365}


def brief_record(ticker: str, made_on: str, price: float, rating: Dict[str, Any],
                 verification: Dict[str, Any]) -> Dict[str, Any]:
    """What to remember about a published brief."""
    claims: Dict[str, List[str]] = {}
    for claim in verification["claims"]:
        if claim["status"] == "verified":
            claims.setdefault(claim["section"], []).append(claim["text"])
    return {
        "ticker": ticker,
        "date": made_on,
        "price": price,
        "rating": rating.get("rating"),
        "score": rating.get("composite_score"),
        "price_target": rating.get("price_target"),
        "coverage": verification["coverage"],
        "claims": claims,
    }


def remember_brief(store, record: Dict[str, Any]) -> None:
    store.put(("memory", record["ticker"]), record["date"], record)


def recent_briefs(store, ticker: str, limit: int = 3, before: Optional[str] = None) -> List[Dict[str, Any]]:
    """Most recent briefs first; `before` excludes records from that date on."""
    records = [item.value for item in store.search(("memory", ticker), limit=1000)]
    records = [r for r in records if before is None or r["date"] < before]
    return sorted(records, key=lambda r: r["date"], reverse=True)[:limit]


def format_memory(briefs: Sequence[Dict[str, Any]], claims_per_section: int = 2) -> str:
    """Prior briefs as prompt text (also the text of the citable memory source)."""
    blocks = []
    for b in briefs:
        lines = [
            f"Brief of {b['date']}: rating {b['rating']} (score {b['score']}), "
            f"12M price target {b['price_target']}, price then {b['price']:.2f}, "
            f"citation coverage {b['coverage']:.0%}."
        ]
        for section, texts in b["claims"].items():
            lines += [f"  [{section}] {t}" for t in texts[:claims_per_section]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ledger_entries_for(ticker: str, made_on: str, price: float, rating: Dict[str, Any],
                       prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The price calls one brief makes, ready to store (`prediction` is PredictiveEngine output)."""
    horizons = prediction.get("predictions", {})
    targets = {
        "7d": horizons.get("short_term", {}).get("target_price"),
        "12m": horizons.get("long_term", {}).get("target_price"),
        "rating_target": rating.get("price_target"),
    }
    due = date.fromisoformat(made_on)
    return [
        {"ticker": ticker, "kind": kind, "made_on": made_on, "price": price, "target": target,
         "due": (due + timedelta(days=HORIZONS[kind])).isoformat()}
        for kind, target in targets.items() if target and price
    ]


def record_predictions(store, entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        store.put(("ledger", entry["ticker"]), f"{entry['made_on']}:{entry['kind']}", entry)


def ledger(store, ticker: str) -> List[Dict[str, Any]]:
    return sorted((item.value for item in store.search(("ledger", ticker), limit=10000)),
                  key=lambda e: (e["made_on"], e["kind"]))


def score_entries(entries: List[Dict[str, Any]], closes: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """
    Score each call against actual closes ((iso date, close) pairs, ascending).

    Due calls use the first close on or after the due date: error is
    (target - actual) / actual, and a direction hit means the target and the
    actual move went the same way from the price when the call was made.
    Calls not yet due show progress against the latest close.
    """
    scored = []
    latest_date, latest_close = closes[-1] if closes else ("", None)
    for entry in entries:
        result = {**entry, "status": "pending", "actual": None, "error_pct": None, "direction_hit": None}
        actual = next((close for day, close in closes if day >= entry["due"]), None)
        if actual is not None:
            predicted_move = entry["target"] - entry["price"]
            actual_move = actual - entry["price"]
            result.update(
                status="scored",
                actual=round(actual, 2),
                error_pct=round((entry["target"] - actual) / actual * 100, 2),
                direction_hit=(predicted_move > 0) == (actual_move > 0),
            )
        elif latest_close is not None:
            result["actual"] = round(latest_close, 2)  # progress so far
        scored.append(result)
    return scored


def summarize(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Hit rate and error over scored calls, overall and by kind."""
    def stats(rows):
        if not rows:
            return {"scored": 0}
        return {
            "scored": len(rows),
            "direction_hit_rate": sum(r["direction_hit"] for r in rows) / len(rows),
            "mean_abs_error_pct": sum(abs(r["error_pct"]) for r in rows) / len(rows),
        }
    done = [r for r in scored if r["status"] == "scored"]
    return {
        "total": len(scored),
        "pending": len(scored) - len(done),
        **stats(done),
        "by_kind": {kind: stats([r for r in done if r["kind"] == kind]) for kind in HORIZONS},
    }


def format_track_record(summary: Dict[str, Any]) -> str:
    """One line for prompts and reports."""
    if not summary.get("scored"):
        return f"No calls have reached their horizon yet ({summary.get('pending', 0)} pending)."
    return (
        f"{summary['scored']} past calls scored: direction right {summary['direction_hit_rate']:.0%} "
        f"of the time, mean absolute error {summary['mean_abs_error_pct']:.1f}% "
        f"({summary['pending']} still pending)."
    )
