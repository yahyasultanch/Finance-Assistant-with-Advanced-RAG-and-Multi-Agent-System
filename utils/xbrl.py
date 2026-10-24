"""Key financial metrics from SEC XBRL company facts."""
import re
from typing import Any, Dict, List, Optional

# Metric name -> (candidate us-gaap concepts in preference order, kind)
# kind "flow" = reported over a period (annual + quarterly), "instant" = balance at a date
KEY_METRICS = {
    "Revenue": ([
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ], "flow"),
    "Gross Profit": (["GrossProfit"], "flow"),
    "Operating Income": (["OperatingIncomeLoss"], "flow"),
    "Net Income": (["NetIncomeLoss"], "flow"),
    "Diluted EPS": (["EarningsPerShareDiluted"], "flow"),
    "R&D Expense": (["ResearchAndDevelopmentExpense"], "flow"),
    "Operating Cash Flow": (["NetCashProvidedByUsedInOperatingActivities"], "flow"),
    "Cash and Equivalents": (["CashAndCashEquivalentsAtCarryingValue"], "instant"),
    "Long-Term Debt": (["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
    "Shareholders' Equity": (["StockholdersEquity"], "instant"),
}

# XBRL "frame" labels are SEC's de-duplicated, calendar-aligned values:
# CY2025 = annual, CY2025Q3 = quarter, CY2025Q3I = balance at quarter end
FRAME_PATTERNS = {
    "annual": re.compile(r"^CY(\d{4})$"),
    "quarter": re.compile(r"^CY(\d{4})Q(\d)$"),
    "instant": re.compile(r"^CY(\d{4})Q(\d)I$"),
}


def _framed(entries: List[Dict[str, Any]], period: str) -> Dict[str, Dict[str, Any]]:
    """Map frame label -> entry for one period type."""
    pattern = FRAME_PATTERNS[period]
    return {e["frame"]: e for e in entries if pattern.match(e.get("frame", ""))}


def _year_ago(frame: str) -> str:
    """CY2025Q3 -> CY2024Q3, CY2025 -> CY2024."""
    return re.sub(r"\d{4}", lambda m: str(int(m.group()) - 1), frame, count=1)


def _point(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame": entry["frame"],
        "end": entry["end"],
        "value": entry["val"],
        "form": entry.get("form", ""),
        "accession_number": entry.get("accn", ""),
        "filed": entry.get("filed", ""),
    }


def _latest_with_prior(framed: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Latest framed value plus the same period a year earlier and YoY change."""
    if not framed:
        return None
    latest = framed[max(framed, key=lambda f: framed[f]["end"])]
    prior = framed.get(_year_ago(latest["frame"]))
    result = {"latest": _point(latest), "prior": _point(prior) if prior else None, "yoy_pct": None}
    if prior and prior["val"]:
        result["yoy_pct"] = round((latest["val"] - prior["val"]) / abs(prior["val"]) * 100, 1)
    return result


def key_metrics(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract key metrics from a companyfacts payload.

    For each metric, the candidate concept with the most recent data wins
    (companies switch concepts over time, e.g. Revenues vs
    RevenueFromContractWithCustomer...). Flow metrics get the latest annual
    and latest quarterly value; instants get the latest balance. Each comes
    with the year-earlier value and YoY change.

    Returns:
        List of dicts: name, concept, unit, and 'annual'/'quarter'/'instant' series
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    metrics = []

    for name, (concepts, kind) in KEY_METRICS.items():
        best = None
        for concept in concepts:
            units = gaap.get(concept, {}).get("units", {})
            if not units:
                continue
            unit, entries = next(iter(units.items()))
            last_end = max((e["end"] for e in entries), default="")
            if best is None or last_end > best[3]:
                best = (concept, unit, entries, last_end)
        if best is None:
            continue

        concept, unit, entries, _ = best
        metric = {"name": name, "concept": concept, "unit": unit}
        periods = ["annual", "quarter"] if kind == "flow" else ["instant"]
        for period in periods:
            series = _latest_with_prior(_framed(entries, period))
            if series:
                metric[period] = series
        if len(metric) > 3:
            metrics.append(metric)

    return metrics


def format_value(value: float, unit: str) -> str:
    """$46.74B (46,743,000,000) for USD; $1.08 for per-share values."""
    if unit == "USD":
        for scale, suffix in ((1e9, "B"), (1e6, "M")):
            if abs(value) >= scale:
                return f"${value / scale:,.2f}{suffix} ({value:,.0f})"
        return f"${value:,.0f}"
    if unit == "USD/shares":
        return f"${value:,.2f}"
    return f"{value:,} {unit}"
